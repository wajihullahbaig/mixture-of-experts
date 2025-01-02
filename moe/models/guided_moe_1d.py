import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Optional, Tuple
from models.base_moe_1d import Expert1D  

class GuidedGatingNetwork1D(nn.Module):
    """Gating network with label-guided routing for 1D inputs"""
    
    def __init__(self, input_size: int, num_experts: int, hidden_size: int, 
                 expert_label_map: Dict[int, List[int]]):
        super().__init__()
        self.expert_label_map = expert_label_map
        self.num_experts = num_experts
        
        # Calculate base probabilities
        total_labels = sum(len(labels) for labels in expert_label_map.values())
        self.base_probs = {
            expert_idx: len(labels) / total_labels 
            for expert_idx, labels in expert_label_map.items()
        }
        
        # Main network
        self.network = nn.Sequential(
            nn.BatchNorm1d(input_size),
            nn.Linear(input_size, hidden_size),
            nn.ReLU(),
            nn.BatchNorm1d(hidden_size),
            nn.Dropout(0.3),
            nn.Linear(hidden_size, num_experts)
        )
        
        # Temperature parameters
        self.train_temperature = nn.Parameter(torch.ones(1))
        self.register_buffer('eval_temperature', torch.ones(1) * 2.0)
        
        self._init_weights()
    
    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight)
                nn.init.constant_(m.bias, 0)
    
    def compute_soft_masks(self, labels: torch.Tensor) -> torch.Tensor:
        """Compute soft assignment masks with proper probability scaling"""
        batch_size = labels.size(0)
        device = labels.device
        
        # Create assignment matrix with scaled background probabilities
        base_probs = torch.tensor([self.base_probs[i] for i in range(self.num_experts)], 
                                device=device)
        soft_mask = base_probs.unsqueeze(0).expand(batch_size, -1)
        
        # Create label-to-expert mapping tensor
        label_expert_map = torch.zeros(max(max(self.expert_label_map.values())) + 1, 
                                     self.num_experts, device=device)
        
        for expert_idx, assigned_labels in self.expert_label_map.items():
            expert_prob = self.base_probs[expert_idx]
            label_expert_map[torch.tensor(assigned_labels, device=device), expert_idx] = expert_prob
        
        # Compute expert assignments with proper scaling
        expert_assignments = label_expert_map[labels]
        soft_mask = torch.where(expert_assignments > 0, 
                              expert_assignments,
                              soft_mask)
        
        # Normalize to ensure probabilities sum to 1
        soft_mask = F.normalize(soft_mask, p=1, dim=1)
        
        return soft_mask
    
    def forward(self, x: torch.Tensor, labels: Optional[torch.Tensor] = None) -> torch.Tensor:
        logits = self.network(x)
        
        # Add small noise during training for robustness
        if self.training:
            noise = torch.randn_like(logits) * 0.1
            logits = logits + noise
        
        if self.training and labels is not None:
            # Get soft assignment masks
            soft_masks = self.compute_soft_masks(labels)
            
            # Combine network output with soft guidance
            routing_weights = F.softmax(logits / self.train_temperature, dim=1)
            
            # Interpolate between learned and guided routing
            mixing_factor = torch.sigmoid(self.train_temperature)
            final_weights = mixing_factor * routing_weights + (1 - mixing_factor) * soft_masks
        else:
            # During inference, use temperature-scaled softmax
            final_weights = F.softmax(logits / self.eval_temperature, dim=1)
        
        return final_weights

class GuidedMoE1D(nn.Module):
    """Guided Mixture of Experts with 1D architecture"""
    
    def __init__(self, input_size: int, hidden_size: int, output_size: int, num_experts: int,
                 expert_label_assignments: Dict[int, List[int]]):
        super().__init__()
        self.num_experts = num_experts
        self.expert_label_assignments = expert_label_assignments
        
        # Initialize experts
        self.experts = nn.ModuleList([
            Expert1D(input_size, hidden_size, output_size)
            for _ in range(num_experts)
        ])
        
        # Initialize guided gating network
        self.gating_network = GuidedGatingNetwork1D(
            input_size,
            num_experts,
            hidden_size,
            expert_label_assignments
        )
        
        # Store loss components
        self.loss_components = {}
    
    def forward(self, x: torch.Tensor, labels: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, torch.Tensor, List[torch.Tensor]]:
        # Get expert weights using label guidance during training
        expert_weights = self.gating_network(x, labels)
        
        # Get expert outputs
        expert_outputs = []
        expert_l2_losses = []
        
        for expert in self.experts:
            output, l2_loss = expert(x)
            expert_outputs.append(output)
            expert_l2_losses.append(l2_loss)
        
        expert_outputs = torch.stack(expert_outputs)
        expert_outputs = expert_outputs.permute(1, 0, 2)
        
        # Combine expert outputs
        final_output = torch.sum(expert_outputs * expert_weights.unsqueeze(-1), dim=1)
        
        return final_output, expert_weights, expert_l2_losses
    
    def compute_loss(self, final_output: torch.Tensor, target: torch.Tensor, 
                    expert_weights: torch.Tensor, expert_l2_losses: List[torch.Tensor]) -> torch.Tensor:
        # Classification loss
        ce_loss = F.cross_entropy(final_output, target)
        
        # Expert assignment loss
        batch_size = target.size(0)
        device = target.device
        
        # Create label-to-expert mapping tensor
        label_expert_map = torch.zeros(max(max(self.expert_label_assignments.values())) + 1, 
                                     self.num_experts, device=device)
        for expert_idx, assigned_labels in self.expert_label_assignments.items():
            label_expert_map[torch.tensor(assigned_labels, device=device), expert_idx] = 1
        
        # Get target expert assignments
        target_assignments = label_expert_map[target]
        
        # Compute assignment loss
        expert_assignment_loss = F.binary_cross_entropy(
            expert_weights,
            target_assignments,
            reduction='mean'
        )
        
        # Expert diversity loss (entropy)
        diversity_loss = -torch.mean(
            torch.sum(expert_weights * torch.log(expert_weights + 1e-6), dim=1)
        )
        
        # Load balancing loss
        mean_expert_weights = expert_weights.mean(0)
        uniform_weights = torch.ones_like(mean_expert_weights) / self.num_experts
        balance_loss = F.kl_div(
            mean_expert_weights.log(),
            uniform_weights,
            reduction='batchmean'
        )
        
        # Combine L2 losses
        total_l2_loss = torch.stack(expert_l2_losses).sum()
        
        # Combine all losses with updated weights
        total_loss = (
            0.3 * ce_loss +                  # Classification
            0.3 * expert_assignment_loss +    # Expert assignment
            0.1 * diversity_loss +           # Expert diversity
            0.2 * balance_loss +             # Load balancing
            0.1 * total_l2_loss              # Regularization
        )
        
        # Store components for monitoring
        self.loss_components = {
            'ce_loss': ce_loss.item(),
            'expert_assignment_loss': expert_assignment_loss.item(),
            'diversity_loss': diversity_loss.item(),
            'balance_loss': balance_loss.item(),
            'l2_loss': total_l2_loss.item(),
            'total_loss': total_loss.item()
        }
        
        return total_loss