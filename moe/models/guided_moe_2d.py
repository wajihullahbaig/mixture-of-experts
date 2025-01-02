import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Optional, Tuple
from models.base_moe_2d import Expert2D
from moe.models.guided_moe_1d import GuidedMoE1D  

class GuidedGatingNetwork2D(nn.Module):
    """CNN-based gating network with label-guided routing"""
    
    def __init__(self, input_channels: int, num_experts: int, 
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
        
        # Feature extraction using CNN
        self.features = nn.Sequential(
            # First conv block
            nn.Conv2d(input_channels, 32, 3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2),
            
            # Second conv block
            nn.Conv2d(32, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2)
        )
        
        # Calculate feature size
        self.feature_size = self._get_feature_size(input_channels)
        
        # Gating network
        self.gate = nn.Sequential(
            nn.Flatten(),
            nn.Linear(self.feature_size, 256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, num_experts)
        )
        
        # Temperature parameters
        self.train_temperature = nn.Parameter(torch.ones(1))
        self.register_buffer('eval_temperature', torch.ones(1) * 2.0)
        
        self._init_weights()
    
    def _get_feature_size(self, input_channels: int) -> int:
        if input_channels == 1:  # MNIST
            return 64 * 7 * 7
        else:  # CIFAR
            return 64 * 8 * 8
    
    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
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
        features = self.features(x)
        logits = self.gate(features)
        
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

class GuidedMoE2D(nn.Module):
    """Guided Mixture of Experts with CNN architecture"""
    
    def __init__(self, input_channels: int, num_classes: int, num_experts: int,
                 expert_label_assignments: Dict[int, List[int]]):
        super().__init__()
        self.num_experts = num_experts
        self.expert_label_assignments = expert_label_assignments
        
        # Initialize CNN experts
        self.experts = nn.ModuleList([
            Expert2D(input_channels, num_classes)
            for _ in range(num_experts)
        ])
        
        # Initialize guided gating network
        self.gating_network = GuidedGatingNetwork2D(
            input_channels,
            num_experts,
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
        # Use same loss computation as 1D version
        # This ensures consistent loss calculation between architectures
        return GuidedMoE1D.compute_loss(self, final_output, target, expert_weights, expert_l2_losses)