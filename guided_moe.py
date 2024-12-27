import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Optional, Tuple

class GuidedExpert(nn.Module):
    """Individual expert network"""
    
    def __init__(self, input_size: int, hidden_size: int, output_size: int, dropout_rate: float = 0.3):
        """
        Initialize expert network
        
        Args:
            input_size: Input feature dimension
            hidden_size: Hidden layer dimension
            output_size: Output dimension (num classes)
            dropout_rate: Dropout probability
        """
        super().__init__()
        self.network = nn.Sequential(
            nn.BatchNorm1d(input_size),
            nn.Linear(input_size, hidden_size),
            nn.ReLU(),
            nn.BatchNorm1d(hidden_size),
            nn.Dropout(dropout_rate),
            nn.Linear(hidden_size, hidden_size // 2),
            nn.ReLU(),
            nn.BatchNorm1d(hidden_size // 2),
            nn.Dropout(dropout_rate),
            nn.Linear(hidden_size // 2, output_size)
        )
        
        # L2 regularization weight
        self.l2_reg = 0.01
        
        # Initialize weights
        self._init_weights()
    
    def _init_weights(self):
        """Initialize network weights"""
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight)
                nn.init.constant_(m.bias, 0)
    
    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Forward pass with L2 regularization"""
        output = self.network(x)
        l2_loss = sum(p.pow(2.0).sum() for p in self.parameters()) * self.l2_reg
        return output, l2_loss

class GuidedGatingNetwork(nn.Module):
    """Gating network with label-guided routing"""
    
    def __init__(self, input_size: int, num_experts: int, hidden_size: int, 
                 expert_label_map: Dict[int, List[int]]):
        super().__init__()
        self.expert_label_map = expert_label_map
        
        # Main network for computing expert weights
        super().__init__()
        self.network = nn.Sequential(
            nn.BatchNorm1d(input_size),
            nn.Linear(input_size, hidden_size),
            nn.ReLU(),
            nn.BatchNorm1d(hidden_size),
            nn.Dropout(0.3),
            nn.Linear(hidden_size, hidden_size // 2),
            nn.ReLU(),
            nn.BatchNorm1d(hidden_size // 2),
            nn.Dropout(0.3),
            nn.Linear(hidden_size // 2, num_experts)
        )
        
        # Temperature parameter for routing
        self.routing_temperature = nn.Parameter(torch.ones(1)*0.75)
        
        # Initialize weights
        self._init_weights()
    
    def _init_weights(self):
        """Initialize network weights"""
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight)
                nn.init.constant_(m.bias, 0)
    
    def compute_soft_masks(self, labels: Optional[torch.Tensor] = None) -> Optional[torch.Tensor]:
        """Compute soft assignment masks based on labels"""
        if labels is None:
            return None
        
        batch_size = labels.size(0)
        num_experts = len(self.expert_label_map)
        device = labels.device
        
        # Create assignment matrix
        soft_mask = torch.full((batch_size, num_experts), 0.1 / (num_experts - 1), device=device)
        
        # Create label-to-expert mapping tensor
        label_expert_map = torch.zeros(max(max(self.expert_label_map.values())) + 1, 
                                     num_experts, device=device)
        for expert_idx, assigned_labels in self.expert_label_map.items():
            label_expert_map[torch.tensor(assigned_labels, device=device), expert_idx] = 1
        
        # Compute expert assignments
        expert_assignments = label_expert_map[labels]
        soft_mask = torch.where(expert_assignments == 1, 
                              torch.tensor(0.9, device=device),
                              soft_mask)
        
        return soft_mask
    
    def forward(self, x: torch.Tensor, labels: Optional[torch.Tensor] = None) -> torch.Tensor:
        """Forward pass with label guidance during training"""
        logits = self.network(x)
        
        if self.training and labels is not None:
            # Get soft assignment masks
            soft_masks = self.compute_soft_masks(labels)
            
            # Combine network output with soft guidance
            routing_weights = F.softmax(logits / self.routing_temperature, dim=1)
            
            # Interpolate between learned and guided routing
            mixing_factor = torch.sigmoid(self.routing_temperature)
            final_weights = mixing_factor * routing_weights + (1 - mixing_factor) * soft_masks
            
            return final_weights
        else:
            # During inference, use pure network output
            return F.softmax(logits / self.routing_temperature, dim=1)

class GuidedMixtureOfExperts(nn.Module):
    """Guided Mixture of Experts with label-based routing"""
    
    def __init__(self, input_size: int, hidden_size: int, output_size: int, num_experts: int, 
                 expert_label_assignments: Dict[int, List[int]]):
        super().__init__()
        self.num_experts = num_experts
        self.expert_label_assignments = expert_label_assignments
        
        # Initialize experts
        self.experts = nn.ModuleList([
            GuidedExpert(input_size, hidden_size, output_size)
            for _ in range(num_experts)
        ])
        
        # Initialize gating network with label map
        self.gating_network = GuidedGatingNetwork(
            input_size, 
            num_experts,
            hidden_size,
            expert_label_assignments
        )
        
        # Store loss components
        self.loss_components = {}
    
    def forward(self, x: torch.Tensor, labels: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, torch.Tensor, List[torch.Tensor]]:
        """Forward pass with optional label guidance"""
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
        """Compute loss with expert assignment guidance"""
        # Classification loss
        ce_loss = F.cross_entropy(final_output, target)
        
        # Expert assignment loss - using label map
        batch_size = target.size(0)
        num_experts = len(self.expert_label_assignments)
        device = target.device
        
        # Create label-to-expert mapping tensor
        label_expert_map = torch.zeros(max(max(self.expert_label_assignments.values())) + 1, 
                                     num_experts, device=device)
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
        
        # Combine L2 losses
        total_l2_loss = torch.stack(expert_l2_losses).sum()
        
        # Combine all losses
        total_loss = (
            ce_loss + 
            0.3 * expert_assignment_loss +  # Higher weight for assignment
            0.5 * diversity_loss +         # Small weight for diversity
            0.2* total_l2_loss          # Very small weight for L2
        )
        
        # Store components for monitoring
        self.loss_components = {
            'ce_loss': ce_loss.item(),
            'expert_assignment_loss': expert_assignment_loss.item(),
            'diversity_loss': diversity_loss.item(),
            'l2_loss': total_l2_loss.item(),
            'total_loss': total_loss.item()
        }
        
        return total_loss

def create_label_assignments(num_classes: int, num_experts: int) -> Dict[int, List[int]]:
    """Create balanced label assignments for experts"""
    assignments = {}
    base_labels_per_expert = num_classes // num_experts
    remaining_labels = num_classes % num_experts
    
    current_idx = 0
    for expert_idx in range(num_experts):
        num_labels = base_labels_per_expert + (1 if expert_idx < remaining_labels else 0)
        assignments[expert_idx] = list(range(current_idx, current_idx + num_labels))
        current_idx += num_labels
    
    return assignments

# Training and evaluation functions remain the same
def train_step(model: GuidedMixtureOfExperts, batch: Tuple[torch.Tensor, torch.Tensor], 
               optimizer: torch.optim.Optimizer) -> Dict:
    """Single training step"""
    model.train()
    inputs, targets = batch
    optimizer.zero_grad()
    
    # Forward pass with label guidance
    outputs, expert_weights, expert_l2_losses = model(inputs, targets)
    loss = model.compute_loss(outputs, targets, expert_weights, expert_l2_losses)
    
    # Backward pass
    loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
    optimizer.step()
    
    # Calculate accuracy
    predictions = outputs.argmax(dim=1)
    accuracy = (predictions == targets).float().mean().item()
    
    return {
        'loss': loss.item(),
        'accuracy': accuracy,
        'expert_weights': expert_weights.detach(),
        'loss_components': model.loss_components
    }

@torch.no_grad()
def evaluate_step(model: GuidedMixtureOfExperts, batch: Tuple[torch.Tensor, torch.Tensor]) -> Dict:
    """Single evaluation step"""
    model.eval()
    inputs, targets = batch
    
    # Forward pass without label guidance
    outputs, expert_weights, expert_l2_losses = model(inputs)
    loss = model.compute_loss(outputs, targets, expert_weights, expert_l2_losses)
    
    # Calculate accuracy
    predictions = outputs.argmax(dim=1)
    accuracy = (predictions == targets).float().mean().item()
    
    return {
        'loss': loss.item(),
        'accuracy': accuracy,
        'expert_weights': expert_weights,
        'predictions': predictions,
        'targets': targets,
        'loss_components': model.loss_components
    }