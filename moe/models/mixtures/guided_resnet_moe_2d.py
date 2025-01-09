from typing import Dict, List, Optional, Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F

from moe.interfaces.moe_interface import MoEInterface
from moe.models.experts.experts_resnet_2d import ResNet18Expert2D
from moe.models.gates.gates_2d import GuidedGating2D

class GuidedResNetMoE2D(MoEInterface):
    """Guided Mixture of Experts implementation using ResNet18 experts for 2D inputs"""
    
    def __init__(self, 
                 input_channels: int, 
                 num_classes: int, 
                 num_experts: int,
                 expert_label_assignments: Dict[int, List[int]],
                 dropout_rate: float = 0.3):
        """
        Initialize guided ResNet18-based MoE model
        
        Args:
            input_channels: Number of input channels (e.g. 3 for RGB)
            num_classes: Number of output classes
            num_experts: Number of ResNet18 experts
            expert_label_assignments: Dictionary mapping expert indices to lists of class labels
            dropout_rate: Dropout rate for regularization
        """
        super().__init__()
        self._num_experts = num_experts
        self.expert_label_assignments = expert_label_assignments
        
        # Initialize ResNet18 experts
        self.experts = nn.ModuleList([
            ResNet18Expert2D(input_channels, num_classes, dropout_rate)
            for _ in range(num_experts)
        ])
        
        # Initialize guided gating network
        self.gating_network = GuidedGating2D(
            input_channels,
            num_experts,
            expert_label_assignments
        )
        
        # Storage for metrics
        self.loss_components = {}
        self.metrics = {}
    
    @property
    def num_experts(self) -> int:
        return self._num_experts
    
    def forward(self, x: torch.Tensor, labels: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, torch.Tensor, List[torch.Tensor]]:
        """
        Forward pass through the MoE model
        
        Args:
            x: Input tensor of shape [batch_size, channels, height, width]
            labels: Optional tensor of labels for guided routing
            
        Returns:
            Tuple containing:
            - Final output tensor
            - Expert weights tensor
            - List of expert L2 losses
        """
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
        
        # Combine expert outputs using routing weights
        final_output = torch.sum(expert_outputs * expert_weights.unsqueeze(-1), dim=1)
        
        # Store routing entropy for monitoring
        if self.training:
            entropy = -(expert_weights * torch.log(expert_weights + 1e-6)).sum(dim=-1).mean()
            self.metrics['routing_entropy'] = entropy.item()
        
        return final_output, expert_weights, expert_l2_losses
    
    def compute_loss(self, final_output: torch.Tensor, target: torch.Tensor, 
                    expert_weights: torch.Tensor, expert_l2_losses: List[torch.Tensor]) -> torch.Tensor:
        """
        Compute the total loss for training with guided expert assignment
        
        Args:
            final_output: Combined output from all experts
            target: Ground truth labels
            expert_weights: Routing weights for each expert
            expert_l2_losses: L2 regularization losses from each expert
            
        Returns:
            Total loss value
        """
        # Classification loss
        ce_loss = F.cross_entropy(final_output, target)
        
        # Expert assignment loss based on label assignments
        label_expert_map = torch.zeros(
            max(max(self.expert_label_assignments.values())) + 1,
            self.num_experts,
            device=target.device
        )
        
        for expert_idx, assigned_labels in self.expert_label_assignments.items():
            label_expert_map[torch.tensor(assigned_labels, device=target.device), expert_idx] = 1
        
        target_assignments = label_expert_map[target]
        expert_assignment_loss = F.binary_cross_entropy(
            expert_weights, target_assignments
        )
        
        # Expert diversity loss (entropy)
        diversity_loss = -torch.mean(
            torch.sum(expert_weights * torch.log(expert_weights + 1e-6), dim=1)
        )
        
        # Load balancing loss
        usage_per_expert = expert_weights.mean(0)
        target_usage = torch.ones_like(usage_per_expert) / self.num_experts
        balance_loss = F.kl_div(
            usage_per_expert.log(),
            target_usage,
            reduction='batchmean'
        )
        
        # Combine L2 losses with expert assignment masking
        masked_l2_losses = []
        for i, l2_loss in enumerate(expert_l2_losses):
            expert_mask = target_assignments[:, i]
            masked_l2_loss = l2_loss * expert_mask.mean()
            masked_l2_losses.append(masked_l2_loss)
        
        total_l2_loss = torch.stack(masked_l2_losses).sum()
        
        # Combine all losses with weights
        total_loss = (
            0.4 * ce_loss +
            0.4 * expert_assignment_loss +
            0.1 * diversity_loss +
            0.05 * balance_loss +
            0.05 * total_l2_loss
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
    
    def get_metrics(self) -> Dict[str, float]:
        """Get current metrics including loss components and routing entropy"""
        metrics = {
            **self.loss_components,
            **self.metrics
        }
        return metrics