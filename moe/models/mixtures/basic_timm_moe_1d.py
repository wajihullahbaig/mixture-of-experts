from typing import Dict, List, Optional, Tuple, Any
import torch
import torch.nn as nn
import torch.nn.functional as F

from moe.interfaces.moe_interface import MoEInterface
from moe.models.experts.experts_timm_1d import TimmExpert1D
from moe.models.gates.gates_1d import BasicGating1D


class BasicTimmMoE1D(MoEInterface):
    """Mixture of Experts implementation using TIMM models for 1D inputs"""
    
    def __init__(self,
                 model_name: str,
                 input_size: int,
                 hidden_size: int,
                 output_size: int,
                 num_experts: int,
                 pretrained: bool = True,
                 dropout_rate: float = 0.3):
        """
        Initialize TIMM-based MoE model
        
        Args:
            model_name: Name of TIMM model to use as feature extractor
            input_size: Size of input features
            hidden_size: Size of hidden layer
            output_size: Number of output classes
            num_experts: Number of experts to use
            pretrained: Whether to use pretrained TIMM models
            dropout_rate: Dropout rate for regularization
        """
        super().__init__()
        self._num_experts = num_experts
        
        # Initialize experts
        self.experts = nn.ModuleList([
            TimmExpert1D(
                model_name=model_name,
                input_size=input_size,
                hidden_size=hidden_size,
                output_size=output_size,
                pretrained=pretrained,
                dropout_rate=dropout_rate
            )
            for _ in range(num_experts)
        ])
        
        # Initialize gating network
        self.gating_network = BasicGating1D(input_size, num_experts, hidden_size)
        
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
            x: Input tensor of shape [batch_size, input_size]
            labels: Optional tensor of labels (not used in basic version)
            
        Returns:
            Tuple containing:
            - Final output tensor
            - Expert weights tensor
            - List of expert L2 losses
        """
        # Get expert weights
        expert_weights = self.gating_network(x)
        
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
        Compute the total loss for training
        
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
        
        # Combine L2 losses
        total_l2_loss = torch.stack(expert_l2_losses).sum()
        
        # Combine all losses with weights
        total_loss = (
            0.6 * ce_loss +
            0.1 * diversity_loss +
            0.2 * balance_loss +
            0.1 * total_l2_loss
        )
        
        # Store components for monitoring
        self.loss_components = {
            'ce_loss': ce_loss.item(),
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