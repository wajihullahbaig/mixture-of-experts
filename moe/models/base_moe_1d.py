import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Optional, Tuple

from moe.interfaces.moe_interface import MoEInterface
from moe.models.experts_1d import Expert1D
from moe.models.gates_1d import BasicGating1D

class BasicMoE1D(MoEInterface):
    """Basic Mixture of Experts implementation for 1D inputs"""
    
    def __init__(self, input_size: int, hidden_size: int, output_size: int, num_experts: int):
        super().__init__()
        self._num_experts = num_experts
        
        # Initialize experts
        self.experts = nn.ModuleList([
            Expert1D(input_size, hidden_size, output_size)
            for _ in range(num_experts)
        ])
        
        # Initialize gating network
        self.gating_network = BasicGating1D(input_size, num_experts, hidden_size)
        
        # Store loss components
        self.loss_components = {}
    
    @property
    def num_experts(self) -> int:
        return self._num_experts
    
    def forward(self, x: torch.Tensor, labels: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, torch.Tensor, List[torch.Tensor]]:
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
        
        # Combine expert outputs
        final_output = torch.sum(expert_outputs * expert_weights.unsqueeze(-1), dim=1)
        
        return final_output, expert_weights, expert_l2_losses
    
    def compute_loss(self, final_output: torch.Tensor, target: torch.Tensor, 
                    expert_weights: torch.Tensor, expert_l2_losses: List[torch.Tensor]) -> torch.Tensor:
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
