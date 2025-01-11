from typing import List, Optional, Tuple
import torch
from torch import nn

from moe.interfaces.moe_interface import MoEInterface
from moe.models.experts.experts_2d import Expert2D
from moe.models.gates.gates_2d import BasicGating2D
from moe.models.mixtures.moe_loss import MoELoss
from moe.models.mixtures.basic_moe_1d import BasicMoE1D


class BasicMoE2D(MoELoss, MoEInterface):
    """Basic Mixture of Experts implementation for 2D inputs"""
    
    def __init__(self, input_channels: Tuple[int,int,int], num_classes: int, num_experts: int):
        # Initialize both parent classes
        MoELoss.__init__(self)
        nn.Module.__init__(self)
        
        self._num_experts = num_experts
        
        # Initialize experts
        self.experts = nn.ModuleList([
            Expert2D(input_channels, num_classes)
            for _ in range(num_experts)
        ])
        
        # Initialize gating network
        self.gating_network = BasicGating2D(input_channels, num_experts)
        
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
                expert_weights: torch.Tensor, expert_l2_losses: List[torch.Tensor], 
                expert_outputs: List[torch.Tensor]) -> torch.Tensor:
        # Use same loss computation as 1D
        return BasicMoE1D.compute_loss(self, final_output, target, expert_weights, expert_l2_losses, expert_outputs)

