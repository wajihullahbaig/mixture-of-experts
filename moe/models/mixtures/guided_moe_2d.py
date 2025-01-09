from typing import Dict, List, Optional, Tuple
import torch
from torch import nn as nn

from moe.interfaces.moe_interface import MoEInterface
from moe.models.experts.experts_2d import Expert2D
from moe.models.gates.gates_2d import GuidedGating2D
from moe.models.mixtures.guided_moe_1d import GuidedMoE1D


class GuidedMoE2D(MoEInterface):
    """Guided Mixture of Experts implementation for 2D inputs"""
    
    def __init__(self, input_channels: int, num_classes: int, num_experts: int,
                 expert_label_assignments: Dict[int, List[int]]):
        super().__init__()
        self._num_experts = num_experts
        self.expert_label_assignments = expert_label_assignments
        
        # Initialize experts
        self.experts = nn.ModuleList([
            Expert2D(input_channels, num_classes)
            for _ in range(num_experts)
        ])
        
        # Initialize guided gating network
        self.gating_network = GuidedGating2D(
            input_channels,
            num_experts,
            expert_label_assignments
        )
        
        # Store loss components
        self.loss_components = {}
    
    @property
    def num_experts(self) -> int:
        return self._num_experts
    
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
        # Use same loss computation as 1D guided version
        return GuidedMoE1D.compute_loss(self, final_output, target, expert_weights, expert_l2_losses)