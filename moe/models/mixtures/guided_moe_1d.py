from typing import Dict, List, Optional, Tuple
import torch
from torch import nn
from torch.nn import functional as F
from moe.interfaces.moe_interface import MoEInterface
from moe.models.experts.experts_1d import Expert1D
from moe.models.gates.gates_1d import GuidedGating1D
import numpy as np

from moe.models.mixtures.base_moe import BaseMoE

class GuidedMoE1D(BaseMoE, MoEInterface):
    """Guided Mixture of Experts implementation for 1D inputs"""
    
    def __init__(self, input_size: int, hidden_size: int, output_size: int, num_experts: int,
                 expert_label_assignments: Dict[int, List[int]]):
        # Initialize both parent classes
        BaseMoE.__init__(self)
        nn.Module.__init__(self)
        
        self._num_experts = num_experts
        self.expert_label_assignments = expert_label_assignments
        
        # Initialize experts
        self.experts = nn.ModuleList([
            Expert1D(input_size, hidden_size, output_size)
            for _ in range(num_experts)
        ])
        
        # Initialize guided gating network
        self.gating_network = GuidedGating1D(
            input_size,
            num_experts,
            hidden_size,
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
        
        return final_output, expert_weights,expert_outputs, expert_l2_losses
    
    def compute_loss(self, final_output: torch.Tensor, target: torch.Tensor,
                    expert_weights: torch.Tensor, expert_l2_losses: List[torch.Tensor],
                    expert_outputs: List[torch.Tensor], temperature: float) -> torch.Tensor:
        # Primary losses
        ce_loss = F.cross_entropy(final_output, target)
        l2_loss = torch.stack(expert_l2_losses).sum()
        
        # Additional losses using BaseMoE utilities
        balance_loss = self.compute_load_balance_loss(expert_weights)
        conf_penalty = self.compute_confidence_penalty(expert_weights, temperature)
        entropy_reg = self.compute_entropy_regularization(expert_weights)
        contr_loss = self.compute_expert_contrastive_loss(expert_outputs, expert_weights)
        
        # Combine all losses
        losses = [ce_loss, l2_loss, balance_loss, conf_penalty, entropy_reg, contr_loss]
        weights = [0.4, 0.1, 0.2, 0.1, 0.1, 0.1]
        
        total_loss, normalized_losses, _ = self.normalize_and_weight_losses(*losses, weights=weights)
        
        # Store components
        self.loss_components = {
            'ce_loss': normalized_losses[0].item(),
            'l2_loss': normalized_losses[1].item(),
            'balance_loss': normalized_losses[2].item(),
            'confidence_penalty': normalized_losses[3].item(),
            'entropy_reg': normalized_losses[4].item(),
            'contrastive_loss': normalized_losses[5].item(),
            'total_loss': total_loss.item()
        }
        
        return total_loss

    
    

