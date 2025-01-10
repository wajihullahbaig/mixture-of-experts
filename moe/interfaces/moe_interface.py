from abc import ABC, abstractmethod
from typing import List, Optional, Tuple
import torch
import torch.nn as nn


class MoEInterface(ABC, nn.Module):
    """Interface for Mixture of Experts models"""
    
    @abstractmethod
    def forward(self, x: torch.Tensor, labels: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, torch.Tensor, List[torch.Tensor]]:
        """Forward pass through the MoE model"""
        pass
    
    @abstractmethod
    def compute_loss(self, final_output: torch.Tensor, target: torch.Tensor,
                    expert_weights: torch.Tensor, expert_l2_losses: List[torch.Tensor],
                    expert_outputs: List[torch.Tensor], temperature: float) -> torch.Tensor:
        """Compute the total loss"""
        pass
    
    @property
    @abstractmethod
    def num_experts(self) -> int:
        """Number of experts"""
        pass