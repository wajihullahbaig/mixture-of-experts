from abc import ABC, abstractmethod
from typing import Dict, List, Optional, Tuple, Union
import torch
import torch.nn as nn


class MoEInterface(ABC, nn.Module):
    """Interface for Mixture of Experts models"""
    
    def __init__(self):
        super().__init__()
        self.loss_components: Dict[str, float] = {}
    
    @abstractmethod
    def forward(self, x: torch.Tensor, labels: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, torch.Tensor, List[torch.Tensor]]:
        """
        Forward pass through the MoE model
        
        Args:
            x: Input tensor
            labels: Optional labels for guided routing
            
        Returns:
            Tuple containing:
            - Final output
            - Expert weights
            - List of expert L2 losses
        """
        pass
    
    @abstractmethod
    def compute_loss(self, final_output: torch.Tensor, target: torch.Tensor,
                    expert_weights: torch.Tensor, expert_l2_losses: List[torch.Tensor]) -> torch.Tensor:
        """
        Compute the total loss for the MoE model
        
        Args:
            final_output: Model predictions
            target: True labels
            expert_weights: Weights assigned to each expert
            expert_l2_losses: L2 regularization losses from each expert
            
        Returns:
            Total loss value
        """
        pass
    
    @property
    @abstractmethod
    def num_experts(self) -> int:
        """Number of experts in the model"""
        pass