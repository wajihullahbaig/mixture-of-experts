from abc import ABC, abstractmethod
from typing import Dict, List, Optional, Tuple, Union
import torch
import torch.nn as nn


class ExpertInterface(ABC, nn.Module):
    """Interface for individual experts"""
    
    @abstractmethod
    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass through the expert
        
        Args:
            x: Input tensor
            
        Returns:
            Tuple containing:
            - Expert output
            - L2 regularization loss
        """
        pass
