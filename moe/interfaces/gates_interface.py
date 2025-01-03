from abc import ABC, abstractmethod
from typing import Dict, List, Optional, Tuple, Union
import torch
import torch.nn as nn

class GatingInterface(ABC, nn.Module):
    """Interface for gating networks"""
    
    @abstractmethod
    def forward(self, x: torch.Tensor, labels: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Forward pass through the gating network
        
        Args:
            x: Input tensor
            labels: Optional labels for guided routing
            
        Returns:
            Expert weights/routing probabilities
        """
        pass