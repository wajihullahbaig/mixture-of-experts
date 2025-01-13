import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple

from moe.interfaces.experts_interface import ExpertInterface

class Expert1D(ExpertInterface):
    """Expert network for 1D inputs"""
    
    def __init__(self, input_size: int, hidden_size: int, output_size: int, dropout_rate: float = 0.3):
        super().__init__()
        self.network = nn.Sequential(
            nn.BatchNorm1d(input_size),
            nn.Linear(input_size, hidden_size),
            nn.ReLU(),
            nn.BatchNorm1d(hidden_size),
            nn.Dropout(dropout_rate),
            nn.Linear(hidden_size, hidden_size // 2),
            nn.ReLU(),
            nn.BatchNorm1d(hidden_size // 2),
            nn.Dropout(dropout_rate),
            nn.Linear(hidden_size // 2, output_size)
        )
        
        self.l2_reg = 0.0001
        self._init_weights()
    
    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight)
                nn.init.constant_(m.bias, 0)
    
    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        output = self.network(x)
        l2_loss = sum(p.pow(2.0).sum() for p in self.parameters()) * self.l2_reg
        return output, l2_loss
