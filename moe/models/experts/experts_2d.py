import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple

from moe.interfaces.experts_interface import ExpertInterface


class Expert2D(ExpertInterface):
    """Expert network for 2D inputs (CNNs)"""
    
    def __init__(self, input_channels: int, num_classes: int, dropout_rate: float = 0.3):
        super().__init__()
        
        self.features = nn.Sequential(
            # First conv block
            nn.Conv2d(input_channels, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Dropout2d(dropout_rate),
            
            # Second conv block
            nn.Conv2d(64, 128, 3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Dropout2d(dropout_rate),
            
            # Third conv block
            nn.Conv2d(128, 256, 3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Dropout2d(dropout_rate)
        )
        
        # Calculate feature size
        self.feature_size = self._get_feature_size(input_channels)
        
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(self.feature_size, 512),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(512, num_classes)
        )
        
        self.l2_reg = 0.0001
        self._init_weights()
    
    def _get_feature_size(self, input_channels: int) -> int:
        return 256 * (3 * 3 if input_channels == 1 else 4 * 4)
    
    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight)
                nn.init.constant_(m.bias, 0)
    
    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        features = self.features(x)
        output = self.classifier(features)
        l2_loss = sum(p.pow(2.0).sum() for p in self.parameters()) * self.l2_reg
        return output, l2_loss