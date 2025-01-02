import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Optional, Tuple
from torchinfo import summary
from pprint import pformat

from moe.models.base_moe_1d import BaseMoE1D

class Expert2D(nn.Module):
    """Individual expert network with CNN layers"""
    
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
        
        # Calculate final feature size based on input size
        self.feature_size = self._get_feature_size(input_channels)
        
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(self.feature_size, 512),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(512, num_classes)
        )
        
        self.l2_reg = 0.01
        self._init_weights()
    
    def _get_feature_size(self, input_channels: int) -> int:
        # Calculate output size after conv layers
        if input_channels == 1:  # MNIST: 28x28
            return 256 * 3 * 3
        else:  # CIFAR: 32x32
            return 256 * 4 * 4
    
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

class GatingNetwork2D(nn.Module):
    """Gating network with CNN for 2D inputs"""
    
    def __init__(self, input_channels: int, num_experts: int):
        super().__init__()
        
        # Lighter CNN for gating
        self.features = nn.Sequential(
            nn.Conv2d(input_channels, 32, 3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2),
            
            nn.Conv2d(32, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2)
        )
        
        # Calculate feature size
        self.feature_size = self._get_feature_size(input_channels)
        
        self.gate = nn.Sequential(
            nn.Flatten(),
            nn.Linear(self.feature_size, 256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, num_experts)
        )
        
        self.temperature = nn.Parameter(torch.ones(1))
        self._init_weights()
    
    def _get_feature_size(self, input_channels: int) -> int:
        if input_channels == 1:  # MNIST
            return 64 * 7 * 7
        else:  # CIFAR
            return 64 * 8 * 8
    
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
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        features = self.features(x)
        logits = self.gate(features)
        return F.softmax(logits / self.temperature, dim=-1)

class BaseMoE2D(nn.Module):
    """Base Mixture of Experts with 2D CNN architecture"""
    
    def __init__(self, input_channels: int, num_classes: int, num_experts: int):
        super().__init__()
        self.num_experts = num_experts
        
        # Initialize experts
        self.experts = nn.ModuleList([
            Expert2D(input_channels, num_classes)
            for _ in range(num_experts)
        ])
        
        # Initialize gating network
        self.gating_network = GatingNetwork2D(input_channels, num_experts)
        
        # Store loss components
        self.loss_components = {}
    
    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, List[torch.Tensor]]:
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
        # Use same loss computation as 1D version
        return BaseMoE1D.compute_loss(self, final_output, target, expert_weights, expert_l2_losses)