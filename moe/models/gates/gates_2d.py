import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Optional, Union

from moe.interfaces.gates_interface import GatingInterface

class BasicGating2D(GatingInterface):
    """Basic gating network for 2D inputs"""
    
    def __init__(self, input_channels: int, num_experts: int, dropout_rate: float = 0.3):
        super().__init__()
        
        self.features = nn.Sequential(
            # First conv block with larger kernel
            nn.Conv2d(input_channels, 32, 5, padding=2),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Dropout2d(dropout_rate),
            
            # Second conv block
            nn.Conv2d(32, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Dropout2d(dropout_rate),
            
            # Third conv block
            nn.Conv2d(64, 128, 3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Dropout2d(dropout_rate),
        )
        
        # Calculate feature size
        self.feature_size = self._get_feature_size(input_channels)
        
        # Gating network
        self.gate = nn.Sequential(
            nn.Flatten(),
            nn.Linear(self.feature_size, 512),
            nn.LayerNorm(512),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(512, 256),
            nn.LayerNorm(256),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(256, num_experts)
        )
        
        self.temperature = nn.Parameter(torch.ones(1)*1.2)
        self._init_weights()
    
    def _get_feature_size(self, input_channels: int) -> int:
        if input_channels == 1:  # MNIST
            return 128 * 3 * 3
        else:  # CIFAR
            return 128 * 4 * 4
    
    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, (nn.BatchNorm2d, nn.LayerNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight, gain=0.5)
                nn.init.constant_(m.bias, 0)
    
    def forward(self, x: torch.Tensor, labels: Optional[torch.Tensor] = None) -> torch.Tensor:
        features = self.features(x)
        logits = self.gate(features)
        
        # Add numerical stability
        temp = torch.clamp(self.temperature, min=0.1)
        scaled_logits = logits / temp
        scaled_logits = scaled_logits - scaled_logits.max(dim=-1, keepdim=True)[0]
        
        return F.softmax(scaled_logits, dim=-1)


class GuidedGating2D(GatingInterface):
    """Guided gating network for 2D inputs with label-based routing"""
    
    def __init__(self, input_channels: int, num_experts: int, 
                 expert_label_map: Dict[int, List[int]], dropout_rate: float = 0.3):
        super().__init__()
        self.expert_label_map = expert_label_map
        self.num_experts = num_experts
        
        # Calculate base probabilities
        total_labels = sum(len(labels) for labels in expert_label_map.values())
        self.base_probs = {
            expert_idx: len(labels) / total_labels 
            for expert_idx, labels in expert_label_map.items()
        }
        
        # Feature extraction - same as BasicGating2D
        self.features = nn.Sequential(
            # First conv block with larger kernel
            nn.Conv2d(input_channels, 32, 5, padding=2),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Dropout2d(dropout_rate),
            
            # Second conv block
            nn.Conv2d(32, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Dropout2d(dropout_rate),
            
            # Third conv block
            nn.Conv2d(64, 128, 3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Dropout2d(dropout_rate),
        )
        
        # Calculate feature size
        self.feature_size = self._get_feature_size(input_channels)
        
        # Gating network - same as BasicGating2D
        self.gate = nn.Sequential(
            nn.Flatten(),
            nn.Linear(self.feature_size, 512),
            nn.LayerNorm(512),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(512, 256),
            nn.LayerNorm(256),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(256, num_experts)
        )
        
        self.temperature = nn.Parameter(torch.ones(1)*5.0)
        self._init_weights()
    
    def _get_feature_size(self, input_channels: int) -> int:
        if input_channels == 1:  # MNIST
            return 128 * 3 * 3
        else:  # CIFAR
            return 128 * 4 * 4
    
    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, (nn.BatchNorm2d, nn.LayerNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight, gain=0.5)
                nn.init.constant_(m.bias, 0)
    
    def compute_soft_masks(self, labels: torch.Tensor) -> torch.Tensor:
        """Compute soft assignment masks with proper probability scaling"""
        batch_size = labels.size(0)
        device = labels.device
        
        # Create assignment matrix with scaled background probabilities
        base_probs = torch.tensor([self.base_probs[i] for i in range(self.num_experts)], 
                                device=device)
        soft_mask = base_probs.unsqueeze(0).expand(batch_size, -1)
        
        # Create label-to-expert mapping tensor
        label_expert_map = torch.zeros(max(max(self.expert_label_map.values())) + 1, 
                                     self.num_experts, device=device)
        
        for expert_idx, assigned_labels in self.expert_label_map.items():
            expert_prob = self.base_probs[expert_idx]
            label_expert_map[torch.tensor(assigned_labels, device=device), expert_idx] = expert_prob
        
        expert_assignments = label_expert_map[labels]
        soft_mask = torch.where(expert_assignments > 0, 
                              expert_assignments,
                              soft_mask)
        
        return F.normalize(soft_mask, p=1, dim=1)
    
    def forward(self, x: torch.Tensor, labels: Optional[torch.Tensor] = None) -> torch.Tensor:
        features = self.features(x)
        logits = self.gate(features)
        
        # Add small noise during training for exploration
        if self.training:
            noise = torch.randn_like(logits) * 0.01
            logits = logits + noise
        
        if self.training and labels is not None:
            # Get soft assignment masks
            soft_masks = self.compute_soft_masks(labels)
            
            # Combine network output with soft guidance
            routing_weights = F.softmax(logits / self.temperature, dim=1)
            
            # Dynamic mixing factor based on training progress
            mixing_factor = torch.sigmoid(self.temperature)
            final_weights = mixing_factor * routing_weights + (1 - mixing_factor) * soft_masks
            
            return final_weights
        
        return F.softmax(logits / self.temperature, dim=-1)