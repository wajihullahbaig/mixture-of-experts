import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Optional, Union

from moe.interfaces.gates_interface import GatingInterface


class BasicGating1D(GatingInterface):
    """Basic gating network for 1D inputs"""
    
    def __init__(self, input_size: int, num_experts: int, hidden_size: int,dropout_rate=0.3):
        super().__init__()
        self.network  = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.ReLU(),
            nn.BatchNorm1d(hidden_size),
            nn.Dropout(dropout_rate),
            nn.Linear(hidden_size, hidden_size // 2),
            nn.ReLU(),
            nn.BatchNorm1d(hidden_size // 2),
            nn.Dropout(dropout_rate),
            nn.Linear(hidden_size // 2, num_experts)
        )
        
        self.temperature = nn.Parameter(torch.ones(1)*1.2)
        self._init_weights()
    
    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight)
                nn.init.constant_(m.bias, 0)
    
    def forward(self, x: torch.Tensor, labels: Optional[torch.Tensor] = None) -> torch.Tensor:
        logits = self.network(x)
        
        # Add numerical stability
        # 1. Clamp temperature to avoid division by very small numbers
        temp = torch.clamp(self.temperature, min=0.1)
        
        # 2. Apply log-space softmax for numerical stability
        scaled_logits = logits / temp
        
        # 3. Subtract max for numerical stability
        scaled_logits = scaled_logits - scaled_logits.max(dim=-1, keepdim=True)[0]
        
        return F.softmax(scaled_logits, dim=-1)


class GuidedGating1D(GatingInterface):
    """Guided gating network for 1D inputs with label-based routing"""
    
    def __init__(self, input_size: int, num_experts: int, hidden_size: int, 
                 expert_label_map: Dict[int, List[int]],dropout_rate = 0.3):
        super().__init__()
        self.expert_label_map = expert_label_map
        self.num_experts = num_experts
        
        # Calculate base probabilities
        total_labels = sum(len(labels) for labels in expert_label_map.values())
        self.base_probs = {
            expert_idx: len(labels) / total_labels 
            for expert_idx, labels in expert_label_map.items()
        }
        
        self.network  = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.ReLU(),
            nn.BatchNorm1d(hidden_size),
            nn.Dropout(dropout_rate),
            nn.Linear(hidden_size, hidden_size // 2),
            nn.ReLU(),
            nn.BatchNorm1d(hidden_size // 2),
            nn.Dropout(dropout_rate),
            nn.Linear(hidden_size // 2, num_experts)
        )
        
        

        self.temperature = nn.Parameter(torch.ones(1)*2.0)
        
        self._init_weights()
    
    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight)
                nn.init.constant_(m.bias, 0)
    
    def compute_soft_masks(self, labels: torch.Tensor) -> torch.Tensor:
        """Improved soft assignment masks with better probability scaling"""
        batch_size = labels.size(0)
        device = labels.device
        
        # Create base probability distribution
        base_probs = torch.tensor([self.base_probs[i] for i in range(self.num_experts)], 
                                device=device)
        soft_mask = base_probs.unsqueeze(0).expand(batch_size, -1)
        
        # Create smoothed label-to-expert mapping
        label_expert_map = torch.zeros(max(max(self.expert_label_map.values())) + 1, 
                                    self.num_experts, device=device)
        
        # Add label smoothing
        smoothing_factor = 0.1
        smoothed_value = smoothing_factor / self.num_experts
        label_expert_map.fill_(smoothed_value)
        
        # Assign expert probabilities with smoothing
        for expert_idx, assigned_labels in self.expert_label_map.items():
            expert_prob = self.base_probs[expert_idx] * (1 - smoothing_factor)
            label_expert_map[torch.tensor(assigned_labels, device=device), expert_idx] = expert_prob
        
        # Compute expert assignments with neighborhood consideration
        expert_assignments = label_expert_map[labels]
        
        # Add locality-based smoothing
        kernel_size = 3
        smoothing_kernel = torch.ones(1, 1, kernel_size, device=device) / kernel_size
        smoothed_assignments = F.conv1d(
            expert_assignments.unsqueeze(1),
            smoothing_kernel,
            padding=kernel_size//2
        ).squeeze(1)
        
        # Combine with original assignments
        soft_mask = torch.where(smoothed_assignments > 0, 
                            smoothed_assignments,
                            soft_mask)
        
        return F.normalize(soft_mask, p=1, dim=1)
    
    def forward(self, x: torch.Tensor, labels: Optional[torch.Tensor] = None) -> torch.Tensor:
        logits = self.network(x)
        
        # Add small noise during training for robustness
        if self.training:
            noise = torch.randn_like(logits) * 0.1
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
