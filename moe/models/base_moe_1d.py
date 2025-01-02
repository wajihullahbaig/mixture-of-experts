import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Optional, Tuple
from torchinfo import summary
from pprint import pformat      

class Expert1D(nn.Module):
    """Individual expert network with fully connected layers"""
    
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
        
        self.l2_reg = 0.01
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

class GatingNetwork1D(nn.Module):
    """Gating network for 1D inputs"""
    
    def __init__(self, input_size: int, num_experts: int, hidden_size: int):
        super().__init__()
        self.network = nn.Sequential(
            nn.BatchNorm1d(input_size),
            nn.Linear(input_size, hidden_size),
            nn.ReLU(),
            nn.BatchNorm1d(hidden_size),
            nn.Dropout(0.3),
            nn.Linear(hidden_size, num_experts)
        )
        
        self.temperature = nn.Parameter(torch.ones(1))
        self._init_weights()
    
    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight)
                nn.init.constant_(m.bias, 0)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        logits = self.network(x)
        return F.softmax(logits / self.temperature, dim=-1)

class BaseMoE1D(nn.Module):
    """Base Mixture of Experts with 1D architecture"""
    
    def __init__(self, input_size: int, hidden_size: int, output_size: int, num_experts: int):
        super().__init__()
        self.num_experts = num_experts
        
        # Initialize experts
        self.experts = nn.ModuleList([
            Expert1D(input_size, hidden_size, output_size)
            for _ in range(num_experts)
        ])
        
        # Initialize gating network
        self.gating_network = GatingNetwork1D(input_size, num_experts, hidden_size)
        
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
        # Classification loss
        ce_loss = F.cross_entropy(final_output, target)
        
        # Expert diversity loss (entropy)
        diversity_loss = -torch.mean(
            torch.sum(expert_weights * torch.log(expert_weights + 1e-6), dim=1)
        )
        
        # Load balancing loss
        usage_per_expert = expert_weights.mean(0)
        target_usage = torch.ones_like(usage_per_expert) / self.num_experts
        balance_loss = F.kl_div(
            usage_per_expert.log(),
            target_usage,
            reduction='batchmean'
        )
        
        # Combine L2 losses
        total_l2_loss = torch.stack(expert_l2_losses).sum()
        
        # Combine all losses
        total_loss = (
            0.6*ce_loss + 
            0.1 * diversity_loss +
            0.2 * balance_loss +
            0.1 * total_l2_loss
        )
        
        # Store components for monitoring
        self.loss_components = {
            'ce_loss': ce_loss.item(),
            'diversity_loss': diversity_loss.item(),
            'balance_loss': balance_loss.item(),
            'l2_loss': total_l2_loss.item(),
            'total_loss': total_loss.item()
        }
        
        return total_loss