import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Optional, Tuple
import math

from moe.interfaces.moe_interface import MoEInterface

class BaseMoE(nn.Module):
    """Base class providing utility functions for MoE implementations"""
    
    def __init__(self):
        super().__init__()
        self.loss_components: Dict[str, float] = {}
        self.metrics: Dict[str, float] = {}
    
    def normalize_and_weight_losses(self, *losses: torch.Tensor, 
                                  weights: Optional[List[float]] = None) -> Tuple[torch.Tensor, torch.Tensor, List[float]]:
        """Normalize and weight multiple loss components"""
        losses = torch.stack(list(losses))
        max_loss = torch.max(losses)
        normalized_losses = losses / max_loss
        
        if weights is None:
            weights = [1.0/len(losses)] * len(losses)
        
        weights = torch.tensor(weights, device=losses.device, dtype=losses.dtype)
        assert torch.isclose(torch.sum(weights), torch.tensor(1.0)), "Weights must sum to 1"
        
        total_loss = max_loss * torch.sum(normalized_losses * weights)
        return total_loss, normalized_losses, weights
    
    def compute_confidence_penalty(self, expert_weights: torch.Tensor, 
                                 temperature: float = 1.0,
                                 confidence_threshold: float = 0.7) -> torch.Tensor:
        """Compute confidence penalty"""
        max_probs = expert_weights.max(dim=-1)[0]
        penalty = torch.relu(max_probs - confidence_threshold)
        penalty = penalty * (1.0 / temperature)
        return penalty.mean()
    
    def compute_expert_contrastive_loss(self, expert_outputs: torch.Tensor, 
                                  expert_weights: torch.Tensor,
                                  temperature: float = 0.1) -> torch.Tensor:
        """
        Compute contrastive loss between expert representations
        
        Args:
            expert_outputs: Stacked expert outputs [num_experts, batch_size, feature_dim]
            expert_weights: Expert assignment weights [batch_size, num_experts]
            temperature: Temperature for similarity scaling
            
        Returns:
            Contrastive loss tensor
        """
        batch_size = expert_outputs.size(1)
        num_experts = expert_outputs.size(0)
        
        # Normalize expert outputs
        expert_outputs = F.normalize(expert_outputs, p=2, dim=-1)
        
        # Compute similarities between all expert pairs
        # [num_experts, batch_size, batch_size]
        similarities = torch.matmul(expert_outputs, expert_outputs.transpose(1, 2))
        
        # Compute positive and negative masks
        expert_assignments = expert_weights.argmax(dim=-1)  # [batch_size]
        pos_mask = torch.eye(batch_size, device=expert_outputs.device).unsqueeze(0).expand(num_experts, -1, -1)
        neg_mask = 1 - pos_mask
        
        # Scale similarities by temperature
        similarities = similarities / temperature
        
        # Compute positive and negative losses
        pos_loss = -(similarities * pos_mask).sum(dim=-1).mean()
        neg_loss = torch.log(1 + (similarities * neg_mask).exp()).sum(dim=-1).mean()
        
        return pos_loss + neg_loss

    def compute_entropy_regularization(self, routing_weights: torch.Tensor,
                                    min_entropy: float = 0.1,
                                    max_entropy: float = 0.5) -> torch.Tensor:
        """Compute entropy regularization"""
        entropy = -(routing_weights * torch.log(routing_weights + 1e-7)).sum(dim=-1)
        mean_entropy = entropy.mean()
        entropy_reg = torch.relu(min_entropy - mean_entropy) + torch.relu(mean_entropy - max_entropy)
        return entropy_reg
    
    def compute_load_balance_loss(self, expert_weights: torch.Tensor) -> torch.Tensor:
        """Compute load balancing loss"""
        usage_per_expert = expert_weights.mean(0)
        target_usage = torch.ones_like(usage_per_expert) / expert_weights.size(1)
        balance_loss = F.kl_div(usage_per_expert.log(), target_usage, reduction='batchmean')
        return balance_loss