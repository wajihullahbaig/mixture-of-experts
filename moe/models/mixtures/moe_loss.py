import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Optional, Tuple
import math

from moe.interfaces.moe_interface import MoEInterface

class MoELoss(nn.Module):
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
    
    def compute_diversity_loss(self,expert_weights: torch.Tensor) -> torch.Tensor:
        """Compute diversity loss"""
        diversity_loss = -torch.mean(
            torch.sum(expert_weights * torch.log(expert_weights + 1e-6), dim=1)
        )
        return diversity_loss
    
    def compute_load_balance_loss(self, expert_weights: torch.Tensor) -> torch.Tensor:
        """Compute load balancing loss"""
        usage_per_expert = expert_weights.mean(0)
        target_usage = torch.ones_like(usage_per_expert) / expert_weights.size(1)
        balance_loss = F.kl_div(usage_per_expert.log(), target_usage, reduction='batchmean')
        return balance_loss
    
    def compute_l2_losses_with_masking(self, expert_l2_losses: List[torch.Tensor], target_assignments: torch.Tensor) -> List[torch.Tensor]:
        """Compute L2 losses with masking"""
        masked_l2_losses = []
        for i, l2_loss in enumerate(expert_l2_losses):
            expert_mask = target_assignments[:, i]
            masked_l2_loss = l2_loss * expert_mask.mean()
            masked_l2_losses.append(masked_l2_loss)
        
        total_l2_loss = torch.stack(masked_l2_losses).sum()
        return total_l2_loss