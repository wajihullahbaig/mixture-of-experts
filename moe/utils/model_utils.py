
from typing import Dict
from moe.configs.default_config import Config
from moe.factories.datasets_factory import DatasetFactory
from moe.models.base_moe_1d import BasicMoE1D
from moe.models.base_moe_2d import BasicMoE2D
from moe.models.guided_moe_1d import GuidedMoE1D
from moe.models.guided_moe_2d import GuidedMoE2D
import torch.nn as nn

def create_expert_assignments(num_classes: int, num_experts: int) -> Dict[int, list]:
    """Create balanced label assignments for experts"""
    assignments = {}
    labels_per_expert = num_classes // num_experts
    remaining = num_classes % num_experts
    
    start_idx = 0
    for expert_idx in range(num_experts):
        num_labels = labels_per_expert + (1 if expert_idx < remaining else 0)
        assignments[expert_idx] = list(range(start_idx, start_idx + num_labels))
        start_idx += num_labels
    
    return assignments