
from typing import Dict
from moe.configs.default_config import Config
from moe.data_handlers.datasets import DatasetFactory
from moe.models.base_moe_1d import BaseMoE1D
from moe.models.base_moe_2d import BaseMoE2D
from moe.models.guided_moe_1d import GuidedMoE1D
from moe.models.guided_moe_2d import GuidedMoE2D
import torch.nn as nn


def create_model(config:Config) -> nn.Module:
    """Create appropriate model based on configuration"""
    input_info = DatasetFactory.get_input_size(config.data.dataset, config.model.architecture)
    dcfg = DatasetFactory.get_dataset_config(config.data.dataset, config.model.architecture)
    if config.model.architecture == '1d':
        input_size = input_info if isinstance(input_info, int) else input_info[0] * input_info[1] * input_info[2]
        if config.model.model_type == 'base':
            return BaseMoE1D(
                input_size=input_size,
                hidden_size=config.model.hidden_size,
                output_size=dcfg["num_classes"],  # MNIST/CIFAR classes
                num_experts=config.model.num_experts
            )
        else:
            expert_assignments = create_expert_assignments(dcfg["num_classes"], config.model.num_experts)
            return GuidedMoE1D(
                input_size=input_size,
                hidden_size=config.model.hidden_size,
                output_size=dcfg["num_classes"],
                num_experts=config.model.num_experts,
                expert_label_assignments=expert_assignments
            )
    else:  # 2d architecture
        input_channels = input_info[0]
        if config.model.model_type == 'base':
            return BaseMoE2D(
                input_channels=input_channels,
                num_classes=dcfg["num_classes"],
                num_experts=config.model.num_experts
            )
        else:
            expert_assignments = create_expert_assignments(dcfg["num_classes"], config.model.num_experts)
            return GuidedMoE2D(
                input_channels=input_channels,
                num_classes=dcfg["num_classes"],
                num_experts=config.model.num_experts,
                expert_label_assignments=expert_assignments
            )

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