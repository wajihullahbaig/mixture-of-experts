from typing import Dict, Union, Type, Tuple, Optional
from dataclasses import dataclass
from enum import Enum, auto
import torch.nn as nn

from moe.configs.default_config import ArchitectureType, MoEConfig, MoEType
from moe.interfaces.moe_interface import MoEInterface
from moe.models.base_moe_1d import BasicMoE1D
from moe.models.base_moe_2d import BasicMoE2D
from moe.models.guided_moe_1d import GuidedMoE1D
from moe.models.guided_moe_2d import GuidedMoE2D



class MoEFactory:
    """Factory for creating MoE models"""
    
    _moe_registry: Dict[Tuple[MoEType, ArchitectureType], Type[MoEInterface]] = {
        (MoEType.BASIC, ArchitectureType.ARCH_1D): BasicMoE1D,
        (MoEType.BASIC, ArchitectureType.ARCH_2D): BasicMoE2D,
        (MoEType.GUIDED, ArchitectureType.ARCH_1D): GuidedMoE1D,
        (MoEType.GUIDED, ArchitectureType.ARCH_2D): GuidedMoE2D
    }
    
    @classmethod
    def register_moe(cls, moe_type: MoEType, arch_type: ArchitectureType, 
                    moe_class: Type[MoEInterface]) -> None:
        """
        Register a new MoE implementation
        
        Args:
            moe_type: Type of MoE model
            arch_type: Type of architecture
            moe_class: MoE class implementation
        """
        if not issubclass(moe_class, MoEInterface):
            raise ValueError(f"Class {moe_class.__name__} must implement MoEInterface")
        
        cls._moe_registry[(moe_type, arch_type)] = moe_class
    
    @classmethod
    def unregister_moe(cls, moe_type: MoEType, arch_type: ArchitectureType) -> None:
        """
        Unregister an MoE implementation
        
        Args:
            moe_type: Type of MoE model
            arch_type: Type of architecture
        """
        if (moe_type, arch_type) in cls._moe_registry:
            del cls._moe_registry[(moe_type, arch_type)]
    
    @classmethod
    def create_moe(cls, config: MoEConfig) -> MoEInterface:
        """
        Create an MoE model based on configuration
        
        Args:
            config: MoE configuration
            
        Returns:
            Configured MoE model
        """
        # Validate configuration
        config.validate()
        
        # Get appropriate MoE class
        key = (config.moe_type, config.architecture)
        if key not in cls._moe_registry:
            raise ValueError(f"No implementation registered for {key}")
        
        moe_class = cls._moe_registry[key]
        
        # Create model based on architecture type
        if config.architecture == ArchitectureType.ARCH_1D:
            if config.moe_type == MoEType.BASIC:
                return moe_class(
                    input_size=config.input_size,
                    hidden_size=config.hidden_size,
                    output_size=config.output_size,
                    num_experts=config.num_experts
                )
            else:  # Guided
                return moe_class(
                    input_size=config.input_size,
                    hidden_size=config.hidden_size,
                    output_size=config.output_size,
                    num_experts=config.num_experts,
                    expert_label_assignments=config.expert_label_map
                )
        else:  # 2D architecture
            if config.moe_type == MoEType.BASIC:
                return moe_class(
                    input_channels=config.input_size[0],
                    num_classes=config.output_size,
                    num_experts=config.num_experts
                )
            else:  # Guided
                return moe_class(
                    input_channels=config.input_size[0],
                    num_classes=config.output_size,
                    num_experts=config.num_experts,
                    expert_label_assignments=config.expert_label_map
                )
    
    @classmethod
    def list_registered_models(cls) -> Dict[Tuple[MoEType, ArchitectureType], Type[MoEInterface]]:
        """Get dictionary of all registered MoE implementations"""
        return cls._moe_registry.copy()

# Example usage:
def create_model_example():
    # Create configuration for a basic 1D MoE
    config_1d = MoEConfig(
        moe_type=MoEType.BASIC,
        architecture=ArchitectureType.ARCH_1D,
        num_experts=5,
        input_size=784,  # MNIST flattened
        hidden_size=256,
        output_size=10
    )
    
    # Create 1D model
    model_1d = MoEFactory.create_moe(config_1d)
    
    # Create configuration for a guided 2D MoE
    config_2d = MoEConfig(
        moe_type=MoEType.GUIDED,
        architecture=ArchitectureType.ARCH_2D,
        num_experts=5,
        input_size=(3, 32, 32),  # CIFAR
        output_size=10,
        expert_label_map={  # Example label assignments
            0: [0, 1],
            1: [2, 3],
            2: [4, 5],
            3: [6, 7],
            4: [8, 9]
        }
    )
    
    # Create 2D model
    model_2d = MoEFactory.create_moe(config_2d)
    
    return model_1d, model_2d