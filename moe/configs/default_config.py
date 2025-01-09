from dataclasses import dataclass, field
from enum import Enum, auto
from pathlib import Path
from typing import Optional, Tuple, Dict, List, Union

@dataclass
class ModelConfig:
    model_type: str = 'guided'
    architecture: str = '1d'
    num_experts: int = 5
    hidden_size: int = 512
    dropout_rate: float = 0.3
    l2_reg: float = 0.01

@dataclass
class DataConfig:
    dataset: str = 'mnist'
    batch_size: int = 128
    num_workers: int = 4
    data_dir: str = 'data'

@dataclass
class TrainingConfig:
    """Training configuration"""
    num_epochs: int
    batch_size: int
    learning_rate: float
    weight_decay: float
    num_workers: int
    clip_grad_norm: float
    early_stopping_patience: int

@dataclass
class Config:    
    model: ModelConfig = field(default_factory=ModelConfig)
    data: DataConfig = field(default_factory=DataConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    seed: int = 42
    device: str = 'cuda'
    output_dir: str = 'outputs'
    dataset_name:str = ""

class MoEType(Enum):
    """Types of MoE models available"""
    BASIC = auto()
    GUIDED = auto()

class ArchitectureType(Enum):
    """Types of architectures available"""
    ARCH_1D = auto()
    ARCH_2D = auto()
    ARCH_RESNET18_2D = auto()
    ARCH_TIMM_2D = auto()


@dataclass
class MoEConfig:
    """Configuration for MoE model creation"""
    moe_type: MoEType
    architecture: ArchitectureType
    num_experts: int
    input_size: Union[int, Tuple[int, int, int]]  # int for 1D, tuple for 2D (channels, height, width)
    hidden_size: Optional[int] = None  # Only used for 1D
    output_size: Optional[int] = None  # Number of classes for classification
    expert_label_map: Optional[Dict[int, list]] = None  # Required for guided MoE
    dropout_rate: float = 0.3
    l2_reg: float = 0.01

    def validate(self) -> None:
        """Validate configuration parameters"""
        # Check if input size matches architecture
        if self.architecture == ArchitectureType.ARCH_1D and not isinstance(self.input_size, int):
            raise ValueError("1D architecture requires integer input_size")
        if self.architecture == ArchitectureType.ARCH_2D and not isinstance(self.input_size, tuple):
            raise ValueError("2D architecture requires tuple input_size (channels, height, width)")        
        if self.architecture == ArchitectureType.ARCH_RESNET18_2D and not isinstance(self.input_size, tuple):
            raise ValueError("2D RESNET18 architecture requires tuple input_size (channels, height, width)")        
        if self.architecture == ArchitectureType.ARCH_TIMM_2D and not isinstance(self.input_size, tuple):
            raise ValueError("TIMM architecture requires tuple input_size (channels, height, width)")
        
        # Check if hidden_size is provided
        if (self.architecture == ArchitectureType.ARCH_1D or self.architecture == ArchitectureType.ARCH_TIMM_2D) and self.hidden_size is None:
            raise ValueError("hidden_size is required for 1D or TIMM architecture")
        
        # Check if output_size is provided
        if self.output_size is None:
            raise ValueError("output_size is required")
        
        # Check if expert_label_map is provided for guided MoE
        if self.moe_type == MoEType.GUIDED and self.expert_label_map is None:
            raise ValueError("expert_label_map is required for guided MoE")
        
        # Validate expert_label_map if provided
        if self.expert_label_map is not None:
            # Check if all labels are valid
            all_labels = set()
            for labels in self.expert_label_map.values():
                all_labels.update(labels)
            if max(all_labels) >= self.output_size:
                raise ValueError("expert_label_map contains invalid label indices")
            
            # Check if all experts are assigned
            if set(self.expert_label_map.keys()) != set(range(self.num_experts)):
                raise ValueError("expert_label_map must contain assignments for all experts")


@dataclass
class ExperimentConfig:
    """Complete experiment configuration"""
    moe_config: MoEConfig
    training_config: TrainingConfig
    dataset_name: str
    data_dir: Path
    output_dir: Path
    device: str
    seed: int    