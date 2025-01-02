from dataclasses import dataclass, field
from typing import Optional, Tuple, Dict, List

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
    num_epochs: int = 100
    learning_rate: float = 0.001
    weight_decay: float = 1e-5
    clip_grad_norm: float = 1.0
    early_stopping_patience: int = 10

@dataclass
class Config:    
    model: ModelConfig = field(default_factory=ModelConfig)
    data: DataConfig = field(default_factory=DataConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    seed: int = 42
    device: str = 'cuda'
    output_dir: str = 'outputs'
    dataset_name:str = ""
    