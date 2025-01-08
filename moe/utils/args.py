import argparse
from pathlib import Path
from typing import Dict, List
from dataclasses import dataclass

from moe.factories.datasets_factory import DatasetFactory
from moe.configs.default_config import ArchitectureType, ExperimentConfig, MoEConfig, MoEType, TrainingConfig
from moe.utils.model_utils import create_expert_assignments




def parse_args() -> ExperimentConfig:
    """Parse command line arguments and create configurations"""
    parser = argparse.ArgumentParser(description='Mixture of Experts Training')
    
    # Model arguments
    parser.add_argument('--moe-type', type=str, default='guided',
                      choices=['basic', 'guided'],
                      help='Type of MoE model')
    parser.add_argument('--architecture', type=str, default='resnet18',
                      choices=['1d', '2d', 'resent18'],
                      help='Network architecture type')
    parser.add_argument('--num-experts', type=int, default=5,
                      help='Number of experts')
    parser.add_argument('--hidden-size', type=int, default=256,
                      help='Hidden layer size (for 1D architecture)')
    parser.add_argument('--dropout-rate', type=float, default=0.3,
                      help='Dropout rate')
    parser.add_argument('--l2-reg', type=float, default=0.01,
                      help='L2 regularization strength')
    
    # Dataset arguments
    parser.add_argument('--dataset', type=str, default='cifar10',
                      choices=['mnist', 'cifar10'],
                      help='Dataset to use')
    parser.add_argument('--data-dir', type=str, default='data',
                      help='Data directory')
    
    # Training arguments
    parser.add_argument('--num-epochs', type=int, default=100,
                      help='Number of epochs')
    parser.add_argument('--batch-size', type=int, default=128,
                      help='Batch size')
    parser.add_argument('--learning-rate', type=float, default=0.001,
                      help='Learning rate')
    parser.add_argument('--weight-decay', type=float, default=1e-5,
                      help='Weight decay')
    parser.add_argument('--num-workers', type=int, default=4,
                      help='Number of data loading workers')
    parser.add_argument('--clip-grad-norm', type=float, default=1.0,
                      help='Gradient clipping norm')
    parser.add_argument('--early-stopping-patience', type=int, default=10,
                      help='Early stopping patience epochs')
    
    # General arguments
    parser.add_argument('--seed', type=int, default=42,
                      help='Random seed')
    parser.add_argument('--device', type=str, default='cuda',
                      help='Device to use')
    parser.add_argument('--output-dir', type=str, default='outputs',
                      help='Output directory')
    
    args = parser.parse_args()
    
    # Get dataset configuration
    dataset_config = DatasetFactory.get_dataset_config(args.dataset, args.architecture)
    
    # Create MoE configuration
    
    moe_config = MoEConfig(
        moe_type=MoEType.BASIC if args.moe_type == 'basic' else MoEType.GUIDED,
        architecture = ArchitectureType.ARCH_RESNET18_2D if args.architecture == 'resnet18' else 
        (
            ArchitectureType.ARCH_1D if args.architecture == '1d' else ArchitectureType.ARCH_2D
        )
        ,
        num_experts=args.num_experts,
        input_size=dataset_config["input_size"],
        hidden_size=args.hidden_size,
        output_size=dataset_config["num_classes"],
        dropout_rate=args.dropout_rate,
        l2_reg=args.l2_reg,
        expert_label_map=(
            create_expert_assignments(dataset_config["num_classes"], args.num_experts)
            if args.moe_type == 'guided' else None
        )
    )
    
    # Create training configuration
    training_config = TrainingConfig(
        num_epochs=args.num_epochs,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        num_workers=args.num_workers,
        clip_grad_norm=args.clip_grad_norm,
        early_stopping_patience=args.early_stopping_patience
    )
    
    # Create complete experiment configuration
    config = ExperimentConfig(
        moe_config=moe_config,
        training_config=training_config,
        dataset_name=args.dataset,
        data_dir=Path(args.data_dir),
        output_dir=Path(args.output_dir),
        device=args.device,
        seed=args.seed
    )
    
    return config

def print_config(config: ExperimentConfig) -> None:
    """Print configuration in a readable format"""
    print("\nExperiment Configuration:")
    print("=" * 50)
    
    print("\nMoE Configuration:")
    print("-" * 30)
    for key, value in vars(config.moe_config).items():
        if key == 'expert_label_map' and value is not None:
            print(f"  {key}:")
            for expert_id, labels in value.items():
                print(f"    Expert {expert_id}: {labels}")
        else:
            print(f"  {key}: {value}")
    
    print("\nTraining Configuration:")
    print("-" * 30)
    for key, value in vars(config.training_config).items():
        print(f"  {key}: {value}")
    
    print("\nGeneral Configuration:")
    print("-" * 30)
    print(f"  dataset_name: {config.dataset_name}")
    print(f"  data_dir: {config.data_dir}")
    print(f"  output_dir: {config.output_dir}")
    print(f"  device: {config.device}")
    print(f"  seed: {config.seed}")
    print("\n" + "=" * 50)