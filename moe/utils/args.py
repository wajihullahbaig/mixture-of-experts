import argparse
from configs.default_config import Config, ModelConfig, DataConfig, TrainingConfig

def parse_args() -> Config:
    parser = argparse.ArgumentParser(description='Mixture of Experts Training')
    
    # Model arguments
    parser.add_argument('--model-type', type=str, default='guided',
                      choices=['base', 'guided'],
                      help='Type of MoE model')
    parser.add_argument('--architecture', type=str, default='2d',
                      choices=['1d', '2d'],
                      help='Network architecture type')
    parser.add_argument('--num-experts', type=int, default=5,
                      help='Number of experts')
    parser.add_argument('--hidden-size', type=int, default=512,
                      help='Hidden layer size')
    
    # Data arguments
    parser.add_argument('--dataset', type=str, default='cifar',
                      choices=['mnist', 'cifar'],
                      help='Dataset to use')
    parser.add_argument('--batch-size', type=int, default=128,
                      help='Batch size')
    parser.add_argument('--num-workers', type=int, default=4,
                      help='Number of data loading workers')
    parser.add_argument('--data-dir', type=str, default='data',
                      help='Data directory')
    
    # Training arguments
    parser.add_argument('--num-epochs', type=int, default=100,
                      help='Number of epochs')
    parser.add_argument('--learning-rate', type=float, default=0.001,
                      help='Learning rate')
    parser.add_argument('--weight-decay', type=float, default=1e-5,
                      help='Weight decay')
    
    # General arguments
    parser.add_argument('--seed', type=int, default=42,
                      help='Random seed')
    parser.add_argument('--device', type=str, default='cuda',
                      help='Device to use')
    parser.add_argument('--output-dir', type=str, default='outputs',
                      help='Output directory')
    
    args = parser.parse_args()
    
    # Create config from arguments
    config = Config(
        model=ModelConfig(
            model_type=args.model_type,
            architecture=args.architecture,
            num_experts=args.num_experts,
            hidden_size=args.hidden_size
        ),
        data=DataConfig(
            dataset=args.dataset,
            batch_size=args.batch_size,
            num_workers=args.num_workers,
            data_dir=args.data_dir
        ),
        training=TrainingConfig(
            num_epochs=args.num_epochs,
            learning_rate=args.learning_rate,
            weight_decay=args.weight_decay
        ),
        seed=args.seed,
        device=args.device,
        output_dir=args.output_dir,
        dataset_name=args.dataset        
    )
    
    return config