"""
train.py - Training scripts for both Base and Guided MoE models
"""

import argparse
import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

from base_moe import BaseMixtureOfExperts
from guided_moe import GuidedMixtureOfExperts, create_label_assignments
from tracking import BaseExpertTracker
from utils import (
    set_seed, print_model_info, print_dataset_info,
    train_epoch, evaluate_epoch, print_epoch_metrics, save_best_model
)

def parse_args():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description='Train MoE models')
    
    parser.add_argument('--model-type', type=str, default='base', choices=['base', 'guided'],
                        help='Type of MoE model to train')
    parser.add_argument('--input-size', type=int, default=784,
                        help='Input feature dimension')
    parser.add_argument('--hidden-size', type=int, default=256,
                        help='Hidden layer dimension')
    parser.add_argument('--output-size', type=int, default=10,
                        help='Output dimension (num classes)')
    parser.add_argument('--num-experts', type=int, default=10,
                        help='Number of experts in mixture')
    parser.add_argument('--batch-size', type=int, default=256,
                        help='Training batch size')
    parser.add_argument('--epochs', type=int, default=15,
                        help='Number of training epochs')
    parser.add_argument('--lr', type=float, default=0.001,
                        help='Learning rate')
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed')
    parser.add_argument('--output-dir', type=str, default='./outputs',
                        help='Directory to save outputs')
    
    return parser.parse_args()

def load_data(batch_size: int) -> tuple[DataLoader, DataLoader]:
    """Load MNIST dataset"""
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])
    
    train_dataset = datasets.MNIST(
        './data',
        train=True,
        download=True,
        transform=transform
    )
    
    test_dataset = datasets.MNIST(
        './data',
        train=False,
        download=True,
        transform=transform
    )
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False
    )
    
    return train_loader, test_loader

def train_model(args):
    """Main training function"""
    # Set random seed
    set_seed(args.seed)
    
    # Setup device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Load data
    train_loader, test_loader = load_data(args.batch_size)
    
    # Print dataset info
    print_dataset_info(train_loader, "Training")
    print_dataset_info(test_loader, "Test")
    
    # Initialize model
    if args.model_type == 'guided':
        # Create label assignments for guided MoE
        expert_label_assignments = create_label_assignments(
            args.output_size,
            args.num_experts
        )
        
        print("\nExpert Label Assignments:")
        print("=" * 50)
        for expert_idx, labels in expert_label_assignments.items():
            print(f"Expert {expert_idx}: Labels {labels}")
        print("=" * 50)
        
        model = GuidedMixtureOfExperts(
            args.input_size,
            args.hidden_size,
            args.output_size,
            args.num_experts,
            expert_label_assignments
        ).to(device)
    else:
        expert_label_assignments = None
        model = BaseMixtureOfExperts(
            args.input_size,
            args.hidden_size,
            args.output_size,
            args.num_experts
        ).to(device)
    
    # Print model info
    print_model_info(model, args.batch_size, args.input_size)
    
    # Initialize optimizer
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    
    # Initialize trackers
    train_tracker = BaseExpertTracker(
        model_type=args.model_type,
        mode='train',
        base_path=args.output_dir,
        num_experts=args.num_experts,
        expert_label_assignments=expert_label_assignments
    )
    
    test_tracker = BaseExpertTracker(
        model_type=args.model_type,
        mode='test',
        base_path=args.output_dir,
        num_experts=args.num_experts,
        expert_label_assignments=expert_label_assignments
    )
    
    # Training loop
    best_test_acc = 0.0
    print("\nStarting training...")
    print("=" * 50)
    
    for epoch in range(args.epochs):
        # Training phase
        train_loss, train_acc, train_expert_usage, train_cm_metrics = train_epoch(
            model,
            train_loader,
            optimizer,
            device,
            train_tracker,
            epoch + 1
        )
        
        # Testing phase
        test_loss, test_acc, test_expert_usage, test_cm_metrics = evaluate_epoch(
            model,
            test_loader,
            device,
            test_tracker,
            epoch + 1
        )
        
        # Update metrics
        train_metrics = {
            'loss': train_loss,
            'accuracy': train_acc,
            'expert_usage': train_expert_usage,
            'per_class_accuracy': train_cm_metrics['per_class_accuracy']
        }
        
        test_metrics = {
            'loss': test_loss,
            'accuracy': test_acc,
            'expert_usage': test_expert_usage,
            'per_class_accuracy': test_cm_metrics['per_class_accuracy']
        }
        
        # Update trackers
        train_tracker.update_metrics(epoch + 1, train_loss, train_acc, train_expert_usage)
        test_tracker.update_metrics(epoch + 1, test_loss, test_acc, test_expert_usage)
        
        # Save checkpoints
        train_tracker.save_model(model, optimizer, epoch + 1, train_loss)
        
        # Save best model
        if test_acc > best_test_acc:
            best_test_acc = test_acc
            save_best_model(
                model,
                optimizer,
                epoch + 1,
                test_acc,
                f"{test_tracker.model_dir}/best_model.pt"
            )
        
        # Print epoch results
        print_epoch_metrics(
            epoch + 1,
            train_metrics,
            test_metrics,
            expert_label_assignments
        )
    
    print(f"\nTraining completed! Best test accuracy: {best_test_acc:.2f}%")
    print(f"Model checkpoints and metrics saved in {args.output_dir}")

def main():
    """Main entry point"""
    args = parse_args()
    # Make the correct output directory
    args.output_dir = args.output_dir + "/" + args.model_type
    train_model(args)

if __name__ == '__main__':
    main()