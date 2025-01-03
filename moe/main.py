import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from typing import Dict
from tqdm import tqdm

from moe.factories.moe_factory import MoEFactory
from utils.args import parse_args, print_config
from utils.seeding import set_seed
from utils.tracking import ExpertTracker
from moe.factories.datasets_factory import DatasetFactory
from data_handlers.data_utils import DataProcessor
from models.guided_moe_1d import GuidedMoE1D
from models.guided_moe_2d import GuidedMoE2D
from utils.model_utils import create_expert_assignments

def train_epoch(model: nn.Module, train_loader: DataLoader, optimizer: optim.Optimizer,
                data_processor: DataProcessor, device: str, tracker: ExpertTracker,
                epoch: int) -> Dict:
    """Train for one epoch"""
    model.train()
    total_loss = 0
    correct = 0
    total = 0
    all_expert_weights = []
    all_labels = []
    all_predictions = []
    
    for batch_idx, (inputs, targets) in tqdm(enumerate(train_loader), total=len(train_loader), desc='Training'):
        # Process batch according to architecture
        inputs, targets = data_processor.process_batch((inputs, targets))
        inputs, targets = inputs.to(device), targets.to(device)
        
        optimizer.zero_grad()
        
        # Forward pass
        if isinstance(model, (GuidedMoE1D, GuidedMoE2D)):
            outputs, expert_weights, expert_l2_losses = model(inputs, targets)
        else:
            outputs, expert_weights, expert_l2_losses = model(inputs)
        
        # Compute loss
        loss = model.compute_loss(outputs, targets, expert_weights, expert_l2_losses)
        
        # Backward pass
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        
        # Update metrics
        total_loss += loss.item()
        _, predicted = outputs.max(1)
        total += targets.size(0)
        correct += predicted.eq(targets).sum().item()
        
        # Store for later analysis
        all_expert_weights.append(expert_weights.detach())
        all_labels.append(targets)
        all_predictions.append(predicted)
        
        if batch_idx % 50 == 0:  # Log periodically
            tracker.log_expert_activations(
                expert_weights.detach(),
                targets,
                epoch,
                'train'
            )
    
    # Compute epoch metrics
    avg_loss = total_loss / len(train_loader)
    accuracy = 100. * correct / total
    
    # Concatenate all batches
    all_expert_weights = torch.cat(all_expert_weights, dim=0)
    all_labels = torch.cat(all_labels, dim=0)
    all_predictions = torch.cat(all_predictions, dim=0)
    
    # Log confusion matrix
    tracker.log_confusion_matrix(all_labels, all_predictions, epoch, 'train')
    
    return {
        'loss': avg_loss,
        'accuracy': accuracy,
        'expert_weights': all_expert_weights,
        'predictions': all_predictions,
        'targets': all_labels
    }

@torch.no_grad()
def evaluate(model: nn.Module, val_loader: DataLoader, data_processor: DataProcessor,
            device: str, tracker: ExpertTracker, epoch: int) -> Dict:
    """Evaluate model"""
    model.eval()
    total_loss = 0
    correct = 0
    total = 0
    all_expert_weights = []
    all_labels = []
    all_predictions = []
    
    for batch_idx, (inputs, targets) in tqdm(enumerate(val_loader), total=len(val_loader), desc='Validation'):
        # Process batch according to architecture
        inputs, targets = data_processor.process_batch((inputs, targets))
        inputs, targets = inputs.to(device), targets.to(device)
        
        # Forward pass
        if isinstance(model, (GuidedMoE1D, GuidedMoE2D)):
            outputs, expert_weights, expert_l2_losses = model(inputs)
        else:
            outputs, expert_weights, expert_l2_losses = model(inputs)
        
        # Compute loss
        loss = model.compute_loss(outputs, targets, expert_weights, expert_l2_losses)
        
        # Update metrics
        total_loss += loss.item()
        _, predicted = outputs.max(1)
        total += targets.size(0)
        correct += predicted.eq(targets).sum().item()
        
        # Store for later analysis
        all_expert_weights.append(expert_weights)
        all_labels.append(targets)
        all_predictions.append(predicted)
        
        if batch_idx % 50 == 0:  # Log periodically
            tracker.log_expert_activations(
                expert_weights,
                targets,
                epoch,
                'val'
            )
    
    # Compute epoch metrics
    avg_loss = total_loss / len(val_loader)
    accuracy = 100. * correct / total
    
    # Concatenate all batches
    all_expert_weights = torch.cat(all_expert_weights, dim=0)
    all_labels = torch.cat(all_labels, dim=0)
    all_predictions = torch.cat(all_predictions, dim=0)
    
    # Log confusion matrix
    tracker.log_confusion_matrix(all_labels, all_predictions, epoch, 'val')
    
    return {
        'loss': avg_loss,
        'accuracy': accuracy,
        'expert_weights': all_expert_weights,
        'predictions': all_predictions,
        'targets': all_labels
    }

def main():
    # Parse arguments
    config = parse_args()
    print_config(config)
    # Set device
    device = torch.device(config.device if torch.cuda.is_available() else "cpu")
    
    set_seed(config.seed)
    
    # Create data loaders
    train_loader, val_loader = DatasetFactory.create_dataloaders(
    dataset=config.dataset_name,
    data_dir=config.data_dir,
    batch_size=config.training_config.batch_size,
    num_workers=config.training_config.num_workers,
    architecture=config.moe_config.architecture
)   
    # Create model using factory
    model = MoEFactory.create_moe(config.moe_config).to(device)
    # Create optimizer
    optimizer = optim.Adam(
        model.parameters(),
        lr=config.training_config.learning_rate,
        weight_decay=config.training_config.weight_decay
    )
    
    # Create tracker
    dcfg = DatasetFactory.get_dataset_config(config.dataset_name, config.moe_config.architecture)
    expert_assignments = create_expert_assignments(dcfg["num_classes"], config.moe_config.num_experts)
    tracker = ExpertTracker(
        model_type=config.moe_config.moe_type.name,
        architecture=config.moe_config.architecture.name,
        base_path=config.output_dir,
        dataset_name=config.dataset_name,
        num_experts=config.moe_config.num_experts,
        expert_label_assignments=expert_assignments if config.moe_config.moe_type.name == 'GUIDED' else None
    )
    
    # Create data processor
    data_processor = DataProcessor(config.moe_config.architecture)
    
    # Training loop
    best_val_acc = 0
    for epoch in range(config.training_config.num_epochs):
        # Train
        train_metrics = train_epoch(
            model, train_loader, optimizer, data_processor, device, tracker, epoch
        )
        
        # Evaluate
        val_metrics = evaluate(
            model, val_loader, data_processor, device, tracker, epoch
        )
        
        # Log metrics
        tracker.log_metrics(epoch, train_metrics, val_metrics)
        
        # Save best model
        if val_metrics['accuracy'] > best_val_acc:
            best_val_acc = val_metrics['accuracy']
            tracker.save_model(model, optimizer, epoch, val_metrics, is_best=True)
        
        # Regular checkpoint
        if epoch % 10 == 0:
            tracker.save_model(model, optimizer, epoch, val_metrics)
        
        # Print progress
        print(f"Epoch {epoch}:")
        print(f"Train Loss: {train_metrics['loss']:.4f}, Train Acc: {train_metrics['accuracy']:.2f}%")
        print(f"Val Loss: {val_metrics['loss']:.4f}, Val Acc: {val_metrics['accuracy']:.2f}%")
        print("-" * 50)

if __name__ == "__main__":
    main()