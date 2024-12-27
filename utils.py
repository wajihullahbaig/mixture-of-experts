"""
utils.py - Shared utility functions and training loops for MoE models
"""

import os
import random
from typing import Optional, Dict, Tuple, Union, List
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import numpy as np
from tqdm import tqdm
from torchinfo import summary

from base_moe import BaseMixtureOfExperts
from guided_moe import GuidedMixtureOfExperts
from tracking import BaseExpertTracker

def set_seed(seed: Optional[int] = 42) -> None:
    """Set all random seeds for reproducibility"""
    if seed is not None:
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        os.environ['PYTHONHASHSEED'] = str(seed)

def print_model_info(model: Union[BaseMixtureOfExperts, GuidedMixtureOfExperts], 
                    batch_size: int, input_size: int):
    """Print model architecture summary"""
    print("\nModel Architecture Summary:")
    print("=" * 50)
    print("\nComplete Model:")
    summary(model, input_size=(batch_size, input_size))
    
    print("\nSingle Expert Architecture:")
    summary(model.experts[0], input_size=(batch_size, input_size))
    
    print("\nGating Network Architecture:")
    summary(model.gating_network, input_size=(batch_size, input_size))

def print_dataset_info(data_loader: DataLoader, dataset_type: str):
    """Print information about the dataset"""
    total_samples = len(data_loader.dataset)
    batch_size = data_loader.batch_size
    num_batches = len(data_loader)
    
    print(f"\n{dataset_type} Dataset Information:")
    print("=" * 50)
    print(f"Total number of samples: {total_samples:,}")
    print(f"Batch size: {batch_size}")
    print(f"Number of batches: {num_batches}")
    print(f"Input shape: {data_loader.dataset[0][0].shape}")
    print(f"Number of classes: {len(set(data_loader.dataset.targets.numpy()))}")
    print("=" * 50)

def train_epoch(model: Union[BaseMixtureOfExperts, GuidedMixtureOfExperts],
               train_loader: DataLoader,
               optimizer: optim.Optimizer,
               device: torch.device,
               tracker: BaseExpertTracker,
               epoch: int) -> Tuple[float, float, np.ndarray, Dict]:
    """
    Train for one epoch and track metrics
    
    Args:
        model: MoE model (base or guided)
        train_loader: Training data loader
        optimizer: PyTorch optimizer
        device: Device to use for computation
        tracker: Expert activation tracker
        epoch: Current epoch number
        
    Returns:
        Tuple containing:
            - Epoch loss
            - Epoch accuracy
            - Expert usage statistics
            - Confusion matrix metrics
    """
    model.train()
    total_loss = 0
    correct = 0
    total = 0
    all_preds = []
    all_labels = []
    expert_weight_accumulator = torch.zeros(model.num_experts).to(device)
    num_batches = 0
    
    progress_bar = tqdm(train_loader, desc=f'Training Epoch {epoch}')
    
    for batch_idx, (data, target) in enumerate(progress_bar):
        data, target = data.to(device), target.to(device)
        data = data.view(data.size(0), -1)
        
        optimizer.zero_grad()
        
        # Forward pass
        if isinstance(model, GuidedMixtureOfExperts):
            final_output, expert_weights, expert_l2_losses = model(data, target)
            loss = model.compute_loss(final_output, target, expert_weights, expert_l2_losses)
        else:
            final_output, expert_weights = model(data)
            loss = model.compute_loss(final_output, target, expert_weights)
        
        # Backward pass
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        
        # Accumulate expert weights - handle both squeezed and unsqueezed weights
        if expert_weights.dim() == 3:  # If weights are [batch_size, num_experts, 1]
            batch_weights = expert_weights.squeeze(-1)
        elif expert_weights.dim() == 2:  # If weights are [batch_size, num_experts]
            batch_weights = expert_weights
        else:
            raise ValueError(f"Unexpected expert weights shape: {expert_weights.shape}")
            
        expert_weight_accumulator += batch_weights.mean(dim=0).detach()
        num_batches += 1
        
        # Accumulate metrics
        total_loss += loss.item()
        pred = final_output.argmax(dim=1)
        correct += pred.eq(target).sum().item()
        total += target.size(0)
        
        all_preds.extend(pred.cpu().numpy())
        all_labels.extend(target.cpu().numpy())
        
        # Save activations periodically
        if batch_idx % 500 == 0:
            tracker.save_expert_activations(
                expert_weights,
                target,
                epoch,
                batch_idx
            )
            
            # Update progress bar
            progress_bar.set_postfix({
                'loss': f'{total_loss/(batch_idx+1):.3f}',
                'acc': f'{100.*correct/total:.2f}%'
            })
    
    # Calculate final metrics
    epoch_loss = total_loss / len(train_loader)
    epoch_acc = 100. * correct / total
    avg_expert_weights = expert_weight_accumulator / num_batches
    
    # Generate confusion matrix
    cm, cm_metrics = tracker.save_confusion_matrix(
        np.array(all_labels),
        np.array(all_preds),
        epoch
    )
    
    return epoch_loss, epoch_acc, avg_expert_weights.cpu().numpy(), cm_metrics

@torch.no_grad()
def evaluate_epoch(model: Union[BaseMixtureOfExperts, GuidedMixtureOfExperts],
                  val_loader: DataLoader,
                  device: torch.device,
                  tracker: BaseExpertTracker,
                  epoch: int) -> Tuple[float, float, np.ndarray, Dict]:
    """
    Evaluate for one epoch
    
    Args:
        model: MoE model (base or guided)
        val_loader: Validation data loader
        device: Device to use for computation
        tracker: Expert activation tracker
        epoch: Current epoch number
        
    Returns:
        Tuple containing:
            - Epoch loss
            - Epoch accuracy
            - Expert usage statistics
            - Confusion matrix metrics
    """
    model.eval()
    total_loss = 0
    correct = 0
    total = 0
    all_preds = []
    all_labels = []
    expert_weight_accumulator = torch.zeros(model.num_experts).to(device)
    num_batches = 0
    
    progress_bar = tqdm(val_loader, desc=f'Validation Epoch {epoch}')
    
    for batch_idx, (data, target) in enumerate(progress_bar):
        data, target = data.to(device), target.to(device)
        data = data.view(data.size(0), -1)
        
        # Forward pass
        if isinstance(model, GuidedMixtureOfExperts):
            final_output, expert_weights, expert_l2_losses = model(data)
            loss = model.compute_loss(final_output, target, expert_weights, expert_l2_losses)
        else:
            final_output, expert_weights = model(data)
            loss = model.compute_loss(final_output, target, expert_weights)
        
        # Accumulate expert weights
        expert_weight_accumulator += expert_weights.mean(dim=0).detach()
        num_batches += 1
        
        # Accumulate metrics
        total_loss += loss.item()
        pred = final_output.argmax(dim=1)
        correct += pred.eq(target).sum().item()
        total += target.size(0)
        
        all_preds.extend(pred.cpu().numpy())
        all_labels.extend(target.cpu().numpy())
        
        # Save activations periodically
        if batch_idx % 100 == 0:
            tracker.save_expert_activations(
                expert_weights,
                target,
                epoch,
                batch_idx
            )
            
            # Update progress bar
            progress_bar.set_postfix({
                'loss': f'{total_loss/(batch_idx+1):.3f}',
                'acc': f'{100.*correct/total:.2f}%'
            })
    
    # Calculate final metrics
    epoch_loss = total_loss / len(val_loader)
    epoch_acc = 100. * correct / total
    avg_expert_weights = expert_weight_accumulator / num_batches
    
    # Generate confusion matrix
    cm, cm_metrics = tracker.save_confusion_matrix(
        np.array(all_labels),
        np.array(all_preds),
        epoch
    )
    
    return epoch_loss, epoch_acc, avg_expert_weights.cpu().numpy(), cm_metrics

def print_epoch_metrics(epoch: int, 
                       train_metrics: Dict[str, Union[float, np.ndarray]], 
                       val_metrics: Dict[str, Union[float, np.ndarray]], 
                       expert_label_assignments: Optional[Dict[int, List[int]]] = None):
    """Print epoch training and validation metrics"""
    print(f'\nEpoch {epoch}:')
    
    print('Training:')
    print(f'  Loss: {train_metrics["loss"]:.4f}')
    print(f'  Accuracy: {train_metrics["accuracy"]:.2f}%')
    
    if expert_label_assignments:
        print('\nTraining Expert Usage:')
        for expert_idx, usage in enumerate(train_metrics["expert_usage"]):
            assigned_labels = expert_label_assignments[expert_idx]
            print(f'  Expert {expert_idx} (Labels {assigned_labels}): {usage:.3f}')
    else:
        print(f'  Expert Usage: {train_metrics["expert_usage"]}')
        
    print('  Per-class Accuracy:', train_metrics["per_class_accuracy"])
    
    print('\nValidation:')
    print(f'  Loss: {val_metrics["loss"]:.4f}')
    print(f'  Accuracy: {val_metrics["accuracy"]:.2f}%')
    
    if expert_label_assignments:
        print('\nValidation Expert Usage:')
        for expert_idx, usage in enumerate(val_metrics["expert_usage"]):
            assigned_labels = expert_label_assignments[expert_idx]
            print(f'  Expert {expert_idx} (Labels {assigned_labels}): {usage:.3f}')
    else:
        print(f'  Expert Usage: {val_metrics["expert_usage"]}')
        
    print('  Per-class Accuracy:', val_metrics["per_class_accuracy"])
    print("=" * 50)

def save_best_model(model: nn.Module, 
                   optimizer: optim.Optimizer,
                   epoch: int, 
                   accuracy: float,
                   save_path: str):
    """Save model checkpoint"""
    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'accuracy': accuracy
    }
    
    if hasattr(model, 'expert_label_assignments'):
        checkpoint['expert_label_assignments'] = model.expert_label_assignments
        
    torch.save(checkpoint, save_path)