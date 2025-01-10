import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from typing import Dict
from tqdm import tqdm
import sys

from moe.models.mixtures.guided_moe_1d import GuidedMoE1D
from moe.models.mixtures.guided_moe_2d import GuidedMoE2D
from moe.models.mixtures.guided_timm_moe_2d import GuidedTimmMoE2D
from moe.utils.model_utils import check_for_nans
from utils.tracking import ExpertTracker
from data_handlers.data_utils import DataProcessor
from utils.app_logger import AppLogger


def train_epoch(model: nn.Module, train_loader: DataLoader, optimizer: optim.Optimizer,
                data_processor: DataProcessor, device: str, tracker: ExpertTracker,
                epoch: int, nan_check:bool) -> Dict:
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
        
        # Forward pass - Guided versions need guidance from labels during training.
        if isinstance(model, (GuidedMoE1D, GuidedMoE2D,GuidedTimmMoE2D)):
            outputs, expert_weights, expert_l2_losses = model(inputs, targets)
        else:
            outputs, expert_weights, expert_l2_losses = model(inputs)
        
        if nan_check:
            nans_list = check_for_nans([outputs,expert_weights,expert_l2_losses])
            if nans_list:
                logger = AppLogger(__name__)            
                logger.error(f"NaNs detected in tensors: {nans_list}")
                raise ValueError("NaNs detected in tensors.")
                
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
    
    if nan_check:
        nans_list = check_for_nans([all_expert_weights,all_labels,all_predictions])
        if nans_list:
            logger = AppLogger(__name__)            
            logger.error(f"NaNs detected in tensors: {nans_list}")
            raise ValueError("NaNs detected in tensors.")

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
            device: str, tracker: ExpertTracker, epoch: int, nan_check:bool) -> Dict:
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

        if nan_check:
            nans_list = check_for_nans([outputs,expert_weights,expert_l2_losses])
            if nans_list:
                logger = AppLogger(__name__)            
                logger.error(f"NaNs detected in tensors: {nans_list}")
                raise ValueError("NaNs detected in tensors.")
        
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
    
    if nan_check:
        nans_list = check_for_nans([all_expert_weights,all_labels,all_predictions])
        if nans_list:
            logger = AppLogger(__name__)            
            logger.error(f"NaNs detected in tensors: {nans_list}")
            raise ValueError("NaNs detected in tensors.")    
    # Log confusion matrix
    tracker.log_confusion_matrix(all_labels, all_predictions, epoch, 'val')
    
    return {
        'loss': avg_loss,
        'accuracy': accuracy,
        'expert_weights': all_expert_weights,
        'predictions': all_predictions,
        'targets': all_labels
    }
