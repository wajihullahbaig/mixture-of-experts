import random
from typing import Optional
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
import os
import json
from tqdm import tqdm
import numpy as np
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from torchinfo import summary

def set_seed(seed: Optional[int] = 42) -> None:
    """
    Set all random seeds for reproducibility
    """
    if seed is not None:
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        os.environ['PYTHONHASHSEED'] = str(seed)

class GuidedExpert(nn.Module):
    """
    Improved expert network with better normalization and architecture
    """
    def __init__(self, input_size, hidden_size, output_size, assigned_labels):
        super(GuidedExpert, self).__init__()
        self.assigned_labels = assigned_labels
        
        # Add batch normalization and better layer sizing
        self.network = nn.Sequential(
            nn.BatchNorm1d(input_size),
            nn.Linear(input_size, hidden_size),
            nn.ReLU(),
            nn.BatchNorm1d(hidden_size),
            nn.Dropout(0.01),
            nn.Linear(hidden_size, hidden_size // 2),
            nn.ReLU(),
            nn.BatchNorm1d(hidden_size // 2),
            nn.Dropout(0.01),
            nn.Linear(hidden_size // 2, output_size)
        )
        
        # Initialize weights using Xavier/Glorot initialization
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight)
                nn.init.constant_(m.bias, 0)
    
    def forward(self, x):
        return self.network(x)

class GuidedGatingNetwork(nn.Module):
    """
    Improved gating network with better regularization and temperature scaling
    """
    def __init__(self, input_size, num_experts, hidden_size, expert_label_map):
        super(GuidedGatingNetwork, self).__init__()
        self.expert_label_map = expert_label_map
        
        # Add batch normalization and adjusted architecture
        self.network = nn.Sequential(
            nn.BatchNorm1d(input_size),
            nn.Linear(input_size, hidden_size),
            nn.ReLU(),
            nn.BatchNorm1d(hidden_size),
            nn.Dropout(0.01),
            nn.Linear(hidden_size, hidden_size // 2),
            nn.ReLU(),
            nn.BatchNorm1d(hidden_size // 2),
            nn.Dropout(0.01),
            nn.Linear(hidden_size // 2, num_experts)
        )
        
        # Learnable temperature with constraint
        self.temperature = nn.Parameter(torch.ones(1) * 1.0)
        
        # Initialize weights
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight)
                nn.init.constant_(m.bias, 0)
    
    def forward(self, x, labels=None):
        logits = self.network(x)
        
        # Get temperature with minimum bound to prevent division by very small numbers
        #temperature = torch.exp(self.log_temperature).clamp(min=0.1)
        
        if self.training and labels is not None:
            # Create expert mask based on label assignments
            batch_size = labels.size(0)
            expert_mask = torch.zeros((batch_size, len(self.expert_label_map)), 
                                   device=labels.device)
            
            for i, label in enumerate(labels):
                for expert_idx, assigned_labels in self.expert_label_map.items():
                    if label.item() in assigned_labels:
                        expert_mask[i, expert_idx] = 1.0
            
            # Add small epsilon to prevent zero probabilities
            expert_mask = expert_mask + 1e-6
            
            # Apply mask and temperature scaling
            masked_logits = logits * expert_mask
            scaled_logits = masked_logits / self.temperature
            
            # Add label smoothing to prevent overconfident predictions
            smoothed_probs = F.softmax(scaled_logits, dim=-1)
            smoothed_probs = smoothed_probs * 0.9 + 0.1 / len(self.expert_label_map)
            
            return smoothed_probs
        else:
            # During inference, use normal softmax with temperature
            scaled_logits = logits / self.temperature
            return F.softmax(scaled_logits, dim=-1)

class GuidedMixtureOfExperts(nn.Module):
    """
    Simplified Guided Mixture of Experts model with basic loss handling
    """
    def __init__(self, input_size, hidden_size, output_size, num_experts, expert_label_assignments):
        super(GuidedMixtureOfExperts, self).__init__()
        self.num_experts = num_experts
        self.expert_label_assignments = expert_label_assignments
        
        # Initialize experts
        self.experts = nn.ModuleList([
            GuidedExpert(input_size, hidden_size, output_size, assigned_labels) 
            for assigned_labels in expert_label_assignments.values()
        ])
        
        # Initialize gating network
        self.gating_network = GuidedGatingNetwork(
            input_size, 
            num_experts, 
            hidden_size,
            expert_label_assignments
        )
    
    def forward(self, x, labels=None):
        # Get expert weights from gating network
        expert_weights = self.gating_network(x, labels)
        
        # Get outputs from each expert
        expert_outputs = torch.stack([expert(x) for expert in self.experts])
        
        # Combine expert outputs
        expert_outputs = expert_outputs.permute(1, 0, 2)
        expert_weights = expert_weights.unsqueeze(-1)
        
        # Add small epsilon to prevent numerical instability
        expert_weights = expert_weights + 1e-6
        expert_weights = expert_weights / expert_weights.sum(dim=1, keepdim=True)
        
        final_output = torch.bmm(
            expert_outputs.transpose(1, 2),
            expert_weights
        ).squeeze(-1)
        
        return final_output, expert_weights
    
    def compute_loss(self, final_output, target):
        # Simple classification loss using cross entropy
        return F.cross_entropy(final_output, target)

class ExpertActivationTracker:
    """
    Tracks and visualizes expert activations and metrics for both training and testing
    """
    def __init__(self, phase='train', base_path=None, num_experts=None, expert_label_assignments=None):
        self.timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        self.phase = phase
        self.base_dir = os.path.join(base_path or f'guided_moe_outputs/{phase}', self.timestamp)
        self.plots_dir = os.path.join(self.base_dir, 'plots')
        self.data_dir = os.path.join(self.base_dir, 'csv')
        self.model_dir = os.path.join(self.base_dir, 'models')
        
        # Create directories
        for directory in [self.plots_dir, self.data_dir, self.model_dir]:
            os.makedirs(directory, exist_ok=True)
        
        self.num_experts = num_experts
        self.expert_label_assignments = expert_label_assignments
        
        # Save configuration
        self.config = {
            'created_at': self.timestamp,
            'base_dir': self.base_dir,
            'num_experts': num_experts,
            'expert_label_assignments': {
                str(k): v for k, v in expert_label_assignments.items()
            } if expert_label_assignments else None
        }
        
        with open(os.path.join(self.base_dir, 'config.json'), 'w') as f:
            json.dump(self.config, f, indent=4)
        
        # Initialize metrics tracking
        self.metrics = {
            'train_loss': [],
            'train_acc': [],
            'expert_usage': [],
            'expert_specialization': []
        }
    
    def save_expert_activations(self, expert_weights, labels, epoch, batch_idx):
        """Save and visualize expert activations"""
        probs_np = expert_weights.squeeze(-1).detach().cpu().numpy()
        labels_np = labels.cpu().numpy()
        
        activation_data = []
        for sample_idx in range(len(probs_np)):
            for expert_idx, prob in enumerate(probs_np[sample_idx]):
                assigned_labels = self.expert_label_assignments[expert_idx]
                is_assigned = labels_np[sample_idx] in assigned_labels
                
                activation_data.append({
                    'Sample': f'Sample_{sample_idx + 1}',
                    'Expert': f'Expert_{expert_idx}',
                    'Probability': float(prob),
                    'Label': int(labels_np[sample_idx]),
                    'Is_Assigned': is_assigned,
                    'Assigned_Labels': str(assigned_labels),
                    'Epoch': epoch,
                    'Batch': batch_idx
                })
        
        df = pd.DataFrame(activation_data)
        
        # Save to CSV
        csv_path = os.path.join(
            self.data_dir, 
            f'activations_epoch_{epoch}_batch_{batch_idx}.csv'
        )
        df.to_csv(csv_path, index=False)
        
        # Create visualizations
        self._create_activation_heatmap(df, epoch, batch_idx)
        self._create_expert_usage_plot(df, epoch, batch_idx)
        self._create_label_distribution_plot(df, epoch, batch_idx)
        
        return df

    def _create_activation_heatmap(self, df, epoch, batch_idx):
        plt.figure(figsize=(12, 8))
        pivot_df = df.pivot(
            index='Sample',
            columns='Expert',
            values='Probability'
        )
        
        sns.heatmap(
            pivot_df,
            annot=True,
            fmt='.3f',
            cmap='YlOrRd'
        )
        
        plt.title(f'Expert Activation Heatmap (Epoch {epoch}, Batch {batch_idx})')
        plt.tight_layout()
        plt.savefig(
            os.path.join(
                self.plots_dir,
                f'heatmap_epoch_{epoch}_batch_{batch_idx}.png'
            )
        )
        plt.close()

    def _create_expert_usage_plot(self, df, epoch, batch_idx):
        plt.figure(figsize=(12, 6))
        
        # Calculate mean activations for assigned vs non-assigned samples
        assigned_activations = df[df['Is_Assigned']].groupby('Expert')['Probability'].mean()
        non_assigned_activations = df[~df['Is_Assigned']].groupby('Expert')['Probability'].mean()
        
        # Plot
        x = np.arange(len(assigned_activations))
        width = 0.35
        
        plt.bar(x - width/2, assigned_activations, width, label='Assigned Labels')
        plt.bar(x + width/2, non_assigned_activations, width, label='Non-assigned Labels')
        
        plt.xlabel('Expert')
        plt.ylabel('Mean Activation Probability')
        plt.title(f'Expert Usage - Assigned vs Non-assigned (Epoch {epoch}, Batch {batch_idx})')
        plt.legend()
        plt.xticks(x, assigned_activations.index)
        
        plt.tight_layout()
        plt.savefig(
            os.path.join(
                self.plots_dir,
                f'expert_usage_epoch_{epoch}_batch_{batch_idx}.png'
            )
        )
        plt.close()

    def _create_label_distribution_plot(self, df, epoch, batch_idx):
        plt.figure(figsize=(12, 6))
        
        pivot_df = df.pivot_table(
            index='Label',
            columns='Expert',
            values='Probability',
            aggfunc='mean'
        )
        
        sns.heatmap(
            pivot_df,
            annot=True,
            fmt='.3f',
            cmap='YlOrRd'
        )
        
        plt.title(f'Expert Specialization by Label (Epoch {epoch}, Batch {batch_idx})')
        plt.xlabel('Expert')
        plt.ylabel('Label')
        
        # Add assigned label annotations
        for expert_idx, labels in self.expert_label_assignments.items():
            plt.text(
                expert_idx + 0.5,
                -0.5,
                f'Assigned: {labels}',
                ha='center',
                va='center',
                rotation=45
            )
        
        plt.tight_layout()
        plt.savefig(
            os.path.join(
                self.plots_dir,
                f'label_distribution_epoch_{epoch}_batch_{batch_idx}.png'
            )
        )
        plt.close()

    def save_confusion_matrix(self, y_true, y_pred, epoch):
        """Generate and save confusion matrix"""
        if torch.is_tensor(y_true):
            y_true = y_true.cpu().numpy()
        if torch.is_tensor(y_pred):
            y_pred = y_pred.cpu().numpy()
            
        cm = confusion_matrix(y_true, y_pred)
        
        plt.figure(figsize=(10, 8))
        disp = ConfusionMatrixDisplay(confusion_matrix=cm)
        disp.plot(cmap='Blues')
        
        plt.title(f'Confusion Matrix - Epoch {epoch}')
        plt.tight_layout()
        plt.savefig(
            os.path.join(
                self.plots_dir,
                f'confusion_matrix_epoch_{epoch}.png'
            )
        )
        plt.close()
        
        # Save metrics
        metrics = {
            'accuracy': np.trace(cm) / np.sum(cm),
            'per_class_accuracy': np.diag(cm) / np.sum(cm, axis=1),
            'per_class_precision': np.diag(cm) / np.sum(cm, axis=0)
        }
        
        return cm, metrics

    def update_metrics(self, epoch, loss, accuracy, expert_usage):
        """
        Update and save training metrics
        
        Args:
            epoch: Current epoch number
            loss: Training loss
            accuracy: Training accuracy
            expert_usage: Expert usage statistics
        """
        # Store metrics
        self.metrics['train_loss'].append((epoch, loss))
        self.metrics['train_acc'].append((epoch, accuracy))
        self.metrics['expert_usage'].append((epoch, expert_usage))
        
        # Save metrics to CSV
        metrics_df = pd.DataFrame({
            'epoch': [epoch],
            'loss': [loss],
            'accuracy': [accuracy],
            'expert_usage': [str(expert_usage.tolist())]
        })
        
        metrics_path = os.path.join(self.data_dir, 'training_metrics.csv')
        if os.path.exists(metrics_path):
            metrics_df.to_csv(metrics_path, mode='a', header=False, index=False)
        else:
            metrics_df.to_csv(metrics_path, index=False)
        
        # Create training progress plots
        self._plot_training_progress()
    
    def _plot_training_progress(self):
        """Create plots showing training progress over epochs"""
        # Loss plot
        plt.figure(figsize=(10, 6))
        losses = np.array(self.metrics['train_loss'])
        plt.plot(losses[:, 0], losses[:, 1], marker='o')
        plt.title('Training Loss Over Time')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.grid(True)
        plt.savefig(os.path.join(self.plots_dir, 'training_loss.png'))
        plt.close()
        
        # Accuracy plot
        plt.figure(figsize=(10, 6))
        accuracies = np.array(self.metrics['train_acc'])
        plt.plot(accuracies[:, 0], accuracies[:, 1], marker='o', color='green')
        plt.title('Training Accuracy Over Time')
        plt.xlabel('Epoch')
        plt.ylabel('Accuracy (%)')
        plt.grid(True)
        plt.savefig(os.path.join(self.plots_dir, 'training_accuracy.png'))
        plt.close()
        
        # Expert usage plot
        plt.figure(figsize=(12, 6))
        expert_usage = np.array([usage[1] for usage in self.metrics['expert_usage']])
        epochs = np.array([usage[0] for usage in self.metrics['expert_usage']])
        
        for expert_idx in range(expert_usage.shape[1]):
            plt.plot(
                epochs, 
                expert_usage[:, expert_idx],
                marker='o',
                label=f'Expert {expert_idx} {self.expert_label_assignments[expert_idx]}'
            )
            
        plt.title('Expert Usage Over Time')
        plt.xlabel('Epoch')
        plt.ylabel('Usage Probability')
        plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        plt.grid(True)
        plt.tight_layout()
        plt.savefig(os.path.join(self.plots_dir, 'expert_usage_over_time.png'))
        plt.close()

    def save_model(self, model, optimizer, epoch, loss):
        """Save model checkpoint"""
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'loss': loss,
            'expert_label_assignments': model.expert_label_assignments
        }
        
        torch.save(
            checkpoint,
            os.path.join(self.model_dir, f'checkpoint_epoch_{epoch}.pt')
        )

def create_label_assignments(num_classes, num_experts, labels_per_expert):
    """
    Create continuous label assignments for experts
    """
    assignments = {}
    labels = list(range(num_classes))
    
    for expert_idx in range(num_experts):
        start_idx = expert_idx * labels_per_expert
        end_idx = min(start_idx + labels_per_expert, num_classes)
        
        if end_idx <= start_idx:
            break
            
        assignments[expert_idx] = labels[start_idx:end_idx]
    
    return assignments

def train_epoch(model, train_loader, optimizer, device, tracker, epoch):
    """
    Simplified training loop for one epoch
    """
    model.train()
    total_loss = 0
    correct = 0
    total = 0
    all_preds = []
    all_labels = []
    
    progress_bar = tqdm(train_loader, desc=f'Epoch {epoch}')
    
    for batch_idx, (data, target) in enumerate(progress_bar):
        data, target = data.to(device), target.to(device)
        data = data.view(data.size(0), -1)
        
        optimizer.zero_grad()
        
        # Forward pass
        final_output, expert_weights = model(data, target)
        
        # Compute simplified loss
        loss = model.compute_loss(final_output, target)
        
        # Backward pass with gradient clipping
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        
        # Accumulate loss
        total_loss += loss.item()
        
        # Calculate accuracy
        pred = final_output.argmax(dim=1)
        correct += pred.eq(target).sum().item()
        total += target.size(0)
        
        # Store predictions and labels for confusion matrix
        all_preds.extend(pred.cpu().numpy())
        all_labels.extend(target.cpu().numpy())
        
        # Save activations and expert usage periodically
        if batch_idx % 200 == 0:
            tracker.save_expert_activations(
                expert_weights,
                target,
                epoch,
                batch_idx
            )
            
            # Update progress bar with metrics
            progress_bar.set_postfix({
                'loss': f'{total_loss/(batch_idx+1):.3f}',
                'acc': f'{100.*correct/total:.2f}%'
            })
    
    # Calculate epoch metrics
    epoch_loss = total_loss / len(train_loader)
    epoch_acc = 100. * correct / total
    
    # Calculate final expert usage for the epoch
    expert_usage = expert_weights.mean(dim=0).squeeze().detach().cpu().numpy()
    
    # Generate and save confusion matrix
    cm, cm_metrics = tracker.save_confusion_matrix(
        np.array(all_labels),
        np.array(all_preds),
        epoch
    )       

    return epoch_loss, epoch_acc, expert_usage, cm_metrics

def print_dataset_info(train_loader):
    """Print information about the dataset"""
    total_samples = len(train_loader.dataset)
    batch_size = train_loader.batch_size
    num_batches = len(train_loader)
    
    print("\nDataset Information:")
    print("=" * 50)
    print(f"Total number of samples: {total_samples:,}")
    print(f"Batch size: {batch_size}")
    print(f"Number of batches: {num_batches}")
    print(f"Input shape: {train_loader.dataset[0][0].shape}")
    print(f"Number of classes: {len(set(train_loader.dataset.targets.numpy()))}")
    print("=" * 50)

def test_epoch(model, val_loader, device, tracker, epoch):
    """
    Simplified validation loop
    """
    model.eval()
    total_loss = 0
    correct = 0
    total = 0
    all_preds = []
    all_labels = []
    
    with torch.no_grad():
        progress_bar = tqdm(val_loader, desc=f'Validation Epoch {epoch}')
        
        for batch_idx, (data, target) in enumerate(progress_bar):
            data, target = data.to(device), target.to(device)
            data = data.view(data.size(0), -1)
            
            # Forward pass
            final_output, expert_weights = model(data)
            
            # Compute loss
            loss = model.compute_loss(final_output, target)
            
            # Accumulate loss
            total_loss += loss.item()
            
            # Calculate accuracy
            pred = final_output.argmax(dim=1)
            correct += pred.eq(target).sum().item()
            total += target.size(0)
            
            # Store predictions and labels
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
    
    # Calculate epoch metrics
    epoch_loss = total_loss / len(val_loader)
    epoch_acc = 100. * correct / total
    
    # Calculate expert usage
    expert_usage = expert_weights.mean(dim=0).squeeze().detach().cpu().numpy()
    
    # Generate and save confusion matrix
    cm, cm_metrics = tracker.save_confusion_matrix(
        np.array(all_labels),
        np.array(all_preds),
        epoch
    )
    
    return epoch_loss, epoch_acc, expert_usage, cm_metrics

def main():
    # Set random seed
    set_seed(42)
    
    # Hyperparameters
    input_size = 784  # 28x28 MNIST images
    hidden_size = 256
    output_size = 10  # 10 digits
    num_experts = 5   # Using 5 experts
    labels_per_expert = 2  # Each expert handles 2 consecutive digits
    learning_rate = 0.001
    num_epochs = 100
    batch_size = 256
    
    # Create label assignments
    expert_label_assignments = create_label_assignments(
        output_size,
        num_experts,
        labels_per_expert
    )
    
    print("\nExpert Label Assignments:")
    print("=" * 50)
    for expert_idx, labels in expert_label_assignments.items():
        print(f"Expert {expert_idx}: Labels {labels}")
    print("=" * 50)
    
    # Setup device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"\nUsing device: {device}")
    
    # Load MNIST dataset
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])
    
    # Training dataset
    train_dataset = datasets.MNIST(
        './data',
        train=True,
        download=True,
        transform=transform
    )
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True
    )
    
    # Test dataset
    test_dataset = datasets.MNIST(
        './data',
        train=False,
        download=True,
        transform=transform
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False
    )
    
    # Initialize model and optimizer
    model = GuidedMixtureOfExperts(
        input_size,
        hidden_size,
        output_size,
        num_experts,
        expert_label_assignments
    ).to(device)
    
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    
    # Print model summary
    print("\nModel Architecture Summary:")
    print("=" * 50)
    print("\nComplete Guided MoE Model:")
    summary(model, input_size=(batch_size, input_size))
    
    print("\nSingle Expert Architecture:")
    summary(model.experts[0], input_size=(batch_size, input_size))
    
    print("\nGating Network Architecture:")
    summary(model.gating_network, input_size=(batch_size, input_size))
    
    # Print dataset information
    print_dataset_info(train_loader)
    
    # Initialize trackers for training and testing
    train_tracker = ExpertActivationTracker(
        phase='train',
        num_experts=num_experts,
        expert_label_assignments=expert_label_assignments
    )
    
    test_tracker = ExpertActivationTracker(
        phase='test',
        num_experts=num_experts,
        expert_label_assignments=expert_label_assignments
    )
    
    # Training and testing loop
    print("\nStarting training and testing...")
    print("=" * 50)
    
    best_test_acc = 0
    
    for epoch in range(num_epochs):
        # Training phase
        train_loss, train_accuracy, train_expert_usage, train_cm_metrics = train_epoch(
            model,
            train_loader,
            optimizer,
            device,
            train_tracker,
            epoch + 1
        )
        
        # Testing phase
        test_loss, test_accuracy, test_expert_usage, test_cm_metrics = test_epoch(
            model,
            test_loader,
            device,
            test_tracker,
            epoch + 1
        )
        
        # Update and save metrics
        train_tracker.update_metrics(epoch + 1, train_loss, train_accuracy, train_expert_usage)
        test_tracker.update_metrics(epoch + 1, test_loss, test_accuracy, test_expert_usage)
        
        # Save model checkpoint (save best model based on test accuracy)
        if test_accuracy > best_test_acc:
            best_test_acc = test_accuracy
            train_tracker.save_model(model, optimizer, epoch + 1, train_loss)
            test_tracker.save_model(model, optimizer, epoch + 1, test_loss)
        
        # Print metrics
        print(f'\nEpoch {epoch + 1}:')
        print(f'Training Loss: {train_loss:.4f}, Accuracy: {train_accuracy:.2f}%')
        print(f'Test Loss: {test_loss:.4f}, Accuracy: {test_accuracy:.2f}%')
        
        print('\nTraining Expert Usage:')
        for expert_idx, usage in enumerate(train_expert_usage):
            assigned_labels = expert_label_assignments[expert_idx]
            print(f'Expert {expert_idx} (Labels {assigned_labels}): {usage:.3f}')
        
        print('\nTest Expert Usage:')
        for expert_idx, usage in enumerate(test_expert_usage):
            assigned_labels = expert_label_assignments[expert_idx]
            print(f'Expert {expert_idx} (Labels {assigned_labels}): {usage:.3f}')
        
        print('\nTraining Per-class Metrics:')
        for class_idx in range(len(train_cm_metrics['per_class_accuracy'])):
            print(f'Class {class_idx}:')
            print(f'  Accuracy: {train_cm_metrics["per_class_accuracy"][class_idx]:.3f}')
            print(f'  Precision: {train_cm_metrics["per_class_precision"][class_idx]:.3f}')
            
        print('\nTest Per-class Metrics:')
        for class_idx in range(len(test_cm_metrics['per_class_accuracy'])):
            print(f'Class {class_idx}:')
            print(f'  Accuracy: {test_cm_metrics["per_class_accuracy"][class_idx]:.3f}')
            print(f'  Precision: {test_cm_metrics["per_class_precision"][class_idx]:.3f}')
        print("=" * 50)

if __name__ == '__main__':
    main()