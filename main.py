import random
from typing import Optional, Tuple, Dict, List
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
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, classification_report
from torchinfo import summary
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

def set_seed(seed: Optional[int] = 42) -> None:
    """Set all random seeds for reproducibility."""
    if seed is not None:
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        os.environ['PYTHONHASHSEED'] = str(seed)

class Expert(nn.Module):
    """Individual expert network"""
    def __init__(self, input_size: int, hidden_size: int, output_size: int, dropout_rate: float = 0.3):
        super(Expert, self).__init__()
        self.network = nn.Sequential(
            nn.BatchNorm1d(input_size),
            nn.Linear(input_size, hidden_size),
            nn.ReLU(),
            nn.BatchNorm1d(hidden_size),
            nn.Dropout(dropout_rate),
            nn.Linear(hidden_size, hidden_size // 2),
            nn.ReLU(),
            nn.BatchNorm1d(hidden_size // 2),
            nn.Dropout(dropout_rate),
            nn.Linear(hidden_size // 2, output_size)
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.network(x)

class GatingNetwork(nn.Module):
    """Gating network that determines expert activation weights"""
    def __init__(self, input_size: int, num_experts: int, hidden_size: int, dropout_rate: float = 0.3):
        super(GatingNetwork, self).__init__()
        self.network = nn.Sequential(
            nn.BatchNorm1d(input_size),
            nn.Linear(input_size, hidden_size),
            nn.ReLU(),
            nn.BatchNorm1d(hidden_size),
            nn.Dropout(dropout_rate),
            nn.Linear(hidden_size, hidden_size // 2),
            nn.ReLU(),
            nn.BatchNorm1d(hidden_size // 2),
            nn.Dropout(dropout_rate),
            nn.Linear(hidden_size // 2, num_experts)
        )
        self.temperature = nn.Parameter(torch.ones(1) * 1.0)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        logits = self.network(x)
        scaled_logits = logits / self.temperature
        return F.softmax(scaled_logits, dim=-1)

class MixtureOfExperts(nn.Module):
    """Complete Mixture of Experts model"""
    def __init__(self, input_size: int, hidden_size: int, output_size: int, num_experts: int):
        super(MixtureOfExperts, self).__init__()
        self.num_experts = num_experts
        
        # Initialize experts
        self.experts = nn.ModuleList([
            Expert(input_size, hidden_size, output_size) 
            for _ in range(num_experts)
        ])
        
        # Initialize gating network
        self.gating_network = GatingNetwork(input_size, num_experts, hidden_size)
        
    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        # Get expert weights from gating network
        expert_weights = self.gating_network(x)
        
        # Get outputs from each expert
        expert_outputs = torch.stack([expert(x) for expert in self.experts])
        
        # Combine expert outputs
        expert_outputs = expert_outputs.permute(1, 0, 2)
        expert_weights = expert_weights.unsqueeze(-1)
        
        final_output = torch.bmm(
            expert_outputs.transpose(1, 2),
            expert_weights
        ).squeeze(-1)
        
        return final_output, expert_weights

class ExpertActivationTracker:
    """Tracks and saves expert activation patterns"""
    def __init__(self, mode: str, base_path: Optional[str] = None, num_experts: Optional[int] = None):
        self.mode = mode
        self.timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        if base_path is None:
            base_path = f'moe_outputs/{mode}'
        
        self.base_dir = os.path.join(base_path, self.timestamp)
        self.plots_dir = os.path.join(self.base_dir, 'plots')
        self.data_dir = os.path.join(self.base_dir, 'csv')
        self.model_dir = os.path.join(self.base_dir, 'models')
        
        # Create directories
        for directory in [self.plots_dir, self.data_dir, self.model_dir]:
            os.makedirs(directory, exist_ok=True)
        
        self.num_experts = num_experts
        self.config = {
            'mode': mode,
            'created_at': self.timestamp,
            'base_dir': self.base_dir,
            'num_experts': num_experts
        }
        
        # Save config
        with open(os.path.join(self.base_dir, 'config.json'), 'w') as f:
            json.dump(self.config, f, indent=4)
        
        # Initialize metrics tracking
        self.metrics = {
            f'{mode}_loss': [],
            f'{mode}_acc': [],
            'expert_usage': []
        }
    
    def save_expert_activations(self, expert_weights: torch.Tensor, 
                              labels: torch.Tensor, 
                              epoch: int, 
                              batch_idx: int) -> pd.DataFrame:
        """Save and visualize expert activations"""
        probs_np = expert_weights.squeeze(-1).detach().cpu().numpy()
        labels_np = labels.cpu().numpy()
        
        # Create DataFrame
        activation_data = []
        for sample_idx in range(len(probs_np)):
            for expert_idx, prob in enumerate(probs_np[sample_idx]):
                activation_data.append({
                    'Sample': f'Sample_{sample_idx + 1}',
                    'Expert': f'Expert_{expert_idx + 1}',
                    'Probability': float(prob),
                    'Label': int(labels_np[sample_idx]),
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

    def _create_activation_heatmap(self, df: pd.DataFrame, epoch: int, batch_idx: int):
        """Create and save activation heatmap"""
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
        
        plt.title(f'{self.mode} Expert Activation Heatmap (Epoch {epoch}, Batch {batch_idx})')
        plt.tight_layout()
        plt.savefig(
            os.path.join(
                self.plots_dir,
                f'heatmap_epoch_{epoch}_batch_{batch_idx}.png'
            )
        )
        plt.close()

    def _create_expert_usage_plot(self, df: pd.DataFrame, epoch: int, batch_idx: int):
        """Create and save expert usage plot"""
        plt.figure(figsize=(10, 6))
        mean_activations = df.groupby('Expert')['Probability'].mean()
        
        sns.barplot(
            x=mean_activations.index,
            y=mean_activations.values,
            palette='viridis'
        )
        
        plt.title(f'{self.mode} Mean Expert Usage (Epoch {epoch}, Batch {batch_idx})')
        plt.xlabel('Expert')
        plt.ylabel('Mean Activation Probability')
        plt.tight_layout()
        plt.savefig(
            os.path.join(
                self.plots_dir,
                f'expert_usage_epoch_{epoch}_batch_{batch_idx}.png'
            )
        )
        plt.close()

    def _create_label_distribution_plot(self, df: pd.DataFrame, epoch: int, batch_idx: int):
        """Create and save label distribution plot"""
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
        
        plt.title(f'{self.mode} Expert Specialization by Label (Epoch {epoch}, Batch {batch_idx})')
        plt.xlabel('Expert')
        plt.ylabel('Label')
        plt.tight_layout()
        plt.savefig(
            os.path.join(
                self.plots_dir,
                f'label_distribution_epoch_{epoch}_batch_{batch_idx}.png'
            )
        )
        plt.close()

    def update_metrics(self, epoch: int, loss: float, accuracy: float, expert_usage: np.ndarray):
        """Update and save training metrics"""
        self.metrics[f'{self.mode}_loss'].append((epoch, loss))
        self.metrics[f'{self.mode}_acc'].append((epoch, accuracy))
        self.metrics['expert_usage'].append((epoch, expert_usage))
        
        # Fix: Create DataFrame with proper structure
        metrics_df = pd.DataFrame({
            'epoch': [epoch],
            'loss': [loss],
            'accuracy': [accuracy],
            'expert_usage': [str(expert_usage.tolist())]  # Convert numpy array to string
        })
        
        metrics_path = os.path.join(self.data_dir, f'{self.mode}_metrics.csv')
        if os.path.exists(metrics_path):
            metrics_df.to_csv(metrics_path, mode='a', header=False, index=False)
        else:
            metrics_df.to_csv(metrics_path, index=False)

    def save_confusion_matrix(self, y_true: np.ndarray, y_pred: np.ndarray, epoch: int) -> Tuple[np.ndarray, Dict]:
        """Generate and save confusion matrix plot and metrics"""
        # Compute confusion matrix
        cm = confusion_matrix(y_true, y_pred)
        
        # Create figure and plot
        plt.figure(figsize=(10, 8))
        disp = ConfusionMatrixDisplay(confusion_matrix=cm)
        disp.plot(cmap='Blues')
        
        plt.title(f'{self.mode} Confusion Matrix - Epoch {epoch}')
        plt.tight_layout()
        
        plt.savefig(
            os.path.join(
                self.plots_dir,
                f'confusion_matrix_epoch_{epoch}.png'
            )
        )
        plt.close()
        
        # Save raw confusion matrix data
        cm_df = pd.DataFrame(
            cm,
            index=[f'True_{i}' for i in range(cm.shape[0])],
            columns=[f'Pred_{i}' for i in range(cm.shape[1])]
        )
        cm_df.to_csv(
            os.path.join(
                self.data_dir,
                f'confusion_matrix_epoch_{epoch}.csv'
            )
        )
        
        # Calculate metrics
        accuracy = np.trace(cm) / np.sum(cm)
        per_class_accuracy = np.diag(cm) / np.sum(cm, axis=1)
        per_class_precision = np.diag(cm) / np.sum(cm, axis=0)
        
        # Get classification report
        report = classification_report(y_true, y_pred, output_dict=True)
        
        # Create separate DataFrames for different metrics
        basic_metrics_df = pd.DataFrame({
            'metric': ['accuracy'],
            'value': [accuracy]
        })
        
        per_class_metrics_df = pd.DataFrame({
            'class': range(len(per_class_accuracy)),
            'accuracy': per_class_accuracy,
            'precision': per_class_precision
        })
        
        # Save metrics separately
        basic_metrics_df.to_csv(
            os.path.join(
                self.data_dir,
                f'basic_metrics_epoch_{epoch}.csv'
            ),
            index=False
        )
        
        per_class_metrics_df.to_csv(
            os.path.join(
                self.data_dir,
                f'per_class_metrics_epoch_{epoch}.csv'
            ),
            index=False
        )
        
        # Save classification report separately
        report_df = pd.DataFrame(report).transpose()
        report_df.to_csv(
            os.path.join(
                self.data_dir,
                f'classification_report_epoch_{epoch}.csv'
            )
        )
        
        # Return metrics as dictionary for use in training loop
        metrics = {
            'accuracy': accuracy,
            'per_class_accuracy': per_class_accuracy,
            'per_class_precision': per_class_precision,
            'classification_report': report
        }
        
        return cm, metrics

    def save_model(self, model: nn.Module, optimizer: optim.Optimizer, epoch: int, loss: float):
        """Save model checkpoint"""
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'loss': loss
        }
        
        torch.save(
            checkpoint,
            os.path.join(self.model_dir, f'checkpoint_epoch_{epoch}.pt')
        )

def train_epoch(model: nn.Module, 
                train_loader: DataLoader, 
                optimizer: optim.Optimizer, 
                device: torch.device,
                tracker: ExpertActivationTracker, 
                epoch: int) -> Tuple[float, float, np.ndarray, Dict]:
    """Train for one epoch and track metrics"""
    model.train()
    total_loss = 0
    correct = 0
    total = 0
    all_preds = []
    all_labels = []
    all_expert_weights = []
    
    progress_bar = tqdm(train_loader, desc=f'Training Epoch {epoch}')
    
    for batch_idx, (data, target) in enumerate(progress_bar):
        data, target = data.to(device), target.to(device)
        data = data.view(data.size(0), -1)
        
        optimizer.zero_grad()
        output, expert_weights = model(data)
        
        loss = F.cross_entropy(output, target)
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
        pred = output.argmax(dim=1)
        correct += pred.eq(target).sum().item()
        total += target.size(0)
        
        all_preds.extend(pred.cpu().numpy())
        all_labels.extend(target.cpu().numpy())
        all_expert_weights.append(expert_weights.cpu())
        
        if batch_idx % 200 == 0:
            tracker.save_expert_activations(
                expert_weights,
                target,
                epoch,
                batch_idx
            )
        
        progress_bar.set_postfix({
            'loss': f'{total_loss/(batch_idx+1):.3f}',
            'acc': f'{100.*correct/total:.2f}%'
        })
    
    epoch_loss = total_loss / len(train_loader)
    epoch_acc = 100. * correct / total
    
    # Calculate average expert usage
    all_expert_weights = torch.cat(all_expert_weights, dim=0)
    expert_usage = all_expert_weights.mean(dim=0).squeeze().detach().cpu().numpy()
    
    # Generate and save confusion matrix
    cm, cm_metrics = tracker.save_confusion_matrix(
        np.array(all_labels),
        np.array(all_preds),
        epoch
    )
    
    return epoch_loss, epoch_acc, expert_usage, cm_metrics

def evaluate_model(model: nn.Module, 
                  test_loader: DataLoader, 
                  device: torch.device,
                  tracker: ExpertActivationTracker,
                  epoch: int) -> Tuple[float, float, np.ndarray, Dict]:
    """
    Evaluate model on test set
    """
    model.eval()
    total_loss = 0
    correct = 0
    total = 0
    all_preds = []
    all_labels = []
    all_expert_weights = []
    
    with torch.no_grad():
        progress_bar = tqdm(test_loader, desc=f'Evaluating Epoch {epoch}')
        
        for batch_idx, (data, target) in enumerate(progress_bar):
            data, target = data.to(device), target.to(device)
            data = data.view(data.size(0), -1)
            
            output, expert_weights = model(data)
            loss = F.cross_entropy(output, target)
            
            total_loss += loss.item()
            pred = output.argmax(dim=1)
            correct += pred.eq(target).sum().item()
            total += target.size(0)
            
            all_preds.extend(pred.cpu().numpy())
            all_labels.extend(target.cpu().numpy())
            all_expert_weights.append(expert_weights.cpu())
            
            if batch_idx % 100 == 0:
                tracker.save_expert_activations(
                    expert_weights,
                    target,
                    epoch,
                    batch_idx
                )
            
            progress_bar.set_postfix({
                'loss': f'{total_loss/(batch_idx+1):.3f}',
                'acc': f'{100.*correct/total:.2f}%'
            })
    
    epoch_loss = total_loss / len(test_loader)
    epoch_acc = 100. * correct / total
    
    # Calculate average expert usage
    all_expert_weights = torch.cat(all_expert_weights, dim=0)
    expert_usage = all_expert_weights.mean(dim=0).squeeze().numpy()
    
    # Generate and save confusion matrix
    cm, cm_metrics = tracker.save_confusion_matrix(
        np.array(all_labels),
        np.array(all_preds),
        epoch
    )
    
    return epoch_loss, epoch_acc, expert_usage, cm_metrics

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

def main():
    # Set random seed for reproducibility
    set_seed(42)
    
    # Hyperparameters
    input_size = 784  # 28x28 MNIST images
    hidden_size = 256
    output_size = 10  # 10 digits
    num_experts = 5
    learning_rate = 0.001
    num_epochs = 50
    batch_size = 256
    
    # Setup device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Load MNIST dataset
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
    
    # Initialize model and optimizer
    model = MixtureOfExperts(
        input_size,
        hidden_size,
        output_size,
        num_experts
    ).to(device)
    
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    
    # Print model summary and dataset info
    print("\nModel Architecture Summary:")
    print("=" * 50)
    summary(model, input_size=(batch_size, input_size))
    
    print("\nDataset Information:")
    print_dataset_info(train_loader, "Training")
    print_dataset_info(test_loader, "Test")
    
    # Initialize trackers for both train and test
    train_tracker = ExpertActivationTracker(mode='train', num_experts=num_experts)
    test_tracker = ExpertActivationTracker(mode='test', num_experts=num_experts)
    
    # Training and evaluation loop
    best_test_acc = 0.0
    for epoch in range(num_epochs):
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
        test_loss, test_acc, test_expert_usage, test_cm_metrics = evaluate_model(
            model,
            test_loader,
            device,
            test_tracker,
            epoch + 1
        )
        
        # Update metrics
        train_tracker.update_metrics(epoch + 1, train_loss, train_acc, train_expert_usage)
        test_tracker.update_metrics(epoch + 1, test_loss, test_acc, test_expert_usage)
        
        # Save model checkpoints
        train_tracker.save_model(model, optimizer, epoch + 1, train_loss)
        
        # Save best model
        if test_acc > best_test_acc:
            best_test_acc = test_acc
            torch.save({
                'epoch': epoch + 1,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'test_acc': test_acc
            }, os.path.join(test_tracker.model_dir, 'best_model.pt'))
        
        # Print epoch results
        print(f'\nEpoch {epoch + 1}:')
        print('Training:')
        print(f'  Loss: {train_loss:.4f}')
        print(f'  Accuracy: {train_acc:.2f}%')
        print(f'  Expert Usage: {train_expert_usage}')
        print('  Per-class Accuracy:', train_cm_metrics['per_class_accuracy'])
        print('Test:')
        print(f'  Loss: {test_loss:.4f}')
        print(f'  Accuracy: {test_acc:.2f}%')
        print(f'  Expert Usage: {test_expert_usage}')
        print('  Per-class Accuracy:', test_cm_metrics['per_class_accuracy'])
        print(f'  Best Test Accuracy: {best_test_acc:.2f}%')

if __name__ == '__main__':
    main()