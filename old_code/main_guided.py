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
    Expert network with improved stability and regularization
    """
    def __init__(self, input_size, hidden_size, output_size):
        super().__init__()
        
        # Shared feature extractor
        self.feature_extractor = nn.Sequential(
            nn.BatchNorm1d(input_size),
            nn.Linear(input_size, hidden_size),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.BatchNorm1d(hidden_size)
        )
        
        # Task-specific layers with skip connections
        self.task_layers = nn.ModuleList([
            nn.Sequential(
                nn.Linear(hidden_size, hidden_size // 2),
                nn.ReLU(),
                nn.Dropout(0.2),
                nn.BatchNorm1d(hidden_size // 2)
            ),
            nn.Sequential(
                nn.Linear(hidden_size // 2, hidden_size // 4),
                nn.ReLU(),
                nn.Dropout(0.2),
                nn.BatchNorm1d(hidden_size // 4)
            )
        ])
        
        # Output layer
        self.output_layer = nn.Linear(hidden_size // 4, output_size)
        
        # L2 regularization weight
        self.l2_reg = 0.01
        
    def forward(self, x):
        # Feature extraction
        features = self.feature_extractor(x)
        
        # Task-specific processing with residuals
        h = features
        for layer in self.task_layers:
            h_new = layer(h)
            # Skip connection if dimensions match
            if h.size(1) == h_new.size(1):
                h = h + h_new
            else:
                h = h_new
        
        # Output with L2 regularization
        output = self.output_layer(h)
        
        # Add L2 regularization
        l2_loss = sum(p.pow(2.0).sum() for p in self.parameters()) * self.l2_reg
        
        return output, l2_loss
class GuidedGatingNetwork(nn.Module):
    """
    Improved gating network with better regularization and smoother transitions
    """
    def __init__(self, input_size, num_experts, hidden_size, expert_label_map):
        super().__init__()
        self.expert_label_map = expert_label_map
        
        # Improved architecture with residual connections
        self.network = nn.Sequential(
            nn.BatchNorm1d(input_size),  # Input normalization defined here
            nn.Linear(input_size, hidden_size),
            nn.ReLU(),
            nn.BatchNorm1d(hidden_size),
            nn.Linear(hidden_size, hidden_size // 2),
            nn.ReLU(),
            nn.BatchNorm1d(hidden_size // 2)
        )
        # Final layer separate for residual connection
        self.fc_out = nn.Linear(hidden_size // 2, num_experts)
        
        # Learnable parameters for soft routing
        self.routing_temperature = nn.Parameter(torch.ones(1) * 1.0)
        self.label_embedding = nn.Parameter(torch.randn(10, hidden_size // 4))
        
        # Initialize weights
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight)
                nn.init.constant_(m.bias, 0)
    
    def compute_soft_masks(self, labels=None):
        """Compute soft assignment masks based on label similarities"""
        if labels is None:
            return None
            
        # Get label embeddings for the batch
        batch_embeddings = self.label_embedding[labels]
        
        # Compute similarity between labels and expert assignments
        expert_similarities = []
        for expert_idx, assigned_labels in self.expert_label_map.items():
            # Average embedding for assigned labels
            expert_embedding = self.label_embedding[torch.tensor(assigned_labels)].mean(0)
            
            # Compute cosine similarity
            similarity = F.cosine_similarity(
                batch_embeddings,
                expert_embedding.unsqueeze(0),
                dim=1
            )
            expert_similarities.append(similarity)
        
        # Stack similarities and apply softmax with temperature
        similarities = torch.stack(expert_similarities, dim=1)
        return F.softmax(similarities / self.routing_temperature, dim=1)
    
    def forward(self, x, labels=None):
        # Forward pass through main network
        features = self.network(x)
        
        # Final layer for expert weights
        logits = self.fc_out(features)
        
        if self.training and labels is not None:
            # Get soft assignment masks
            soft_masks = self.compute_soft_masks(labels)
            
            # Combine network output with soft guidance
            routing_weights = F.softmax(logits / self.routing_temperature, dim=1)
            
            # Interpolate between learned and guided routing based on training progress
            mixing_factor = torch.sigmoid(self.routing_temperature)
            final_weights = mixing_factor * routing_weights + (1 - mixing_factor) * soft_masks
            
            return final_weights
        else:
            # During inference, use pure network output with temperature
            return F.softmax(logits / self.routing_temperature, dim=1)

class GuidedMixtureOfExperts(nn.Module):
    """
    Improved MoE with proper loss handling
    """
    def __init__(self, input_size, hidden_size, output_size, num_experts, expert_label_assignments):
        super().__init__()
        self.num_experts = num_experts
        self.expert_label_assignments = expert_label_assignments
        
        # Initialize experts - now returning L2 losses
        self.experts = nn.ModuleList([
            GuidedExpert(input_size, hidden_size, output_size)
            for _ in range(num_experts)
        ])
        
        # Initialize gating network
        self.gating_network = GuidedGatingNetwork(
            input_size, 
            num_experts,
            hidden_size,
            expert_label_assignments
        )
        
    def forward(self, x, labels=None):
        batch_size = x.size(0)
        
        # Get expert weights from gating network
        expert_weights = self.gating_network(x, labels)
        
        # Get outputs from each expert and collect L2 losses
        expert_outputs = []
        expert_l2_losses = []
        
        for expert in self.experts:
            output, l2_loss = expert(x)  # Now returns both output and L2 loss
            expert_outputs.append(output)
            expert_l2_losses.append(l2_loss)
        
        expert_outputs = torch.stack(expert_outputs)
        expert_outputs = expert_outputs.permute(1, 0, 2)
        
        # Combine expert outputs with attention mechanism
        attention_weights = expert_weights.unsqueeze(-1)
        final_output = torch.bmm(expert_outputs.transpose(1, 2), attention_weights).squeeze(-1)
        
        return final_output, expert_weights, expert_l2_losses
    
    def compute_loss(self, final_output, target, expert_weights, expert_l2_losses):
        """
        Compute loss with all components
        """
        # Base classification loss
        ce_loss = F.cross_entropy(final_output, target)
        
        # Expert diversity loss (entropy)
        diversity_loss = -torch.mean(
            torch.sum(expert_weights * torch.log(expert_weights + 1e-6), dim=1)
        )
        
        # Load balancing loss
        usage_per_expert = expert_weights.mean(0)
        target_usage = torch.ones_like(usage_per_expert) / self.num_experts
        balance_loss = F.kl_div(
            usage_per_expert.log(),
            target_usage,
            reduction='batchmean'
        )
        
        # Combine L2 losses from experts
        total_l2_loss = sum(expert_l2_losses)
        
        # Combine all losses with weights
        total_loss = (
            ce_loss + 
            0.01 * diversity_loss +  # Small weight for diversity
            0.01 * balance_loss +    # Small weight for balance
            0.001 * total_l2_loss    # Very small weight for L2
        )
        
        # Store components for monitoring
        self.loss_components = {
            'ce_loss': ce_loss.item(),
            'diversity_loss': diversity_loss.item(),
            'balance_loss': balance_loss.item(),
            'l2_loss': total_l2_loss.item(),
            'total_loss': total_loss.item()
        }
        
        return total_loss

def create_improved_label_assignments(num_classes, num_experts):
    """
    Create overlapping label assignments for better generalization
    """
    assignments = {}
    labels_per_expert = num_classes // (num_experts - 1)  # Allow overlap
    
    for expert_idx in range(num_experts):
        start_idx = expert_idx * (labels_per_expert - 1)
        end_idx = min(start_idx + labels_per_expert, num_classes)
        
        # Add overlapping labels
        assignments[expert_idx] = list(range(start_idx, end_idx))
        
        # Add some random labels from other ranges for diversity
        other_labels = list(set(range(num_classes)) - set(assignments[expert_idx]))
        if other_labels:
            num_extra = min(2, len(other_labels))
            assignments[expert_idx].extend(
                random.sample(other_labels, num_extra)
            )
    
    return assignments

def create_improved_label_assignments(num_classes, num_experts):
    """
    Create overlapping label assignments for better generalization
    """
    assignments = {}
    labels_per_expert = num_classes // (num_experts - 1)  # Allow overlap
    
    for expert_idx in range(num_experts):
        start_idx = expert_idx * (labels_per_expert - 1)
        end_idx = min(start_idx + labels_per_expert, num_classes)
        
        # Add overlapping labels
        assignments[expert_idx] = list(range(start_idx, end_idx))
        
        # Add some random labels from other ranges for diversity
        other_labels = list(set(range(num_classes)) - set(assignments[expert_idx]))
        if other_labels:
            num_extra = min(2, len(other_labels))
            assignments[expert_idx].extend(
                random.sample(other_labels, num_extra)
            )
    
    return assignments

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
    Improved training loop supporting expert weights and enhanced loss computation
    """
    model.train()
    total_loss = 0
    correct = 0
    total = 0
    all_preds = []
    all_labels = []
    expert_weight_accumulator = torch.zeros(model.num_experts).to(device)
    num_batches = 0
    
    progress_bar = tqdm(train_loader, desc=f'Epoch {epoch}')
    
    for batch_idx, (data, target) in enumerate(progress_bar):
        data, target = data.to(device), target.to(device)
        data = data.view(data.size(0), -1)
        
        optimizer.zero_grad()
        
        # Forward pass with expert losses
        final_output, expert_weights, expert_l2_losses = model(data, target)
    
        # Compute loss with all components
        loss = model.compute_loss(final_output, target, expert_weights, expert_l2_losses)
        
        # Backward pass with gradient clipping
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        
        # Accumulate expert weights for tracking
        expert_weight_accumulator += expert_weights.mean(dim=0).detach()
        num_batches += 1
        
        # Accumulate metrics
        total_loss += loss.item()
        pred = final_output.argmax(dim=1)
        correct += pred.eq(target).sum().item()
        total += target.size(0)
        
        # Store predictions and labels for confusion matrix
        all_preds.extend(pred.cpu().numpy())
        all_labels.extend(target.cpu().numpy())
        
        # Save activations and expert usage periodically
        if batch_idx % 300 == 0:
            tracker.save_expert_activations(
                expert_weights,
                target,
                epoch,
                batch_idx
            )
            
            # Calculate expert utilization metrics
            expert_usage = expert_weights.mean(dim=0).detach()
            expert_entropy = -(expert_usage * torch.log(expert_usage + 1e-10)).sum()
            
            # Update progress bar with enhanced metrics
            progress_bar.set_postfix({
                'loss': f'{total_loss/(batch_idx+1):.3f}',
                'acc': f'{100.*correct/total:.2f}%',
                'expert_entropy': f'{expert_entropy.item():.2f}'
            })
    
    # Calculate final metrics
    epoch_loss = total_loss / len(train_loader)
    epoch_acc = 100. * correct / total
    
    # Calculate average expert usage
    avg_expert_weights = expert_weight_accumulator / num_batches
    
    # Generate confusion matrix and metrics
    cm, cm_metrics = tracker.save_confusion_matrix(
        np.array(all_labels),
        np.array(all_preds),
        epoch
    )
    
    return epoch_loss, epoch_acc, avg_expert_weights.cpu().numpy(), cm_metrics

def test_epoch(model, val_loader, device, tracker, epoch):
    """
    Improved validation loop supporting expert weights and enhanced metrics
    """
    model.eval()
    total_loss = 0
    correct = 0
    total = 0
    all_preds = []
    all_labels = []
    expert_weight_accumulator = torch.zeros(model.num_experts).to(device)
    num_batches = 0
    
    with torch.no_grad():
        progress_bar = tqdm(val_loader, desc=f'Validation Epoch {epoch}')
        
        for batch_idx, (data, target) in enumerate(progress_bar):
            data, target = data.to(device), target.to(device)
            data = data.view(data.size(0), -1)
            
            # Forward pass with expert losses
            final_output, expert_weights, expert_l2_losses = model(data)
            
            # Compute loss with all components
            loss = model.compute_loss(final_output, target, expert_weights, expert_l2_losses)
                    
                    # Accumulate expert weights for tracking
            expert_weight_accumulator += expert_weights.mean(dim=0).detach()
            num_batches += 1
            
            # Accumulate metrics
            total_loss += loss.item()
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
                
                # Calculate expert utilization metrics
                expert_usage = expert_weights.mean(dim=0).detach()
                expert_entropy = -(expert_usage * torch.log(expert_usage + 1e-10)).sum()
                
                # Update progress bar with enhanced metrics
                progress_bar.set_postfix({
                    'loss': f'{total_loss/(batch_idx+1):.3f}',
                    'acc': f'{100.*correct/total:.2f}%',
                    'expert_entropy': f'{expert_entropy.item():.2f}'
                })
    
    # Calculate final metrics
    epoch_loss = total_loss / len(val_loader)
    epoch_acc = 100. * correct / total
    
    # Calculate average expert usage
    avg_expert_weights = expert_weight_accumulator / num_batches
    
    # Generate confusion matrix and metrics
    cm, cm_metrics = tracker.save_confusion_matrix(
        np.array(all_labels),
        np.array(all_preds),
        epoch
    )
    
    return epoch_loss, epoch_acc, avg_expert_weights.cpu().numpy(), cm_metrics

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
    num_epochs = 10
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
        expert_label_assignments=create_improved_label_assignments(10, 5)
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