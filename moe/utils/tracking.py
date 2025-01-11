import os
import json
from datetime import datetime
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Tuple, Optional
import torch
from sklearn.metrics import confusion_matrix, classification_report, ConfusionMatrixDisplay

class ExpertTracker:
    """Track and visualize expert behavior and metrics"""
    
    def __init__(self, model_type: str, architecture: str, base_path: str,dataset_name:str,
                 num_experts: int, expert_label_assignments: Optional[Dict[int, List[int]]] = None, fold_no:str = None):
        """
        Initialize the tracker
        
        Args:
            model_type: Type of MoE model ('base' or 'guided')
            architecture: Network architecture ('1d' or '2d')
            base_path: Base directory for saving outputs
            num_experts: Number of experts in the model
            expert_label_assignments: Dictionary mapping expert indices to their assigned labels
        """
        plt.switch_backend('agg')
        self.model_type = model_type
        self.architecture = architecture
        self.timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        self.num_experts = num_experts
        self.expert_label_assignments = expert_label_assignments
        
        # Setup directory structure
        if fold_no is not None:
            self.base_dir = os.path.join(base_path, f'{dataset_name}/{model_type}/{architecture}/fold_{fold_no}', self.timestamp)
        else:
            self.base_dir = os.path.join(base_path, f'{dataset_name}/{model_type}/{architecture}', self.timestamp)
        self.plots_dir = os.path.join(self.base_dir, 'plots')
        self.data_dir = os.path.join(self.base_dir, 'data')
        self.model_dir = os.path.join(self.base_dir, 'models')
        
        # Create directories
        for directory in [self.plots_dir, self.data_dir, self.model_dir]:
            os.makedirs(directory, exist_ok=True)
        
        # Save configuration
        self._save_config()
        
        # Initialize metrics
        self.metrics = {
            'train_loss': [],
            'train_acc': [],
            'val_loss': [],
            'val_acc': [],
            'expert_usage': []
        }
    
    def _save_config(self):
        """Save tracker configuration"""
        config = {
            'model_type': self.model_type,
            'architecture': self.architecture,
            'created_at': self.timestamp,
            'num_experts': self.num_experts
        }
        
        if self.expert_label_assignments:
            config['expert_label_assignments'] = {
                str(k): v for k, v in self.expert_label_assignments.items()
            }
        
        with open(os.path.join(self.base_dir, 'config.json'), 'w') as f:
            json.dump(config, f, indent=4)
    
    def log_expert_activations(self, expert_weights: torch.Tensor, labels: torch.Tensor,
                            epoch: int, mode: str) -> None:
        """
        Log and visualize expert activations
        
        Args:
            expert_weights: Tensor of expert weights [batch_size, num_experts]
            labels: Tensor of true labels [batch_size]
            epoch: Current epoch number
            mode: 'train' or 'val'
        """
        # Convert to numpy for processing
        weights_np = expert_weights.detach().cpu().numpy()
        labels_np = labels.cpu().numpy()
        
        # Create DataFrame with proper indexing
        data = []
        for i in range(len(labels_np)):
            for expert_idx in range(self.num_experts):
                is_assigned = False
                if self.expert_label_assignments:
                    is_assigned = labels_np[i] in self.expert_label_assignments.get(expert_idx, [])
                
                data.append({
                    'Sample': i,
                    'Expert': expert_idx,
                    'Weight': weights_np[i, expert_idx],
                    'Label': int(labels_np[i]),  # Ensure labels are integers
                    'Is_Assigned': is_assigned
                })
        
        df = pd.DataFrame(data)
        
        # Save to CSV
        df.to_csv(os.path.join(self.data_dir, f'{mode}_activations_epoch_{epoch}.csv'), index=False)
        
        # Create visualizations
        self._plot_expert_heatmap(df, epoch, mode)
        self._plot_expert_usage(df, epoch, mode)
        self._plot_expert_label_distribution(df, epoch, mode)
    
    def _plot_expert_heatmap(self, df: pd.DataFrame, epoch: int, mode: str):
        """Plot expert activation heatmap with proper aggregation"""
        plt.figure(figsize=(12, 8))
        
        # Aggregate weights by taking the mean for each Label-Expert combination
        pivot_df = df.groupby(['Label', 'Expert'])['Weight'].mean().unstack()
        
        # Create heatmap
        sns.heatmap(pivot_df, annot=True, fmt='.3f', cmap='YlOrRd')
        
        plt.title(f'{mode} Expert Specialization by Label (Epoch {epoch})')
        plt.xlabel('Expert')
        plt.ylabel('Label')
        
        # Add expert assignments if available
        if self.expert_label_assignments:
            for expert_idx, labels in self.expert_label_assignments.items():
                plt.text(expert_idx + 0.5, -0.5, f'Assigned: {labels}',
                        ha='center', va='center', rotation=45)
        
        plt.tight_layout()
        plt.savefig(os.path.join(self.plots_dir, f'{mode}_heatmap_epoch_{epoch}.png'))
        plt.close()
    
    def _plot_expert_usage(self, df: pd.DataFrame, epoch: int, mode: str):
        """Plot expert usage statistics"""
        plt.figure(figsize=(10, 6))
        
        if self.expert_label_assignments:
            assigned = df[df['Is_Assigned']].groupby('Expert')['Weight'].mean()
            non_assigned = df[~df['Is_Assigned']].groupby('Expert')['Weight'].mean()
            
            x = np.arange(self.num_experts)
            width = 0.35
            
            plt.bar(x - width/2, assigned, width, label='Assigned Labels')
            plt.bar(x + width/2, non_assigned, width, label='Non-assigned Labels')
            plt.legend()
        else:
            usage = df.groupby('Expert')['Weight'].mean()
            plt.bar(range(self.num_experts), usage)
        
        plt.title(f'{mode} Expert Usage (Epoch {epoch})')
        plt.xlabel('Expert')
        plt.ylabel('Mean Weight')
        plt.tight_layout()
        plt.savefig(os.path.join(self.plots_dir, f'{mode}_usage_epoch_{epoch}.png'))
        plt.close()
    
    def _plot_expert_label_distribution(self, df: pd.DataFrame, epoch: int, mode: str):
        """Plot label distribution for each expert"""
        plt.figure(figsize=(15, 5))
        
        for expert_idx in range(self.num_experts):
            plt.subplot(1, self.num_experts, expert_idx + 1)
            expert_df = df[df['Expert'] == expert_idx]
            sns.boxplot(data=expert_df, x='Label', y='Weight')
            plt.title(f'Expert {expert_idx}')
            plt.xticks(rotation=45)
        
        plt.suptitle(f'{mode} Label Distribution per Expert (Epoch {epoch})')
        plt.tight_layout()
        plt.savefig(os.path.join(self.plots_dir, f'{mode}_label_dist_epoch_{epoch}.png'))
        plt.close()
    
    def log_metrics(self, epoch: int, train_metrics: Dict, val_metrics: Dict):
        """Log training and validation metrics"""
        metrics_df = pd.DataFrame({
            'epoch': [epoch],
            'train_loss': [train_metrics['loss']],
            'train_acc': [train_metrics['accuracy']],
            'val_loss': [val_metrics['loss']],
            'val_acc': [val_metrics['accuracy']],
            'train_expert_usage': [train_metrics['expert_weights'].mean(0).tolist()],
            'val_expert_usage': [val_metrics['expert_weights'].mean(0).tolist()]
        })
        
        metrics_path = os.path.join(self.data_dir, 'metrics.csv')
        if os.path.exists(metrics_path):
            metrics_df.to_csv(metrics_path, mode='a', header=False, index=False)
        else:
            metrics_df.to_csv(metrics_path, index=False)
        
        # Update plots
        self._plot_training_curves()
    
    def _plot_training_curves(self):
        """Plot training and validation curves"""
        metrics_df = pd.read_csv(os.path.join(self.data_dir, 'metrics.csv'))
        
        # Plot loss curves
        plt.figure(figsize=(10, 5))
        plt.plot(metrics_df['epoch'], metrics_df['train_loss'], label='Train Loss')
        plt.plot(metrics_df['epoch'], metrics_df['val_loss'], label='Val Loss')
        plt.title('Loss Curves')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.legend()
        plt.savefig(os.path.join(self.plots_dir, 'loss_curves.png'))
        plt.close()
        
        # Plot accuracy curves
        plt.figure(figsize=(10, 5))
        plt.plot(metrics_df['epoch'], metrics_df['train_acc'], label='Train Accuracy')
        plt.plot(metrics_df['epoch'], metrics_df['val_acc'], label='Val Accuracy')
        plt.title('Accuracy Curves')
        plt.xlabel('Epoch')
        plt.ylabel('Accuracy')
        plt.legend()
        plt.savefig(os.path.join(self.plots_dir, 'accuracy_curves.png'))
        plt.close()
    
    def log_confusion_matrix(self, y_true: torch.Tensor, y_pred: torch.Tensor, 
                           epoch: int, mode: str) -> Dict:
        """Log confusion matrix and classification metrics"""
        y_true = y_true.cpu().numpy()
        y_pred = y_pred.cpu().numpy()
        
        # Compute confusion matrix
        cm = confusion_matrix(y_true, y_pred)
        
        # Plot confusion matrix
        plt.figure(figsize=(10, 8))
        disp = ConfusionMatrixDisplay(confusion_matrix=cm)
        disp.plot(cmap='Blues')
        plt.title(f'{mode} Confusion Matrix (Epoch {epoch})')
        plt.tight_layout()
        plt.savefig(os.path.join(self.plots_dir, f'{mode}_confusion_matrix_epoch_{epoch}.png'))
        plt.close()
        
        # Get classification metrics
        report = classification_report(y_true, y_pred, output_dict=True)
        
        # Save metrics
        metrics = {
            'confusion_matrix': cm.tolist(),
            'classification_report': report
        }
        
        with open(os.path.join(self.data_dir, f'{mode}_classification_metrics_epoch_{epoch}.json'), 'w') as f:
            json.dump(metrics, f, indent=4)
        
        return metrics
    
    def save_model(self, model: torch.nn.Module, optimizer: torch.optim.Optimizer, 
                  epoch: int, metrics: Dict, is_best: bool = False):
        """Save model checkpoint"""
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'metrics': metrics
        }
        
        # Save regular checkpoint
        checkpoint_path = os.path.join(self.model_dir, f'checkpoint_epoch_{epoch}.pt')
        torch.save(checkpoint, checkpoint_path)
        
        # Save best model if specified
        if is_best:
            best_path = os.path.join(self.model_dir, 'best_model.pt')
            torch.save(checkpoint, best_path)
    
    def load_model(self, model: torch.nn.Module, optimizer: torch.optim.Optimizer, 
                  checkpoint_path: str) -> Tuple[int, Dict]:
        """Load model checkpoint"""
        checkpoint = torch.load(checkpoint_path)
        
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        
        return checkpoint['epoch'], checkpoint['metrics']