"""
tracking.py - Expert activation and metrics tracking for MoE models with improved figure handling
"""

import os
import json
from datetime import datetime
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')  # Set the backend to Agg before importing pyplot
import seaborn as sns
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, classification_report
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

class BaseExpertTracker:
    """Base class for tracking expert activations and metrics in MoE models
    
    Features:
    - Zero-based expert indexing in visualizations
    - Consistent expert naming across all plots
    - Enhanced visualization formatting
    """
    
    def __init__(self, model_type: str, mode: str, base_path: str = None, num_experts: int = None, 
                 expert_label_assignments: dict = None):
        """Initialize the expert tracker with zero-based indexing"""
        self.model_type = model_type
        self.mode = mode
        self.timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        
        # Set up directory structure
        if base_path is None:
            base_name = 'moe_guided_outputs' if model_type == 'guided' else 'moe_outputs'
            base_path = f'{base_name}/{mode}'
            
        self.base_dir = os.path.join(base_path, self.timestamp)
        self.plots_dir = os.path.join(self.base_dir, 'plots')
        self.data_dir = os.path.join(self.base_dir, 'csv')
        self.model_dir = os.path.join(self.base_dir, 'models')
        
        # Create directories
        for directory in [self.plots_dir, self.data_dir, self.model_dir]:
            os.makedirs(directory, exist_ok=True)
            
        self.num_experts = num_experts
        self.expert_label_assignments = expert_label_assignments
        
        # Save configuration
        self._save_config()
        
        # Initialize metrics tracking
        self.metrics = {
            f'{mode}_loss': [],
            f'{mode}_acc': [],
            'expert_usage': []
        }

    def _save_config(self):
        """Save tracker configuration to JSON file"""
        config = {
            'model_type': self.model_type,
            'mode': self.mode,
            'created_at': self.timestamp,
            'base_dir': self.base_dir,
            'num_experts': self.num_experts,
        }
        
        if self.expert_label_assignments:
            config['expert_label_assignments'] = {
                str(k): v for k, v in self.expert_label_assignments.items()
            }
            
        with open(os.path.join(self.base_dir, 'config.json'), 'w') as f:
            json.dump(config, f, indent=4)

    def save_expert_activations(self, expert_weights: torch.Tensor, labels: torch.Tensor,
                              epoch: int, batch_idx: int) -> pd.DataFrame:
        """Save and visualize expert activations with zero-based indexing"""
        probs_np = expert_weights.squeeze(-1).detach().cpu().numpy()
        labels_np = labels.cpu().numpy()
        
        # Create DataFrame with zero-based expert indexing
        activation_data = []
        for sample_idx in range(len(probs_np)):
            for expert_idx in range(probs_np.shape[1]):
                data = {
                    'Sample': f'Sample_{sample_idx}',  # Zero-based sample indexing
                    'Expert': f'Expert_{expert_idx}',  # Zero-based expert indexing
                    'Probability': float(probs_np[sample_idx, expert_idx]),
                    'Label': int(labels_np[sample_idx]),
                    'Epoch': epoch,
                    'Batch': batch_idx
                }
                
                if self.expert_label_assignments:
                    data['Is_Assigned'] = labels_np[sample_idx] in self.expert_label_assignments[expert_idx]
                    data['Assigned_Labels'] = str(self.expert_label_assignments[expert_idx])
                    
                activation_data.append(data)
        
        df = pd.DataFrame(activation_data)
        
        # Save to CSV
        csv_path = os.path.join(self.data_dir, f'activations_epoch_{epoch}_batch_{batch_idx}.csv')
        df.to_csv(csv_path, index=False)
        
        self._create_activation_plots(df, epoch, batch_idx)
        
        return df


    def _create_activation_plots(self, df: pd.DataFrame, epoch: int, batch_idx: int):
        """Create activation plots with zero-based expert indexing"""
        # Activation heatmap
        fig_heatmap = plt.figure(figsize=(12, 8))
        pivot_df = df.pivot(index='Sample', columns='Expert', values='Probability')
        
        # Ensure expert columns are properly ordered
        expert_cols = [f'Expert_{i}' for i in range(self.num_experts)]
        pivot_df = pivot_df.reindex(columns=expert_cols)
        
        # Create heatmap with modified expert labels
        sns.heatmap(pivot_df, annot=True, fmt='.3f', cmap='YlOrRd')
        plt.title(f'{self.mode} Expert Activation Heatmap (Epoch {epoch}, Batch {batch_idx})')
        
        # Customize x-axis labels to show zero-based indexing
        plt.xticks(np.arange(self.num_experts) + 0.5, [str(i) for i in range(self.num_experts)])
        
        plt.tight_layout()
        plt.savefig(os.path.join(self.plots_dir, f'heatmap_epoch_{epoch}_batch_{batch_idx}.png'))
        plt.close(fig_heatmap)

        # Expert usage plot
        fig_usage = plt.figure(figsize=(12, 6))
        if self.expert_label_assignments:
            assigned = df[df['Is_Assigned']].groupby('Expert')['Probability'].mean()
            non_assigned = df[~df['Is_Assigned']].groupby('Expert')['Probability'].mean()
            
            # Ensure proper ordering of experts
            assigned = assigned.reindex([f'Expert_{i}' for i in range(self.num_experts)])
            non_assigned = non_assigned.reindex([f'Expert_{i}' for i in range(self.num_experts)])
            
            x = np.arange(self.num_experts)
            width = 0.35
            plt.bar(x - width/2, assigned, width, label='Assigned Labels')
            plt.bar(x + width/2, non_assigned, width, label='Non-assigned Labels')
            plt.legend()
            plt.xticks(x, [str(i) for i in range(self.num_experts)])
        else:
            mean_activations = df.groupby('Expert')['Probability'].mean()
            mean_activations = mean_activations.reindex([f'Expert_{i}' for i in range(self.num_experts)])
            plt.bar(range(self.num_experts), mean_activations.values)
            plt.xticks(range(self.num_experts), [str(i) for i in range(self.num_experts)])
        
        plt.title(f'{self.mode} Expert Usage (Epoch {epoch}, Batch {batch_idx})')
        plt.xlabel('Expert')
        plt.ylabel('Mean Activation Probability')
        plt.tight_layout()
        plt.savefig(os.path.join(self.plots_dir, f'expert_usage_epoch_{epoch}_batch_{batch_idx}.png'))
        plt.close(fig_usage)

        # Label distribution plot
        fig_dist = plt.figure(figsize=(12, 6))
        pivot_df = df.pivot_table(index='Label', columns='Expert', values='Probability', aggfunc='mean')
        
        # Ensure proper ordering of expert columns
        pivot_df = pivot_df.reindex(columns=[f'Expert_{i}' for i in range(self.num_experts)])
        
        sns.heatmap(pivot_df, annot=True, fmt='.3f', cmap='YlOrRd')
        plt.title(f'{self.mode} Expert Specialization by Label (Epoch {epoch}, Batch {batch_idx})')
        
        # Customize x-axis labels
        plt.xticks(np.arange(self.num_experts) + 0.5, [str(i) for i in range(self.num_experts)])
        
        if self.expert_label_assignments:
            for expert_idx, labels in self.expert_label_assignments.items():
                plt.text(expert_idx + 0.5, -0.5, f'Assigned: {labels}',
                        ha='center', va='center', rotation=45)
        
        plt.xlabel('Expert')
        plt.ylabel('Label')
        plt.tight_layout()
        plt.savefig(os.path.join(self.plots_dir, f'label_distribution_epoch_{epoch}_batch_{batch_idx}.png'))
        plt.close(fig_dist)

    def update_metrics(self, epoch: int, loss: float, accuracy: float, expert_usage: np.ndarray):
        """
        Update and save training metrics
        
        Args:
            epoch (int): Current epoch
            loss (float): Current loss value
            accuracy (float): Current accuracy value
            expert_usage (np.ndarray): Expert usage statistics
        """
        self.metrics[f'{self.mode}_loss'].append((epoch, loss))
        self.metrics[f'{self.mode}_acc'].append((epoch, accuracy))
        self.metrics['expert_usage'].append((epoch, expert_usage))
        
        metrics_df = pd.DataFrame({
            'epoch': [epoch],
            'loss': [loss],
            'accuracy': [accuracy],
            'expert_usage': [str(expert_usage.tolist())]
        })
        
        metrics_path = os.path.join(self.data_dir, f'{self.mode}_metrics.csv')
        if os.path.exists(metrics_path):
            metrics_df.to_csv(metrics_path, mode='a', header=False, index=False)
        else:
            metrics_df.to_csv(metrics_path, index=False)

    def save_confusion_matrix(self, y_true: np.ndarray, y_pred: np.ndarray, 
                            epoch: int) -> tuple[np.ndarray, dict]:
        """
        Generate and save confusion matrix plot and metrics
        
        Args:
            y_true (np.ndarray): True labels
            y_pred (np.ndarray): Predicted labels
            epoch (int): Current epoch
            
        Returns:
            tuple[np.ndarray, dict]: Confusion matrix and metrics dictionary
        """
        if torch.is_tensor(y_true):
            y_true = y_true.cpu().numpy()
        if torch.is_tensor(y_pred):
            y_pred = y_pred.cpu().numpy()
            
        # Compute confusion matrix
        cm = confusion_matrix(y_true, y_pred)
        
        # Create and save plot with proper figure handling
        fig_cm = plt.figure(figsize=(10, 8))
        disp = ConfusionMatrixDisplay(confusion_matrix=cm)
        disp.plot(cmap='Blues')
        plt.title(f'{self.mode} Confusion Matrix - Epoch {epoch}')
        plt.tight_layout()
        plt.savefig(os.path.join(self.plots_dir, f'confusion_matrix_epoch_{epoch}.png'))
        plt.close(fig_cm)
        
        # Calculate metrics
        metrics = {
            'accuracy': np.trace(cm) / np.sum(cm),
            'per_class_accuracy': np.diag(cm) / np.sum(cm, axis=1),
            'per_class_precision': np.diag(cm) / np.sum(cm, axis=0),
            'classification_report': classification_report(y_true, y_pred, output_dict=True)
        }
        
        # Save detailed metrics
        metrics_df = pd.DataFrame({
            'metric': ['accuracy'],
            'value': [metrics['accuracy']]
        })
        metrics_df.to_csv(os.path.join(self.data_dir, f'metrics_epoch_{epoch}.csv'), index=False)
        
        return cm, metrics

    def save_model(self, model: nn.Module, optimizer: optim.Optimizer, epoch: int, loss: float):
        """
        Save model checkpoint
        
        Args:
            model (nn.Module): PyTorch model
            optimizer (optim.Optimizer): PyTorch optimizer
            epoch (int): Current epoch
            loss (float): Current loss value
        """
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'loss': loss
        }
        
        if hasattr(model, 'expert_label_assignments'):
            checkpoint['expert_label_assignments'] = model.expert_label_assignments
            
        torch.save(checkpoint, os.path.join(self.model_dir, f'checkpoint_epoch_{epoch}.pt'))