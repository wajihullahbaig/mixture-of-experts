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

class Expert(nn.Module):
    """
    Individual expert network
    """
    def __init__(self, input_size, hidden_size, output_size):
        super(Expert, self).__init__()
        self.network = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(hidden_size, hidden_size // 2),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(hidden_size // 2, output_size)
        )
    
    def forward(self, x):
        return self.network(x)

class GatingNetwork(nn.Module):
    """
    Gating network that determines expert activation weights
    """
    def __init__(self, input_size, num_experts, hidden_size):
        super(GatingNetwork, self).__init__()
        self.network = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(hidden_size, hidden_size // 2),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(hidden_size // 2, num_experts)
        )
        self.temperature = nn.Parameter(torch.ones(1) * 1.0)
    
    def forward(self, x):
        logits = self.network(x)
        scaled_logits = logits / self.temperature
        return F.softmax(scaled_logits, dim=-1)

class MixtureOfExperts(nn.Module):
    """
    Complete Mixture of Experts model
    """
    def __init__(self, input_size, hidden_size, output_size, num_experts):
        super(MixtureOfExperts, self).__init__()
        self.num_experts = num_experts
        
        # Initialize experts
        self.experts = nn.ModuleList([
            Expert(input_size, hidden_size, output_size) 
            for _ in range(num_experts)
        ])
        
        # Initialize gating network
        self.gating_network = GatingNetwork(input_size, num_experts, hidden_size)
        
    def forward(self, x):
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
    """
    Tracks and saves expert activation patterns
    """
    def __init__(self, base_path=None, num_experts=None):
        self.timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        if base_path is None:
            base_path = 'moe_outputs'
        
        self.base_dir = os.path.join(base_path, self.timestamp)
        self.plots_dir = os.path.join(self.base_dir, 'plots')
        self.data_dir = os.path.join(self.base_dir, 'data')
        self.model_dir = os.path.join(self.base_dir, 'models')
        
        # Create directories
        for directory in [self.plots_dir, self.data_dir, self.model_dir]:
            os.makedirs(directory, exist_ok=True)
        
        self.num_experts = num_experts
        self.config = {
            'created_at': self.timestamp,
            'base_dir': self.base_dir,
            'num_experts': num_experts
        }
        
        # Save config
        with open(os.path.join(self.base_dir, 'config.json'), 'w') as f:
            json.dump(self.config, f, indent=4)
        
        # Initialize metrics tracking
        self.metrics = {
            'train_loss': [],
            'train_acc': [],
            'expert_usage': []
        }
    
    def save_expert_activations(self, expert_weights, labels, epoch, batch_idx):
        """
        Save and visualize expert activations
        """
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
        plt.figure(figsize=(10, 6))
        mean_activations = df.groupby('Expert')['Probability'].mean()
        
        sns.barplot(
            x=mean_activations.index,
            y=mean_activations.values,
            palette='viridis'
        )
        
        plt.title(f'Mean Expert Usage (Epoch {epoch}, Batch {batch_idx})')
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
    
    def _create_label_distribution_plot(self, df, epoch, batch_idx):
        plt.figure(figsize=(12, 6))
        
        # Calculate mean expert activation per label
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
        plt.tight_layout()
        plt.savefig(
            os.path.join(
                self.plots_dir,
                f'label_distribution_epoch_{epoch}_batch_{batch_idx}.png'
            )
        )
        plt.close()
    
    def update_metrics(self, epoch, loss, accuracy, expert_usage):
        """
        Update training metrics
        """
        self.metrics['train_loss'].append((epoch, loss))
        self.metrics['train_acc'].append((epoch, accuracy))
        self.metrics['expert_usage'].append((epoch, expert_usage))
        
        # Save metrics
        metrics_df = pd.DataFrame({
            'epoch': epoch,
            'loss': loss,
            'accuracy': accuracy,
            'expert_usage': str(expert_usage)
        }, index=[0])
        
        metrics_path = os.path.join(self.data_dir, 'training_metrics.csv')
        if os.path.exists(metrics_path):
            metrics_df.to_csv(metrics_path, mode='a', header=False, index=False)
        else:
            metrics_df.to_csv(metrics_path, index=False)
    
    def save_model(self, model, optimizer, epoch, loss):
        """
        Save model checkpoint
        """
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

def train_epoch(model, train_loader, optimizer, device, tracker, epoch):
    """
    Train for one epoch
    """
    model.train()
    total_loss = 0
    correct = 0
    total = 0
    
    progress_bar = tqdm(train_loader, desc=f'Epoch {epoch}')
    
    for batch_idx, (data, target) in enumerate(progress_bar):
        data, target = data.to(device), target.to(device)
        
        # Flatten images for MLP
        data = data.view(data.size(0), -1)
        
        optimizer.zero_grad()
        output, expert_weights = model(data)
        
        loss = F.cross_entropy(output, target)
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
        
        # Calculate accuracy
        pred = output.argmax(dim=1)
        correct += pred.eq(target).sum().item()
        total += target.size(0)
        
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
    epoch_loss = total_loss / len(train_loader)
    epoch_acc = 100. * correct / total
    
    # Calculate expert usage
    expert_usage = expert_weights.mean(dim=0).squeeze().detach().cpu().numpy()
    
    return epoch_loss, epoch_acc, expert_usage

def main():
    # Hyperparameters
    input_size = 784  # 28x28 MNIST images
    hidden_size = 256
    output_size = 10  # 10 digits
    num_experts = 4
    learning_rate = 0.001
    num_epochs = 10
    batch_size = 64
    
    # Setup device
    #device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    device = torch.device('cpu')
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
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True
    )
    
    # Initialize model and optimizer
    model = MixtureOfExperts(
        input_size,
        hidden_size,
        output_size,
        num_experts
    ).to(device)
    
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    
    # Initialize tracker
    tracker = ExpertActivationTracker(num_experts=num_experts)
    
    # Training loop
    for epoch in range(num_epochs):
        loss, accuracy, expert_usage = train_epoch(
            model,
            train_loader,
            optimizer,
            device,
            tracker,
            epoch + 1
        )
        
        # Update and save metrics
        tracker.update_metrics(epoch + 1, loss, accuracy, expert_usage)
        
        # Save model checkpoint
        tracker.save_model(model, optimizer, epoch + 1, loss)
        
        print(f'\nEpoch {epoch + 1}:')
        print(f'Average Loss: {loss:.4f}')
        print(f'Accuracy: {accuracy:.2f}%')
        print('Expert Usage:', expert_usage)

if __name__ == '__main__':
    main()