"""
guided_moe.py - Guided Mixture of Experts implementation with label-based expert assignment
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import random
from typing import Dict, List, Optional, Tuple

class GuidedExpert(nn.Module):
    """Expert network with improved stability and regularization"""
    
    def __init__(self, input_size: int, hidden_size: int, output_size: int, dropout_rate: float = 0.3):
        """
        Initialize guided expert network
        
        Args:
            input_size: Input feature dimension
            hidden_size: Hidden layer dimension
            output_size: Output dimension (num classes)
            dropout_rate: Dropout probability
        """
        super().__init__()
        
        # Shared feature extractor
        self.feature_extractor = nn.Sequential(
            nn.BatchNorm1d(input_size),
            nn.Linear(input_size, hidden_size),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.BatchNorm1d(hidden_size)
        )
        
        # Task-specific layers with skip connections
        self.task_layers = nn.ModuleList([
            nn.Sequential(
                nn.Linear(hidden_size, hidden_size // 2),
                nn.ReLU(),
                nn.Dropout(dropout_rate),
                nn.BatchNorm1d(hidden_size // 2)
            ),
            nn.Sequential(
                nn.Linear(hidden_size // 2, hidden_size // 4),
                nn.ReLU(),
                nn.Dropout(dropout_rate),
                nn.BatchNorm1d(hidden_size // 4)
            )
        ])
        
        # Output layer
        self.output_layer = nn.Linear(hidden_size // 4, output_size)
        
        # L2 regularization weight
        self.l2_reg = 0.01
        
        # Initialize weights
        self._init_weights()
    
    def _init_weights(self):
        """Initialize network weights"""
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight)
                nn.init.constant_(m.bias, 0)
    
    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass through expert network
        
        Args:
            x: Input tensor
            
        Returns:
            Tuple containing:
                - Output logits
                - L2 regularization loss
        """
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
        l2_loss = sum(p.pow(2.0).sum() for p in self.parameters()) * self.l2_reg
        
        return output, l2_loss

class GuidedGatingNetwork(nn.Module):
    """Improved gating network with label-guided routing"""
    
    def __init__(self, input_size: int, num_experts: int, hidden_size: int, 
                 expert_label_map: Dict[int, List[int]]):
        """
        Initialize guided gating network
        
        Args:
            input_size: Input feature dimension
            num_experts: Number of experts to weight
            hidden_size: Hidden layer dimension
            expert_label_map: Dict mapping expert indices to their assigned labels
        """
        super().__init__()
        self.expert_label_map = expert_label_map
        
        # Main network
        self.network = nn.Sequential(
            nn.BatchNorm1d(input_size),
            nn.Linear(input_size, hidden_size),
            nn.ReLU(),
            nn.BatchNorm1d(hidden_size),
            nn.Linear(hidden_size, hidden_size // 2),
            nn.ReLU(),
            nn.BatchNorm1d(hidden_size // 2)
        )
        
        # Final layer for expert weights
        self.fc_out = nn.Linear(hidden_size // 2, num_experts)
        
        # Learnable parameters for soft routing
        self.routing_temperature = nn.Parameter(torch.ones(1))
        self.label_embedding = nn.Parameter(torch.randn(10, hidden_size // 4))
        
        # Initialize weights
        self._init_weights()
    
    def _init_weights(self):
        """Initialize network weights"""
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight)
                nn.init.constant_(m.bias, 0)
    
    def compute_soft_masks(self, labels: Optional[torch.Tensor] = None) -> Optional[torch.Tensor]:
        """
        Compute soft assignment masks based on label similarities
        
        Args:
            labels: Ground truth labels (optional)
            
        Returns:
            Soft assignment mask tensor or None if labels not provided
        """
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
    
    def forward(self, x: torch.Tensor, labels: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Forward pass through gating network
        
        Args:
            x: Input tensor
            labels: Optional ground truth labels for guided routing
            
        Returns:
            Expert weights tensor
        """
        # Forward pass through main network
        features = self.network(x)
        logits = self.fc_out(features)
        
        if self.training and labels is not None:
            # Get soft assignment masks
            soft_masks = self.compute_soft_masks(labels)
            
            # Combine network output with soft guidance
            routing_weights = F.softmax(logits / self.routing_temperature, dim=1)
            
            # Interpolate between learned and guided routing
            mixing_factor = torch.sigmoid(self.routing_temperature)
            final_weights = mixing_factor * routing_weights + (1 - mixing_factor) * soft_masks
            
            return final_weights
        else:
            # During inference, use pure network output
            return F.softmax(logits / self.routing_temperature, dim=1)

class GuidedMixtureOfExperts(nn.Module):
    """Complete Guided Mixture of Experts model"""
    
    def __init__(self, input_size: int, hidden_size: int, output_size: int, num_experts: int, 
                 expert_label_assignments: Dict[int, List[int]]):
        """
        Initialize Guided MoE model
        
        Args:
            input_size: Input feature dimension
            hidden_size: Hidden layer dimension
            output_size: Output dimension (num classes)
            num_experts: Number of experts in mixture
            expert_label_assignments: Dict mapping experts to their assigned labels
        """
        super().__init__()
        self.num_experts = num_experts
        self.expert_label_assignments = expert_label_assignments
        
        # Initialize experts
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
        
        # Store loss components for monitoring
        self.loss_components = {}
    
    def forward(self, x: torch.Tensor, labels: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, torch.Tensor, List[torch.Tensor]]:
        """
        Forward pass through Guided MoE
        
        Args:
            x: Input tensor
            labels: Optional ground truth labels for guided routing
            
        Returns:
            Tuple containing:
                - Final output tensor
                - Expert weights tensor
                - List of expert L2 losses
        """
        # Get expert weights from gating network
        expert_weights = self.gating_network(x, labels)
        
        # Get outputs from each expert
        expert_outputs = []
        expert_l2_losses = []
        
        for expert in self.experts:
            output, l2_loss = expert(x)
            expert_outputs.append(output)
            expert_l2_losses.append(l2_loss)
        
        expert_outputs = torch.stack(expert_outputs)
        expert_outputs = expert_outputs.permute(1, 0, 2)
        
        # Combine expert outputs with attention mechanism
        attention_weights = expert_weights.unsqueeze(-1)
        final_output = torch.bmm(expert_outputs.transpose(1, 2), attention_weights).squeeze(-1)
        
        return final_output, expert_weights, expert_l2_losses
    
    def compute_loss(self, final_output: torch.Tensor, target: torch.Tensor, 
                    expert_weights: torch.Tensor, expert_l2_losses: List[torch.Tensor]) -> torch.Tensor:
        """
        Compute total loss with all components
        
        Args:
            final_output: Model predictions
            target: Ground truth labels
            expert_weights: Expert combination weights
            expert_l2_losses: List of expert L2 regularization losses
            
        Returns:
            Total loss value
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
            0.01 * diversity_loss +   # Small weight for diversity
            0.01 * balance_loss +     # Small weight for balance
            0.01 * total_l2_loss     # Very small weight for L2
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

def create_improved_label_assignments(num_classes: int, num_experts: int) -> Dict[int, List[int]]:
    """
    Create balanced label assignments for experts with no overlap
    
    Args:
        num_classes: Total number of classes (e.g., 10)
        num_experts: Number of experts to assign (e.g., 2, 3, or 5)
        
    Returns:
        Dict mapping expert indices to their assigned labels
    """
    assignments = {}
    base_labels_per_expert = num_classes // num_experts
    remaining_labels = num_classes % num_experts
    
    current_idx = 0
    
    # Distribute labels sequentially
    for expert_idx in range(num_experts):
        # Calculate number of labels for this expert
        # Give one extra label to earlier experts if we have remainders
        num_labels = base_labels_per_expert + (1 if expert_idx < remaining_labels else 0)
        
        # Assign consecutive labels
        assignments[expert_idx] = list(range(current_idx, current_idx + num_labels))
        current_idx += num_labels
    
    return assignments

def train_step(model: GuidedMixtureOfExperts, batch: Tuple[torch.Tensor, torch.Tensor], 
               optimizer: torch.optim.Optimizer) -> Dict:
    """
    Single training step
    
    Args:
        model: Guided MoE model
        batch: Tuple of (inputs, targets)
        optimizer: PyTorch optimizer
        
    Returns:
        Dict containing batch metrics
    """
    model.train()
    inputs, targets = batch
    optimizer.zero_grad()
    
    # Forward pass
    outputs, expert_weights, expert_l2_losses = model(inputs, targets)
    loss = model.compute_loss(outputs, targets, expert_weights, expert_l2_losses)
    
    # Backward pass
    loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
    optimizer.step()
    
    # Calculate accuracy
    predictions = outputs.argmax(dim=1)
    accuracy = (predictions == targets).float().mean().item()
    
    return {
        'loss': loss.item(),
        'accuracy': accuracy,
        'expert_weights': expert_weights.detach(),
        'loss_components': model.loss_components
    }

@torch.no_grad()
def evaluate_step(model: GuidedMixtureOfExperts, batch: Tuple[torch.Tensor, torch.Tensor]) -> Dict:
    """
    Single evaluation step
    
    Args:
        model: Guided MoE model
        batch: Tuple of (inputs, targets)
        
    Returns:
        Dict containing batch metrics
    """
    model.eval()
    inputs, targets = batch
    
    # Forward pass
    outputs, expert_weights, expert_l2_losses = model(inputs)
    loss = model.compute_loss(outputs, targets, expert_weights, expert_l2_losses)
    
    # Calculate accuracy
    predictions = outputs.argmax(dim=1)
    accuracy = (predictions == targets).float().mean().item()
    
    return {
        'loss': loss.item(),
        'accuracy': accuracy,
        'expert_weights': expert_weights,
        'predictions': predictions,
        'targets': targets,
        'loss_components': model.loss_components
    }