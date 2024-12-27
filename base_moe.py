"""
base_moe.py - Base Mixture of Experts implementation
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

class Expert(nn.Module):
    """Individual expert network"""
    
    def __init__(self, input_size: int, hidden_size: int, output_size: int, dropout_rate: float = 0.3):
        """
        Initialize expert network
        
        Args:
            input_size: Input feature dimension
            hidden_size: Hidden layer dimension
            output_size: Output dimension (num classes)
            dropout_rate: Dropout probability
        """
        super().__init__()
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
        """Forward pass through expert network"""
        return self.network(x)

class GatingNetwork(nn.Module):
    """Gating network that determines expert weights"""
    
    def __init__(self, input_size: int, num_experts: int, hidden_size: int, dropout_rate: float = 0.3):
        """
        Initialize gating network
        
        Args:
            input_size: Input feature dimension
            num_experts: Number of experts to weight
            hidden_size: Hidden layer dimension
            dropout_rate: Dropout probability
        """
        super().__init__()
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
        
        # Learnable temperature parameter for expert selection
        self.temperature = nn.Parameter(torch.ones(1))
        
        # Initialize weights
        self._init_weights()
    
    def _init_weights(self):
        """Initialize network weights"""
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight)
                nn.init.constant_(m.bias, 0)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through gating network
        
        Args:
            x: Input tensor of shape (batch_size, input_size)
            
        Returns:
            Expert weights tensor of shape (batch_size, num_experts)
        """
        logits = self.network(x)
        # Apply temperature scaling and softmax for expert weights
        return F.softmax(logits / self.temperature, dim=-1)

class BaseMixtureOfExperts(nn.Module):
    """Base Mixture of Experts model"""
    
    def __init__(self, input_size: int, hidden_size: int, output_size: int, num_experts: int):
        """
        Initialize MoE model
        
        Args:
            input_size: Input feature dimension
            hidden_size: Hidden layer dimension
            output_size: Output dimension (num classes)
            num_experts: Number of experts in the mixture
        """
        super().__init__()
        self.num_experts = num_experts
        
        # Initialize experts
        self.experts = nn.ModuleList([
            Expert(input_size, hidden_size, output_size)
            for _ in range(num_experts)
        ])
        
        # Initialize gating network
        self.gating_network = GatingNetwork(input_size, num_experts, hidden_size)
        
        # Store loss components for monitoring
        self.loss_components = {}
    
    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass through MoE
        
        Args:
            x: Input tensor of shape (batch_size, input_size)
            
        Returns:
            tuple containing:
                - Final output tensor of shape (batch_size, output_size)
                - Expert weights tensor of shape (batch_size, num_experts)
        """
        # Get expert weights from gating network
        expert_weights = self.gating_network(x)
        
        # Get outputs from each expert
        expert_outputs = torch.stack([expert(x) for expert in self.experts])
        
        # Combine expert outputs with weights
        expert_outputs = expert_outputs.permute(1, 0, 2)  # (batch_size, num_experts, output_size)
        
        # Ensure expert weights has shape (batch_size, num_experts)
        if expert_weights.dim() == 3:
            expert_weights = expert_weights.squeeze(-1)
        
        # Weighted sum of expert outputs
        final_output = (expert_outputs * expert_weights.unsqueeze(-1)).sum(dim=1)
        
        return final_output, expert_weights
    
    def compute_loss(self, final_output: torch.Tensor, target: torch.Tensor, 
                    expert_weights: torch.Tensor) -> torch.Tensor:
        """
        Compute total loss with regularization terms
        
        Args:
            final_output: Model predictions
            target: Ground truth labels
            expert_weights: Expert combination weights
            
        Returns:
            Total loss value
        """
        # Classification loss
        ce_loss = F.cross_entropy(final_output, target)
        
        # Expert diversity loss (entropy of expert weights)
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
        
        # Combine losses with weights
        total_loss = (
            ce_loss + 
            0.1 * diversity_loss +  # Encourage expert specialization
            0.1 * balance_loss      # Encourage balanced expert usage
        )
        
        # Store components for monitoring
        self.loss_components = {
            'ce_loss': ce_loss.item(),
            'diversity_loss': diversity_loss.item(),
            'balance_loss': balance_loss.item(),
            'total_loss': total_loss.item()
        }
        
        return total_loss

def train_step(model: BaseMixtureOfExperts, batch: tuple[torch.Tensor, torch.Tensor], 
               optimizer: torch.optim.Optimizer) -> dict:
    """
    Single training step
    
    Args:
        model: MoE model
        batch: Tuple of (inputs, targets)
        optimizer: PyTorch optimizer
        
    Returns:
        Dict containing batch metrics
    """
    model.train()
    inputs, targets = batch
    optimizer.zero_grad()
    
    # Forward pass
    outputs, expert_weights = model(inputs)
    loss = model.compute_loss(outputs, targets, expert_weights)
    
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
def evaluate_step(model: BaseMixtureOfExperts, batch: tuple[torch.Tensor, torch.Tensor]) -> dict:
    """
    Single evaluation step
    
    Args:
        model: MoE model
        batch: Tuple of (inputs, targets)
        
    Returns:
        Dict containing batch metrics
    """
    model.eval()
    inputs, targets = batch
    
    # Forward pass
    outputs, expert_weights = model(inputs)
    loss = model.compute_loss(outputs, targets, expert_weights)
    
    # Calculate accuracy
    predictions = outputs.argmax(dim=1)
    accuracy = (predictions == targets).float().mean().item()
    
    return {
        'loss': loss.item(),
        'accuracy': accuracy,
        'expert_weights': expert_weights.detach(),
        'predictions': predictions,
        'targets': targets,
        'loss_components': model.loss_components
    }