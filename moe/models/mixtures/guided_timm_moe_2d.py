from typing import Dict, List, Optional, Tuple, Any
import torch
import torch.nn as nn
import torch.nn.functional as F
import timm

from moe.interfaces.moe_interface import MoEInterface
from moe.models.experts.experts_1d import Expert1D
from moe.models.gates.gates_1d import GuidedGating1D
from moe.models.mixtures.moe_loss import MoELoss
from moe.models.mixtures.guided_moe_1d import GuidedMoE1D

class GuidedTimmMoE2D(MoELoss,MoEInterface):
    """Guided Mixture of Experts implementation using TIMM models for 1D inputs"""
    
    def __init__(self,
                 model_name: str,
                 input_channels: Tuple[int,int,int],
                 hidden_size: int,
                 output_size: int,
                 num_experts: int,
                 expert_label_assignments: Dict[int, List[int]],
                 pretrained: bool = True,
                 dropout_rate: float = 0.3):
        """
        Initialize guided TIMM-based MoE model
        
        Args:
            model_name: Name of TIMM model to use as feature extractor
            input_size: Size of input features
            hidden_size: Size of hidden layer
            output_size: Number of output classes
            num_experts: Number of experts to use
            expert_label_assignments: Dictionary mapping expert indices to lists of class labels
            pretrained: Whether to use pretrained TIMM models
            dropout_rate: Dropout rate for regularization
        """
        # Initialize both parent classes
        MoELoss.__init__(self)
        nn.Module.__init__(self)
        
        self._num_experts = num_experts
        self.expert_label_assignments = expert_label_assignments
        self.model_name = model_name
        try:
            # Initialize TIMM model for feature extraction with specific settings
            self.feature_extractor = timm.create_model(
                model_name,
                pretrained=pretrained,
                num_classes=0,  # Remove classification head
                in_chans=input_channels[0],  # Number of input channels
                global_pool='avg',  # Use global average pooling
                features_only=False  # Get the final features
            )
            
            # Get feature dimension by checking model's last layer
            if hasattr(self.feature_extractor, 'num_features'):
                self._feature_dim = self.feature_extractor.num_features
            else:
                # For models without num_features attribute, try to determine from the model
                self._feature_dim = self._determine_feature_dim(self.feature_extractor)
            
        except Exception as e:            
            raise f"Error initializing TIMM model: {str(e)}"
        
        # Initialize experts
        self.experts = nn.ModuleList([
            Expert1D(self._feature_dim, hidden_size, output_size,dropout_rate)
            for _ in range(num_experts)
        ])
        
        # Initialize guided gating network
        self.gating_network = GuidedGating1D(
            self._feature_dim,
            num_experts,
            hidden_size,
            expert_label_assignments
        )        
        # Storage for metrics
        self.loss_components = {}
        self.metrics = {}
    
    @property
    def num_experts(self) -> int:
        return self._num_experts
    
    def forward(self, x: torch.Tensor, labels: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, torch.Tensor, List[torch.Tensor]]:
        """
        Forward pass through the MoE model
        
        Args:
            x: Input tensor of shape [batch_size, input_size]
            labels: Optional tensor of labels for guided routing
            
        Returns:
            Tuple containing:
            - Final output tensor
            - Expert weights tensor
            - List of expert L2 losses
        """
        x = self.feature_extractor(x) # Get the TIMM features
        
        # Get expert weights using label guidance during training
        expert_weights = self.gating_network(x, labels)
        
        # Get expert outputs
        expert_outputs = []
        expert_l2_losses = []
        
        for expert in self.experts:
            output, l2_loss = expert(x)
            expert_outputs.append(output)
            expert_l2_losses.append(l2_loss)
        
        expert_outputs = torch.stack(expert_outputs)
        expert_outputs = expert_outputs.permute(1, 0, 2)
        
        # Combine expert outputs using routing weights
        final_output = torch.sum(expert_outputs * expert_weights.unsqueeze(-1), dim=1)
        
        # Store routing entropy for monitoring
        if self.training:
            entropy = -(expert_weights * torch.log(expert_weights + 1e-6)).sum(dim=-1).mean()
            self.metrics['routing_entropy'] = entropy.item()
        
        return final_output, expert_weights,expert_l2_losses
    
    def compute_loss(self, final_output: torch.Tensor, target: torch.Tensor,
                expert_weights: torch.Tensor, expert_l2_losses: List[torch.Tensor]) -> torch.Tensor:
        # Use same loss computation as 1D guided version
        return GuidedMoE1D.compute_loss(self, final_output, target, expert_weights, expert_l2_losses)
    
    def get_metrics(self) -> Dict[str, float]:
        """Get current metrics including loss components and routing entropy"""
        metrics = {
            **self.loss_components,
            **self.metrics
        }
        return metrics
    

    def _determine_feature_dim(self, model) -> int:
        """Helper method to determine feature dimension from model architecture"""
        # Common feature dimension attributes in TIMM models
        feature_attributes = ['num_features', 'feature_dim', 'num_channels', 'embed_dim']
        
        for attr in feature_attributes:
            if hasattr(model, attr):
                return getattr(model, attr)
        
        # If no attribute found, try to find the last conv/linear layer
        last_layer = None
        for module in model.modules():
            if isinstance(module, (nn.Conv2d, nn.Linear)):
                last_layer = module
        
        if last_layer is not None:
            if isinstance(last_layer, nn.Conv2d):
                return last_layer.out_channels
            else:
                return last_layer.out_features
        
        # Default fallback
        raise ValueError(f"Could not determine feature dimension for model {self.model_name}")
    
