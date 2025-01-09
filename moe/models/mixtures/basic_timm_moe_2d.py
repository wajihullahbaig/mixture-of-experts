from typing import Any, Dict, List, Optional, Tuple
import torch
from torch import nn
import timm

from moe.interfaces.moe_interface import MoEInterface
from moe.models.experts.experts_1d import Expert1D
from moe.models.gates.gates_1d import BasicGating1D
from moe.models.mixtures.basic_moe_1d import BasicMoE1D



class BasicTimmMoE2D(MoEInterface):
    """Basic Mixture of Experts implementation for 2D inputs"""
    
    def __init__(self,
                 model_name: str,
                 input_channels: Tuple[int,int,int],
                 hidden_size: int,
                 output_size: int,
                 num_experts: int,
                 pretrained: bool = True,
                 dropout_rate: float = 0.3):
        super().__init__()
        self._num_experts = num_experts
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
        # Initialize gating network
        self.gating_network = BasicGating1D(self._feature_dim, num_experts,hidden_size)
        
        # Store loss components
        self.loss_components = {}
    
    @property
    def num_experts(self) -> int:
        return self._num_experts
    
    def forward(self, x: torch.Tensor, labels: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, torch.Tensor, List[torch.Tensor]]:
        # Extract features using TIMM model
        x = self.feature_extractor(x)
        # Get expert weights
        expert_weights = self.gating_network(x)
        
        # Get expert outputs
        expert_outputs = []
        expert_l2_losses = []
        
        for expert in self.experts:
            output, l2_loss = expert(x)
            expert_outputs.append(output)
            expert_l2_losses.append(l2_loss)
        
        expert_outputs = torch.stack(expert_outputs)
        expert_outputs = expert_outputs.permute(1, 0, 2)
        
        # Combine expert outputs
        final_output = torch.sum(expert_outputs * expert_weights.unsqueeze(-1), dim=1)
        
        return final_output, expert_weights, expert_l2_losses
    
    def compute_loss(self, final_output: torch.Tensor, target: torch.Tensor, 
                    expert_weights: torch.Tensor, expert_l2_losses: List[torch.Tensor]) -> torch.Tensor:
        # Use same loss computation as 1D
        return BasicMoE1D.compute_loss(self, final_output, target, expert_weights, expert_l2_losses)

