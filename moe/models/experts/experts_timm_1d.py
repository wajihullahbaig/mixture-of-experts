import torch
import torch.nn as nn
import timm
from typing import Tuple, Optional, Dict, Any

from moe.interfaces.experts_interface import ExpertInterface

class TimmExpert1D(ExpertInterface):
    """Expert network using TIMM models for feature extraction and 1D classification"""
    
    def __init__(self, 
                 model_name: str,
                 input_size: int,
                 hidden_size: int,
                 output_size: int,
                 pretrained: bool = True,
                 dropout_rate: float = 0.3,
                 l2_reg: float = 0.01):
        """
        Initialize TIMM-based expert
        
        Args:
            model_name: Name of TIMM model to use as feature extractor
            input_size: Size of input features
            hidden_size: Size of hidden layer
            output_size: Number of output classes
            pretrained: Whether to use pretrained weights
            dropout_rate: Dropout rate for regularization
            l2_reg: L2 regularization coefficient
        """
        super().__init__()
        
        # Store parameters
        self.model_name = model_name
        self.input_size = input_size
        self.l2_reg = l2_reg
        
        # Initialize TIMM model for feature extraction
        self.feature_extractor = timm.create_model(
            model_name,
            pretrained=pretrained,
            num_classes=0  # Remove classification head
        )
        
        # Get feature dimension from TIMM model
        with torch.no_grad():
            dummy_input = torch.randn(1, input_size)
            features = self.feature_extractor(dummy_input)
            feature_dim = features.shape[1]
        
        # Create classifier network
        self.classifier = nn.Sequential(
            nn.BatchNorm1d(feature_dim),
            nn.Linear(feature_dim, hidden_size),
            nn.ReLU(),
            nn.BatchNorm1d(hidden_size),
            nn.Dropout(dropout_rate),
            nn.Linear(hidden_size, hidden_size // 2),
            nn.ReLU(),
            nn.BatchNorm1d(hidden_size // 2),
            nn.Dropout(dropout_rate),
            nn.Linear(hidden_size // 2, output_size)
        )
        
        self._init_weights()
    
    def _init_weights(self):
        """Initialize the weights of the classifier network"""
        for m in self.classifier.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm1d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
    
    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass through the expert
        
        Args:
            x: Input tensor of shape [batch_size, input_size]
            
        Returns:
            Tuple of (output logits, l2_loss)
        """
        # Extract features using TIMM model
        with torch.set_grad_enabled(self.training):
            features = self.feature_extractor(x)
        
        # Apply classifier
        output = self.classifier(features)
        
        # Calculate L2 regularization loss
        l2_loss = sum(p.pow(2.0).sum() for p in self.parameters()) * self.l2_reg
        
        return output, l2_loss
    
    def get_config(self) -> Dict[str, Any]:
        """Get expert configuration for serialization"""
        return {
            'model_name': self.model_name,
            'input_size': self.input_size,
            'l2_reg': self.l2_reg
        }