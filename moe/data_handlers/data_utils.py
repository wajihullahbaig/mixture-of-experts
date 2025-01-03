# data/data_utils.py

import torch
from typing import Tuple

from moe.configs.default_config import ArchitectureType
class DataProcessor:
    """Process data according to architecture"""
    
    def __init__(self, architecture: ArchitectureType):
        """
        Initialize data processor
        Args:
            architecture: '1d' or '2d'
        """
        self.architecture = architecture
    
    def process_batch(self, 
                     batch: Tuple[torch.Tensor, torch.Tensor]
                    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Process a batch of data based on architecture"""
        inputs, targets = batch
        
        if self.architecture == ArchitectureType.ARCH_1D:
            # Flatten input for 1D architecture
            inputs = inputs.view(inputs.size(0), -1)
        
        return inputs, targets