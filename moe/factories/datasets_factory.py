from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from typing import Tuple, Optional, Union
from torch.utils.data import random_split, Dataset

from moe.configs.default_config import ArchitectureType

class DatasetFactory:
    """Factory for creating datasets and dataloaders"""
   
    @staticmethod
    def get_dataset_config(dataset: str, architecture: Union[ArchitectureType,str]) -> dict:
        """Get dataset-specific configuration"""
        configs = {
            'mnist': {
                'input_size': 784 if architecture == ArchitectureType.ARCH_1D or architecture == '1d' else (1, 28, 28),
                'input_channels': 1,
                'num_classes': 10,
                'transforms': [
                    transforms.ToTensor(),
                    transforms.Normalize((0.1307,), (0.3081,))
                ]
            },
            'cifar10': {
                'input_size': 3072 if architecture == ArchitectureType.ARCH_1D or architecture == '1d' else (3, 32, 32),
                'input_channels': 3,
                'num_classes': 10,
                'transforms': [
                    transforms.ToTensor(),
                    transforms.Normalize(mean=[0.4914, 0.4822, 0.4465], 
                                        std=[0.2023, 0.1994, 0.2010])
                ]
            }
        }
        
        if dataset not in configs:
            raise ValueError(f"Dataset {dataset} not supported. Choose from {list(configs.keys())}")
            
        return configs[dataset]
    
    @staticmethod
    def create_dataset(dataset: str, data_dir: str, train: bool = True, 
                      architecture: Optional[Union[ArchitectureType, str]] = None) -> Dataset:
        """Create a dataset without DataLoader wrapping"""
        config = DatasetFactory.get_dataset_config(dataset, architecture) if architecture else \
                DatasetFactory.get_dataset_config(dataset, ArchitectureType.ARCH_2D)
        transform = transforms.Compose(config['transforms'])
        
        if dataset == 'mnist':
            return datasets.MNIST(data_dir, train=train, download=True, transform=transform)
        elif dataset == 'cifar10':
            return datasets.CIFAR10(data_dir, train=train, download=True, transform=transform)
        elif dataset == 'cifar100':
            return datasets.CIFAR100(data_dir, train=train, download=True, transform=transform)
        else:
            raise ValueError(f"Dataset {dataset} not supported")
    
    @staticmethod
    def create_dataloaders(dataset: str, data_dir: str, batch_size: int,
                        num_workers: int, architecture: str, subset_fraction: float = 1.0) -> tuple:
        """Create train and test dataloaders with optional subset"""
        config = DatasetFactory.get_dataset_config(dataset, architecture)
        transform = transforms.Compose(config['transforms'])
        
        if dataset == 'mnist':
            train_dataset = datasets.MNIST(data_dir, train=True, download=True, transform=transform)
            test_dataset = datasets.MNIST(data_dir, train=False, download=True, transform=transform)
        elif dataset == 'cifar10':
            train_dataset = datasets.CIFAR10(data_dir, train=True, download=True, transform=transform)
            test_dataset = datasets.CIFAR10(data_dir, train=False, download=True, transform=transform)
        elif dataset == 'cifar100':
            train_dataset = datasets.CIFAR100(data_dir, train=True, download=True, transform=transform)
            test_dataset = datasets.CIFAR100(data_dir, train=False, download=True, transform=transform)
        
        if subset_fraction < 1.0:
            train_subset_size = int(len(train_dataset) * subset_fraction)
            test_subset_size = int(len(test_dataset) * subset_fraction)
            
            train_dataset, _ = random_split(train_dataset, [train_subset_size, len(train_dataset) - train_subset_size])
            test_dataset, _ = random_split(test_dataset, [test_subset_size, len(test_dataset) - test_subset_size])
        
        train_loader = DataLoader(train_dataset, batch_size=batch_size, 
                                shuffle=True, num_workers=num_workers)
        test_loader = DataLoader(test_dataset, batch_size=batch_size, 
                                shuffle=False, num_workers=num_workers)
        
        return train_loader, test_loader