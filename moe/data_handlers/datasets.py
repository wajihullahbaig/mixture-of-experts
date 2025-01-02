import torch
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from typing import Tuple, Optional

class DatasetFactory:
    """Factory for creating datasets and dataloaders"""
    
    @staticmethod
    def get_transforms(dataset: str, architecture: str) -> transforms.Compose:
        """Get transforms for specific dataset and architecture"""
        if dataset == 'mnist':
            transform = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize((0.1307,), (0.3081,))
            ])
        elif dataset == 'cifar':
            transform = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=[0.485, 0.456, 0.406],
                    std=[0.229, 0.224, 0.225]
                )
            ])
        else:
            raise ValueError(f"Dataset {dataset} not supported")
        
        return transform
    
    @staticmethod
    def get_input_size(dataset: str, architecture: str) -> tuple:
        """Get input size for specific dataset and architecture"""
        if dataset == 'mnist':
            return (1, 28, 28) if architecture == '2d' else 784
        elif dataset == 'cifar':
            return (3, 32, 32) if architecture == '2d' else 3072
        else:
            raise ValueError(f"Dataset {dataset} not supported")
    
   
    @staticmethod
    def get_dataset_config(dataset: str, architecture: str) -> dict:
        """Get dataset-specific configuration"""
        configs = {
            'mnist': {
                'input_size': 784 if architecture == '1d' else (1, 28, 28),
                'input_channels': 1,
                'num_classes': 10,
                'transforms': [
                    transforms.ToTensor(),
                    transforms.Normalize((0.1307,), (0.3081,))
                ]
            },
            'cifar': {
                'input_size': 3072 if architecture == '1d' else (3, 32, 32),
                'input_channels': 3,
                'num_classes': 10,
                'transforms': [
                    transforms.ToTensor(),
                    transforms.Normalize(
                        mean=[0.485, 0.456, 0.406],
                        std=[0.229, 0.224, 0.225]
                    )
                ]
            }
        }
        
        if dataset not in configs:
            raise ValueError(f"Dataset {dataset} not supported. Choose from {list(configs.keys())}")
            
        return configs[dataset]
    
    @staticmethod
    def create_dataloaders(dataset: str, data_dir: str, batch_size: int,
                          num_workers: int, architecture: str) -> tuple:
        """Create train and test dataloaders"""
        config = DatasetFactory.get_dataset_config(dataset, architecture)
        transform = transforms.Compose(config['transforms'])
        
        if dataset == 'mnist':
            train_dataset = datasets.MNIST(data_dir, train=True, download=True, transform=transform)
            test_dataset = datasets.MNIST(data_dir, train=False, download=True, transform=transform)
        elif dataset == 'cifar':
            train_dataset = datasets.CIFAR10(data_dir, train=True, download=True, transform=transform)
            test_dataset = datasets.CIFAR10(data_dir, train=False, download=True, transform=transform)
            
        train_loader = DataLoader(train_dataset, batch_size=batch_size, 
                                shuffle=True, num_workers=num_workers)
        test_loader = DataLoader(test_dataset, batch_size=batch_size, 
                               shuffle=False, num_workers=num_workers)
        
        return train_loader, test_loader
    