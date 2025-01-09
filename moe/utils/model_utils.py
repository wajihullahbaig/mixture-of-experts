import torch
import numpy as np
from typing import List, Dict, Any

def create_expert_assignments(num_classes: int, num_experts: int) -> Dict[int, list]:
    """Create balanced label assignments for experts"""
    assignments = {}
    labels_per_expert = num_classes // num_experts
    remaining = num_classes % num_experts
    
    start_idx = 0
    for expert_idx in range(num_experts):
        num_labels = labels_per_expert + (1 if expert_idx < remaining else 0)
        assignments[expert_idx] = list(range(start_idx, start_idx + num_labels))
        start_idx += num_labels
    
    return assignments


def check_for_nans(items: List[Any]) -> Dict[int, Dict[str, Any]]:
    """
    Check if any item in the list contains NaN values, moving tensors to CPU if necessary.

    Args:
    items (List[Any]): A list of PyTorch tensors, numpy arrays, or other numeric objects to check.

    Returns:
    Dict[int, Dict[str, Any]]: A dictionary where keys are item indices (1-based)
                               and values are dictionaries containing details about NaN occurrences.
    """
    nan_details = {}
    for idx, item in enumerate(items, start=1):
        if isinstance(item, torch.Tensor):
            # Move tensor to CPU if it's on another device
            item = item.cpu().detach()
            if torch.isnan(item).any():
                nan_positions = torch.nonzero(torch.isnan(item), as_tuple=False)
                nan_details[idx] = {
                    "item_number": idx,
                    "type": "PyTorch Tensor",
                    "original_device": item.device,
                    "input_shape": item.shape,
                    "nan_count": torch.isnan(item).sum().item(),
                    "nan_positions": nan_positions.tolist()
                }
        elif isinstance(item, np.ndarray):
            if np.isnan(item).any():
                nan_positions = np.argwhere(np.isnan(item))
                nan_details[idx] = {
                    "item_number": idx,
                    "type": "NumPy Array",
                    "input_shape": item.shape,
                    "nan_count": np.isnan(item).sum(),
                    "nan_positions": nan_positions.tolist()
                }
        elif isinstance(item, (list, tuple)):
            # Check if the list/tuple contains any torch.Tensor
            contains_tensor = any(isinstance(x, torch.Tensor) for x in item)
            if contains_tensor:
                # Convert all tensors to CPU and detach
                cpu_item = [x.cpu().detach() if isinstance(x, torch.Tensor) else x for x in item]
                # Convert to numpy array
                np_item = np.array([x.numpy() if isinstance(x, torch.Tensor) else x for x in cpu_item])
            else:
                np_item = np.array(item)
            
            if np.isnan(np_item).any():
                nan_positions = np.argwhere(np.isnan(np_item))
                nan_details[idx] = {
                    "item_number": idx,
                    "type": f"{type(item).__name__} containing {'tensors' if contains_tensor else 'non-tensors'}",
                    "input_shape": np_item.shape,
                    "nan_count": np.isnan(np_item).sum(),
                    "nan_positions": nan_positions.tolist()
                }
        elif isinstance(item, (int, float)):
            if np.isnan(item):
                nan_details[idx] = {
                    "item_number": idx,
                    "type": f"{type(item).__name__}",
                    "nan_count": 1
                }
        else:
            nan_details[idx] = {
                "item_number": idx,
                "type": f"{type(item).__name__}",
                "error": "Unsupported type for NaN checking"
            }
    
    return nan_details
