"""
Utility functions for QLoRA fine-tuning.
"""
import sys
from pathlib import Path
from typing import List, Set

import torch
import torch.nn as nn


def find_mamba_target_modules(model) -> List[str]:
    """
    Auto-detect target modules for LoRA in Mamba models.
    
    Mamba models use State Space Model (SSM) layers instead of attention.
    This function identifies linear layers in the Mamba SSM blocks.
    
    Args:
        model: The Mamba model to inspect
        
    Returns:
        List of module names that can be targeted for LoRA
    """
    target_modules = set()
    
    # Common patterns for Mamba SSM layers
    # Based on mamba_ssm structure: in_proj, out_proj, x_proj, dt_proj, A_proj, B_proj, D_proj
    mamba_patterns = [
        "in_proj",
        "out_proj", 
        "x_proj",
        "dt_proj",
        "A_proj",
        "B_proj",
        "D_proj",
        "conv1d",  # Convolutional layers in SSM
    ]
    
    # Also check for projector layers in CobraVLM
    projector_patterns = [
        "projector",
        "gate_proj",
        "up_proj",
        "down_proj",
    ]
    
    # Check all named modules
    excluded_modules = {"in_proj"}  # Exclude in_proj to avoid shape issues with quantized models
    for name, module in model.named_modules():
        if isinstance(module, nn.Linear):
            # Extract just the module name (last part)
            module_name = name.split('.')[-1]
            # Skip excluded modules
            if module_name in excluded_modules:
                continue
            # Check if it matches Mamba SSM patterns
            for pattern in mamba_patterns + projector_patterns:
                if pattern in name.lower():
                    # Use just the module name, not the full path
                    target_modules.add(module_name)
                    break
    
    # If no specific patterns found, try to find linear layers in mixer/SSM blocks
    if not target_modules:
        for name, module in model.named_modules():
            if isinstance(module, nn.Linear):
                # Include linear layers in layers (Mamba blocks)
                if "layers" in name or "mixer" in name or "ssm" in name:
                    target_modules.add(name)
    
    # Fallback: if still nothing, return common linear layer names
    if not target_modules:
        # Try to find any linear layers in the backbone
        for name, module in model.named_modules():
            if isinstance(module, nn.Linear) and "backbone" in name:
                target_modules.add(name)
    
    return sorted(list(target_modules))


def get_model_parameter_count(model) -> dict:
    """
    Get parameter count statistics for a model.
    
    Args:
        model: The model to analyze
        
    Returns:
        Dictionary with total, trainable, and frozen parameter counts
    """
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    frozen_params = total_params - trainable_params
    
    return {
        "total": total_params,
        "trainable": trainable_params,
        "frozen": frozen_params,
        "trainable_percent": (trainable_params / total_params * 100) if total_params > 0 else 0.0,
    }


def add_parent_to_path():
    """Add parent directory to Python path to import cobra modules."""
    parent_dir = Path(__file__).parent.parent
    if str(parent_dir) not in sys.path:
        sys.path.insert(0, str(parent_dir))


def format_number(num: int) -> str:
    """Format large numbers with K/M/B suffixes."""
    if num >= 1_000_000_000:
        return f"{num / 1_000_000_000:.2f}B"
    elif num >= 1_000_000:
        return f"{num / 1_000_000:.2f}M"
    elif num >= 1_000:
        return f"{num / 1_000:.2f}K"
    else:
        return str(num)

