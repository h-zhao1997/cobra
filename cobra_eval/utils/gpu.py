"""
GPU management utilities.
"""
import torch
import gc
from typing import Optional

def clear_gpu_memory() -> Optional[float]:
    """Clear GPU cache and run garbage collection."""
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()
        gc.collect()
        if torch.cuda.is_available():
            allocated = torch.cuda.memory_allocated(0) / 1024**3  # GB
            reserved = torch.cuda.memory_reserved(0) / 1024**3  # GB
            total = torch.cuda.get_device_properties(0).total_memory / 1024**3  # GB
            free = total - reserved
            print(f"GPU Memory - Allocated: {allocated:.2f} GB, Reserved: {reserved:.2f} GB, Free: {free:.2f} GB / {total:.2f} GB")
            return free
    return None

def check_gpu_memory(min_free_gb: float = 8.0) -> bool:
    """
    Check if there's enough free GPU memory.
    
    Args:
        min_free_gb: Minimum free memory required in GB
    
    Returns:
        True if enough memory is available, False otherwise
    """
    if not torch.cuda.is_available():
        return True
    
    total = torch.cuda.get_device_properties(0).total_memory / 1024**3  # GB
    reserved = torch.cuda.memory_reserved(0) / 1024**3  # GB
    free = total - reserved
    
    print(f"GPU Memory Status:")
    print(f"  Total: {total:.2f} GB")
    print(f"  Reserved: {reserved:.2f} GB")
    print(f"  Free: {free:.2f} GB")
    print(f"  Required: {min_free_gb:.2f} GB")
    
    if free < min_free_gb:
        print(f"\n⚠️  WARNING: Only {free:.2f} GB free, but {min_free_gb:.2f} GB recommended!")
        print("   Consider:")
        print("   1. Closing other processes using the GPU")
        print("   2. Running: nvidia-smi to check GPU usage")
        print("   3. Killing processes: kill <PID> (from nvidia-smi)")
        response = input(f"\nContinue anyway? (y/n): ")
        return response.lower() == 'y'
    
    return True

