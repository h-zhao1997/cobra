"""
Utility script to check and clear GPU memory.

Usage:
    python clear_gpu.py              # Check GPU memory status
    python clear_gpu.py --clear      # Clear GPU cache
    python clear_gpu.py --kill-all   # Kill all Python processes using GPU (DANGEROUS!)
"""

import argparse
import gc
import subprocess
import sys

import torch


def check_gpu_memory():
    """Check and display GPU memory status."""
    if not torch.cuda.is_available():
        print("CUDA is not available. No GPU detected.")
        return
    
    print("=" * 60)
    print("GPU Memory Status")
    print("=" * 60)
    
    for i in range(torch.cuda.device_count()):
        props = torch.cuda.get_device_properties(i)
        total = props.total_memory / 1024**3  # GB
        
        allocated = torch.cuda.memory_allocated(i) / 1024**3  # GB
        reserved = torch.cuda.memory_reserved(i) / 1024**3  # GB
        free = total - reserved
        
        print(f"\nGPU {i}: {props.name}")
        print(f"  Total Memory:    {total:.2f} GB")
        print(f"  Allocated:        {allocated:.2f} GB ({allocated/total*100:.1f}%)")
        print(f"  Reserved:         {reserved:.2f} GB ({reserved/total*100:.1f}%)")
        print(f"  Free:             {free:.2f} GB ({free/total*100:.1f}%)")
    
    print("\n" + "=" * 60)
    print("Processes using GPU (from nvidia-smi):")
    print("=" * 60)
    try:
        result = subprocess.run(
            ["nvidia-smi", "--query-compute-apps=pid,process_name,used_memory", "--format=csv,noheader"],
            capture_output=True,
            text=True,
            check=True
        )
        if result.stdout.strip():
            print(result.stdout)
        else:
            print("No processes found using GPU compute.")
    except (subprocess.CalledProcessError, FileNotFoundError):
        print("Could not run nvidia-smi. Make sure nvidia-smi is installed and in PATH.")


def clear_gpu_cache():
    """Clear PyTorch GPU cache."""
    if not torch.cuda.is_available():
        print("CUDA is not available. Nothing to clear.")
        return
    
    print("Clearing GPU cache...")
    torch.cuda.empty_cache()
    torch.cuda.synchronize()
    gc.collect()
    
    allocated = torch.cuda.memory_allocated(0) / 1024**3  # GB
    reserved = torch.cuda.memory_reserved(0) / 1024**3  # GB
    total = torch.cuda.get_device_properties(0).total_memory / 1024**3  # GB
    free = total - reserved
    
    print(f"✓ GPU cache cleared")
    print(f"  Free memory: {free:.2f} GB / {total:.2f} GB")


def kill_python_processes():
    """Kill all Python processes using GPU (DANGEROUS - use with caution!)."""
    if not torch.cuda.is_available():
        print("CUDA is not available.")
        return
    
    print("⚠️  WARNING: This will kill ALL Python processes using the GPU!")
    print("   This includes other users' processes if on a shared machine.")
    response = input("Are you sure? Type 'yes' to continue: ")
    
    if response.lower() != 'yes':
        print("Cancelled.")
        return
    
    try:
        # Get PIDs of processes using GPU
        result = subprocess.run(
            ["nvidia-smi", "--query-compute-apps=pid", "--format=csv,noheader"],
            capture_output=True,
            text=True,
            check=True
        )
        
        pids = [pid.strip() for pid in result.stdout.strip().split('\n') if pid.strip()]
        
        if not pids:
            print("No processes found using GPU.")
            return
        
        # Filter to only Python processes
        python_pids = []
        for pid in pids:
            try:
                proc_result = subprocess.run(
                    ["ps", "-p", pid, "-o", "comm="],
                    capture_output=True,
                    text=True,
                    check=True
                )
                if "python" in proc_result.stdout.lower():
                    python_pids.append(pid)
            except subprocess.CalledProcessError:
                continue
        
        if not python_pids:
            print("No Python processes found using GPU.")
            return
        
        print(f"\nFound {len(python_pids)} Python process(es) using GPU:")
        for pid in python_pids:
            print(f"  PID: {pid}")
        
        response = input(f"\nKill these {len(python_pids)} process(es)? Type 'yes' to confirm: ")
        if response.lower() != 'yes':
            print("Cancelled.")
            return
        
        # Kill processes
        killed = 0
        for pid in python_pids:
            try:
                subprocess.run(["kill", "-9", pid], check=True)
                print(f"✓ Killed process {pid}")
                killed += 1
            except subprocess.CalledProcessError as e:
                print(f"✗ Failed to kill process {pid}: {e}")
        
        print(f"\n✓ Killed {killed} process(es)")
        print("Waiting 2 seconds for memory to free...")
        import time
        time.sleep(2)
        
        # Clear cache
        clear_gpu_cache()
        
    except (subprocess.CalledProcessError, FileNotFoundError) as e:
        print(f"Error: {e}")
        print("Make sure nvidia-smi and ps are installed and in PATH.")


def main():
    parser = argparse.ArgumentParser(description="GPU memory management utility")
    parser.add_argument(
        "--clear",
        action="store_true",
        help="Clear GPU cache"
    )
    parser.add_argument(
        "--kill-all",
        action="store_true",
        help="Kill all Python processes using GPU (DANGEROUS!)"
    )
    
    args = parser.parse_args()
    
    if args.kill_all:
        kill_python_processes()
    elif args.clear:
        check_gpu_memory()
        clear_gpu_cache()
        print()
        check_gpu_memory()
    else:
        check_gpu_memory()


if __name__ == "__main__":
    main()

