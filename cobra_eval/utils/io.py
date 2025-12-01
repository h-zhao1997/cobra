"""
JSON Input/Output utilities.
"""
import json
from pathlib import Path
from typing import Dict, Any, Union, Optional
from datetime import datetime

class DateTimeEncoder(json.JSONEncoder):
    """Custom JSON encoder for datetime objects."""
    def default(self, obj):
        if isinstance(obj, datetime):
            return obj.isoformat()
        return super().default(obj)

def save_json_results(
    data: Dict[str, Any], 
    output_dir: Union[str, Path], 
    filename_prefix: str = "results"
) -> Path:
    """
    Save dictionary data to a JSON file with timestamp.
    
    Args:
        data: Dictionary to save
        output_dir: Directory to save to
        filename_prefix: Prefix for the filename
        
    Returns:
        Path to the saved file
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    filename = f"{filename_prefix}_{timestamp}.json"
    output_path = output_dir / filename
    
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(data, f, cls=DateTimeEncoder, indent=2, ensure_ascii=False)
        
    print(f"Results saved to {output_path}")
    return output_path

def load_json_results(file_path: Union[str, Path]) -> Dict[str, Any]:
    """Load results from a JSON file."""
    with open(file_path, 'r', encoding='utf-8') as f:
        return json.load(f)

def find_latest_result_file(
    output_dir: Union[str, Path], 
    filename_prefix: str
) -> Optional[Path]:
    """Find the most recent results file matching the prefix."""
    output_dir = Path(output_dir)
    if not output_dir.exists():
        return None
        
    # Look for {prefix}_*.json
    pattern = f"{filename_prefix}_*.json"
    files = list(output_dir.glob(pattern))
    
    if not files:
        return None
        
    # Sort by name (which includes timestamp)
    return sorted(files)[-1]

def save_checkpoint(
    data: Dict[str, Any], 
    output_dir: Union[str, Path], 
    filename_prefix: str = "checkpoint",
    sample_count: int = 0,
    overwrite: bool = False
) -> Path:
    """
    Save checkpoint data to a JSON file.
    
    Args:
        data: Dictionary to save
        output_dir: Directory to save to
        filename_prefix: Prefix for the filename
        sample_count: Number of samples processed (for filename)
        overwrite: If True, overwrites a single 'latest' file instead of creating new timestamped files
        
    Returns:
        Path to the saved checkpoint file
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    if overwrite:
        filename = f"{filename_prefix}_latest.json"
    else:
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        filename = f"{filename_prefix}_{sample_count}_{timestamp}.json"
        
    output_path = output_dir / filename
    
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(data, f, cls=DateTimeEncoder, indent=2, ensure_ascii=False)
        
    return output_path

def find_latest_checkpoint(
    output_dir: Union[str, Path], 
    filename_prefix: str = "checkpoint"
) -> Optional[Path]:
    """Find the most recent checkpoint file matching the prefix."""
    output_dir = Path(output_dir)
    if not output_dir.exists():
        return None
        
    # Look for {prefix}_*.json
    pattern = f"{filename_prefix}_*.json"
    files = list(output_dir.glob(pattern))
    
    if not files:
        return None
        
    # Sort by modification time (most recent first)
    return max(files, key=lambda p: p.stat().st_mtime)
