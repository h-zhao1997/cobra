"""
Configuration for QLoRA fine-tuning.
"""
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional


@dataclass
class QLoRAConfig:
    """Configuration for QLoRA fine-tuning of Cobra VLM."""
    
    # Model configuration
    model_id: str = "cobra+3b"  # Cobra model ID to load
    pretrained_checkpoint: Optional[Path] = None  # Path to pretrained checkpoint (if loading from disk)
    hf_token: Optional[str] = None  # HuggingFace token (or path to .hf_token file)
    
    # LoRA configuration
    lora_r: int = 8  # LoRA rank
    lora_alpha: int = 16  # LoRA alpha
    lora_dropout: float = 0.05  # LoRA dropout
    lora_target_modules: Optional[List[str]] = None  # Target modules for LoRA (None = auto-detect)
    lora_bias: str = "none"  # LoRA bias: "none", "all", or "lora_only"
    
    # Dataset configuration
    dataset_name: str = "Xkev/LLaVA-CoT-100k"  # HuggingFace dataset name
    dataset_root: Optional[Path] = None  # Local dataset root (if using local data)
    dataset_proportion: Optional[float] = None  # Proportion of dataset to use (0.0 to 1.0)
    dataset_max_samples: Optional[int] = None  # Maximum number of samples to use (overrides proportion if set)
    dataset_seed: int = 42  # Random seed for dataset sampling
    
    # Training configuration
    output_dir: Path = Path("./qlora_outputs")  # Output directory
    per_device_train_batch_size: int = 1  # Batch size per device
    gradient_accumulation_steps: int = 1  # Gradient accumulation steps
    learning_rate: float = 2e-5  # Learning rate
    num_train_epochs: int = 2  # Number of training epochs
    max_steps: Optional[int] = None  # Maximum training steps (overrides epochs if set)
    warmup_ratio: float = 0.03  # Warmup ratio
    weight_decay: float = 0.01  # Weight decay
    max_grad_norm: float = 1.0  # Maximum gradient norm
    
    # Training settings
    fp16: bool = True  # Use FP16 training
    bf16: bool = False  # Use BF16 training (if supported)
    logging_steps: int = 10  # Logging frequency
    save_steps: int = 500  # Save checkpoint frequency
    eval_steps: Optional[int] = None  # Evaluation frequency
    save_total_limit: int = 3  # Maximum number of checkpoints to keep
    
    # Other settings
    seed: int = 42  # Random seed
    dataloader_num_workers: int = 4  # Number of dataloader workers
    remove_unused_columns: bool = False  # Remove unused columns from dataset
    
    # Vision encoder freezing
    freeze_vision_encoder: bool = True  # Freeze vision encoder
    
    # Tracking
    report_to: List[str] = field(default_factory=lambda: ["wandb"])  # Tracking backends
    wandb_project: str = "cobra-qlora"  # Weights & Biases project name
    wandb_entity: Optional[str] = None  # Weights & Biases entity
    
    def __post_init__(self):
        """Validate configuration."""
        # Set default proportion if neither proportion nor max_samples is set
        if self.dataset_proportion is None and self.dataset_max_samples is None:
            self.dataset_proportion = 1.0
        
        # Validate proportion if set
        if self.dataset_proportion is not None:
            if not 0.0 < self.dataset_proportion <= 1.0:
                raise ValueError(f"dataset_proportion must be between 0.0 and 1.0, got {self.dataset_proportion}")
        
        # Validate max_samples if set
        if self.dataset_max_samples is not None:
            if self.dataset_max_samples <= 0:
                raise ValueError(f"dataset_max_samples must be > 0, got {self.dataset_max_samples}")
        
        if self.hf_token is not None and Path(self.hf_token).exists():
            # Read token from file
            with open(self.hf_token, "r") as f:
                self.hf_token = f.read().strip()
        
        # Set default target modules if not specified (will be auto-detected)
        if self.lora_target_modules is None:
            self.lora_target_modules = []  # Empty list triggers auto-detection

