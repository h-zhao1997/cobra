"""
Main training script for QLoRA fine-tuning of Cobra VLM on LLaVA-CoT-100k.
"""
import os
import sys
from pathlib import Path

# Workaround for Triton ldconfig issue
# Set environment variables before importing torch/transformers
os.environ.setdefault("TRITON_CACHE_DIR", "/tmp/triton_cache")
# Try to find CUDA libraries in common locations
cuda_paths = [
    "/usr/local/cuda/lib64",
    "/usr/lib/x86_64-linux-gnu",
    "/opt/conda/lib",
]
if "LD_LIBRARY_PATH" not in os.environ:
    existing_paths = []
else:
    existing_paths = os.environ["LD_LIBRARY_PATH"].split(":")
os.environ["LD_LIBRARY_PATH"] = ":".join([p for p in cuda_paths + existing_paths if p and os.path.exists(p)])

import torch
from transformers import TrainingArguments, Trainer
from trl import SFTTrainer

# Add parent directory to path so we can import qlora_finetune modules
# This allows the script to work when run from within qlora_finetune/ or from parent directory
script_dir = Path(__file__).parent
parent_dir = script_dir.parent
if str(parent_dir) not in sys.path:
    sys.path.insert(0, str(parent_dir))

from qlora_finetune.config import QLoRAConfig
from qlora_finetune.dataset_loader import load_llava_cot_dataset, format_for_sft
from qlora_finetune.model_loader import load_and_prepare_model
from qlora_finetune.utils import add_parent_to_path, get_model_parameter_count

# Add parent to path for cobra imports
add_parent_to_path()

# Disable tokenizers parallelism
os.environ["TOKENIZERS_PARALLELISM"] = "false"


def main(config: QLoRAConfig):
    """Main training function."""
    print("=" * 80)
    print("QLoRA Fine-tuning for Cobra VLM")
    print("=" * 80)
    
    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Create output directory
    config.output_dir.mkdir(parents=True, exist_ok=True)
    
    # Load and prepare model
    print("\n" + "=" * 80)
    print("Loading and preparing model...")
    print("=" * 80)
    
    vlm, llm_backbone = load_and_prepare_model(
        model_id=config.model_id,
        pretrained_checkpoint=config.pretrained_checkpoint,
        hf_token=config.hf_token,
        freeze_vision_encoder=config.freeze_vision_encoder,
        lora_r=config.lora_r,
        lora_alpha=config.lora_alpha,
        lora_dropout=config.lora_dropout,
        lora_target_modules=config.lora_target_modules,
        lora_bias=config.lora_bias,
    )
    
    # Get tokenizer - either from VLM or load it separately if using quantized loader
    if vlm is not None:
        tokenizer = vlm.llm_backbone.tokenizer
    else:
        # Load tokenizer separately when using quantized loader
        from transformers import AutoTokenizer
        mamba_model_map = {
            "cobra+3b": "xiuyul/mamba-2.8b-zephyr",
            "cobra+7b": "state-spaces/mamba-2.8b",
        }
        mamba_model_id = mamba_model_map.get(config.model_id, "xiuyul/mamba-2.8b-zephyr")
        hf_token = config.hf_token
        if hf_token is not None and Path(hf_token).exists():
            with open(hf_token, "r") as f:
                hf_token = f.read().strip()
        tokenizer = AutoTokenizer.from_pretrained(mamba_model_id, token=hf_token)
        print(f"Loaded tokenizer from {mamba_model_id}")
    
    # Print model info
    param_info = get_model_parameter_count(llm_backbone)
    print(f"\nModel Parameters:")
    print(f"  Total: {param_info['total']:,} ({param_info['total']/1e9:.2f}B)")
    print(f"  Trainable: {param_info['trainable']:,} ({param_info['trainable']/1e9:.2f}B)")
    print(f"  Trainable: {param_info['trainable_percent']:.2f}%")
    
    # Load dataset
    print("\n" + "=" * 80)
    print("Loading dataset...")
    print("=" * 80)
    
    dataset = load_llava_cot_dataset(
        dataset_name=config.dataset_name,
        dataset_root=config.dataset_root,
        dataset_proportion=config.dataset_proportion,
        dataset_max_samples=config.dataset_max_samples,
        dataset_seed=config.dataset_seed,
    )
    
    print(f"Loaded dataset with {len(dataset)} samples")
    if config.dataset_max_samples is not None:
        print(f"Limited to {config.dataset_max_samples} samples")
    elif config.dataset_proportion is not None and config.dataset_proportion < 1.0:
        print(f"Using {config.dataset_proportion*100:.1f}% of the dataset")
    
    # Format dataset for SFT
    print("Formatting dataset for SFT...")
    dataset = dataset.map(
        lambda x: format_for_sft(x, tokenizer),
        remove_columns=[col for col in dataset.column_names if col != "text"],
    )
    
    # Note: SFTTrainer handles tokenization internally when dataset_text_field is provided
    # So we don't need a custom data collator
    
    # Print model info for debugging
    print(f"\nModel info:")
    print(f"  Model type: {type(llm_backbone)}")
    if hasattr(llm_backbone, 'config'):
        print(f"  Hidden size: {llm_backbone.config.d_model if hasattr(llm_backbone.config, 'd_model') else 'N/A'}")
        print(f"  Vocab size: {llm_backbone.config.vocab_size if hasattr(llm_backbone.config, 'vocab_size') else 'N/A'}")
    if hasattr(llm_backbone, 'backbone') and hasattr(llm_backbone.backbone, 'embedding'):
        emb = llm_backbone.backbone.embedding
        print(f"  Embedding shape: {emb.weight.shape if hasattr(emb, 'weight') else 'N/A'}")
    
    # Enable gradient checkpointing to save memory
    # Try different methods depending on the model type
    try:
        if hasattr(llm_backbone, 'gradient_checkpointing_enable'):
            llm_backbone.gradient_checkpointing_enable()
            print("Enabled gradient checkpointing to reduce memory usage")
        elif hasattr(llm_backbone, 'enable_gradient_checkpointing'):
            llm_backbone.enable_gradient_checkpointing()
            print("Enabled gradient checkpointing to reduce memory usage")
        elif hasattr(llm_backbone, 'base_model') and hasattr(llm_backbone.base_model, 'gradient_checkpointing_enable'):
            llm_backbone.base_model.gradient_checkpointing_enable()
            print("Enabled gradient checkpointing to reduce memory usage")
    except Exception as e:
        print(f"Warning: Could not enable gradient checkpointing: {e}")
        print("Continuing without gradient checkpointing")
    
    # Check if WandB is available before setting report_to
    report_to = config.report_to
    if "wandb" in report_to:
        wandb_api_key = os.environ.get("WANDB_API_KEY")
        wandb_logged_in = False
        try:
            from wandb.sdk.lib.apikey import api_key
            if api_key(settings=None):
                wandb_logged_in = True
        except:
            pass
        
        if not wandb_api_key and not wandb_logged_in:
            print("Warning: WandB API key not found. Disabling WandB tracking.")
            print("To enable WandB, set WANDB_API_KEY environment variable or run: wandb login")
            report_to = [r for r in report_to if r != "wandb"]
            if not report_to:
                report_to = ["none"]
    
    # Training arguments
    # Build kwargs dict, only including optional parameters if they're not None
    training_kwargs = {
        "output_dir": str(config.output_dir),
        "per_device_train_batch_size": config.per_device_train_batch_size,
        "gradient_accumulation_steps": config.gradient_accumulation_steps,
        "learning_rate": config.learning_rate,
        "num_train_epochs": config.num_train_epochs,
        "warmup_ratio": config.warmup_ratio,
        "weight_decay": config.weight_decay,
        "max_grad_norm": config.max_grad_norm,
        "fp16": config.fp16,
        "bf16": config.bf16,
        # Note: Mamba models don't support standard gradient checkpointing
        # "gradient_checkpointing": True,  # Disabled for Mamba compatibility
        "logging_steps": config.logging_steps,
        "save_steps": config.save_steps,
        "save_total_limit": config.save_total_limit,
        "seed": config.seed,
        "dataloader_num_workers": config.dataloader_num_workers,
        "remove_unused_columns": config.remove_unused_columns,
        "report_to": report_to,
        "run_name": f"qlora-{config.model_id}-{config.dataset_max_samples or config.dataset_proportion or 'all'}",
    }
    
    # Only add optional parameters if they're not None
    if config.max_steps is not None:
        training_kwargs["max_steps"] = config.max_steps
    if config.eval_steps is not None:
        training_kwargs["eval_steps"] = config.eval_steps
    
    training_args = TrainingArguments(**training_kwargs)
    
    # Initialize W&B if specified
    if "wandb" in report_to:
        import wandb
        wandb.init(
            project=config.wandb_project,
            entity=config.wandb_entity,
            name=training_args.run_name,
            config={
                "model_id": config.model_id,
                "dataset_proportion": config.dataset_proportion,
                "dataset_max_samples": config.dataset_max_samples,
                "lora_r": config.lora_r,
                "lora_alpha": config.lora_alpha,
                "learning_rate": config.learning_rate,
                "batch_size": config.per_device_train_batch_size * config.gradient_accumulation_steps,
            },
        )
    
    # Create trainer
    print("\n" + "=" * 80)
    print("Creating trainer...")
    print("=" * 80)
    
    # Create a custom data collator to ensure proper input formatting
    from transformers import DataCollatorForLanguageModeling
    
    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer,
        mlm=False,  # Causal LM, not masked LM
        pad_to_multiple_of=8,  # Pad to multiple of 8 for efficiency
    )
    
    trainer = SFTTrainer(
        model=llm_backbone,
        args=training_args,
        train_dataset=dataset,
        tokenizer=tokenizer,
        dataset_text_field="text",  # Specify the text field in the dataset
        packing=False,  # Don't pack sequences (we handle formatting manually)
        max_seq_length=512,  # Reduced sequence length to save memory
        data_collator=data_collator,  # Use custom collator
    )
    
    # Train
    print("\n" + "=" * 80)
    print("Starting training...")
    print("=" * 80)
    
    trainer.train()
    
    # Save final model
    print("\n" + "=" * 80)
    print("Saving final model...")
    print("=" * 80)
    
    trainer.save_model()
    tokenizer.save_pretrained(config.output_dir)
    
    print(f"\nTraining complete! Model saved to {config.output_dir}")
    print(f"To load the model, use:")
    print(f"  from peft import PeftModel")
    print(f"  model = PeftModel.from_pretrained(base_model, '{config.output_dir}')")


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="QLoRA fine-tuning for Cobra VLM")
    parser.add_argument("--config", type=str, help="Path to config file (JSON)")
    parser.add_argument("--model_id", type=str, default="cobra+3b", help="Cobra model ID")
    parser.add_argument("--dataset_proportion", type=float, default=None, help="Proportion of dataset to use (0.0-1.0)")
    parser.add_argument("--dataset_max_samples", type=int, default=None, help="Maximum number of samples to use (overrides proportion)")
    parser.add_argument("--output_dir", type=str, default="./qlora_outputs", help="Output directory")
    parser.add_argument("--hf_token", type=str, default=".hf_token", help="HuggingFace token or path to file")
    parser.add_argument("--per_device_train_batch_size", type=int, default=None, help="Batch size per device")
    parser.add_argument("--gradient_accumulation_steps", type=int, default=None, help="Gradient accumulation steps")
    
    args = parser.parse_args()
    
    # Create config
    if args.config and Path(args.config).exists():
        import json
        with open(args.config, "r") as f:
            config_dict = json.load(f)
        config = QLoRAConfig(**config_dict)
    else:
        config = QLoRAConfig()
        # Override with command line args
        if args.model_id:
            config.model_id = args.model_id
        if args.dataset_proportion is not None:
            config.dataset_proportion = args.dataset_proportion
        if args.dataset_max_samples is not None:
            config.dataset_max_samples = args.dataset_max_samples
        if args.output_dir:
            config.output_dir = Path(args.output_dir)
        if args.hf_token:
            config.hf_token = args.hf_token
        if args.per_device_train_batch_size is not None:
            config.per_device_train_batch_size = args.per_device_train_batch_size
        if args.gradient_accumulation_steps is not None:
            config.gradient_accumulation_steps = args.gradient_accumulation_steps
    
    main(config)

