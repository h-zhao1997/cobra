# LoRA Fine-tuning for Cobra VLM

This directory contains a complete implementation for **parameter-efficient LoRA fine-tuning** of Cobra VLM on the LLaVA-CoT-100k dataset.

## Overview

LoRA enables efficient fine-tuning by:
- **LoRA (Low-Rank Adaptation)**: Only trains a small number of additional parameters (~0.1-1% of original model)
- **Parameter Efficiency**: Typically uses <1% of trainable parameters compared to full fine-tuning

This implementation is designed to work with Cobra VLM's Mamba-based architecture and supports:
- Custom dataset proportion sampling (use only a fraction of the dataset)
- Automatic target module detection for Mamba SSM layers
- Vision encoder freezing (matches Cobra's original recipe)
- Integration with Weights & Biases for tracking

## Installation

### 1. Setup Virtual Environment

This package uses its own isolated virtual environment in `./env`.

**Recommended (scripted) setup:**

```bash
# From the project root
cd qlora_finetune
bash install_requirements.sh

# Activate the environment (Linux/Mac)
source env/bin/activate
```

**Manual setup (alternative):**

```bash
cd qlora_finetune
python -m venv env
source env/bin/activate
pip install -r requirements.txt
```

### 2. Verify Installation

```bash
python -c "import peft, bitsandbytes, trl; print('All packages installed successfully!')"
```

## Quick Start

### Basic Training

```bash
# Activate virtual environment first
source env/bin/activate

# Train with default settings (uses 100% of dataset)
python train_qlora.py \
    --model_id cobra+3b \
    --dataset_proportion 1.0 \
    --output_dir ./qlora_outputs
```

### Using a Fraction of the Dataset

You can specify either a proportion or an absolute number of samples:

```bash
# Use only 10% of the dataset (proportion)
python train_qlora.py \
    --model_id cobra+3b \
    --dataset_proportion 0.1 \
    --output_dir ./qlora_outputs_10pct

# Use exactly 100 samples (absolute number)
python train_qlora.py \
    --model_id cobra+3b \
    --dataset_max_samples 100 \
    --output_dir ./qlora_outputs_100samples

# Use 50% of the dataset
python train_qlora.py \
    --model_id cobra+3b \
    --dataset_proportion 0.5 \
    --output_dir ./qlora_outputs_50pct
```

**Note**: `--dataset_max_samples` takes precedence over `--dataset_proportion` if both are specified.

### Using a Configuration File

Create a JSON configuration file:

```json
{
    "model_id": "cobra+3b",
    "dataset_proportion": 0.25,
    "lora_r": 16,
    "lora_alpha": 32,
    "learning_rate": 2e-5,
    "per_device_train_batch_size": 4,
    "gradient_accumulation_steps": 4,
    "num_train_epochs": 2,
    "output_dir": "./qlora_outputs",
    "wandb_project": "cobra-qlora"
}
```

Then run:

```bash
python train_qlora.py --config config.json
```

## Configuration Options

### Dataset Configuration

- `dataset_proportion` (float, 0.0-1.0, optional): Proportion of dataset to use
  - Example: `0.1` uses 10% of the dataset, `0.5` uses 50%
  - If not set and `dataset_max_samples` is not set, uses 100% of dataset
- `dataset_max_samples` (int, optional): Maximum number of samples to use (overrides proportion if set)
  - Example: `100` uses exactly 100 samples, `1000` uses 1000 samples
  - Useful for quick tests or when you want a specific number regardless of dataset size
- `dataset_seed` (int): Random seed for dataset sampling. Default: 42
- `dataset_name` (str): HuggingFace dataset name. Default: "Xkev/LLaVA-CoT-100k"
- `dataset_root` (Path, optional): Local dataset root if using local data

### LoRA Configuration

- `lora_r` (int): LoRA rank. Default: 8
  - Higher values = more parameters, better capacity but slower training (and more VRAM)
- `lora_alpha` (int): LoRA alpha. Default: 16
  - Scaling factor for LoRA weights (often 2x rank)
- `lora_dropout` (float): LoRA dropout. Default: 0.05
- `lora_target_modules` (list, optional): Target modules for LoRA
  - If None, auto-detects Mamba SSM layers
  - Common targets: `["in_proj", "out_proj", "x_proj", "dt_proj"]`

### Quantization / Precision

This implementation currently **runs the Cobra/Mamba backbone in full precision (FP16)** and does **not**
apply 4-bit quantization. LoRA gives parameter-efficiency, but VRAM usage is closer to full fine-tuning
of the base model. If you want true QLoRA (4-bit) behavior, you would need to extend `model_loader.py`
to load the backbone with a 4-bit `bitsandbytes` configuration.

### Training Configuration

- `per_device_train_batch_size` (int): Batch size per device. Default: 1
- `gradient_accumulation_steps` (int): Gradient accumulation. Default: 1
  - Effective batch size = `per_device_train_batch_size * gradient_accumulation_steps * num_gpus`
- `learning_rate` (float): Learning rate. Default: 2e-5
- `num_train_epochs` (int): Number of epochs. Default: 2
- `max_steps` (int, optional): Maximum training steps (overrides epochs if set)
- `warmup_ratio` (float): Warmup ratio. Default: 0.03
- `weight_decay` (float): Weight decay. Default: 0.01
- `max_grad_norm` (float): Maximum gradient norm. Default: 1.0
- `fp16` (bool): Use FP16 training. Default: True
- `bf16` (bool): Use BF16 training (if supported). Default: False
- `logging_steps` (int): How often to log training metrics. Default: 10
- `save_steps` (int): How often to save checkpoints. Default: 500
- `eval_steps` (int, optional): How often to run evaluation (if enabled)
- `save_total_limit` (int): Maximum number of checkpoints to keep. Default: 3

### Other Settings

- `freeze_vision_encoder` (bool): Freeze vision encoder. Default: True
- `seed` (int): Random seed. Default: 42
- `output_dir` (Path): Output directory for checkpoints
- `dataloader_num_workers` (int): Number of workers for data loading. Default: 4
- `remove_unused_columns` (bool): Whether to drop unused dataset columns. Default: False
- `report_to` (list of str): Tracking backends (e.g., `["wandb"]` or `["none"]`). Default: `["wandb"]`
- `wandb_project` (str): Weights & Biases project name. Default: `"cobra-qlora"`
- `wandb_entity` (str, optional): Weights & Biases entity (organization/user)
- `hf_token` (str): HuggingFace token or path to `.hf_token` file.
  - Default in `QLoRAConfig` is `None`; when running `train_qlora.py` via CLI, the default CLI argument is `.hf_token`

## Usage Examples

### Example 1: Quick Test Run (100 samples)

```bash
python train_qlora.py \
    --model_id cobra+3b \
    --dataset_max_samples 100 \
    --num_train_epochs 1 \
    --output_dir ./test_run
```

### Example 1b: Quick Test Run (10% of dataset)

```bash
python train_qlora.py \
    --model_id cobra+3b \
    --dataset_proportion 0.1 \
    --num_train_epochs 1 \
    --output_dir ./test_run
```

### Example 2: Full Training with Custom LoRA Settings

```bash
python train_qlora.py \
    --model_id cobra+3b \
    --dataset_proportion 1.0 \
    --lora_r 32 \
    --lora_alpha 64 \
    --learning_rate 1e-5 \
    --per_device_train_batch_size 8 \
    --gradient_accumulation_steps 2 \
    --output_dir ./full_training
```

### Example 3: Using Local Dataset

```python
from qlora_finetune.config import QLoRAConfig
from qlora_finetune.train_qlora import main

config = QLoRAConfig(
    model_id="cobra+3b",
    dataset_root=Path("./data/llava_cot"),
    dataset_proportion=0.5,
    output_dir=Path("./outputs"),
)

main(config)
```

## Inference

After training, you can use the fine-tuned model for inference:

```python
from qlora_finetune.inference import load_qlora_model, generate_with_qlora
from PIL import Image

# Load model (with LoRA adapters)
model, vlm = load_qlora_model(
    base_model_id="cobra+3b",
    lora_adapter_path=Path("./qlora_outputs"),
    merge_weights=False,  # Set to True to merge LoRA weights
)

# Generate
image = Image.open("path/to/image.jpg")
prompt = "What is going on in this image?"
response = generate_with_qlora(model, vlm, image, prompt)
print(response)
```

### Merging LoRA Weights

To merge LoRA weights into the base model (for faster inference):

```python
from qlora_finetune.inference import load_qlora_model, save_merged_model

# Load and merge
model, vlm = load_qlora_model(
    base_model_id="cobra+3b",
    lora_adapter_path=Path("./qlora_outputs"),
    merge_weights=True,  # Merge LoRA weights
)

# Save merged model
save_merged_model(
    model,
    output_path=Path("./merged_model"),
    tokenizer=vlm.llm_backbone.tokenizer,
)
```

## Architecture Notes

### Mamba SSM Target Modules

Mamba uses State Space Model (SSM) layers instead of attention. The auto-detection function identifies linear layers in:
- SSM mixer layers: `in_proj`, `out_proj`, `x_proj`, `dt_proj`, `A_proj`, `B_proj`, `D_proj`
- Projector layers: `gate_proj`, `up_proj`, `down_proj`

You can manually specify target modules if auto-detection doesn't work:

```python
config = QLoRAConfig(
    lora_target_modules=["in_proj", "out_proj", "x_proj", "dt_proj"],
)
```

### Vision Encoder

By default, the vision encoder is frozen (not trained). This matches Cobra's original training recipe and significantly reduces memory usage.

## Monitoring Training

Training progress is logged to:
- **Console**: Real-time training metrics
- **Weights & Biases**: If `report_to` includes "wandb"
  - Project: `wandb_project` (default: "cobra-qlora")
  - Run name: `qlora-{model_id}-{dataset_proportion}`

## Troubleshooting

### Out of Memory (OOM)

- Reduce `per_device_train_batch_size`
- Increase `gradient_accumulation_steps` to maintain effective batch size
- Reduce `lora_r` (e.g., from 16 to 8)

### Slow Training

- Increase `per_device_train_batch_size` if memory allows
- Reduce `dataloader_num_workers` if CPU-bound
- Use `fp16=True` (or `bf16=True` if supported)

### Target Modules Not Found

If auto-detection fails, manually specify:

```python
config.lora_target_modules = ["in_proj", "out_proj", "x_proj"]
```

### Dataset Loading Issues

- Ensure you have internet connection for HuggingFace datasets
- For local datasets, ensure `dataset_root` points to directory with `train.jsonl`
- Check that `hf_token` is set if using gated models

## File Structure

```
qlora_finetune/
├── __init__.py              # Package initialization
├── config.py                # Configuration dataclass
├── model_loader.py          # Model loading and QLoRA setup
├── dataset_loader.py        # Dataset loading with proportion support
├── train_qlora.py           # Main training script
├── inference.py             # Inference utilities
├── utils.py                 # Helper functions
├── requirements.txt         # Python dependencies
├── install_requirements.sh  # Virtual environment and dependency setup script
├── .gitignore               # Git ignore rules
└── README.md                # This file
```



## License

This implementation follows the same license as the main Cobra project (MIT License).

