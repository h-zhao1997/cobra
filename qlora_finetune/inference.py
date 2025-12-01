"""
Inference utilities for QLoRA fine-tuned models.
"""
import sys
from pathlib import Path
from typing import Optional

import torch
from PIL import Image
from peft import PeftModel

# Add parent directory to path so we can import qlora_finetune modules
script_dir = Path(__file__).parent
parent_dir = script_dir.parent
if str(parent_dir) not in sys.path:
    sys.path.insert(0, str(parent_dir))

from qlora_finetune.utils import add_parent_to_path

# Add parent to path
add_parent_to_path()

from cobra.models.load import load as load_cobra_model


def load_qlora_model(
    base_model_id: str,
    lora_adapter_path: Path,
    hf_token: Optional[str] = None,
    merge_weights: bool = False,
) -> torch.nn.Module:
    """
    Load QLoRA fine-tuned model.
    
    Args:
        base_model_id: Base Cobra model ID
        lora_adapter_path: Path to LoRA adapter weights
        hf_token: HuggingFace token
        merge_weights: Whether to merge LoRA weights into base model
        
    Returns:
        Loaded model (with or without merged weights)
    """
    # Load base model
    vlm = load_cobra_model(base_model_id, hf_token=hf_token)
    llm_backbone = vlm.llm_backbone.llm
    
    # Load LoRA adapter
    model = PeftModel.from_pretrained(llm_backbone, str(lora_adapter_path))
    
    # Merge weights if requested
    if merge_weights:
        print("Merging LoRA weights into base model...")
        model = model.merge_and_unload()
        print("Weights merged successfully")
    
    return model, vlm


def generate_with_qlora(
    model: torch.nn.Module,
    vlm,
    image: Image,
    prompt: str,
    max_new_tokens: int = 512,
    temperature: float = 0.7,
    do_sample: bool = True,
) -> str:
    """
    Generate text using QLoRA fine-tuned model.
    
    Args:
        model: The fine-tuned model
        vlm: CobraVLM instance (for image processing)
        image: Input image
        prompt: Text prompt
        max_new_tokens: Maximum number of tokens to generate
        temperature: Sampling temperature
        do_sample: Whether to use sampling
        
    Returns:
        Generated text
    """
    # Use the VLM's generation method, but with the fine-tuned LLM backbone
    # Note: This is a simplified version. For full integration, you may need
    # to replace the LLM backbone in the VLM with the fine-tuned one.
    
    # For now, we'll use the VLM's generate method directly
    # The fine-tuned weights should be in the model already
    generated_text = vlm.generate(
        image=image,
        prompt_text=prompt,
        max_new_tokens=max_new_tokens,
        temperature=temperature,
        do_sample=do_sample,
    )
    
    return generated_text


def save_merged_model(
    model: torch.nn.Module,
    output_path: Path,
    tokenizer=None,
):
    """
    Save merged model (after merging LoRA weights).
    
    Args:
        model: Merged model
        output_path: Path to save the model
        tokenizer: Tokenizer to save (optional)
    """
    output_path = Path(output_path)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Save model
    model.save_pretrained(str(output_path))
    
    # Save tokenizer if provided
    if tokenizer is not None:
        tokenizer.save_pretrained(str(output_path))
    
    print(f"Merged model saved to {output_path}")

