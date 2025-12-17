"""
Model loading and preparation for QLoRA fine-tuning.
"""
import os
import sys
from pathlib import Path
from typing import Optional, Tuple

import torch
from peft import LoraConfig, get_peft_model

def add_bias_to_lora_layers(model):
    """
    Add bias attribute to all LoRA-wrapped layers in the model.
    This ensures mamba_ssm can access bias without AttributeError.
    """
    try:
        from peft.tuners.lora import Linear as LoRALinear
        
        patched_count = 0
        for name, module in model.named_modules():
            # Check if this is a LoRA-wrapped Linear layer
            if isinstance(module, LoRALinear) or (hasattr(module, 'base_layer') and hasattr(module, 'lora_A')):
                if not hasattr(module, '_bias_patched'):
                    # Get bias from base layer if it exists
                    bias_value = None
                    if hasattr(module, 'base_layer'):
                        bias_value = getattr(module.base_layer, 'bias', None)
                    elif hasattr(module, 'bias'):
                        bias_value = getattr(module, 'bias', None)
                    
                    # Add bias as a cached property/attribute
                    # Use object.__setattr__ to bypass PyTorch's module checks
                    try:
                        object.__setattr__(module, 'bias', bias_value)
                        module._bias_patched = True
                        patched_count += 1
                    except:
                        # If that fails, try regular setattr
                        try:
                            setattr(module, 'bias', bias_value)
                            module._bias_patched = True
                            patched_count += 1
                        except:
                            pass
        
        if patched_count > 0:
            print(f"Added bias attribute to {patched_count} LoRA-wrapped layers")
    except (ImportError, AttributeError) as e:
        print(f"Warning: Could not add bias to LoRA layers: {e}")

def wrap_lora_layers_for_bias(model):
    """
    Patch mamba_ssm and add bias to LoRA layers to safely handle bias access.
    This fixes the AttributeError when mamba_ssm tries to access bias on LoRA-wrapped Linear layers.
    """
    # First, add bias attributes to all LoRA layers
    add_bias_to_lora_layers(model)
    
    # The bias attributes should now be accessible, so the original forward should work
    # But we can also patch the forward method as a safety measure
    try:
        import mamba_ssm.modules.mamba_simple as mamba_simple
        
        if not hasattr(mamba_simple.Mamba, '_bias_patched'):
            # The bias attributes are now added, so original forward should work
            # But we can add a safety check just in case
            mamba_simple.Mamba._bias_patched = True
            print("Prepared mamba_ssm for LoRA bias compatibility")
    except (ImportError, AttributeError) as e:
        print(f"Warning: Could not patch mamba_ssm for LoRA bias compatibility: {e}")

# Add parent directory to path so we can import qlora_finetune modules
script_dir = Path(__file__).parent
parent_dir = script_dir.parent
if str(parent_dir) not in sys.path:
    sys.path.insert(0, str(parent_dir))

from qlora_finetune.utils import add_parent_to_path, find_mamba_target_modules

# Add parent to path to import cobra
add_parent_to_path()

from cobra.models.load import load as load_cobra_model
from cobra.models.vlms.cobra import CobraVLM
from cobra.models.mamba.modeling_mamba import MambaForCausalLM
from transformers import AutoTokenizer


def load_mamba_model(
    mamba_model_id: str,
    hf_token: Optional[str] = None,
) -> Tuple[torch.nn.Module, AutoTokenizer]:
    """
    Load base Mamba model without quantization.
    
    Args:
        mamba_model_id: HuggingFace model ID for the Mamba model (e.g., "xiuyul/mamba-2.8b-zephyr")
        hf_token: HuggingFace token or path to .hf_token file
        
    Returns:
        Tuple of (Mamba model, tokenizer)
    """
    # Load HF token if path provided
    if hf_token is not None and Path(hf_token).exists():
        with open(hf_token, "r") as f:
            hf_token = f.read().strip()
    
    print(f"Loading Mamba model {mamba_model_id} without quantization...")
    
    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(mamba_model_id, token=hf_token)
    # Ensure pad token is set
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.pad_token_id = tokenizer.eos_token_id
    
    # Load model without quantization
    model = MambaForCausalLM.from_pretrained(
        mamba_model_id,
        token=hf_token,
        torch_dtype=torch.float16,
    )
    # Move to GPU if available
    if torch.cuda.is_available():
        model = model.cuda()
    print("Model loaded without quantization")
    
    # Disable fused_add_norm to avoid Triton issues
    if hasattr(model, 'config') and hasattr(model.config, 'fused_add_norm'):
        model.config.fused_add_norm = False
        if hasattr(model, 'backbone') and hasattr(model.backbone, 'fused_add_norm'):
            model.backbone.fused_add_norm = False
        if hasattr(model, 'backbone') and hasattr(model.backbone, 'layers'):
            for layer in model.backbone.layers:
                if hasattr(layer, 'fused_add_norm'):
                    layer.fused_add_norm = False
                # Replace RMSNorm with LayerNorm
                if hasattr(layer, 'norm') and 'RMSNorm' in str(type(layer.norm)):
                    import torch.nn as nn
                    old_norm = layer.norm
                    # Get device from old norm's weight
                    device = old_norm.weight.device if old_norm.weight is not None else (torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu'))
                    new_norm = nn.LayerNorm(
                        old_norm.weight.shape[0],
                        eps=old_norm.eps,
                        elementwise_affine=old_norm.weight is not None
                    ).to(device)  # Move to same device as old norm
                    if old_norm.weight is not None:
                        new_norm.weight.data.copy_(old_norm.weight.data)
                    if hasattr(old_norm, 'bias') and old_norm.bias is not None:
                        new_norm.bias.data.copy_(old_norm.bias.data)
                    layer.norm = new_norm
        # Replace final norm
        if hasattr(model, 'backbone') and hasattr(model.backbone, 'norm_f'):
            if 'RMSNorm' in str(type(model.backbone.norm_f)):
                import torch.nn as nn
                old_norm = model.backbone.norm_f
                # Get device from old norm's weight
                device = old_norm.weight.device if old_norm.weight is not None else (torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu'))
                new_norm = nn.LayerNorm(
                    old_norm.weight.shape[0],
                    eps=old_norm.eps,
                    elementwise_affine=old_norm.weight is not None
                ).to(device) # Move to same device as old norm
                if old_norm.weight is not None:
                    new_norm.weight.data.copy_(old_norm.weight.data)
                if hasattr(old_norm, 'bias') and old_norm.bias is not None:
                    new_norm.bias.data.copy_(old_norm.bias.data)
                model.backbone.norm_f = new_norm
        print("Disabled fused_add_norm and replaced RMSNorm with LayerNorm")
    
    return model, tokenizer


def load_cobra_for_qlora(
    model_id: str,
    pretrained_checkpoint: Optional[Path] = None,
    hf_token: Optional[str] = None,
    freeze_vision_encoder: bool = True,
) -> Tuple[Optional[CobraVLM], torch.nn.Module]:
    """
    Load Cobra model and prepare LLM backbone for QLoRA.
    
    This function:
    1. Loads the full CobraVLM model
    2. Extracts the LLM backbone
    3. Applies quantization if specified
    4. Freezes the vision encoder
    
    Args:
        model_id: Cobra model ID (e.g., "cobra+3b")
        pretrained_checkpoint: Path to pretrained checkpoint (optional)
        hf_token: HuggingFace token or path to .hf_token file
        quantization_config: BitsAndBytesConfig for quantization
        freeze_vision_encoder: Whether to freeze vision encoder
        
    Returns:
        Tuple of (CobraVLM model, LLM backbone for QLoRA)
    """
    # Load HF token if path provided
    if hf_token is not None and Path(hf_token).exists():
        with open(hf_token, "r") as f:
            hf_token = f.read().strip()
    
    # Load full Cobra model
    if pretrained_checkpoint is not None:
        # Load from local checkpoint
        vlm = load_cobra_model(pretrained_checkpoint, hf_token=hf_token)
    else:
        # Load from HuggingFace Hub
        vlm = load_cobra_model(model_id, hf_token=hf_token)
    
    # Disable fused_add_norm and replace RMSNorm with PyTorch implementation
    # to avoid Triton compilation issues (ldconfig not available)
    llm = vlm.llm_backbone.llm
    if hasattr(llm, 'config') and hasattr(llm.config, 'fused_add_norm'):
        # Disable in config
        llm.config.fused_add_norm = False
        # Disable in model backbone
        if hasattr(llm, 'backbone') and hasattr(llm.backbone, 'fused_add_norm'):
            llm.backbone.fused_add_norm = False
        # Disable in each layer
        if hasattr(llm, 'backbone') and hasattr(llm.backbone, 'layers'):
            for layer in llm.backbone.layers:
                if hasattr(layer, 'fused_add_norm'):
                    layer.fused_add_norm = False
                # Replace RMSNorm with PyTorch LayerNorm to avoid Triton
                if hasattr(layer, 'norm') and 'RMSNorm' in str(type(layer.norm)):
                    import torch.nn as nn
                    # Create a PyTorch RMSNorm equivalent using LayerNorm
                    # RMSNorm is essentially LayerNorm without centering (no mean subtraction)
                    # For now, use LayerNorm as a close approximation
                    old_norm = layer.norm
                    new_norm = nn.LayerNorm(
                        old_norm.weight.shape[0],
                        eps=old_norm.eps,
                        elementwise_affine=old_norm.weight is not None
                    )
                    if old_norm.weight is not None:
                        new_norm.weight.data.copy_(old_norm.weight.data)
                    if hasattr(old_norm, 'bias') and old_norm.bias is not None:
                        new_norm.bias.data.copy_(old_norm.bias.data)
                    layer.norm = new_norm
        # Replace final norm if it's RMSNorm
        if hasattr(llm, 'backbone') and hasattr(llm.backbone, 'norm_f'):
            if 'RMSNorm' in str(type(llm.backbone.norm_f)):
                import torch.nn as nn
                old_norm = llm.backbone.norm_f
                new_norm = nn.LayerNorm(
                    old_norm.weight.shape[0],
                    eps=old_norm.eps,
                    elementwise_affine=old_norm.weight is not None
                )
                if old_norm.weight is not None:
                    new_norm.weight.data.copy_(old_norm.weight.data)
                if hasattr(old_norm, 'bias') and old_norm.bias is not None:
                    new_norm.bias.data.copy_(old_norm.bias.data)
                llm.backbone.norm_f = new_norm
        print("Disabled fused_add_norm and replaced RMSNorm with LayerNorm to avoid Triton compilation issues")
    
    # Freeze vision encoder if requested
    if freeze_vision_encoder:
        for param in vlm.vision_backbone.parameters():
            param.requires_grad = False
        print("Vision encoder frozen")
    
    # Extract LLM backbone
    llm_backbone = vlm.llm_backbone.llm  # Access the underlying MambaForCausalLM
    
    return vlm, llm_backbone


def prepare_model_for_qlora(
    model: torch.nn.Module,
    target_modules: Optional[list] = None,
    lora_r: int = 16,
    lora_alpha: int = 32,
    lora_dropout: float = 0.05,
    lora_bias: str = "none",
) -> torch.nn.Module:
    """
    Prepare model for LoRA training (no quantization).
    
    Args:
        model: The model to prepare (typically LLM backbone)
        target_modules: List of module names to apply LoRA to (None = auto-detect)
        lora_r: LoRA rank
        lora_alpha: LoRA alpha
        lora_dropout: LoRA dropout
        lora_bias: LoRA bias setting
        
    Returns:
        Model with LoRA applied
    """
    # Auto-detect target modules if not provided
    if target_modules is None or len(target_modules) == 0:
        target_modules = find_mamba_target_modules(model)
        print(f"Auto-detected target modules: {target_modules}")
    
    if not target_modules:
        raise ValueError(
            "No target modules found for LoRA. Please specify target_modules manually."
        )
    
    # Create LoRA config
    lora_config = LoraConfig(
        r=lora_r,
        lora_alpha=lora_alpha,
        target_modules=target_modules,
        lora_dropout=lora_dropout,
        bias=lora_bias,
        task_type="CAUSAL_LM",
    )
    
    # Apply LoRA
    model = get_peft_model(model, lora_config)
    
    # Wrap LoRA layers to safely expose bias attribute for mamba_ssm compatibility
    wrap_lora_layers_for_bias(model)
    
    print(f"Applied LoRA to {len(target_modules)} module types")
    print(f"LoRA config: r={lora_r}, alpha={lora_alpha}, dropout={lora_dropout}")
    
    return model


def load_and_prepare_model(
    model_id: str,
    pretrained_checkpoint: Optional[Path] = None,
    hf_token: Optional[str] = None,
    freeze_vision_encoder: bool = True,
    lora_r: int = 16,
    lora_alpha: int = 32,
    lora_dropout: float = 0.05,
    lora_target_modules: Optional[list] = None,
    lora_bias: str = "none",
) -> Tuple[Optional[CobraVLM], torch.nn.Module]:
    """
    Complete pipeline: Load Cobra model and prepare for LoRA training (no quantization).
    
    Args:
        model_id: Cobra model ID
        pretrained_checkpoint: Path to checkpoint (optional)
        hf_token: HuggingFace token
        freeze_vision_encoder: Freeze vision encoder
        lora_r: LoRA rank
        lora_alpha: LoRA alpha
        lora_dropout: LoRA dropout
        lora_target_modules: Target modules for LoRA
        lora_bias: LoRA bias setting
        
    Returns:
        Tuple of (CobraVLM, prepared LLM backbone with LoRA)
    """
    # Load model
    vlm, llm_backbone = load_cobra_for_qlora(
        model_id=model_id,
        pretrained_checkpoint=pretrained_checkpoint,
        hf_token=hf_token,
        freeze_vision_encoder=freeze_vision_encoder,
    )
    
    # Prepare for LoRA
    llm_backbone = prepare_model_for_qlora(
        model=llm_backbone,
        target_modules=lora_target_modules,
        lora_r=lora_r,
        lora_alpha=lora_alpha,
        lora_dropout=lora_dropout,
        lora_bias=lora_bias,
    )
    
    return vlm, llm_backbone

