"""
mamba.py

Class definition for all LLMs derived from MambaForCausalLM.
"""
from typing import Optional, Type

import torch
from torch import nn as nn
from cobra.models.mamba.modeling_mamba import MambaForCausalLM
from cobra.models.mamba.modeling_mamba import Block as MambaMixerLayer

from cobra.models.backbones.llm.base_llm import HFCausalLLMBackbone
from cobra.models.backbones.llm.prompting import (
    ZephyrChatPromptBuilder,
    PromptBuilder,
    MambaPromptBuilder
)

# Registry =>> Support Mamba Models
# fmt: off
MAMBA_MODELS = {
    # === Pure Mamba (non-instruct/chat-tuned) Models ===
    "mamba-2.8b-slimpj": {
        "llm_family": "mamba", "llm_cls": MambaForCausalLM, "hf_hub_path": "state-spaces/mamba-2.8b-slimpj"
    },

    "mamba-2.8b": {
        "llm_family": "mamba", "llm_cls": MambaForCausalLM, "hf_hub_path": "state-spaces/mamba-2.8b"
    },

    "mamba-1.4b": {
        "llm_family": "mamba", "llm_cls": MambaForCausalLM, "hf_hub_path": "state-spaces/mamba-1.4b"
    },

    # === Finetuned Mamba Chat Model Based on mamba-2.8b-slimpj ===
    "mamba-2.8b-zephyr": {
        "llm_family": "mamba", "llm_cls": MambaForCausalLM, "hf_hub_path": "xiuyul/mamba-2.8b-zephyr"
    },
}


class MambaLLMBackbone(HFCausalLLMBackbone):
    def __init__(
        self,
        llm_backbone_id: str,
        llm_max_length: int = 2048,
        hf_token: Optional[str] = None,
        inference_mode: bool = False,
        use_flash_attention_2: bool = True, # Add for compatibility, Mamba does not have any attention
    ) -> None:
        super().__init__(
            llm_backbone_id,
            llm_max_length=llm_max_length,
            hf_token=hf_token,
            inference_mode=inference_mode,
            use_flash_attention_2=False,
            **MAMBA_MODELS[llm_backbone_id],
        )

        self.llm.config.pad_token_id = self.tokenizer.pad_token_id

    @property
    def prompt_builder_fn(self) -> Type[PromptBuilder]:
        if self.identifier.endswith("zephyr"):
            return ZephyrChatPromptBuilder
        
        elif self.identifier.startswith("mamba"):
            return MambaPromptBuilder

        raise ValueError(f"No PromptBuilder defined for LLM Backbone `{self.identifier}`")

    @property
    def transformer_layer_cls(self) -> Type[nn.Module]:
        return MambaMixerLayer

    @property
    def half_precision_dtype(self) -> torch.dtype:
        """Mamba was trained in AMP with BF16; see https://github.com/state-spaces/mamba/issues/6."""
        return torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
