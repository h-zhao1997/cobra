"""
nn_utils.py

Utility functions and PyTorch submodule definitions.
"""

import torch
import torch.nn as nn
import math
from cobra.models.mamba.modeling_mamba import create_block, rms_norm_fn, layer_norm_fn, _init_weights, RMSNorm
from functools import partial


# === Definitions for Various Projection Modules, with Signature :: [..., in_dim] --> [..., out_dim] ===
class LinearProjector(nn.Module):
    def __init__(self, vision_dim: int, llm_dim: int) -> None:
        super().__init__()
        self.projector = nn.Linear(vision_dim, llm_dim, bias=True)

    def forward(self, img_patches: torch.Tensor) -> torch.Tensor:
        return self.projector(img_patches)


class MLPProjector(nn.Module):
    def __init__(self, vision_dim: int, llm_dim: int, mlp_type: str = "gelu-mlp") -> None:
        super().__init__()
        if mlp_type == "gelu-mlp":
            self.projector = nn.Sequential(
                nn.Linear(vision_dim, llm_dim, bias=True),
                nn.GELU(),
                nn.Linear(llm_dim, llm_dim, bias=True),
            )
        else:
            raise ValueError(f"Projector with `{mlp_type = }` is not supported!")

    def forward(self, img_patches: torch.Tensor) -> torch.Tensor:
        return self.projector(img_patches)


class FusedMLPProjector(nn.Module):
    def __init__(self, fused_vision_dim: int, llm_dim: int, mlp_type: str = "fused-gelu-mlp") -> None:
        super().__init__()
        self.initial_projection_dim = fused_vision_dim * 4
        if mlp_type == "fused-gelu-mlp":
            self.projector = nn.Sequential(
                nn.Linear(fused_vision_dim, self.initial_projection_dim, bias=True),
                nn.GELU(),
                nn.Linear(self.initial_projection_dim, llm_dim, bias=True),
                nn.GELU(),
                nn.Linear(llm_dim, llm_dim, bias=True),
            )
        else:
            raise ValueError(f"Fused Projector with `{mlp_type = }` is not supported!")

    def forward(self, fused_img_patches: torch.Tensor) -> torch.Tensor:
        return self.projector(fused_img_patches)


# LDPv2 Projector: https://github.com/Meituan-AutoML/MobileVLM/blob/main/mobilevlm/model/vision_projector.py
class TokenDownLayer(nn.Module):
    def __init__(self, shape) -> None:
        super().__init__()
        self.dwn = nn.Sequential(
            nn.AdaptiveAvgPool2d(shape)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, num_tokens, c = x.shape
        h = int(math.sqrt(num_tokens))
        assert h * h == num_tokens
        x = x.permute(0, 2, 1).reshape(b, -1, h, h)
        x = self.dwn(x)
        x = x.flatten(2).transpose(1, 2)
        return x


class PosInjectLayer(nn.Module):
    # https://github.com/Meituan-AutoML/Twins/blob/main/gvt.py
    def __init__(self, in_dim: int, out_dim: int, stride: int = 1) -> None:
        super().__init__()
        self.peg = nn.Sequential(
            nn.Conv2d(in_dim, out_dim, 3, stride, 1, bias=True, groups=out_dim)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, num_tokens, c = x.shape
        h = int(math.sqrt(num_tokens))
        assert h * h == num_tokens
        cnn_feat = x.transpose(1, 2).view(b, c, h, h)
        x = self.peg(cnn_feat) + cnn_feat
        x = x.flatten(2).transpose(1, 2)
        return x
       

class LDPProjector(nn.Module):
    def __init__(self, vision_dim: int, llm_dim: int, mlp_type: str = "ldpnet") -> None:
        super().__init__()
        if mlp_type == "ldpnet":
            self.projector = nn.Sequential(
                nn.Linear(vision_dim, llm_dim, bias=True),
                nn.GELU(),
                nn.Linear(llm_dim, llm_dim, bias=True),
                TokenDownLayer((14, 14)),
                PosInjectLayer(llm_dim, llm_dim, stride=1),
            )
        else:
            raise ValueError(f"Projector with `{mlp_type = }` is not supported!")

    def forward(self, img_patches: torch.Tensor) -> torch.Tensor:
        return self.projector(img_patches)
    
        
class FusedLDPProjector(nn.Module):
    def __init__(self, fused_vision_dim: int, llm_dim: int, mlp_type: str = "fused-ldpnet") -> None:
        super().__init__()
        self.initial_projection_dim = fused_vision_dim * 4
        if mlp_type == "fused-ldpnet":
            self.projector = nn.Sequential(
                nn.Linear(fused_vision_dim, self.initial_projection_dim, bias=True), 
                nn.GELU(), 
                nn.Linear(self.initial_projection_dim, llm_dim, bias=True),
                TokenDownLayer((14, 14)),
                PosInjectLayer(llm_dim, llm_dim, stride=1),
            )
            
        else:
            raise ValueError(f"Fused Projector with `{mlp_type = }` is not supported!")
        
    def forward(self, img_patches: torch.Tensor) -> torch.Tensor:
        return self.projector(img_patches)
    