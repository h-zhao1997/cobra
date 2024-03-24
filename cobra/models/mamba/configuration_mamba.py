# coding=utf-8
# Copyright (c) 2023 Jean-Loup Tastet

from transformers.configuration_utils import PretrainedConfig


MAMBA_PRETRAINED_CONFIG_ARCHIVE_MAP = {
    "state-spaces/mamba-2.8b": "https://huggingface.co/state-spaces/mamba-2.8b/resolve/main/config.json",
    "state-spaces/mamba-1.4b": "https://huggingface.co/state-spaces/mamba-1.4b/resolve/main/config.json",
    "state-spaces/mamba-790m": "https://huggingface.co/state-spaces/mamba-790m/resolve/main/config.json",
    "state-spaces/mamba-370m": "https://huggingface.co/state-spaces/mamba-370m/resolve/main/config.json",
    "state-spaces/mamba-130m": "https://huggingface.co/state-spaces/mamba-130m/resolve/main/config.json",
    "state-spaces/mamba-2.8b-slimpj": "https://huggingface.co/state-spaces/mamba-2.8b-slimpj/resolve/main/config.json",
}

DEFAULT_SSM_CONFIG = {
    "d_state": 16,
    "d_conv": 4,
    "expand": 2,
    "dt_rank": "auto",
    "dt_min": 0.001,
    "dt_max": 0.1,
    "dt_init": "random",
    "dt_scale": 1.0,
    "dt_init_floor": 1e-4,
    "conv_bias": True,
    "bias": False,
    "use_fast_path": True,
}


class MambaConfig(PretrainedConfig):

    model_type = "mamba"

    def __init__(
        self,
        vocab_size=50277,
        n_layer=64,
        d_model=2560,
        ssm_cfg=DEFAULT_SSM_CONFIG,  # TODO: refactor?
        norm_epsilon=1e-5,
        rms_norm=True,
        residual_in_fp32=True,
        fused_add_norm=True,
        pad_vocab_size_multiple=8,
        initializer_cfg=None,
        bos_token_id=0,
        eos_token_id=0,
        pad_token_id=1,
        **kwargs,
    ):
        super().__init__(bos_token_id=bos_token_id, eos_token_id=eos_token_id, pad_token_id=pad_token_id, **kwargs)

        self.vocab_size = vocab_size
        self.n_layer = n_layer
        self.d_model = d_model
        self.hidden_size = d_model
        self.norm_epsilon = norm_epsilon
        self.rms_norm = rms_norm
        self.residual_in_fp32 = residual_in_fp32
        self.fused_add_norm = fused_add_norm
        self.pad_vocab_size_multiple = pad_vocab_size_multiple

        self.ssm_cfg = ssm_cfg
        self.initializer_cfg = initializer_cfg