"""From-scratch frozen T5 encoder for symbolic music token sequences."""

from typing import Any, Tuple

import torch

from elf.modules.t5_encoder import T5Encoder, T5EncoderConfig
from elf.utils.logging_utils import log_for_0


def get_music_encoder(
    vocab_size: int,
    d_model: int = 512,
    seed: int = 0,
    dtype: Any = torch.float32,
) -> Tuple[T5EncoderConfig, T5Encoder]:
    """Build a randomly initialized T5 encoder over the music vocabulary."""
    from transformers import T5Config

    log_for_0(
        f"Building from-scratch music encoder: vocab_size={vocab_size}, "
        f"d_model={d_model}, seed={seed}"
    )
    torch.manual_seed(seed)
    hf_cfg = T5Config(
        vocab_size=vocab_size,
        d_model=d_model,
        d_kv=64,
        d_ff=d_model * 4,
        num_layers=6,
        num_heads=8,
        is_gated_act=False,
    )
    cfg = T5EncoderConfig("music-t5-scratch", dtype)
    cfg.vocab_size = vocab_size
    cfg.d_model = d_model
    cfg.d_kv = hf_cfg.d_kv
    cfg.d_ff = hf_cfg.d_ff
    cfg.num_layers = hf_cfg.num_layers
    cfg.num_heads = hf_cfg.num_heads
    cfg.is_gated_act = False

    model = T5Encoder(cfg, pretrained=False, hf_config=hf_cfg)
    if dtype is not None:
        model = model.to(dtype)
    return cfg, model
