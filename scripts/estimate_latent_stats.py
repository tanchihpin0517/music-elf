#!/usr/bin/env python
"""Estimate latent_mean / latent_std for a music encoder over the training dataset."""

import argparse

import numpy as np
import torch

from elf.configs.config import load_config_from_yaml, apply_config_overrides
from elf.modules.music_encoder import get_music_encoder
from elf.utils.data_utils import get_dataloader, get_pad_token_id, load_dataset
from elf.utils.encoder_utils import encode_text
from elf.utils.music_tokenizer import get_music_tokenizer


@torch.no_grad()
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--config_override", action="append", default=[])
    parser.add_argument("--num_batches", type=int, default=50)
    parser.add_argument("--use_cpu", action="store_true")
    args = parser.parse_args()

    config = load_config_from_yaml(args.config)
    if args.config_override:
        config = apply_config_overrides(config, args.config_override)

    device = torch.device("cpu" if args.use_cpu or not torch.cuda.is_available() else "cuda")
    tokenizer = get_music_tokenizer(config.tokenizer_path)
    pad_token_id = get_pad_token_id(tokenizer, config.pad_token)

    train_dataset, _ = load_dataset(config)
    loader = get_dataloader(
        train_dataset,
        batch_size=min(8, len(train_dataset)),
        shuffle=True,
        num_workers=0,
        drop_last=False,
        max_seq_length=config.max_length,
        pad_token_id=pad_token_id,
    )

    encoder_config, encoder = get_music_encoder(
        vocab_size=len(tokenizer),
        d_model=config.encoder_d_model,
        seed=config.encoder_seed,
        dtype=torch.float32,
    )
    encoder = encoder.to(device).eval()

    values = []
    for i, batch in enumerate(loader):
        if i >= args.num_batches:
            break
        input_ids = torch.from_numpy(batch["input_ids"]).to(device)
        attention_mask = torch.from_numpy(batch["attention_mask"]).to(device)
        latents = encode_text(
            input_ids, attention_mask, encoder,
            latent_mean=0.0, latent_std=1.0, use_bf16=False,
        )
        values.append(latents.float().cpu().numpy())

    if not values:
        raise SystemExit("No batches sampled; is the dataset empty?")

    all_latents = np.concatenate(values, axis=0)
    latent_mean = float(all_latents.mean())
    latent_std = float(all_latents.std())
    print(f"latent_mean: {latent_mean:.6f}")
    print(f"latent_std: {latent_std:.6f}")


if __name__ == "__main__":
    main()
