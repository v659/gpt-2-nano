"""Hyperparameters for GPT-2 small (~124M) on T4 x2 with PyTorch DDP.

Defaults target Kaggle's free 2x T4 environment training a GPT-2 small
architecture on a ~1.2B-token slice of FineWeb-Edu. Estimated wall time:
~12–18 hours, fits in 1–2 Kaggle sessions with the time-based checkpointing
in train.py.

For the original 10M nano-on-TinyShakespeare path, override on the CLI:
    python train.py --preset nano --dataset tinyshakespeare
"""
from dataclasses import dataclass


@dataclass
class ModelConfig:
    """GPT-2 small spec by default. Override n_layer / n_head / n_embd for other sizes."""
    block_size: int = 1024
    vocab_size: int = 50304        # GPT-2 BPE, padded to multiple of 128 for matmul throughput
    n_layer: int = 12
    n_head: int = 12
    n_embd: int = 768
    dropout: float = 0.0           # off for pretraining
    bias: bool = False


# Convenience presets — train.py picks one via --preset.
MODEL_PRESETS = {
    "nano":  ModelConfig(block_size=256, n_layer=6,  n_head=6,  n_embd=384, dropout=0.2),
    "small": ModelConfig(block_size=1024, n_layer=12, n_head=12, n_embd=768, dropout=0.0),
    "medium": ModelConfig(block_size=1024, n_layer=24, n_head=16, n_embd=1024, dropout=0.0),
}


@dataclass
class TrainConfig:
    # data
    dataset: str = "fineweb_edu"   # "fineweb_edu" | "tinyshakespeare"
    data_dir: str = "data"
    target_train_tokens: int = 1_200_000_000
    target_val_tokens: int = 5_000_000

    # checkpointing — time-based, not step-based, so it survives Kaggle's 12 h session limit
    ckpt_dir: str = "checkpoints"
    ckpt_interval_hours: float = 3.0
    keep_n_checkpoints: int = 3

    # i/o + eval
    out_dir: str = "out"
    eval_interval: int = 250
    log_interval: int = 10
    eval_iters: int = 50

    # optimization (per-rank values; effective batch = batch_size * grad_accum * world_size)
    batch_size: int = 8                         # per-GPU sequences (T4 16 GB fits 8 @ block=1024)
    gradient_accumulation_steps: int = 32
    max_iters: int = 6000                       # ~3.1B tokens trained ≈ 2.6 epochs of 1.2B
    learning_rate: float = 6e-4
    weight_decay: float = 0.1
    beta1: float = 0.9
    beta2: float = 0.95
    grad_clip: float = 1.0

    # lr schedule
    decay_lr: bool = True
    warmup_iters: int = 200
    lr_decay_iters: int = 6000
    min_lr: float = 6e-5

    # system
    device: str = "cuda"
    dtype: str = "float16"          # T4 has no bf16 — fp16 + GradScaler
    compile: bool = False           # flaky on T4 / Kaggle, leave off
    seed: int = 1337
