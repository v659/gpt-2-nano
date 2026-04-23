"""Hyperparameters for the JAX/Flax GPT-2 small reproduction on TPU.

Defaults target ~124M params (GPT-2 small architecture) trained on a
~1.2B-token slice of FineWeb-Edu — the smallest workload that produces
"coherent ordinary English."
"""
from dataclasses import dataclass


@dataclass
class ModelConfig:
    block_size: int = 1024
    vocab_size: int = 50304        # GPT-2 BPE, padded to multiple of 128 for TPU tile alignment
    n_layer: int = 12
    n_head: int = 12
    n_embd: int = 768
    dropout: float = 0.0           # off for pretraining (Karpathy/GPT-2 convention)


@dataclass
class TrainConfig:
    # data
    data_dir: str = "data/fineweb_edu"
    dataset_name: str = "HuggingFaceFW/fineweb-edu"
    dataset_config: str = "sample-10BT"
    target_train_tokens: int = 1_200_000_000     # ~5 GB of GPT-2 BPE
    target_val_tokens: int = 5_000_000

    # checkpointing — saves on a wall-clock interval, not a step interval,
    # so a long run survives Colab/Kaggle disconnects without losing > N hours.
    ckpt_dir: str = "checkpoints"
    ckpt_interval_hours: float = 3.0
    keep_n_checkpoints: int = 3

    # training
    batch_size: int = 16                    # per-device sequences
    grad_accum_steps: int = 32              # effective batch = 16 * 32 * 1024 = 524k tokens
    max_iters: int = 6000                   # ~3.1B tokens trained ≈ 2.6 epochs of 1.2B
    eval_interval: int = 250
    eval_iters: int = 50
    log_interval: int = 10

    # optimization (matches GPT-2 small recipe)
    learning_rate: float = 6e-4
    weight_decay: float = 0.1
    beta1: float = 0.9
    beta2: float = 0.95
    grad_clip: float = 1.0
    warmup_iters: int = 200
    lr_decay_iters: int = 6000
    min_lr: float = 6e-5

    # system
    seed: int = 1337
    dtype: str = "bfloat16"                 # TPU native; no GradScaler needed
