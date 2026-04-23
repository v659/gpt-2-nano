"""Generate text from a JAX/Flax GPT-2 nano checkpoint.

Usage:
    python -m tpu.sample --ckpt_dir checkpoints --prompt "Once upon a time"
"""
from __future__ import annotations

import argparse

import jax
import jax.numpy as jnp
from flax.training import train_state
import optax

from .checkpoint import TimeCheckpointer, abstract_state
from .config import ModelConfig, TrainConfig
from .model import GPT


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt_dir", type=str, default="checkpoints")
    p.add_argument("--step", type=int, default=None, help="specific step; default = latest")
    p.add_argument("--prompt", type=str, default="Once upon a time")
    p.add_argument("--max_new_tokens", type=int, default=300)
    p.add_argument("--temperature", type=float, default=0.8)
    p.add_argument("--top_k", type=int, default=200)
    p.add_argument("--num_samples", type=int, default=1)
    p.add_argument("--seed", type=int, default=1337)
    return p.parse_args()


def top_k_sample(logits: jnp.ndarray, top_k: int, temperature: float, rng) -> jnp.ndarray:
    logits = logits / temperature
    if top_k is not None and top_k > 0:
        kth = jnp.sort(logits, axis=-1)[..., -top_k:][..., 0:1]
        logits = jnp.where(logits < kth, -jnp.inf, logits)
    return jax.random.categorical(rng, logits, axis=-1)


def main() -> None:
    args = parse_args()

    mcfg = ModelConfig()
    tcfg = TrainConfig()
    model = GPT(mcfg, dtype=jnp.bfloat16)

    # Build a placeholder state to define the restore tree shape.
    rng = jax.random.PRNGKey(args.seed)
    dummy = jnp.zeros((1, mcfg.block_size), dtype=jnp.int32)
    params = model.init(rng, dummy, deterministic=True)["params"]
    state = train_state.TrainState.create(
        apply_fn=model.apply, params=params, tx=optax.adamw(1e-4),
    )

    ckpt = TimeCheckpointer(args.ckpt_dir, interval_hours=99999, keep_n=tcfg.keep_n_checkpoints)
    abstract = {"state": abstract_state(state), "step": jnp.zeros((), jnp.int32)}
    if args.step is not None:
        payload = ckpt.restore(args.step, abstract)
    else:
        restored = ckpt.restore_latest(abstract)
        if restored is None:
            raise FileNotFoundError(f"no checkpoint in {args.ckpt_dir}")
        payload, _ = restored
    state = payload["state"]

    import tiktoken
    enc = tiktoken.get_encoding("gpt2")
    start_ids = enc.encode_ordinary(args.prompt) or [enc.eot_token]
    idx = jnp.asarray([start_ids], dtype=jnp.int32)

    @jax.jit
    def step(idx_in, rng_in):
        # Truncate to block_size — model has no KV cache, regenerates each step.
        cond = idx_in[:, -mcfg.block_size:]
        logits = state.apply_fn({"params": state.params}, cond, deterministic=True)
        next_logits = logits[:, -1, :]
        next_token = top_k_sample(next_logits, args.top_k, args.temperature, rng_in)
        return jnp.concatenate([idx_in, next_token[:, None]], axis=1)

    for s in range(args.num_samples):
        rng, _ = jax.random.split(rng)
        cur = idx
        for _ in range(args.max_new_tokens):
            rng, sub = jax.random.split(rng)
            cur = step(cur, sub)
        out = enc.decode(cur[0].tolist())
        print(f"--- sample {s + 1} ---\n{out}\n")


if __name__ == "__main__":
    main()
