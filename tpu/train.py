"""Train GPT-2 small (~124M) in JAX/Flax on TPU.

Usage:
    python -m tpu.train
    python -m tpu.train --max_iters 8000 --ckpt_dir /content/drive/MyDrive/gpt-2-nano/checkpoints

Key behaviors:
  * bf16 on TPU (no GradScaler).
  * Gradient accumulation to hit GPT-2's ~524k-token effective batch.
  * Checkpoint every `--ckpt_interval_hours` hours.
  * On startup, the latest checkpoint in `--ckpt_dir` is restored automatically.
  * Works on TPU (any generation), GPU (CUDA), or CPU — JAX picks.
"""
from __future__ import annotations

import argparse
import math
import time
from dataclasses import asdict
from functools import partial

import jax
import jax.numpy as jnp
import numpy as np
import optax
from flax import struct
from flax.training import train_state

from .checkpoint import TimeCheckpointer, abstract_state
from .config import ModelConfig, TrainConfig
from .data import get_datasets
from .model import GPT, count_params


# ───────────── argparse ────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--max_iters", type=int, default=None)
    p.add_argument("--batch_size", type=int, default=None)
    p.add_argument("--grad_accum_steps", type=int, default=None)
    p.add_argument("--learning_rate", type=float, default=None)
    p.add_argument("--ckpt_dir", type=str, default=None)
    p.add_argument("--ckpt_interval_hours", type=float, default=None)
    p.add_argument("--data_dir", type=str, default=None)
    return p.parse_args()


def apply_overrides(tcfg: TrainConfig, args: argparse.Namespace) -> TrainConfig:
    for k in ("max_iters", "batch_size", "grad_accum_steps", "learning_rate",
              "ckpt_dir", "ckpt_interval_hours", "data_dir"):
        v = getattr(args, k)
        if v is not None:
            setattr(tcfg, k, v)
    return tcfg


# ───────────── train state ─────────────────────────────────────────────────────

class TrainState(train_state.TrainState):
    """Standard Flax TrainState — included here so the checkpoint payload is unambiguous."""
    pass


def make_lr_schedule(tcfg: TrainConfig):
    return optax.warmup_cosine_decay_schedule(
        init_value=0.0,
        peak_value=tcfg.learning_rate,
        warmup_steps=tcfg.warmup_iters,
        decay_steps=tcfg.lr_decay_iters,
        end_value=tcfg.min_lr,
    )


def weight_decay_mask(params):
    """Decay only ≥2D matmul weights. Skip embeddings, norms, biases."""
    def mask_leaf(path, p):
        name = "/".join(getattr(k, "key", str(k)) for k in path)
        if p.ndim < 2:
            return False
        if "wte" in name or "wpe" in name or "LayerNorm" in name or "ln_" in name:
            return False
        return True
    return jax.tree_util.tree_map_with_path(mask_leaf, params)


def make_optimizer(tcfg: TrainConfig, params) -> optax.GradientTransformation:
    schedule = make_lr_schedule(tcfg)
    return optax.chain(
        optax.clip_by_global_norm(tcfg.grad_clip),
        optax.adamw(
            learning_rate=schedule,
            b1=tcfg.beta1,
            b2=tcfg.beta2,
            weight_decay=tcfg.weight_decay,
            mask=weight_decay_mask(params),
        ),
    )


# ───────────── jitted train / eval ─────────────────────────────────────────────

def loss_fn(params, apply_fn, batch_x, batch_y, dropout_rng, deterministic):
    logits = apply_fn(
        {"params": params}, batch_x,
        deterministic=deterministic,
        rngs={"dropout": dropout_rng} if not deterministic else None,
    )
    return optax.softmax_cross_entropy_with_integer_labels(logits, batch_y).mean()


@partial(jax.jit, static_argnames=("grad_accum_steps",))
def train_step(state: TrainState, batch_x, batch_y, dropout_rng, grad_accum_steps: int):
    """Gradient-accumulated train step. Splits the (micro * accum) batch along axis 0."""
    micro_x = batch_x.reshape(grad_accum_steps, -1, batch_x.shape[-1])
    micro_y = batch_y.reshape(grad_accum_steps, -1, batch_y.shape[-1])

    def micro_step(carry, micro):
        x, y = micro
        sub_rng, _ = jax.random.split(carry["rng"])
        loss, grads = jax.value_and_grad(loss_fn)(
            state.params, state.apply_fn, x, y, sub_rng, False
        )
        carry["loss"] = carry["loss"] + loss
        carry["grads"] = jax.tree_util.tree_map(lambda a, b: a + b, carry["grads"], grads)
        carry["rng"] = sub_rng
        return carry, None

    init_grads = jax.tree_util.tree_map(jnp.zeros_like, state.params)
    init = {"loss": jnp.zeros((), dtype=jnp.float32), "grads": init_grads, "rng": dropout_rng}
    out, _ = jax.lax.scan(micro_step, init, (micro_x, micro_y))

    avg_loss = out["loss"] / grad_accum_steps
    avg_grads = jax.tree_util.tree_map(lambda g: g / grad_accum_steps, out["grads"])
    new_state = state.apply_gradients(grads=avg_grads)
    return new_state, avg_loss


@jax.jit
def eval_step(state: TrainState, batch_x, batch_y):
    return loss_fn(state.params, state.apply_fn, batch_x, batch_y, jax.random.PRNGKey(0), True)


# ───────────── batch loader ────────────────────────────────────────────────────

def get_batch(ds, batch_size: int, grad_accum_steps: int, rng: np.random.Generator):
    x, y = ds.get_batch(batch_size * grad_accum_steps, rng)
    return jnp.asarray(x), jnp.asarray(y)


def estimate_loss(state, train_ds, val_ds, tcfg, np_rng):
    out = {}
    for split, ds in [("train", train_ds), ("val", val_ds)]:
        losses = []
        for _ in range(tcfg.eval_iters):
            x, y = ds.get_batch(tcfg.batch_size, np_rng)
            losses.append(float(eval_step(state, jnp.asarray(x), jnp.asarray(y))))
        out[split] = float(np.mean(losses))
    return out


# ───────────── main ────────────────────────────────────────────────────────────

def main() -> None:
    args = parse_args()
    mcfg = ModelConfig()
    tcfg = apply_overrides(TrainConfig(), args)

    print(f"[jax] devices: {jax.devices()}")
    print(f"[jax] default backend: {jax.default_backend()}")

    # Init model.
    rng = jax.random.PRNGKey(tcfg.seed)
    rng, init_rng, dropout_rng = jax.random.split(rng, 3)

    dtype = {"bfloat16": jnp.bfloat16, "float16": jnp.float16, "float32": jnp.float32}[tcfg.dtype]
    model = GPT(mcfg, dtype=dtype)

    # Use a tiny dummy batch for init so we don't waste TPU memory at startup.
    dummy = jnp.zeros((1, mcfg.block_size), dtype=jnp.int32)
    params = model.init(init_rng, dummy, deterministic=True)["params"]
    print(f"[model] parameters: {count_params(params) / 1e6:.2f}M")

    optimizer = make_optimizer(tcfg, params)
    state = TrainState.create(apply_fn=model.apply, params=params, tx=optimizer)

    # Resume from latest checkpoint, if any.
    ckpt = TimeCheckpointer(tcfg.ckpt_dir, tcfg.ckpt_interval_hours, tcfg.keep_n_checkpoints)
    abstract = {"state": abstract_state(state), "step": jnp.zeros((), jnp.int32)}
    restored = ckpt.restore_latest(abstract)
    start_iter = 0
    if restored is not None:
        payload, latest = restored
        state = payload["state"]
        start_iter = int(payload["step"])
        print(f"[resume] continuing from iter {start_iter}")
    else:
        print("[resume] no checkpoint found, training from scratch")

    # Datasets — prep happens here on first run; subsequent runs hit the cached .bin files.
    train_ds, val_ds = get_datasets(tcfg, mcfg)
    np_rng = np.random.default_rng(tcfg.seed + start_iter)

    print(f"[train] start_iter={start_iter}  max_iters={tcfg.max_iters}  "
          f"micro_batch={tcfg.batch_size}  grad_accum={tcfg.grad_accum_steps}  "
          f"effective_batch_tokens={tcfg.batch_size * tcfg.grad_accum_steps * mcfg.block_size:,}")

    t_run_start = time.time()
    for it in range(start_iter, tcfg.max_iters + 1):
        # Eval.
        if it % tcfg.eval_interval == 0:
            losses = estimate_loss(state, train_ds, val_ds, tcfg, np_rng)
            elapsed_h = (time.time() - t_run_start) / 3600
            print(f"[eval] iter {it:6d} | train {losses['train']:.4f} | "
                  f"val {losses['val']:.4f} | elapsed {elapsed_h:.2f}h")

        # Time-based checkpoint (also saves at iter 0 only if forced explicitly).
        ckpt.maybe_save(it, {"state": state, "step": jnp.asarray(it, jnp.int32)})

        if it == tcfg.max_iters:
            break

        # Train step.
        x, y = get_batch(train_ds, tcfg.batch_size, tcfg.grad_accum_steps, np_rng)
        rng, dropout_rng = jax.random.split(rng)
        state, loss = train_step(state, x, y, dropout_rng, tcfg.grad_accum_steps)

        if it % tcfg.log_interval == 0:
            print(f"iter {it:6d} | loss {float(loss):.4f}")

    # Final, forced save.
    ckpt.save(tcfg.max_iters, {"state": state, "step": jnp.asarray(tcfg.max_iters, jnp.int32)})
    ckpt.wait()
    total_h = (time.time() - t_run_start) / 3600
    print(f"[done] iters {start_iter}→{tcfg.max_iters} in {total_h:.2f}h")


if __name__ == "__main__":
    main()
