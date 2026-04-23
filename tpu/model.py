"""GPT-2 small (124M) in Flax.

Same architecture as the PyTorch version at the project root: pre-LN blocks,
GELU MLP, weight-tied input/output embeddings, scaled init on residual
projections. Uses JAX's fused causal SDPA (`jax.nn.dot_product_attention`)
which dispatches to a Pallas Flash Attention kernel on TPU.
"""
from __future__ import annotations

from typing import Any

import jax
import jax.numpy as jnp
import flax.linen as nn

from .config import ModelConfig


def _kernel_init(scale: float = 0.02):
    return nn.initializers.normal(stddev=scale)


class CausalSelfAttention(nn.Module):
    cfg: ModelConfig
    dtype: Any = jnp.bfloat16

    @nn.compact
    def __call__(self, x: jnp.ndarray, deterministic: bool) -> jnp.ndarray:
        cfg = self.cfg
        B, T, C = x.shape
        head_dim = C // cfg.n_head

        qkv = nn.Dense(
            3 * cfg.n_embd, use_bias=False, dtype=self.dtype,
            kernel_init=_kernel_init(), name="c_attn",
        )(x)
        q, k, v = jnp.split(qkv, 3, axis=-1)
        # (B, T, n_head, head_dim) — the layout jax.nn.dot_product_attention expects.
        q = q.reshape(B, T, cfg.n_head, head_dim)
        k = k.reshape(B, T, cfg.n_head, head_dim)
        v = v.reshape(B, T, cfg.n_head, head_dim)

        y = jax.nn.dot_product_attention(q, k, v, is_causal=True)
        y = y.reshape(B, T, C)

        # Scaled init on the residual projection (per GPT-2 paper).
        y = nn.Dense(
            cfg.n_embd, use_bias=False, dtype=self.dtype,
            kernel_init=_kernel_init(0.02 / (2 * cfg.n_layer) ** 0.5),
            name="c_proj",
        )(y)
        if cfg.dropout > 0:
            y = nn.Dropout(cfg.dropout)(y, deterministic=deterministic)
        return y


class MLP(nn.Module):
    cfg: ModelConfig
    dtype: Any = jnp.bfloat16

    @nn.compact
    def __call__(self, x: jnp.ndarray, deterministic: bool) -> jnp.ndarray:
        cfg = self.cfg
        x = nn.Dense(4 * cfg.n_embd, use_bias=False, dtype=self.dtype,
                     kernel_init=_kernel_init(), name="c_fc")(x)
        x = nn.gelu(x, approximate=True)
        x = nn.Dense(cfg.n_embd, use_bias=False, dtype=self.dtype,
                     kernel_init=_kernel_init(0.02 / (2 * cfg.n_layer) ** 0.5),
                     name="c_proj")(x)
        if cfg.dropout > 0:
            x = nn.Dropout(cfg.dropout)(x, deterministic=deterministic)
        return x


class Block(nn.Module):
    cfg: ModelConfig
    dtype: Any = jnp.bfloat16

    @nn.compact
    def __call__(self, x: jnp.ndarray, deterministic: bool) -> jnp.ndarray:
        x = x + CausalSelfAttention(self.cfg, dtype=self.dtype, name="attn")(
            nn.LayerNorm(dtype=self.dtype, use_bias=False, name="ln_1")(x), deterministic
        )
        x = x + MLP(self.cfg, dtype=self.dtype, name="mlp")(
            nn.LayerNorm(dtype=self.dtype, use_bias=False, name="ln_2")(x), deterministic
        )
        return x


class GPT(nn.Module):
    cfg: ModelConfig
    dtype: Any = jnp.bfloat16

    @nn.compact
    def __call__(self, idx: jnp.ndarray, deterministic: bool = True) -> jnp.ndarray:
        cfg = self.cfg
        B, T = idx.shape
        assert T <= cfg.block_size, f"sequence length {T} > block_size {cfg.block_size}"

        # Embeddings live in fp32 (params), cast activations to compute dtype.
        wte = self.param("wte", _kernel_init(), (cfg.vocab_size, cfg.n_embd))
        wpe = self.param("wpe", _kernel_init(), (cfg.block_size, cfg.n_embd))

        x = wte[idx] + wpe[jnp.arange(T)]
        x = x.astype(self.dtype)

        for i in range(cfg.n_layer):
            x = Block(cfg, dtype=self.dtype, name=f"block_{i}")(x, deterministic)

        x = nn.LayerNorm(dtype=self.dtype, use_bias=False, name="ln_f")(x)

        # Weight-tied logits. Compute in fp32 for numerical safety on the loss.
        logits = jnp.einsum("btc,vc->btv", x, wte.astype(self.dtype))
        return logits.astype(jnp.float32)


def count_params(params) -> int:
    return sum(int(jnp.size(x)) for x in jax.tree_util.tree_leaves(params))
