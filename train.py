"""Train GPT-2 small (~124M) on Kaggle T4 x2 with PyTorch DDP.

Single-GPU usage:
    python train.py
    python train.py --preset nano --dataset tinyshakespeare    # legacy nano path

DDP usage (Kaggle T4 x2 / any multi-GPU box):
    torchrun --standalone --nproc_per_node=2 train.py

Behavior:
  * Auto-detects DDP via torchrun env vars (RANK / LOCAL_RANK / WORLD_SIZE).
  * Time-based checkpointing — saves every --ckpt_interval_hours hours.
  * On startup, restores the latest checkpoint in --ckpt_dir and continues.
  * fp16 + GradScaler on T4 (no bf16 support).
"""
from __future__ import annotations

import argparse
import math
import os
import time
from contextlib import nullcontext
from pathlib import Path

import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP

from config import MODEL_PRESETS, ModelConfig, TrainConfig
from data import get_datasets
from model import GPT


# ───────────── argparse ────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--preset", choices=list(MODEL_PRESETS), default="small")
    p.add_argument("--dataset", choices=["fineweb_edu", "tinyshakespeare"], default=None)
    p.add_argument("--data_dir", type=str, default=None)
    p.add_argument("--ckpt_dir", type=str, default=None)
    p.add_argument("--ckpt_interval_hours", type=float, default=None)
    p.add_argument("--max_iters", type=int, default=None)
    p.add_argument("--batch_size", type=int, default=None)
    p.add_argument("--gradient_accumulation_steps", type=int, default=None)
    p.add_argument("--learning_rate", type=float, default=None)
    p.add_argument("--device", type=str, default=None)
    p.add_argument("--compile", action="store_true")
    return p.parse_args()


def apply_overrides(tcfg: TrainConfig, args: argparse.Namespace) -> TrainConfig:
    for k in (
        "dataset", "data_dir", "ckpt_dir", "ckpt_interval_hours",
        "max_iters", "batch_size", "gradient_accumulation_steps",
        "learning_rate", "device",
    ):
        v = getattr(args, k)
        if v is not None:
            setattr(tcfg, k, v)
    if args.compile:
        tcfg.compile = True
    return tcfg


# ───────────── DDP setup ───────────────────────────────────────────────────────

class DDPInfo:
    def __init__(self):
        self.enabled = int(os.environ.get("RANK", -1)) != -1
        if self.enabled:
            self.rank = int(os.environ["RANK"])
            self.local_rank = int(os.environ["LOCAL_RANK"])
            self.world_size = int(os.environ["WORLD_SIZE"])
        else:
            self.rank = 0
            self.local_rank = 0
            self.world_size = 1
        self.is_master = self.rank == 0

    def init(self, device_type: str) -> None:
        if not self.enabled:
            return
        backend = "nccl" if device_type == "cuda" else "gloo"
        dist.init_process_group(backend=backend)
        if device_type == "cuda":
            torch.cuda.set_device(self.local_rank)

    def destroy(self) -> None:
        if self.enabled and dist.is_initialized():
            dist.destroy_process_group()


def log(ddp: DDPInfo, msg: str) -> None:
    if ddp.is_master:
        print(msg)


# ───────────── checkpointing ───────────────────────────────────────────────────

class TimeCheckpointer:
    """Wall-clock-interval checkpoint saver with rotation. Master-only writes."""

    def __init__(self, ckpt_dir: str | Path, interval_hours: float, keep_n: int, is_master: bool):
        self.dir = Path(ckpt_dir)
        if is_master:
            self.dir.mkdir(parents=True, exist_ok=True)
        self.interval_seconds = interval_hours * 3600.0
        self.keep_n = keep_n
        self.is_master = is_master
        self.last_save_time = time.time()

    def latest(self) -> tuple[Path, int] | None:
        ckpts = sorted(self.dir.glob("ckpt_*.pt"))
        if not ckpts:
            return None
        path = ckpts[-1]
        try:
            step = int(path.stem.split("_")[1])
        except (IndexError, ValueError):
            return None
        return path, step

    def maybe_save(self, step: int, payload_fn, force: bool = False) -> bool:
        elapsed = time.time() - self.last_save_time
        if not force and elapsed < self.interval_seconds:
            return False
        if not self.is_master:
            self.last_save_time = time.time()
            return True
        self._save(step, payload_fn())
        return True

    def _save(self, step: int, payload: dict) -> None:
        path = self.dir / f"ckpt_{step:08d}.pt"
        tmp = path.with_suffix(".pt.tmp")
        torch.save(payload, tmp)
        tmp.rename(path)             # atomic on POSIX
        self._prune()
        self.last_save_time = time.time()
        print(f"[ckpt] saved → {path}")

    def _prune(self) -> None:
        ckpts = sorted(self.dir.glob("ckpt_*.pt"))
        for p in ckpts[:-self.keep_n]:
            try:
                p.unlink()
            except OSError:
                pass


# ───────────── lr schedule ─────────────────────────────────────────────────────

def get_lr(it: int, tcfg: TrainConfig) -> float:
    if not tcfg.decay_lr:
        return tcfg.learning_rate
    if it < tcfg.warmup_iters:
        return tcfg.learning_rate * (it + 1) / (tcfg.warmup_iters + 1)
    if it > tcfg.lr_decay_iters:
        return tcfg.min_lr
    decay_ratio = (it - tcfg.warmup_iters) / (tcfg.lr_decay_iters - tcfg.warmup_iters)
    coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio))
    return tcfg.min_lr + coeff * (tcfg.learning_rate - tcfg.min_lr)


# ───────────── eval ────────────────────────────────────────────────────────────

@torch.no_grad()
def estimate_loss(model, datasets, tcfg, ctx, device_type, batch_gens):
    model.eval()
    out = {}
    for split, ds in datasets.items():
        losses = torch.zeros(tcfg.eval_iters)
        for k in range(tcfg.eval_iters):
            X, Y = ds.get_batch(tcfg.batch_size, tcfg.device, device_type, batch_gens[split])
            with ctx:
                _, loss = model(X, Y)
            losses[k] = loss.item()
        out[split] = losses.mean().item()
    model.train()
    return out


# ───────────── main ────────────────────────────────────────────────────────────

def main() -> None:
    args = parse_args()
    mcfg: ModelConfig = MODEL_PRESETS[args.preset]
    tcfg = apply_overrides(TrainConfig(), args)

    # If user picked the legacy nano preset and didn't specify a dataset, default to TinyShakespeare.
    if args.preset == "nano" and args.dataset is None:
        tcfg.dataset = "tinyshakespeare"

    ddp = DDPInfo()

    # Device + dtype.
    if tcfg.device.startswith("cuda") and not torch.cuda.is_available():
        log(ddp, "[warn] CUDA not available, falling back to CPU.")
        tcfg.device = "cpu"
    device_type = "cuda" if tcfg.device.startswith("cuda") else tcfg.device
    if ddp.enabled and device_type == "cuda":
        tcfg.device = f"cuda:{ddp.local_rank}"

    ddp.init(device_type)

    torch.manual_seed(tcfg.seed + ddp.rank)         # different seed per rank for batch independence
    torch.cuda.manual_seed_all(tcfg.seed + ddp.rank)
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True

    if device_type == "cpu":
        ptdtype = torch.float32
        ctx = nullcontext()
    else:
        ptdtype = {"float16": torch.float16, "bfloat16": torch.bfloat16, "float32": torch.float32}[tcfg.dtype]
        ctx = torch.amp.autocast(device_type=device_type, dtype=ptdtype)
    scaler = torch.amp.GradScaler(device_type, enabled=(ptdtype == torch.float16))

    # Datasets — only rank 0 actually downloads/tokenizes; other ranks block on a sentinel.
    train_ds, val_ds = get_datasets(
        tcfg.data_dir, tcfg.dataset, mcfg.block_size,
        target_train_tokens=tcfg.target_train_tokens,
        target_val_tokens=tcfg.target_val_tokens,
        rank=ddp.rank,
    )
    if ddp.enabled:
        dist.barrier()              # everyone synchronizes once data is ready
    datasets = {"train": train_ds, "val": val_ds}

    # Per-rank generators so each GPU samples independent batches.
    batch_gens = {
        "train": torch.Generator().manual_seed(tcfg.seed + ddp.rank * 7919),
        "val":   torch.Generator().manual_seed(tcfg.seed + ddp.rank * 7919 + 1),
    }

    # Model.
    model = GPT(mcfg).to(tcfg.device)
    log(ddp, f"[model] preset={args.preset}  parameters: {model.num_parameters() / 1e6:.2f}M (non-embedding)")

    optimizer = model.configure_optimizers(
        weight_decay=tcfg.weight_decay,
        lr=tcfg.learning_rate,
        betas=(tcfg.beta1, tcfg.beta2),
    )

    if tcfg.compile:
        log(ddp, "[model] torch.compile() — first iter will be slow")
        model = torch.compile(model)

    # Wrap in DDP after compile, before checkpoint restore.
    if ddp.enabled:
        model = DDP(model, device_ids=[ddp.local_rank] if device_type == "cuda" else None)

    # Resume from latest checkpoint if present.
    ckpt_mgr = TimeCheckpointer(tcfg.ckpt_dir, tcfg.ckpt_interval_hours, tcfg.keep_n_checkpoints, ddp.is_master)
    start_iter = 0
    best_val_loss = float("inf")
    latest = ckpt_mgr.latest()
    if latest is not None:
        path, step = latest
        log(ddp, f"[resume] loading {path}")
        ckpt = torch.load(path, map_location=tcfg.device, weights_only=False)
        # DDP wraps the model; unwrap for state_dict load.
        raw_model = model.module if isinstance(model, DDP) else model
        if hasattr(raw_model, "_orig_mod"):           # also unwrap torch.compile
            raw_model._orig_mod.load_state_dict(ckpt["model"])
        else:
            raw_model.load_state_dict(ckpt["model"])
        optimizer.load_state_dict(ckpt["optimizer"])
        scaler.load_state_dict(ckpt["scaler"])
        start_iter = ckpt["step"] + 1
        best_val_loss = ckpt.get("best_val_loss", float("inf"))
        log(ddp, f"[resume] continuing from iter {start_iter} (best_val_loss={best_val_loss:.4f})")
    else:
        log(ddp, "[resume] no checkpoint found, training from scratch")

    log(ddp,
        f"[train] iters {start_iter}→{tcfg.max_iters}  "
        f"per-rank micro_batch={tcfg.batch_size}  grad_accum={tcfg.gradient_accumulation_steps}  "
        f"world_size={ddp.world_size}  "
        f"effective_tokens/step={tcfg.batch_size * tcfg.gradient_accumulation_steps * mcfg.block_size * ddp.world_size:,}")

    raw_model = model.module if isinstance(model, DDP) else model

    def build_payload() -> dict:
        return {
            "model": (raw_model._orig_mod if hasattr(raw_model, "_orig_mod") else raw_model).state_dict(),
            "optimizer": optimizer.state_dict(),
            "scaler": scaler.state_dict(),
            "model_config": mcfg.__dict__,
            "step": it,
            "best_val_loss": best_val_loss,
        }

    # Training loop.
    t_run_start = time.time()
    X, Y = train_ds.get_batch(tcfg.batch_size, tcfg.device, device_type, batch_gens["train"])

    for it in range(start_iter, tcfg.max_iters + 1):
        lr = get_lr(it, tcfg)
        for pg in optimizer.param_groups:
            pg["lr"] = lr

        # Eval (master only logs, but all ranks must run forward to keep DDP in sync — easier to skip on non-master).
        if it % tcfg.eval_interval == 0 and ddp.is_master:
            losses = estimate_loss(model, datasets, tcfg, ctx, device_type, batch_gens)
            elapsed_h = (time.time() - t_run_start) / 3600
            log(ddp, f"[eval] iter {it:6d} | train {losses['train']:.4f} | "
                     f"val {losses['val']:.4f} | lr {lr:.2e} | elapsed {elapsed_h:.2f}h")
            if losses["val"] < best_val_loss:
                best_val_loss = losses["val"]
        if ddp.enabled:
            dist.barrier()           # other ranks wait while master evals

        # Time-based checkpoint.
        ckpt_mgr.maybe_save(it, build_payload)

        if it == tcfg.max_iters:
            break

        # Forward / backward, with grad accumulation.
        for micro_step in range(tcfg.gradient_accumulation_steps):
            # Skip DDP all-reduce on all but the last micro-step (perf optimization).
            if ddp.enabled:
                model.require_backward_grad_sync = (micro_step == tcfg.gradient_accumulation_steps - 1)
            with ctx:
                _, loss = model(X, Y)
                loss = loss / tcfg.gradient_accumulation_steps
            X, Y = train_ds.get_batch(tcfg.batch_size, tcfg.device, device_type, batch_gens["train"])
            scaler.scale(loss).backward()

        if tcfg.grad_clip != 0.0:
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), tcfg.grad_clip)
        scaler.step(optimizer)
        scaler.update()
        optimizer.zero_grad(set_to_none=True)

        if it % tcfg.log_interval == 0 and ddp.is_master:
            lossf = loss.item() * tcfg.gradient_accumulation_steps
            log(ddp, f"iter {it:6d} | loss {lossf:.4f} | lr {lr:.2e}")

    # Force a final save at the end of training.
    if ddp.is_master:
        ckpt_mgr._save(it, build_payload())
    total_h = (time.time() - t_run_start) / 3600
    log(ddp, f"[done] iters {start_iter}→{tcfg.max_iters} in {total_h:.2f}h, best_val_loss={best_val_loss:.4f}")

    ddp.destroy()


if __name__ == "__main__":
    main()
