"""Dataset preparation and on-the-fly batch sampling.

Two datasets:
  * tinyshakespeare — ~1 MB, 20 min total training, used by the nano preset
  * fineweb_edu     — ~1.2B tokens streamed from HuggingFaceFW/fineweb-edu,
                      used by the GPT-2 small (124M) DDP path

Both produce uint16 .bin files on disk that the trainer mmap-samples from.
"""
from __future__ import annotations

import os
from pathlib import Path
from typing import Iterator

import numpy as np
import requests
import torch

TINY_SHAKESPEARE_URL = (
    "https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt"
)


# ───────────── TinyShakespeare (legacy nano path) ──────────────────────────────

def prepare_tinyshakespeare(data_dir: str = "data") -> tuple[Path, Path]:
    """Download, tokenize, and serialize TinyShakespeare. Idempotent."""
    out_dir = Path(data_dir) / "tinyshakespeare"
    out_dir.mkdir(parents=True, exist_ok=True)
    train_path = out_dir / "train.bin"
    val_path = out_dir / "val.bin"

    if train_path.exists() and val_path.exists():
        print(f"[data] cached: {train_path}, {val_path}")
        return train_path, val_path

    raw_path = out_dir / "input.txt"
    if not raw_path.exists():
        print(f"[data] downloading TinyShakespeare from {TINY_SHAKESPEARE_URL}")
        resp = requests.get(TINY_SHAKESPEARE_URL, timeout=30)
        resp.raise_for_status()
        raw_path.write_text(resp.text, encoding="utf-8")

    text = raw_path.read_text(encoding="utf-8")
    n = len(text)
    train_text = text[: int(n * 0.9)]
    val_text = text[int(n * 0.9):]

    import tiktoken
    enc = tiktoken.get_encoding("gpt2")
    train_ids = np.array(enc.encode_ordinary(train_text), dtype=np.uint16)
    val_ids = np.array(enc.encode_ordinary(val_text), dtype=np.uint16)
    print(f"[data] train: {len(train_ids):,} tokens   val: {len(val_ids):,} tokens")

    train_ids.tofile(train_path)
    val_ids.tofile(val_path)
    return train_path, val_path


# ───────────── FineWeb-Edu (GPT-2 small path) ──────────────────────────────────

FINEWEB_EDU_NAME = "HuggingFaceFW/fineweb-edu"
FINEWEB_EDU_CONFIG = "sample-10BT"


def _bin_token_count(path: Path) -> int:
    return path.stat().st_size // 2 if path.exists() else 0


def _stream_fineweb(name: str, config: str) -> Iterator[str]:
    from datasets import load_dataset
    ds = load_dataset(name, name=config, split="train", streaming=True)
    for ex in ds:
        text = ex.get("text")
        if text:
            yield text


def prepare_fineweb_edu(
    data_dir: str = "data",
    target_train_tokens: int = 1_200_000_000,
    target_val_tokens: int = 5_000_000,
    num_proc: int | None = None,
    rank: int = 0,
) -> tuple[Path, Path]:
    """Stream + tokenize FineWeb-Edu until enough tokens are written. Idempotent.

    Only rank 0 actually writes; other ranks block on a sentinel file. This keeps
    DDP launches from racing on the same .bin file.
    """
    out = Path(data_dir) / "fineweb_edu"
    out.mkdir(parents=True, exist_ok=True)
    train_path = out / "train.bin"
    val_path = out / "val.bin"
    done_marker = out / ".prep_done"

    have_train = _bin_token_count(train_path)
    have_val = _bin_token_count(val_path)
    if have_train >= target_train_tokens and have_val >= target_val_tokens:
        if not done_marker.exists() and rank == 0:
            done_marker.touch()
        print(f"[data] cached: train={have_train:,} val={have_val:,}")
        return train_path, val_path

    if rank != 0:
        # Other ranks wait — only rank 0 should hit HuggingFace and the disk.
        import time as _time
        print(f"[data] rank {rank} waiting for rank 0 to finish prep...")
        while not done_marker.exists():
            _time.sleep(15)
        return train_path, val_path

    if num_proc is None:
        num_proc = max(1, (os.cpu_count() or 2) - 1)

    import tiktoken
    enc = tiktoken.get_encoding("gpt2")
    eot = enc.eot_token

    def encode(text: str) -> np.ndarray:
        ids = enc.encode_ordinary(text)
        ids.append(eot)
        return np.asarray(ids, dtype=np.uint16)

    docs = _stream_fineweb(FINEWEB_EDU_NAME, FINEWEB_EDU_CONFIG)

    val_needed = max(0, target_val_tokens - have_val)
    train_needed = max(0, target_train_tokens - have_train)

    if val_needed > 0:
        print(f"[data] writing {val_needed:,} val tokens → {val_path}")
        _write_until(val_path, val_needed, docs, encode, num_proc)

    if train_needed > 0:
        print(f"[data] writing {train_needed:,} train tokens → {train_path}")
        _write_until(train_path, train_needed, docs, encode, num_proc)

    done_marker.touch()
    print(f"[data] done: train={_bin_token_count(train_path):,} val={_bin_token_count(val_path):,}")
    return train_path, val_path


def _write_until(path: Path, needed: int, docs: Iterator[str], encode, num_proc: int) -> None:
    from multiprocessing import Pool
    written = 0
    chunk = 1024
    f = open(path, "ab")
    try:
        with Pool(num_proc) as pool:
            buf: list[str] = []
            for doc in docs:
                buf.append(doc)
                if len(buf) >= chunk:
                    written += _flush(pool, buf, encode, f)
                    buf.clear()
                    if (written // chunk) % 10 == 0:
                        print(f"  [data] +{written:,} tokens")
                    if written >= needed:
                        break
            if buf and written < needed:
                written += _flush(pool, buf, encode, f)
    finally:
        f.close()


def _flush(pool, buf: list[str], encode, f) -> int:
    arrs = pool.map(encode, buf)
    n = 0
    for a in arrs:
        f.write(a.tobytes())
        n += a.size
    return n


# ───────────── batch sampler ───────────────────────────────────────────────────

class BinDataset:
    """Memory-mapped uint16 token store with random fixed-length sampling."""

    def __init__(self, path: str | os.PathLike, block_size: int):
        self.path = Path(path)
        self.block_size = block_size
        self.data = np.memmap(self.path, dtype=np.uint16, mode="r")
        if len(self.data) <= block_size + 1:
            raise ValueError(
                f"Dataset at {self.path} has only {len(self.data)} tokens, "
                f"need > {block_size + 1}."
            )

    def get_batch(
        self,
        batch_size: int,
        device: str,
        device_type: str,
        generator: torch.Generator | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        # Per-rank generator so DDP ranks see different batches.
        ix = torch.randint(
            len(self.data) - self.block_size - 1,
            (batch_size,),
            generator=generator,
        )
        x = torch.stack([
            torch.from_numpy((self.data[i : i + self.block_size]).astype(np.int64))
            for i in ix
        ])
        y = torch.stack([
            torch.from_numpy((self.data[i + 1 : i + 1 + self.block_size]).astype(np.int64))
            for i in ix
        ])
        if device_type == "cuda":
            x = x.pin_memory().to(device, non_blocking=True)
            y = y.pin_memory().to(device, non_blocking=True)
        else:
            x, y = x.to(device), y.to(device)
        return x, y


def get_datasets(
    data_dir: str,
    dataset: str,
    block_size: int,
    target_train_tokens: int = 1_200_000_000,
    target_val_tokens: int = 5_000_000,
    rank: int = 0,
) -> tuple[BinDataset, BinDataset]:
    if dataset == "tinyshakespeare":
        train_path, val_path = prepare_tinyshakespeare(data_dir)
    elif dataset == "fineweb_edu":
        train_path, val_path = prepare_fineweb_edu(
            data_dir, target_train_tokens, target_val_tokens, rank=rank,
        )
    else:
        raise ValueError(f"unknown dataset {dataset!r}")
    return BinDataset(train_path, block_size), BinDataset(val_path, block_size)


if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("--dataset", choices=["tinyshakespeare", "fineweb_edu"], default="fineweb_edu")
    p.add_argument("--data_dir", default="data")
    args = p.parse_args()
    if args.dataset == "tinyshakespeare":
        prepare_tinyshakespeare(args.data_dir)
    else:
        prepare_fineweb_edu(args.data_dir)
