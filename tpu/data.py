"""FineWeb-Edu streaming + tokenization → memmapped uint16 .bin files.

Streams `HuggingFaceFW/fineweb-edu` (sample-10BT config), encodes with
GPT-2 BPE in parallel, and writes `train.bin` / `val.bin`. Idempotent:
if the targets exist with at least the requested number of tokens, the
download is skipped entirely.

This is one-time prep — runs in ~15–25 min on a Colab CPU runtime, then
the .bin files persist (e.g. on Google Drive) and never need to be
regenerated.
"""
from __future__ import annotations

import os
from pathlib import Path
from typing import Iterator

import numpy as np


def _bin_token_count(path: Path) -> int:
    if not path.exists():
        return 0
    return path.stat().st_size // 2  # uint16 = 2 bytes


def _stream_documents(dataset_name: str, dataset_config: str) -> Iterator[str]:
    """Yield the `text` field of every document in the streamed dataset."""
    from datasets import load_dataset
    ds = load_dataset(dataset_name, name=dataset_config, split="train", streaming=True)
    for ex in ds:
        text = ex.get("text")
        if text:
            yield text


def prepare_fineweb_edu(
    data_dir: str,
    dataset_name: str,
    dataset_config: str,
    target_train_tokens: int,
    target_val_tokens: int,
    num_proc: int | None = None,
) -> tuple[Path, Path]:
    """Download + tokenize until we have enough tokens. Idempotent."""
    out = Path(data_dir)
    out.mkdir(parents=True, exist_ok=True)
    train_path = out / "train.bin"
    val_path = out / "val.bin"

    have_train = _bin_token_count(train_path)
    have_val = _bin_token_count(val_path)
    if have_train >= target_train_tokens and have_val >= target_val_tokens:
        print(f"[data] cached: train={have_train:,} tokens, val={have_val:,} tokens")
        return train_path, val_path

    import tiktoken
    enc = tiktoken.get_encoding("gpt2")
    eot = enc.eot_token

    if num_proc is None:
        num_proc = max(1, (os.cpu_count() or 2) - 1)

    # Encode each document and append <|endoftext|> as a document separator.
    def encode(text: str) -> np.ndarray:
        ids = enc.encode_ordinary(text)
        ids.append(eot)
        return np.asarray(ids, dtype=np.uint16)

    # Write val first (small), then train. Open in append mode so this is
    # restartable: if the prep dies mid-write, you keep what you had.
    val_needed = max(0, target_val_tokens - have_val)
    train_needed = max(0, target_train_tokens - have_train)

    docs = _stream_documents(dataset_name, dataset_config)

    if val_needed > 0:
        print(f"[data] writing {val_needed:,} val tokens → {val_path}")
        _write_until(val_path, val_needed, docs, encode, num_proc)

    if train_needed > 0:
        print(f"[data] writing {train_needed:,} train tokens → {train_path}")
        _write_until(train_path, train_needed, docs, encode, num_proc)

    print(f"[data] done: train={_bin_token_count(train_path):,}, val={_bin_token_count(val_path):,}")
    return train_path, val_path


def _write_until(path: Path, needed: int, docs: Iterator[str], encode, num_proc: int) -> None:
    """Tokenize documents in a process pool until `needed` tokens are written."""
    from multiprocessing import Pool

    written = 0
    chunk_size = 1024  # docs per pool batch
    f = open(path, "ab")
    try:
        with Pool(num_proc) as pool:
            buf: list[str] = []
            for doc in docs:
                buf.append(doc)
                if len(buf) >= chunk_size:
                    written += _flush(pool, buf, encode, f)
                    buf.clear()
                    if written % (10 * chunk_size) < chunk_size:
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


# ───────────── batch sampler (memory-mapped, identical contract to the PyTorch version) ──────────

class BinDataset:
    def __init__(self, path: str | os.PathLike, block_size: int):
        self.path = Path(path)
        self.block_size = block_size
        self.data = np.memmap(self.path, dtype=np.uint16, mode="r")
        if len(self.data) <= block_size + 1:
            raise ValueError(f"{self.path} has only {len(self.data)} tokens")

    def get_batch(self, batch_size: int, rng: np.random.Generator) -> tuple[np.ndarray, np.ndarray]:
        ix = rng.integers(0, len(self.data) - self.block_size - 1, size=batch_size)
        x = np.stack([self.data[i : i + self.block_size].astype(np.int32) for i in ix])
        y = np.stack([self.data[i + 1 : i + 1 + self.block_size].astype(np.int32) for i in ix])
        return x, y


def get_datasets(cfg, model_cfg) -> tuple[BinDataset, BinDataset]:
    train_path, val_path = prepare_fineweb_edu(
        cfg.data_dir,
        cfg.dataset_name,
        cfg.dataset_config,
        cfg.target_train_tokens,
        cfg.target_val_tokens,
    )
    return (
        BinDataset(train_path, model_cfg.block_size),
        BinDataset(val_path, model_cfg.block_size),
    )
