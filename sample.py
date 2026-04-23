"""Generate text from a trained GPT-2 checkpoint.

Usage:
    # auto-pick the latest checkpoint in a directory
    python sample.py --ckpt_dir checkpoints --prompt "The history of mathematics"

    # or point at one specific file
    python sample.py --ckpt checkpoints/ckpt_00006000.pt --prompt "ROMEO:"
"""
from __future__ import annotations

import argparse
from pathlib import Path

import torch

from config import ModelConfig
from model import GPT


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt", type=str, default=None,
                   help="path to a specific checkpoint .pt file")
    p.add_argument("--ckpt_dir", type=str, default="checkpoints",
                   help="directory of rotating checkpoints — picks the highest step (used if --ckpt is unset)")
    p.add_argument("--prompt", type=str, default="\n")
    p.add_argument("--max_new_tokens", type=int, default=300)
    p.add_argument("--temperature", type=float, default=0.8)
    p.add_argument("--top_k", type=int, default=200)
    p.add_argument("--num_samples", type=int, default=1)
    p.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--seed", type=int, default=1337)
    return p.parse_args()


def resolve_ckpt(args: argparse.Namespace) -> Path:
    if args.ckpt:
        path = Path(args.ckpt)
        if not path.exists():
            raise FileNotFoundError(f"checkpoint not found: {path}")
        return path
    d = Path(args.ckpt_dir)
    if not d.is_dir():
        raise FileNotFoundError(f"checkpoint dir not found: {d}")
    ckpts = sorted(d.glob("ckpt_*.pt"))
    if not ckpts:
        # Fall back to legacy single-file layout from the original nano notebook.
        legacy = d / "ckpt.pt"
        if legacy.exists():
            return legacy
        raise FileNotFoundError(f"no ckpt_*.pt files in {d}")
    return ckpts[-1]


def main() -> None:
    args = parse_args()
    torch.manual_seed(args.seed)
    if args.device.startswith("cuda"):
        torch.cuda.manual_seed_all(args.seed)

    ckpt_path = resolve_ckpt(args)
    print(f"[sample] loading {ckpt_path}")
    ckpt = torch.load(ckpt_path, map_location=args.device, weights_only=False)

    mcfg = ModelConfig(**ckpt["model_config"])
    model = GPT(mcfg).to(args.device)
    model.load_state_dict(ckpt["model"])
    model.eval()

    import tiktoken
    enc = tiktoken.get_encoding("gpt2")
    start_ids = enc.encode_ordinary(args.prompt)
    if not start_ids:
        start_ids = [enc.eot_token]
    x = torch.tensor(start_ids, dtype=torch.long, device=args.device)[None, ...]

    for i in range(args.num_samples):
        y = model.generate(x, args.max_new_tokens, temperature=args.temperature, top_k=args.top_k)
        print(f"--- sample {i + 1} ---")
        print(enc.decode(y[0].tolist()))
        print()


if __name__ == "__main__":
    main()
