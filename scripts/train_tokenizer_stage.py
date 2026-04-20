#!/usr/bin/env python3
from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from vocabaudio.training.stage_a_train import train_stage_a
from vocabaudio.utils.io import load_yaml


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train VocabAudio Stage A tokenizer")
    parser.add_argument("--config", type=Path, default=None)
    parser.add_argument("--train-jsonl", type=Path, default=None)
    parser.add_argument("--feature-root", type=Path, default=None)
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--audio-hz", type=int, default=1)
    parser.add_argument("--epochs", type=int, default=15)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-2)
    parser.add_argument("--hidden-dim", type=int, default=768)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--delta-loss-weight", type=float, default=0.2)
    parser.add_argument("--align-loss-weight", type=float, default=0.3)
    parser.add_argument("--usage-loss-weight", type=float, default=0.05)
    parser.add_argument("--temperature", type=float, default=0.07)
    parser.add_argument("--saliency-pos-weight", type=float, default=4.0)
    parser.add_argument("--boundary-pos-weight", type=float, default=6.0)
    parser.add_argument("--boundary-radius", type=int, default=1)
    parser.add_argument("--num-codebooks", type=int, default=8)
    parser.add_argument("--codebook-size", type=int, default=1024)
    parser.add_argument("--sample-frames", type=int, default=250000)
    parser.add_argument("--foreground-ratio", type=float, default=0.45)
    parser.add_argument("--boundary-ratio", type=float, default=0.20)
    parser.add_argument("--per-row-pos-cap", type=int, default=64)
    parser.add_argument("--per-row-boundary-cap", type=int, default=32)
    parser.add_argument("--per-row-neg-cap", type=int, default=128)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    cfg = load_yaml(args.config) if args.config else {}
    runtime = {k: v for k, v in vars(args).items() if v is not None}
    cfg.update(runtime)
    if cfg.get("config") is not None:
        cfg.pop("config")
    train_stage_a(cfg)


if __name__ == "__main__":
    main()
