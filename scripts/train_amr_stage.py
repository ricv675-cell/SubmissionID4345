#!/usr/bin/env python3
from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from vocabaudio.training.stage_b_train import train_stage_b
from vocabaudio.utils.io import load_yaml


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train VocabAudio Stage B AMR model")
    parser.add_argument("--config", type=Path, default=None)
    parser.add_argument("--model-path", type=Path, default=None)
    parser.add_argument("--train-jsonl", type=Path, default=None)
    parser.add_argument("--val-jsonl", type=Path, default=None)
    parser.add_argument("--feature-root", type=Path, default=None)
    parser.add_argument("--stage-a-dir", type=Path, default=None)
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--audio-hz", type=int, default=1)
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--eval-batch-size", type=int, default=4)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--learning-rate", type=float, default=3e-5)
    parser.add_argument("--weight-decay", type=float, default=0.05)
    parser.add_argument("--grad-clip", type=float, default=1.0)
    parser.add_argument("--hidden-dim", type=int, default=768)
    parser.add_argument("--lora-rank", type=int, default=8)
    parser.add_argument("--lora-alpha", type=int, default=32)
    parser.add_argument("--lora-dropout", type=float, default=0.1)
    parser.add_argument("--score-loss-weight", type=float, default=0.3)
    parser.add_argument("--window-iou-threshold", type=float, default=0.3)
    parser.add_argument("--audio-window-sec", type=float, default=30.0)
    parser.add_argument("--stride-sec", type=float, default=15.0)
    parser.add_argument("--max-new-tokens", type=int, default=64)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    cfg = load_yaml(args.config) if args.config else {}
    runtime = {k: v for k, v in vars(args).items() if v is not None}
    cfg.update(runtime)
    if cfg.get("config") is not None:
        cfg.pop("config")
    train_stage_b(cfg)


if __name__ == "__main__":
    main()
