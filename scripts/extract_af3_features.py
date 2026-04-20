#!/usr/bin/env python3
from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from vocabaudio.data.feature_extract import extract_af3_features


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Extract AF3 audio/text features for VocabAudio")
    parser.add_argument("--model-path", type=Path, required=True)
    parser.add_argument("--input-jsonl", nargs="+", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--audio-batch-size", type=int, default=24)
    parser.add_argument("--max-chunks-per-batch", type=int, default=96)
    parser.add_argument("--text-batch-size", type=int, default=128)
    parser.add_argument("--coarse-hz", type=int, default=1)
    parser.add_argument("--fine-hz", type=int, default=5)
    parser.add_argument("--force", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    extract_af3_features(
        model_path=args.model_path,
        input_jsonl=args.input_jsonl,
        output_root=args.output_root,
        audio_batch_size=args.audio_batch_size,
        max_chunks_per_batch=args.max_chunks_per_batch,
        text_batch_size=args.text_batch_size,
        coarse_hz=args.coarse_hz,
        fine_hz=args.fine_hz,
        force=args.force,
    )


if __name__ == "__main__":
    main()
