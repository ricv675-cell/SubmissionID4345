#!/usr/bin/env python3
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from vocabaudio.models.stage_b import AF3HybridGenerativeLocalizer
from vocabaudio.training.stage_b_train import evaluate_model
from vocabaudio.utils.io import load_jsonl, write_json, write_jsonl


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate VocabAudio AMR model")
    parser.add_argument("--model-path", type=Path, required=True)
    parser.add_argument("--model-ckpt", type=Path, required=True)
    parser.add_argument("--input-jsonl", type=Path, required=True)
    parser.add_argument("--feature-root", type=Path, required=True)
    parser.add_argument("--stage-a-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--audio-hz", type=int, default=1)
    parser.add_argument("--eval-batch-size", type=int, default=4)
    parser.add_argument("--max-new-tokens", type=int, default=64)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    payload = torch.load(args.model_ckpt, map_location="cpu", weights_only=False)
    run_args = payload["args"]
    rvq_npz = np.load(args.stage_a_dir / "rvq_codebooks.npz")
    context_payload = torch.load(args.stage_a_dir / "taskaware_context_encoder.pt", map_location="cpu", weights_only=False)

    model = AF3HybridGenerativeLocalizer(
        model_path=str(args.model_path),
        hidden_dim=int(run_args["hidden_dim"]),
        num_codebooks=int(rvq_npz["centers"].shape[0]),
        codebook_size=int(rvq_npz["centers"].shape[1]),
    )
    model.enable_language_lora(
        rank=int(run_args["lora_rank"]),
        alpha=int(run_args["lora_alpha"]),
        dropout=float(run_args["lora_dropout"]),
    )
    model.freeze_backbone()
    model.load_pretrained_context(context_payload["context_encoder"])
    model.load_codebooks(torch.from_numpy(rvq_npz["centers"]).float())
    model.load_state_dict(payload["trainable_state"], strict=False)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)

    rows = load_jsonl(args.input_jsonl)
    metrics, predictions = evaluate_model(
        model=model,
        rows=rows,
        feature_root=args.feature_root,
        audio_hz=args.audio_hz,
        device=device,
        batch_size=args.eval_batch_size,
        max_new_tokens=args.max_new_tokens,
        window_sec=float(run_args["audio_window_sec"]),
        stride_sec=float(run_args["stride_sec"]),
    )
    write_json(args.output_dir / "metrics.json", metrics)
    write_jsonl(args.output_dir / "submission.jsonl", predictions)
    print(metrics["brief"])


if __name__ == "__main__":
    main()
