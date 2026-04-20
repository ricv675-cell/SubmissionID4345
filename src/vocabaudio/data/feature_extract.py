from __future__ import annotations

import json
import math
import os
import time
from collections import defaultdict
from pathlib import Path

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

import numpy as np
import torch
import torchaudio
from transformers import AutoProcessor, AudioFlamingo3ForConditionalGeneration

from vocabaudio.utils.io import load_many_jsonl, write_json


def pool_by_factor(features: np.ndarray, factor: int) -> np.ndarray:
    if factor == 1 or len(features) == 0:
        return features
    n_chunks = int(math.ceil(len(features) / factor))
    pooled = []
    for idx in range(n_chunks):
        start = idx * factor
        end = min(len(features), (idx + 1) * factor)
        pooled.append(features[start:end].mean(axis=0))
    return np.stack(pooled, axis=0).astype(np.float16)


def load_audio_16k(path: str, cache: dict[int, torchaudio.transforms.Resample]) -> np.ndarray:
    waveform, sampling_rate = torchaudio.load(path)
    if waveform.shape[0] > 1:
        waveform = waveform.mean(dim=0, keepdim=True)
    if sampling_rate != 16000:
        if sampling_rate not in cache:
            cache[sampling_rate] = torchaudio.transforms.Resample(sampling_rate, 16000)
        waveform = cache[sampling_rate](waveform)
    return waveform.squeeze(0).numpy()


def chunk_audio(audio: np.ndarray, chunk_seconds: float = 30.0, sample_rate: int = 16000) -> list[np.ndarray]:
    chunk_size = int(chunk_seconds * sample_rate)
    chunks = []
    for start in range(0, len(audio), chunk_size):
        chunks.append(audio[start : start + chunk_size])
    return chunks


def save_feature_npz(path: Path, features: np.ndarray, clip_length: float, duration: float) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        path,
        features=features.astype(np.float16),
        clip_length=np.array([clip_length], dtype=np.float32),
        duration=np.array([duration], dtype=np.float32),
    )


def save_query_npz(path: Path, hidden_states: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(path, last_hidden_state=hidden_states.astype(np.float16))


def extract_af3_features(
    model_path: Path,
    input_jsonl: list[Path],
    output_root: Path,
    audio_batch_size: int = 24,
    max_chunks_per_batch: int = 96,
    text_batch_size: int = 128,
    coarse_hz: int = 1,
    fine_hz: int = 5,
    force: bool = False,
) -> dict:
    output_root.mkdir(parents=True, exist_ok=True)
    rows = load_many_jsonl(input_jsonl)
    unique_audio: dict[str, dict] = {}
    unique_queries: dict[str, dict] = {}
    for row in rows:
        unique_audio[row["audio_path"]] = {
            "dataset": row["dataset"],
            "vid": row["vid"],
            "duration": row["duration"],
            "audio_path": row["audio_path"],
        }
        unique_queries[row["qid"]] = {
            "dataset": row["dataset"],
            "qid": row["qid"],
            "query": row["query"],
        }

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    processor = AutoProcessor.from_pretrained(model_path)
    model = AudioFlamingo3ForConditionalGeneration.from_pretrained(
        model_path,
        torch_dtype=torch.bfloat16 if device.type == "cuda" else torch.float32,
    ).to(device)
    model.eval()

    coarse_factor = 25 // coarse_hz
    fine_factor = 25 // fine_hz
    resampler_cache: dict[int, torchaudio.transforms.Resample] = {}
    summary = defaultdict(dict)
    t_start = time.time()

    audio_items = list(unique_audio.values())
    audio_items.sort(key=lambda x: x["duration"])
    audio_index = 0
    while audio_index < len(audio_items):
        batch_items = audio_items[audio_index : audio_index + audio_batch_size]
        pending_items = []
        raw_chunks: list[np.ndarray] = []
        chunk_ranges: list[tuple[int, int]] = []
        consumed = 0
        for item in batch_items:
            dataset = item["dataset"]
            coarse_path = output_root / dataset / f"audio_{coarse_hz}hz" / f"{item['vid']}.npz"
            fine_path = output_root / dataset / f"audio_{fine_hz}hz" / f"{item['vid']}.npz"
            if coarse_path.exists() and fine_path.exists() and not force:
                consumed += 1
                continue
            audio = load_audio_16k(item["audio_path"], resampler_cache)
            chunks = chunk_audio(audio)
            if pending_items and len(raw_chunks) + len(chunks) > max_chunks_per_batch:
                break
            st = len(raw_chunks)
            raw_chunks.extend(chunks)
            ed = len(raw_chunks)
            pending_items.append(item)
            chunk_ranges.append((st, ed))
            consumed += 1

        if consumed == 0:
            consumed = 1
        audio_index += consumed

        if not pending_items:
            continue

        batch = processor.feature_extractor(
            raw_chunks,
            sampling_rate=16000,
            return_tensors="pt",
            return_attention_mask=True,
            padding=True,
        )
        input_features = batch["input_features"].to(
            device=device,
            dtype=torch.bfloat16 if device.type == "cuda" else torch.float32,
        )
        input_mask = batch["attention_mask"].to(device=device)

        with torch.inference_mode():
            tower_out = model.audio_tower(
                input_features,
                input_features_mask=input_mask,
                return_dict=True,
            )
            audio_embeds = model.multi_modal_projector(tower_out.last_hidden_state)
            post_lengths = ((input_mask.sum(-1) - 2) // 2 + 1).tolist()

        audio_embeds = audio_embeds.detach().cpu().to(torch.float16).numpy()
        for item, (st, ed) in zip(pending_items, chunk_ranges):
            pieces = [audio_embeds[idx, : post_lengths[idx]] for idx in range(st, ed)]
            full = np.concatenate(pieces, axis=0).astype(np.float16)
            dataset = item["dataset"]
            save_feature_npz(
                output_root / dataset / f"audio_{coarse_hz}hz" / f"{item['vid']}.npz",
                pool_by_factor(full, coarse_factor),
                clip_length=1.0 / coarse_hz,
                duration=item["duration"],
            )
            save_feature_npz(
                output_root / dataset / f"audio_{fine_hz}hz" / f"{item['vid']}.npz",
                pool_by_factor(full, fine_factor),
                clip_length=1.0 / fine_hz,
                duration=item["duration"],
            )

        print(
            json.dumps(
                {
                    "stage": "audio",
                    "done_items": min(audio_index, len(audio_items)),
                    "total_items": len(audio_items),
                    "elapsed_sec": round(time.time() - t_start, 1),
                },
                ensure_ascii=False,
            )
        )

    query_items = list(unique_queries.values())
    for batch_start in range(0, len(query_items), text_batch_size):
        batch_items = query_items[batch_start : batch_start + text_batch_size]
        pending_items = []
        texts = []
        for item in batch_items:
            query_path = output_root / item["dataset"] / "text" / f"qid{item['qid']}.npz"
            if query_path.exists() and not force:
                continue
            pending_items.append(item)
            texts.append(item["query"])
        if not pending_items:
            continue

        tokenized = processor.tokenizer(
            texts,
            padding=True,
            truncation=True,
            max_length=64,
            return_tensors="pt",
        )
        tokenized = {k: v.to(device) for k, v in tokenized.items()}
        with torch.inference_mode():
            outputs = model.language_model.model(
                input_ids=tokenized["input_ids"],
                attention_mask=tokenized["attention_mask"],
                return_dict=True,
            )
        hidden = outputs.last_hidden_state.detach().cpu().to(torch.float16).numpy()
        masks = tokenized["attention_mask"].detach().cpu().numpy()
        for item, hs, mask in zip(pending_items, hidden, masks):
            save_query_npz(
                output_root / item["dataset"] / "text" / f"qid{item['qid']}.npz",
                hs[: int(mask.sum())],
            )

    final_summary = {
        "audio_items": len(unique_audio),
        "query_items": len(unique_queries),
        "coarse_hz": coarse_hz,
        "fine_hz": fine_hz,
        "elapsed_sec": round(time.time() - t_start, 1),
    }
    write_json(output_root / "feature_summary.json", final_summary)
    return final_summary

