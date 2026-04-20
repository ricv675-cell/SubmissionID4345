from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader, Dataset

from vocabaudio.metrics.amr_metrics import evaluate_predictions
from vocabaudio.models.stage_b import AF3HybridGenerativeLocalizer, build_generation_prompt, parse_json_timestamps
from vocabaudio.utils.features import load_feature_npz, resolve_audio_feature_path, resolve_query_feature_path
from vocabaudio.utils.io import load_jsonl, write_json, write_jsonl
from vocabaudio.utils.seed import set_seed
from vocabaudio.utils.windows import normalize_windows, temporal_iou


def windows_to_json_text(windows: list[list[float]], query: str, duration: float) -> str:
    norm = normalize_windows(windows, duration)
    payload = {"event": query, "timestamps": [[s, e] for s, e in norm]}
    return json.dumps(payload, ensure_ascii=False)


def best_window_iou(pred_window: list[float], gt_windows: list[list[float]]) -> float:
    if not gt_windows:
        return 0.0
    return max(temporal_iou(pred_window, gt) for gt in gt_windows)


def coarse_score_target(duration: float, gt_windows: list[list[float]], threshold: float) -> float:
    full_window = [0.0, float(duration)]
    return 1.0 if best_window_iou(full_window, gt_windows) >= threshold else 0.0


def intersect_window(gt_window: list[float], clip_window: list[float]) -> list[float] | None:
    start = max(float(gt_window[0]), float(clip_window[0]))
    end = min(float(gt_window[1]), float(clip_window[1]))
    if end <= start:
        return None
    return [start, end]


def build_window_grid(duration: float, window_sec: float, stride_sec: float) -> list[list[float]]:
    if duration <= window_sec:
        return [[0.0, float(duration)]]
    out = []
    start = 0.0
    while start < duration:
        end = min(duration, start + window_sec)
        out.append([round(start, 4), round(end, 4)])
        if end >= duration:
            break
        start += stride_sec
    if out and out[-1][1] < duration:
        out.append([max(0.0, duration - window_sec), duration])
    dedup = []
    seen = set()
    for st, ed in out:
        key = (round(st, 4), round(ed, 4))
        if key not in seen:
            seen.add(key)
            dedup.append([st, ed])
    return dedup


class AF3GenerativeAMRDataset(Dataset):
    def __init__(
        self,
        jsonl_path: Path,
        feature_root: Path,
        audio_hz: int,
        tokenizer,
        window_sec: float,
        stride_sec: float,
        score_iou_threshold: float,
    ) -> None:
        self.rows = load_jsonl(jsonl_path)
        self.feature_root = feature_root
        self.audio_hz = audio_hz
        self.tokenizer = tokenizer
        self.window_sec = window_sec
        self.stride_sec = stride_sec
        self.score_iou_threshold = score_iou_threshold
        self.samples: list[dict] = []
        self._build_samples()

    def _build_samples(self) -> None:
        for row in self.rows:
            duration = float(row["duration"])
            for window_start, window_end in build_window_grid(duration, self.window_sec, self.stride_sec):
                gt_rel = []
                gt_global = normalize_windows(row["relevant_windows"], duration)
                for gt in gt_global:
                    inter = intersect_window(gt, [window_start, window_end])
                    if inter is None:
                        continue
                    gt_rel.append([inter[0] - window_start, inter[1] - window_start])
                label = coarse_score_target(
                    duration=window_end - window_start,
                    gt_windows=gt_rel,
                    threshold=self.score_iou_threshold,
                )
                self.samples.append(
                    {
                        "row": row,
                        "window_start": window_start,
                        "window_end": window_end,
                        "target_windows": gt_rel,
                        "score_target": label,
                    }
                )

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> dict:
        sample = self.samples[idx]
        row = sample["row"]
        full_audio_feat = load_feature_npz(
            resolve_audio_feature_path(self.feature_root, row["dataset"], self.audio_hz, row["vid"]),
            "features",
        )
        total_steps = int(full_audio_feat.shape[0])
        duration = float(row["duration"])
        start_idx = int(round(sample["window_start"] / max(duration, 1e-6) * total_steps))
        end_idx = int(round(sample["window_end"] / max(duration, 1e-6) * total_steps))
        end_idx = max(end_idx, start_idx + 1)
        audio_feat = full_audio_feat[start_idx:end_idx]
        query_feat = load_feature_npz(
            resolve_query_feature_path(self.feature_root, row["dataset"], row["qid"]),
            "last_hidden_state",
        )
        local_duration = sample["window_end"] - sample["window_start"]
        prompt = build_generation_prompt(row["query"], local_duration)
        answer = windows_to_json_text(sample["target_windows"], row["query"], local_duration)
        prompt_ids = self.tokenizer(prompt, add_special_tokens=False, return_tensors="pt")["input_ids"][0]
        answer_ids = self.tokenizer(answer, add_special_tokens=False, return_tensors="pt")["input_ids"][0]
        answer_label_mask = torch.ones_like(answer_ids, dtype=torch.bool)
        return {
            "row": row,
            "window_start": sample["window_start"],
            "window_end": sample["window_end"],
            "score_target": torch.tensor(sample["score_target"], dtype=torch.float32),
            "audio_feat": audio_feat,
            "query_feat": query_feat,
            "prompt_ids": prompt_ids,
            "answer_ids": answer_ids,
            "answer_label_mask": answer_label_mask,
        }


def collate_batch(batch: list[dict], pad_id: int) -> dict:
    audio = pad_sequence([x["audio_feat"] for x in batch], batch_first=True)
    query = pad_sequence([x["query_feat"] for x in batch], batch_first=True)
    prompt = pad_sequence([x["prompt_ids"] for x in batch], batch_first=True, padding_value=pad_id)
    answer = pad_sequence([x["answer_ids"] for x in batch], batch_first=True, padding_value=pad_id)
    answer_label_mask = pad_sequence([x["answer_label_mask"] for x in batch], batch_first=True, padding_value=False)
    audio_mask = torch.zeros(audio.shape[:2], dtype=torch.float32)
    query_mask = torch.zeros(query.shape[:2], dtype=torch.float32)
    prompt_mask = torch.zeros(prompt.shape, dtype=torch.float32)
    for i, item in enumerate(batch):
        audio_mask[i, : len(item["audio_feat"])] = 1.0
        query_mask[i, : len(item["query_feat"])] = 1.0
        prompt_mask[i, : len(item["prompt_ids"])] = 1.0
    return {
        "rows": [x["row"] for x in batch],
        "window_start": torch.tensor([x["window_start"] for x in batch], dtype=torch.float32),
        "window_end": torch.tensor([x["window_end"] for x in batch], dtype=torch.float32),
        "score_target": torch.stack([x["score_target"] for x in batch], dim=0),
        "audio_feat": audio,
        "audio_mask": audio_mask,
        "query_feat": query,
        "query_mask": query_mask,
        "prompt_ids": prompt,
        "prompt_mask": prompt_mask,
        "answer_ids": answer,
        "answer_label_mask": answer_label_mask,
    }


@torch.no_grad()
def evaluate_model(
    model: AF3HybridGenerativeLocalizer,
    rows: list[dict],
    feature_root: Path,
    audio_hz: int,
    device: torch.device,
    batch_size: int,
    max_new_tokens: int,
    window_sec: float,
    stride_sec: float,
) -> tuple[dict, list[dict]]:
    class EvalDataset(Dataset):
        def __init__(self) -> None:
            self.samples = []
            for row in rows:
                for window_start, window_end in build_window_grid(float(row["duration"]), window_sec, stride_sec):
                    self.samples.append(
                        {
                            "row": row,
                            "window_start": window_start,
                            "window_end": window_end,
                        }
                    )

        def __len__(self) -> int:
            return len(self.samples)

        def __getitem__(self, idx: int) -> dict:
            sample = self.samples[idx]
            row = sample["row"]
            full_audio_feat = load_feature_npz(
                resolve_audio_feature_path(feature_root, row["dataset"], audio_hz, row["vid"]),
                "features",
            )
            total_steps = int(full_audio_feat.shape[0])
            duration = float(row["duration"])
            start_idx = int(round(sample["window_start"] / max(duration, 1e-6) * total_steps))
            end_idx = int(round(sample["window_end"] / max(duration, 1e-6) * total_steps))
            end_idx = max(end_idx, start_idx + 1)
            prompt_ids = model.tokenizer(
                build_generation_prompt(row["query"], sample["window_end"] - sample["window_start"]),
                add_special_tokens=False,
                return_tensors="pt",
            )["input_ids"][0]
            return {
                "row": row,
                "window_start": sample["window_start"],
                "window_end": sample["window_end"],
                "score_target": torch.tensor(0.0, dtype=torch.float32),
                "audio_feat": full_audio_feat[start_idx:end_idx],
                "query_feat": load_feature_npz(
                    resolve_query_feature_path(feature_root, row["dataset"], row["qid"]),
                    "last_hidden_state",
                ),
                "prompt_ids": prompt_ids,
                "answer_ids": torch.tensor([model.tokenizer.pad_token_id], dtype=torch.long),
                "answer_label_mask": torch.tensor([False], dtype=torch.bool),
            }

    loader = DataLoader(
        EvalDataset(),
        batch_size=batch_size,
        shuffle=False,
        collate_fn=lambda x: collate_batch(x, model.tokenizer.pad_token_id),
    )
    pred_by_qid: dict[str, dict] = {}
    model.eval()
    for batch in loader:
        audio_feat = batch["audio_feat"].to(device)
        audio_mask = batch["audio_mask"].to(device)
        query_feat = batch["query_feat"].to(device)
        query_mask = batch["query_mask"].to(device)
        prompt_ids = batch["prompt_ids"].to(device)
        prompt_mask = batch["prompt_mask"].to(device)
        prefix_embeds, prefix_mask, score_logits = model.build_hybrid_prefix(
            audio_feat,
            audio_mask,
            query_feat,
            query_mask,
        )
        outputs = model.generate_with_prefix(
            prefix_embeds=prefix_embeds,
            prefix_mask=prefix_mask,
            prompt_ids=prompt_ids,
            prompt_mask=prompt_mask,
            max_new_tokens=max_new_tokens,
        )
        gen_ids = outputs.cpu()
        for row, ids, score, window_start, window_end in zip(
            batch["rows"],
            gen_ids,
            score_logits.detach().cpu().tolist(),
            batch["window_start"].tolist(),
            batch["window_end"].tolist(),
        ):
            local_duration = window_end - window_start
            text = model.tokenizer.decode(ids, skip_special_tokens=True)
            windows = parse_json_timestamps(text, local_duration)
            global_windows = []
            score_prob = round(float(torch.sigmoid(torch.tensor(score))), 6)
            for w in windows:
                global_windows.append(
                    [
                        round(w[0] + window_start, 4),
                        round(w[1] + window_start, 4),
                        score_prob,
                    ]
                )
            qid = row["qid"]
            if qid not in pred_by_qid:
                pred_by_qid[qid] = {
                    "qid": qid,
                    "query": row["query"],
                    "vid": row["vid"],
                    "pred_relevant_windows": [],
                    "raw_text_by_window": [],
                }
            pred_by_qid[qid]["pred_relevant_windows"].extend(global_windows)
            pred_by_qid[qid]["raw_text_by_window"].append(
                {
                    "window_start": window_start,
                    "window_end": window_end,
                    "score": score_prob,
                    "text": text,
                }
            )
    predictions = []
    for row in rows:
        pred = pred_by_qid.get(
            row["qid"],
            {
                "qid": row["qid"],
                "query": row["query"],
                "vid": row["vid"],
                "pred_relevant_windows": [],
                "raw_text_by_window": [],
            },
        )
        pred["pred_relevant_windows"] = sorted(
            pred["pred_relevant_windows"],
            key=lambda x: float(x[2]) if len(x) > 2 else 0.0,
            reverse=True,
        )[:10]
        predictions.append(pred)
    metrics = evaluate_predictions(rows, predictions)
    return metrics, predictions


def extract_trainable_state(model: torch.nn.Module) -> dict[str, torch.Tensor]:
    trainable_names = {name for name, param in model.named_parameters() if param.requires_grad}
    state = {}
    for key, value in model.state_dict().items():
        if any(key == name or key.startswith(name + ".") for name in trainable_names):
            state[key] = value.detach().cpu()
    return state


def train_stage_b(args: dict) -> dict:
    set_seed(int(args["seed"]))
    output_dir = Path(args["output_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)

    rvq_npz = np.load(Path(args["stage_a_dir"]) / "rvq_codebooks.npz")
    context_payload = torch.load(
        Path(args["stage_a_dir"]) / "taskaware_context_encoder.pt",
        map_location="cpu",
        weights_only=False,
    )

    model = AF3HybridGenerativeLocalizer(
        model_path=str(args["model_path"]),
        hidden_dim=int(args["hidden_dim"]),
        num_codebooks=int(rvq_npz["centers"].shape[0]),
        codebook_size=int(rvq_npz["centers"].shape[1]),
    )
    model.enable_gradient_checkpointing()
    model.enable_language_lora(
        rank=int(args["lora_rank"]),
        alpha=int(args["lora_alpha"]),
        dropout=float(args["lora_dropout"]),
    )
    model.freeze_backbone()
    model.load_pretrained_context(context_payload["context_encoder"])
    model.load_codebooks(torch.from_numpy(rvq_npz["centers"]).float())

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)

    train_dataset = AF3GenerativeAMRDataset(
        jsonl_path=Path(args["train_jsonl"]),
        feature_root=Path(args["feature_root"]),
        audio_hz=int(args["audio_hz"]),
        tokenizer=model.tokenizer,
        window_sec=float(args["audio_window_sec"]),
        stride_sec=float(args["stride_sec"]),
        score_iou_threshold=float(args["window_iou_threshold"]),
    )
    train_loader = DataLoader(
        train_dataset,
        batch_size=int(args["batch_size"]),
        shuffle=True,
        num_workers=int(args["num_workers"]),
        pin_memory=True,
        collate_fn=lambda x: collate_batch(x, model.tokenizer.pad_token_id),
    )
    val_rows = load_jsonl(Path(args["val_jsonl"]))

    trainable_params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(
        trainable_params,
        lr=float(args["learning_rate"]),
        weight_decay=float(args["weight_decay"]),
    )

    write_json(
        output_dir / "run_info.json",
        {
            "args": args,
            "num_codebooks": int(rvq_npz["centers"].shape[0]),
            "codebook_size": int(rvq_npz["centers"].shape[1]),
        },
    )

    best_metric = -1.0
    history = []
    t_start = time.time()
    for epoch in range(int(args["epochs"])):
        model.train()
        sums = {"loss_total": 0.0, "loss_gen": 0.0, "loss_score": 0.0}
        batch_count = 0
        for batch in train_loader:
            audio_feat = batch["audio_feat"].to(device)
            audio_mask = batch["audio_mask"].to(device)
            query_feat = batch["query_feat"].to(device)
            query_mask = batch["query_mask"].to(device)
            prompt_ids = batch["prompt_ids"].to(device)
            prompt_mask = batch["prompt_mask"].to(device)
            answer_ids = batch["answer_ids"].to(device)
            answer_label_mask = batch["answer_label_mask"].to(device)
            score_targets = batch["score_target"].to(device)

            prefix_embeds, prefix_mask, score_logits = model.build_hybrid_prefix(
                audio_feat,
                audio_mask,
                query_feat,
                query_mask,
            )
            outputs = model.forward_with_prefix(
                prefix_embeds=prefix_embeds,
                prefix_mask=prefix_mask,
                prompt_ids=prompt_ids,
                prompt_mask=prompt_mask,
                answer_ids=answer_ids,
                answer_label_mask=answer_label_mask,
            )
            gen_loss = outputs.loss
            score_loss = F.binary_cross_entropy_with_logits(score_logits, score_targets)
            loss = gen_loss + float(args["score_loss_weight"]) * score_loss

            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(trainable_params, float(args["grad_clip"]))
            optimizer.step()

            sums["loss_total"] += float(loss.detach().cpu())
            sums["loss_gen"] += float(gen_loss.detach().cpu())
            sums["loss_score"] += float(score_loss.detach().cpu())
            batch_count += 1

        metrics, predictions = evaluate_model(
            model=model,
            rows=val_rows,
            feature_root=Path(args["feature_root"]),
            audio_hz=int(args["audio_hz"]),
            device=device,
            batch_size=int(args["eval_batch_size"]),
            max_new_tokens=int(args["max_new_tokens"]),
            window_sec=float(args["audio_window_sec"]),
            stride_sec=float(args["stride_sec"]),
        )
        record = {
            "epoch": epoch + 1,
            "elapsed_sec": round(time.time() - t_start, 1),
            "loss_total": round(sums["loss_total"] / max(batch_count, 1), 6),
            "loss_gen": round(sums["loss_gen"] / max(batch_count, 1), 6),
            "loss_score": round(sums["loss_score"] / max(batch_count, 1), 6),
            "metrics": metrics,
        }
        history.append(record)
        write_json(output_dir / f"epoch_{epoch + 1:02d}_metrics.json", metrics)
        write_jsonl(output_dir / f"epoch_{epoch + 1:02d}_submission.jsonl", predictions)
        print(json.dumps(record, ensure_ascii=False))

        metric_key = metrics["mAP@avg"]
        payload = {
            "args": args,
            "trainable_state": extract_trainable_state(model),
        }
        torch.save(payload, output_dir / f"checkpoint_epoch{epoch + 1:02d}.pt")
        if metric_key > best_metric:
            best_metric = metric_key
            torch.save(payload, output_dir / "best.pt")
            write_json(output_dir / "best_metrics.json", metrics)
            write_jsonl(output_dir / "best_submission.jsonl", predictions)

    write_json(output_dir / "train_summary.json", {"history": history, "best_mAP@avg": best_metric})
    return {"history": history, "best_mAP@avg": best_metric}
