from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np
import torch
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader, Dataset

from vocabaudio.models.context_rvq import fit_residual_kmeans
from vocabaudio.models.stage_a import TaskAwareContextRVQ, contrastive_align_loss, masked_bce_loss
from vocabaudio.utils.features import load_feature_npz, resolve_audio_feature_path, resolve_query_feature_path
from vocabaudio.utils.io import load_many_jsonl, write_json
from vocabaudio.utils.seed import set_seed
from vocabaudio.utils.windows import normalize_windows, seconds_to_step_idx


def build_targets(
    windows: list[list[float]],
    duration: float,
    num_steps: int,
    boundary_radius: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    saliency = torch.zeros(num_steps, dtype=torch.float32)
    boundary = torch.zeros(num_steps, dtype=torch.float32)
    for start, end in normalize_windows(windows, duration):
        st_idx = seconds_to_step_idx(start, duration, num_steps, use_end=False)
        ed_idx = seconds_to_step_idx(end, duration, num_steps, use_end=True)
        saliency[st_idx : ed_idx + 1] = 1.0
        for idx in (st_idx, ed_idx):
            lo = max(0, idx - boundary_radius)
            hi = min(num_steps, idx + boundary_radius + 1)
            boundary[lo:hi] = 1.0
    return saliency, boundary


class QueryAwareTemporalDataset(Dataset):
    def __init__(
        self,
        rows: list[dict],
        feature_root: Path,
        audio_hz: int,
        boundary_radius: int,
    ) -> None:
        self.rows = rows
        self.feature_root = feature_root
        self.audio_hz = audio_hz
        self.boundary_radius = boundary_radius

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, idx: int) -> dict:
        row = self.rows[idx]
        audio_feat = load_feature_npz(
            resolve_audio_feature_path(self.feature_root, row["dataset"], self.audio_hz, row["vid"]),
            "features",
        )
        query_feat = load_feature_npz(
            resolve_query_feature_path(self.feature_root, row["dataset"], row["qid"]),
            "last_hidden_state",
        )
        saliency, boundary = build_targets(
            row["relevant_windows"],
            duration=float(row["duration"]),
            num_steps=int(audio_feat.shape[0]),
            boundary_radius=self.boundary_radius,
        )
        return {
            "audio_feat": audio_feat,
            "query_feat": query_feat,
            "saliency": saliency,
            "boundary": boundary,
        }


def collate_batch(batch: list[dict]) -> dict:
    audio = pad_sequence([x["audio_feat"] for x in batch], batch_first=True)
    query = pad_sequence([x["query_feat"] for x in batch], batch_first=True)
    saliency = pad_sequence([x["saliency"] for x in batch], batch_first=True, padding_value=0.0)
    boundary = pad_sequence([x["boundary"] for x in batch], batch_first=True, padding_value=0.0)
    audio_mask = torch.zeros(audio.shape[:2], dtype=torch.float32)
    query_mask = torch.zeros(query.shape[:2], dtype=torch.float32)
    for i, item in enumerate(batch):
        audio_mask[i, : len(item["audio_feat"])] = 1.0
        query_mask[i, : len(item["query_feat"])] = 1.0
    return {
        "audio_feat": audio,
        "audio_mask": audio_mask,
        "query_feat": query,
        "query_mask": query_mask,
        "saliency": saliency,
        "boundary": boundary,
    }


def sample_pool(arrays: list[np.ndarray], desired: int, rng: np.random.Generator, dim: int) -> np.ndarray:
    if desired <= 0 or not arrays:
        return np.zeros((0, dim), dtype=np.float32)
    pool = np.concatenate(arrays, axis=0).astype(np.float32)
    if len(pool) <= desired:
        return pool
    idx = rng.choice(len(pool), size=desired, replace=False)
    return pool[idx]


def train_stage_a(args: dict) -> dict:
    set_seed(int(args["seed"]))
    output_dir = Path(args["output_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)

    rows = load_many_jsonl([Path(args["train_jsonl"])])
    dataset = QueryAwareTemporalDataset(
        rows=rows,
        feature_root=Path(args["feature_root"]),
        audio_hz=int(args["audio_hz"]),
        boundary_radius=int(args["boundary_radius"]),
    )
    loader = DataLoader(
        dataset,
        batch_size=int(args["batch_size"]),
        shuffle=True,
        num_workers=int(args["num_workers"]),
        pin_memory=True,
        collate_fn=collate_batch,
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = TaskAwareContextRVQ(
        hidden_dim=int(args["hidden_dim"]),
        dropout=float(args["dropout"]),
        dilations=tuple(int(x) for x in args.get("dilations", [1, 2, 4])),
    ).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(args["learning_rate"]),
        weight_decay=float(args["weight_decay"]),
    )

    history = []
    t_start = time.time()
    for epoch in range(int(args["epochs"])):
        model.train()
        sums = {
            "loss_total": 0.0,
            "loss_recon": 0.0,
            "loss_delta": 0.0,
            "loss_align": 0.0,
            "loss_saliency": 0.0,
            "loss_boundary": 0.0,
        }
        batch_count = 0
        for batch in loader:
            audio_feat = batch["audio_feat"].to(device)
            audio_mask = batch["audio_mask"].to(device)
            query_feat = batch["query_feat"].to(device)
            query_mask = batch["query_mask"].to(device)
            saliency = batch["saliency"].to(device)
            boundary = batch["boundary"].to(device)

            z, recon, saliency_logits, boundary_logits, global_audio, global_text = model(
                audio_feat=audio_feat,
                audio_mask=audio_mask,
                query_feat=query_feat,
                query_mask=query_mask,
            )
            base = model.context_encoder.input_proj(audio_feat).detach()
            loss_recon = (((recon - audio_feat) ** 2) * audio_mask.unsqueeze(-1)).sum() / audio_mask.sum().clamp_min(1.0)
            if audio_feat.shape[1] > 1:
                z_delta = z[:, 1:] - z[:, :-1]
                base_delta = model.context_encoder.delta_proj(audio_feat[:, 1:] - audio_feat[:, :-1])
                delta_mask = audio_mask[:, 1:].unsqueeze(-1)
                loss_delta = (((z_delta - base_delta) ** 2) * delta_mask).sum() / delta_mask.sum().clamp_min(1.0)
            else:
                loss_delta = torch.zeros((), device=device)
            loss_align = contrastive_align_loss(global_audio, global_text, temperature=float(args["temperature"]))
            loss_saliency = masked_bce_loss(
                saliency_logits,
                saliency,
                audio_mask,
                pos_weight=float(args["saliency_pos_weight"]),
            )
            loss_boundary = masked_bce_loss(
                boundary_logits,
                boundary,
                audio_mask,
                pos_weight=float(args["boundary_pos_weight"]),
            )
            loss = (
                loss_recon
                + float(args["delta_loss_weight"]) * loss_delta
                + float(args["align_loss_weight"]) * loss_align
                + float(args["usage_loss_weight"]) * torch.zeros((), device=device)
                + 0.5 * loss_saliency
                + 0.3 * loss_boundary
            )

            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()

            sums["loss_total"] += float(loss.detach().cpu())
            sums["loss_recon"] += float(loss_recon.detach().cpu())
            sums["loss_delta"] += float(loss_delta.detach().cpu())
            sums["loss_align"] += float(loss_align.detach().cpu())
            sums["loss_saliency"] += float(loss_saliency.detach().cpu())
            sums["loss_boundary"] += float(loss_boundary.detach().cpu())
            batch_count += 1

        record = {"epoch": epoch + 1, "elapsed_sec": round(time.time() - t_start, 1)}
        for key, value in sums.items():
            record[key] = round(value / max(batch_count, 1), 6)
        history.append(record)
        print(json.dumps(record, ensure_ascii=False))

    payload = {
        "context_encoder": model.context_encoder.state_dict(),
        "query_proj": model.query_proj.state_dict(),
        "saliency_head": model.saliency_head.state_dict(),
        "boundary_head": model.boundary_head.state_dict(),
        "global_proj": model.global_proj.state_dict(),
        "text_proj": model.text_proj.state_dict(),
        "args": args,
        "history": history,
    }
    torch.save(payload, output_dir / "taskaware_context_encoder.pt")
    write_json(output_dir / "taskaware_train_history.json", {"history": history})

    rng = np.random.default_rng(int(args["seed"]))
    pos_pool: list[np.ndarray] = []
    boundary_pool: list[np.ndarray] = []
    neg_pool: list[np.ndarray] = []
    model.eval()
    with torch.no_grad():
        for item in dataset:
            audio_feat = item["audio_feat"].unsqueeze(0).to(device)
            query_feat = item["query_feat"].unsqueeze(0).to(device)
            audio_mask = torch.ones(audio_feat.shape[:2], dtype=torch.float32, device=device)
            query_mask = torch.ones(query_feat.shape[:2], dtype=torch.float32, device=device)
            z, _, _, _, _, _ = model(
                audio_feat=audio_feat,
                audio_mask=audio_mask,
                query_feat=query_feat,
                query_mask=query_mask,
            )
            z_np = z[0].cpu().numpy().astype(np.float32)
            sal_mask = item["saliency"].numpy() > 0.5
            bnd_mask = item["boundary"].numpy() > 0.5
            neg_mask = ~(sal_mask | bnd_mask)
            for mask, pool, cap in (
                (sal_mask, pos_pool, int(args["per_row_pos_cap"])),
                (bnd_mask, boundary_pool, int(args["per_row_boundary_cap"])),
                (neg_mask, neg_pool, int(args["per_row_neg_cap"])),
            ):
                idx = np.where(mask)[0]
                if len(idx) == 0:
                    continue
                if len(idx) > cap:
                    idx = rng.choice(idx, size=cap, replace=False)
                pool.append(z_np[idx])

    total = int(args["sample_frames"])
    n_boundary = int(total * float(args["boundary_ratio"]))
    n_foreground = int(total * float(args["foreground_ratio"]))
    n_negative = max(total - n_boundary - n_foreground, 0)
    dim = int(args["hidden_dim"])
    sampled = [
        sample_pool(pos_pool, n_foreground, rng, dim),
        sample_pool(boundary_pool, n_boundary, rng, dim),
        sample_pool(neg_pool, n_negative, rng, dim),
    ]
    sample_np = np.concatenate([x for x in sampled if len(x) > 0], axis=0)
    if len(sample_np) > total:
        keep = rng.choice(len(sample_np), size=total, replace=False)
        sample_np = sample_np[keep]

    codebooks = fit_residual_kmeans(
        sample_np,
        num_codebooks=int(args["num_codebooks"]),
        codebook_size=int(args["codebook_size"]),
        random_state=int(args["seed"]),
    )
    np.savez_compressed(output_dir / "rvq_codebooks.npz", centers=codebooks.centers)
    summary = {
        "num_rows": len(rows),
        "num_samples": int(len(sample_np)),
        "num_codebooks": int(args["num_codebooks"]),
        "codebook_size": int(args["codebook_size"]),
        "hidden_dim": int(args["hidden_dim"]),
        "foreground_pool_frames": int(sum(len(x) for x in pos_pool)),
        "boundary_pool_frames": int(sum(len(x) for x in boundary_pool)),
        "negative_pool_frames": int(sum(len(x) for x in neg_pool)),
    }
    write_json(output_dir / "rvq_summary.json", summary)
    return summary

