from __future__ import annotations

from pathlib import Path

import numpy as np
import torch


def l2_normalize_np_array(np_array: np.ndarray, eps: float = 1e-5) -> np.ndarray:
    return np_array / (np.linalg.norm(np_array, axis=-1, keepdims=True) + eps)


def load_feature_npz(path: Path, key: str) -> torch.Tensor:
    arr = np.load(path)[key].astype(np.float32)
    arr = l2_normalize_np_array(arr)
    return torch.from_numpy(arr)


def resolve_audio_feature_path(feature_root: Path, dataset_name: str, audio_hz: int, vid: str) -> Path:
    return feature_root / dataset_name / f"audio_{audio_hz}hz" / f"{vid}.npz"


def resolve_query_feature_path(feature_root: Path, dataset_name: str, qid: str) -> Path:
    return feature_root / dataset_name / "text" / f"qid{qid}.npz"

