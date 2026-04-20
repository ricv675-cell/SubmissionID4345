from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.cluster import MiniBatchKMeans


class ResidualTemporalBlock(nn.Module):
    def __init__(self, dim: int, dilation: int, dropout: float) -> None:
        super().__init__()
        self.conv = nn.Conv1d(dim, dim, kernel_size=3, padding=dilation, dilation=dilation)
        self.norm = nn.BatchNorm1d(dim)
        self.proj = nn.Conv1d(dim, dim, kernel_size=1)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = x.transpose(1, 2)
        y = self.conv(y)
        y = self.norm(y)
        y = F.gelu(y)
        y = self.proj(y)
        y = self.dropout(y)
        y = y.transpose(1, 2)
        return x + y


class TemporalContextEncoder(nn.Module):
    def __init__(
        self,
        input_dim: int = 3584,
        hidden_dim: int = 768,
        dropout: float = 0.1,
        dilations: tuple[int, ...] = (1, 2, 4),
    ) -> None:
        super().__init__()
        self.input_proj = nn.Linear(input_dim, hidden_dim)
        self.blocks = nn.ModuleList(
            [ResidualTemporalBlock(hidden_dim, dilation=d, dropout=dropout) for d in dilations]
        )
        self.output_norm = nn.LayerNorm(hidden_dim)
        self.recon = nn.Linear(hidden_dim, input_dim)
        self.delta_proj = nn.Linear(input_dim, hidden_dim)

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        z = self.input_proj(x)
        for block in self.blocks:
            z = block(z)
        return self.output_norm(z)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        z = self.encode(x)
        recon = self.recon(z)
        return z, recon


@dataclass
class RVQCodebooks:
    centers: np.ndarray

    @property
    def num_codebooks(self) -> int:
        return int(self.centers.shape[0])

    @property
    def codebook_size(self) -> int:
        return int(self.centers.shape[1])

    @property
    def dim(self) -> int:
        return int(self.centers.shape[2])


def fit_residual_kmeans(
    samples: np.ndarray,
    num_codebooks: int,
    codebook_size: int,
    batch_size: int = 8192,
    max_iter: int = 100,
    random_state: int = 42,
) -> RVQCodebooks:
    residual = samples.astype(np.float32).copy()
    centers = []
    for m in range(num_codebooks):
        kmeans = MiniBatchKMeans(
            n_clusters=codebook_size,
            batch_size=batch_size,
            max_iter=max_iter,
            n_init=3,
            random_state=random_state + m,
            compute_labels=True,
        )
        labels = kmeans.fit_predict(residual)
        layer_centers = kmeans.cluster_centers_.astype(np.float32)
        centers.append(layer_centers)
        residual = residual - layer_centers[labels]
    return RVQCodebooks(centers=np.stack(centers, axis=0))


def quantize_with_rvq(z: torch.Tensor, codebooks: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    residual = z
    all_indices = []
    quantized_sum = torch.zeros_like(z)
    for m in range(codebooks.shape[0]):
        centers = codebooks[m]
        dist = torch.cdist(residual.reshape(-1, residual.shape[-1]), centers)
        idx = torch.argmin(dist, dim=-1)
        q = centers[idx].reshape_as(z)
        quantized_sum = quantized_sum + q
        residual = residual - q
        all_indices.append(idx.reshape(z.shape[0], z.shape[1]))
    return torch.stack(all_indices, dim=1), quantized_sum


def rvq_usage_loss(indices: torch.Tensor, codebook_size: int) -> torch.Tensor:
    losses = []
    uniform = None
    for m in range(indices.shape[1]):
        hist = torch.bincount(indices[:, m, :].reshape(-1), minlength=codebook_size).float()
        probs = hist / hist.sum().clamp_min(1.0)
        if uniform is None:
            uniform = torch.full_like(probs, 1.0 / len(probs))
        losses.append(torch.sum(probs * torch.log((probs + 1e-8) / (uniform + 1e-8))))
    return torch.stack(losses).mean()

