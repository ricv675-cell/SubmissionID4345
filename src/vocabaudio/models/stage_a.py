from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from vocabaudio.models.context_rvq import TemporalContextEncoder


class TaskAwareContextRVQ(nn.Module):
    def __init__(
        self,
        input_dim: int = 3584,
        hidden_dim: int = 768,
        dropout: float = 0.1,
        dilations: tuple[int, ...] = (1, 2, 4),
    ) -> None:
        super().__init__()
        self.context_encoder = TemporalContextEncoder(
            input_dim=input_dim,
            hidden_dim=hidden_dim,
            dropout=dropout,
            dilations=dilations,
        )
        self.query_proj = nn.Linear(input_dim, hidden_dim)
        self.global_proj = nn.Linear(hidden_dim, hidden_dim)
        self.text_proj = nn.Linear(input_dim, hidden_dim)
        self.saliency_head = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, 1),
        )
        self.boundary_head = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, 1),
        )

    def forward(
        self,
        audio_feat: torch.Tensor,
        audio_mask: torch.Tensor,
        query_feat: torch.Tensor,
        query_mask: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        z, recon = self.context_encoder(audio_feat)
        query_hidden = self.query_proj(query_feat)
        query_ctx = (query_hidden * query_mask.unsqueeze(-1)).sum(dim=1) / query_mask.sum(dim=1, keepdim=True).clamp_min(1.0)
        query_ctx_expanded = query_ctx.unsqueeze(1).expand_as(z)
        fused = torch.cat([z, query_ctx_expanded], dim=-1)
        saliency_logits = self.saliency_head(fused).squeeze(-1)
        boundary_logits = self.boundary_head(fused).squeeze(-1)
        global_audio = (z * audio_mask.unsqueeze(-1)).sum(dim=1) / audio_mask.sum(dim=1, keepdim=True).clamp_min(1.0)
        global_audio = self.global_proj(global_audio)
        global_text = self.text_proj(
            (query_feat * query_mask.unsqueeze(-1)).sum(dim=1) / query_mask.sum(dim=1, keepdim=True).clamp_min(1.0)
        )
        return z, recon, saliency_logits, boundary_logits, global_audio, global_text


def masked_bce_loss(logits: torch.Tensor, targets: torch.Tensor, mask: torch.Tensor, pos_weight: float) -> torch.Tensor:
    pos_weight_tensor = torch.full((), pos_weight, dtype=logits.dtype, device=logits.device)
    loss = F.binary_cross_entropy_with_logits(logits, targets, reduction="none", pos_weight=pos_weight_tensor)
    loss = loss * mask
    return loss.sum() / mask.sum().clamp_min(1.0)


def contrastive_align_loss(audio_embed: torch.Tensor, text_embed: torch.Tensor, temperature: float) -> torch.Tensor:
    audio_embed = F.normalize(audio_embed, dim=-1)
    text_embed = F.normalize(text_embed, dim=-1)
    logits = audio_embed @ text_embed.transpose(0, 1) / temperature
    labels = torch.arange(logits.shape[0], device=logits.device)
    loss_a = F.cross_entropy(logits, labels)
    loss_t = F.cross_entropy(logits.transpose(0, 1), labels)
    return 0.5 * (loss_a + loss_t)

