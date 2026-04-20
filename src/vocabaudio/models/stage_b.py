from __future__ import annotations

import json
import re

import torch
import torch.nn as nn
from peft import LoraConfig, get_peft_model
from transformers import AudioFlamingo3ForConditionalGeneration, AutoProcessor

from vocabaudio.models.context_rvq import TemporalContextEncoder, quantize_with_rvq
from vocabaudio.utils.windows import normalize_windows


JSON_TIME_RE = re.compile(r"\[\s*([0-9]+(?:\.[0-9]+)?)\s*,\s*([0-9]+(?:\.[0-9]+)?)\s*\]")


class AF3HybridGenerativeLocalizer(nn.Module):
    def __init__(
        self,
        model_path: str,
        hidden_dim: int = 768,
        num_codebooks: int = 8,
        codebook_size: int = 1024,
        prefix_dropout: float = 0.1,
        max_prefix_len: int = 4096,
    ) -> None:
        super().__init__()
        self.processor = AutoProcessor.from_pretrained(model_path)
        self.af3 = AudioFlamingo3ForConditionalGeneration.from_pretrained(
            model_path,
            torch_dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32,
            device_map=None,
        )
        self.context_encoder = TemporalContextEncoder(hidden_dim=hidden_dim, dropout=prefix_dropout)
        self.codebooks = nn.Parameter(torch.zeros(num_codebooks, codebook_size, hidden_dim), requires_grad=False)
        d_model = self.af3.config.text_config.hidden_size
        self.query_proj = nn.Linear(d_model, d_model)
        self.cont_proj = nn.Linear(hidden_dim, d_model)
        self.global_proj = nn.Linear(hidden_dim, d_model)
        self.discrete_embeddings = nn.ModuleList(
            [nn.Embedding(codebook_size, d_model) for _ in range(num_codebooks)]
        )
        self.hybrid_norm = nn.LayerNorm(d_model)
        self.score_head = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, 1),
        )
        self.type_embed = nn.Embedding(4, d_model)
        self.pos_embed = nn.Embedding(max_prefix_len, d_model)
        self.prefix_dropout = nn.Dropout(prefix_dropout)

    def lm_dtype(self) -> torch.dtype:
        return next(self.af3.language_model.parameters()).dtype

    @property
    def tokenizer(self):
        return self.processor.tokenizer

    def resize_token_embeddings(self, new_size: int) -> None:
        self.af3.resize_token_embeddings(new_size, mean_resizing=False)

    def enable_gradient_checkpointing(self) -> None:
        if hasattr(self.af3, "gradient_checkpointing_enable"):
            self.af3.gradient_checkpointing_enable()
        if hasattr(self.af3.language_model, "gradient_checkpointing_enable"):
            self.af3.language_model.gradient_checkpointing_enable()

    def enable_language_lora(
        self,
        rank: int = 8,
        alpha: int = 32,
        dropout: float = 0.1,
        target_suffixes: tuple[str, ...] = ("q_proj", "v_proj"),
    ) -> None:
        target_modules = []
        for name, _ in self.af3.language_model.named_modules():
            if any(name.endswith(suffix) for suffix in target_suffixes):
                target_modules.append(name)
        cfg = LoraConfig(
            r=rank,
            lora_alpha=alpha,
            lora_dropout=dropout,
            bias="none",
            task_type="CAUSAL_LM",
            target_modules=sorted(set(target_modules)),
        )
        self.af3.language_model = get_peft_model(self.af3.language_model, cfg)

    def freeze_backbone(self) -> None:
        for param in self.af3.parameters():
            param.requires_grad = False
        projector = getattr(self.af3, "multi_modal_projector", None)
        if projector is not None:
            for param in projector.parameters():
                param.requires_grad = True
        for name, param in self.af3.named_parameters():
            if "lora_" in name:
                param.requires_grad = True
        embeddings = self.af3.get_input_embeddings()
        for param in embeddings.parameters():
            param.requires_grad = True
        out_head = self.af3.get_output_embeddings()
        if out_head is not None:
            for param in out_head.parameters():
                param.requires_grad = True

    def load_pretrained_context(self, state_dict: dict) -> None:
        self.context_encoder.load_state_dict(state_dict, strict=False)

    def load_codebooks(self, centers: torch.Tensor) -> None:
        self.codebooks.data.copy_(centers)

    def text_token_embeddings(self, input_ids: torch.Tensor) -> torch.Tensor:
        return self.af3.get_input_embeddings()(input_ids).to(dtype=self.lm_dtype())

    def build_hybrid_prefix(
        self,
        audio_feat: torch.Tensor,
        audio_mask: torch.Tensor,
        query_feat: torch.Tensor,
        query_mask: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        z = self.context_encoder.encode(audio_feat)
        indices, _ = quantize_with_rvq(z, self.codebooks)
        cont_tokens = self.cont_proj(z)
        disc_tokens = sum(
            emb(indices[:, i, :]) for i, emb in enumerate(self.discrete_embeddings)
        )
        local_tokens = self.hybrid_norm(cont_tokens + disc_tokens)
        global_vec = (z * audio_mask.unsqueeze(-1)).sum(dim=1) / audio_mask.sum(dim=1, keepdim=True).clamp_min(1.0)
        global_token = self.global_proj(global_vec).unsqueeze(1)
        query_tokens = self.query_proj(query_feat)

        prefix = torch.cat([global_token, query_tokens, cont_tokens, local_tokens], dim=1)
        prefix_mask = torch.cat(
            [
                torch.ones((audio_feat.shape[0], 1), dtype=audio_mask.dtype, device=audio_mask.device),
                query_mask,
                audio_mask,
                audio_mask,
            ],
            dim=1,
        )
        type_ids = torch.cat(
            [
                torch.zeros((audio_feat.shape[0], 1), dtype=torch.long, device=audio_feat.device),
                torch.ones(query_tokens.shape[:2], dtype=torch.long, device=audio_feat.device),
                torch.full(audio_mask.shape, 2, dtype=torch.long, device=audio_feat.device),
                torch.full(audio_mask.shape, 3, dtype=torch.long, device=audio_feat.device),
            ],
            dim=1,
        )
        pos_ids = torch.arange(prefix.shape[1], device=prefix.device).unsqueeze(0)
        prefix = prefix + self.type_embed(type_ids) + self.pos_embed(pos_ids)
        prefix = self.prefix_dropout(prefix).to(dtype=self.lm_dtype())

        query_ctx = (query_feat * query_mask.unsqueeze(-1)).sum(dim=1) / query_mask.sum(dim=1, keepdim=True).clamp_min(1.0)
        pooled_z = global_vec
        score_logits = self.score_head(torch.cat([pooled_z, query_ctx], dim=-1)).squeeze(-1)
        return prefix, prefix_mask, score_logits

    def forward_with_prefix(
        self,
        prefix_embeds: torch.Tensor,
        prefix_mask: torch.Tensor,
        prompt_ids: torch.Tensor,
        prompt_mask: torch.Tensor,
        answer_ids: torch.Tensor | None = None,
        answer_label_mask: torch.Tensor | None = None,
    ):
        prompt_embeds = self.text_token_embeddings(prompt_ids)
        if answer_ids is None:
            inputs_embeds = torch.cat([prefix_embeds, prompt_embeds], dim=1)
            attention_mask = torch.cat([prefix_mask, prompt_mask], dim=1)
            return self.af3.language_model(
                inputs_embeds=inputs_embeds,
                attention_mask=attention_mask,
                return_dict=True,
            )

        answer_embeds = self.text_token_embeddings(answer_ids)
        answer_mask = (answer_ids != self.tokenizer.pad_token_id).to(dtype=prompt_mask.dtype)
        inputs_embeds = torch.cat([prefix_embeds, prompt_embeds, answer_embeds], dim=1)
        attention_mask = torch.cat([prefix_mask, prompt_mask, answer_mask], dim=1)
        labels = torch.full(
            (answer_ids.shape[0], inputs_embeds.shape[1]),
            -100,
            dtype=torch.long,
            device=answer_ids.device,
        )
        if answer_label_mask is None:
            masked_answer = torch.where(answer_mask.bool(), answer_ids, torch.full_like(answer_ids, -100))
        else:
            masked_answer = torch.where(
                answer_label_mask.bool(),
                answer_ids,
                torch.full_like(answer_ids, -100),
            )
        labels[:, prefix_embeds.shape[1] + prompt_ids.shape[1] :] = masked_answer
        return self.af3.language_model(
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
            labels=labels,
            return_dict=True,
        )

    @torch.no_grad()
    def generate_with_prefix(
        self,
        prefix_embeds: torch.Tensor,
        prefix_mask: torch.Tensor,
        prompt_ids: torch.Tensor,
        prompt_mask: torch.Tensor,
        max_new_tokens: int,
    ) -> torch.Tensor:
        prompt_embeds = self.text_token_embeddings(prompt_ids)
        inputs_embeds = torch.cat([prefix_embeds, prompt_embeds], dim=1)
        attention_mask = torch.cat([prefix_mask, prompt_mask], dim=1)
        return self.af3.language_model.generate(
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
            max_new_tokens=max_new_tokens,
            do_sample=False,
        )


def build_generation_prompt(query: str, duration: float) -> str:
    return (
        "Locate the audio segment containing the queried sound.\n"
        f"Audio duration: {duration:.1f} seconds.\n"
        f"Query: {query}\n"
        'Return a JSON object like {"event": "<query>", "timestamps": [[start, end], ...]}.\n'
        "All timestamps must be in seconds."
    )


def parse_json_timestamps(text: str, duration: float) -> list[list[float]]:
    clean = text.strip()
    try:
        parsed = json.loads(clean)
        timestamps = parsed.get("timestamps", []) if isinstance(parsed, dict) else []
        if isinstance(timestamps, list):
            out = []
            for item in timestamps:
                if isinstance(item, list) and len(item) >= 2:
                    out.append([float(item[0]), float(item[1])])
            if out:
                return [[window[0], window[1], 1.0] for window in normalize_windows(out, duration)]
    except Exception:
        pass

    matches = []
    for m in JSON_TIME_RE.finditer(clean):
        start = float(m.group(1))
        end = float(m.group(2))
        matches.append([start, end])
    norm = normalize_windows(matches, duration)
    return [[w[0], w[1], 1.0] for w in norm]
