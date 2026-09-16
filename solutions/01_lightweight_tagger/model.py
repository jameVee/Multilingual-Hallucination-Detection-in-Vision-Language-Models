"""Lightweight multimodal span tagger: frozen CLIP ViT-B/32 patch features +
trainable XLM-RoBERTa-base text encoder + a small cross-attention fusion +
a per-token 5-way sigmoid head (multi-label, since gold spans can overlap
with different labels on the same characters -- see analysis notes on
annotator disagreement).

Designed to be cheap enough to train on a single modest GPU (even an 11 GB
2080 Ti, or the 6 GB laptop card used for local smoke tests) in well under
an hour, as a fast reference point alongside the VLM LoRA fine-tunes
(solutions 02/03). CLIP is frozen and never runs during training -- image
features are precomputed once via common/image_cache.py.
"""
from __future__ import annotations

import sys
from pathlib import Path

import torch
import torch.nn as nn

_THIS_DIR = Path(__file__).resolve().parent
_REPO_ROOT = _THIS_DIR.parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from solutions.common import data_io

NUM_LABELS = len(data_io.LABEL_ORDER)

# Rarer classes get a lower decision threshold by default (favor recall over
# precision where positives are scarce); tune further via a held-out slice
# of *train* if needed -- never tune thresholds against val itself.
DEFAULT_THRESHOLDS = {
    "invention": 0.5,
    "mischaracterization": 0.5,
    "OCR": 0.4,
    "miscounting": 0.4,
    "other": 0.4,
}


class CrossAttentionBlock(nn.Module):
    def __init__(self, dim=768, n_heads=8, ffn_mult=4, dropout=0.1):
        super().__init__()
        self.ln1 = nn.LayerNorm(dim)
        self.attn = nn.MultiheadAttention(dim, n_heads, dropout=dropout, batch_first=True)
        self.ln2 = nn.LayerNorm(dim)
        self.ffn = nn.Sequential(
            nn.Linear(dim, dim * ffn_mult), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(dim * ffn_mult, dim),
        )
        self.dropout = nn.Dropout(dropout)

    def forward(self, text_hidden, image_hidden):
        q = self.ln1(text_hidden)
        attn_out, _ = self.attn(q, image_hidden, image_hidden, need_weights=False)
        text_hidden = text_hidden + self.dropout(attn_out)
        text_hidden = text_hidden + self.dropout(self.ffn(self.ln2(text_hidden)))
        return text_hidden


class SpanTagger(nn.Module):
    def __init__(self, text_model_name="xlm-roberta-base", image_dim=768, hidden=768,
                 n_fusion_layers=2, n_heads=8, dropout=0.1):
        super().__init__()
        from transformers import XLMRobertaModel

        self.text_encoder = XLMRobertaModel.from_pretrained(text_model_name)
        self.image_proj = nn.Sequential(nn.Linear(image_dim, hidden), nn.LayerNorm(hidden))
        self.fusion = nn.ModuleList([
            CrossAttentionBlock(hidden, n_heads, dropout=dropout) for _ in range(n_fusion_layers)
        ])
        self.head = nn.Linear(hidden, NUM_LABELS)

    def forward(self, input_ids, attention_mask, image_tokens):
        text_out = self.text_encoder(input_ids=input_ids, attention_mask=attention_mask).last_hidden_state
        img = self.image_proj(image_tokens)
        hidden = text_out
        for block in self.fusion:
            hidden = block(hidden, img)
        return self.head(hidden)  # [B, L, NUM_LABELS] raw logits


def _finalize_span(run, label):
    start = min(item[2] for item in run)
    end = max(item[3] for item in run)
    prob = sum(item[1] for item in run) / len(run)
    return {"start": start, "end": end, "prob": round(float(prob), 4), "label": label}


def decode_spans(token_probs, offsets, seq_ids, thresholds=None, max_token_gap=2):
    """token_probs: [n_tokens, NUM_LABELS] sigmoid probabilities.
    offsets: per-token (char_start, char_end) tuples, as returned by the
    tokenizer for the *response* segment (seq_id == 1); other segments are
    ignored. Returns a submission-format list of {start, end, prob, label}.
    """
    thresholds = thresholds or DEFAULT_THRESHOLDS
    spans = []
    for label_idx, label in enumerate(data_io.LABEL_ORDER):
        thresh = thresholds.get(label, 0.5)
        active = []
        for i, (seq_id, (a, b)) in enumerate(zip(seq_ids, offsets)):
            if seq_id != 1 or a == b:
                continue
            prob = float(token_probs[i, label_idx])
            if prob >= thresh:
                active.append((i, prob, a, b))
        if not active:
            continue
        active.sort(key=lambda x: x[0])
        run = [active[0]]
        for item in active[1:]:
            if item[0] - run[-1][0] <= max_token_gap:
                run.append(item)
            else:
                spans.append(_finalize_span(run, label))
                run = [item]
        spans.append(_finalize_span(run, label))
    return sorted(spans, key=lambda s: s["start"])
