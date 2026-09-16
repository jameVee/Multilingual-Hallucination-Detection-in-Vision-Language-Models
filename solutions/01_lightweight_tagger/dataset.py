"""PyTorch Dataset for the lightweight tagger: tokenizes (prompt, response)
pairs and builds per-token multi-label soft targets from gold char spans.

Soft targets (values in [0,1], the annotator-agreement `prob`) are used
directly as BCE regression targets rather than binarized -- this trains the
model to output calibrated per-character probabilities, which is exactly
what the Spearman-correlation half of the official metric rewards.
"""
from __future__ import annotations

import sys
from pathlib import Path

import torch
from torch.utils.data import Dataset

_THIS_DIR = Path(__file__).resolve().parent
_REPO_ROOT = _THIS_DIR.parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from solutions.common import data_io

LABEL_TO_IDX = {label: i for i, label in enumerate(data_io.LABEL_ORDER)}


class SpanTaggingDataset(Dataset):
    def __init__(self, records, tokenizer, image_cache, max_length=512):
        self.records = records
        self.tokenizer = tokenizer
        self.image_cache = image_cache
        self.max_length = max_length
        any_tensor = next(iter(image_cache.values())) if image_cache else None
        self.default_image_tokens = torch.zeros_like(any_tensor) if any_tensor is not None else None

    def __len__(self):
        return len(self.records)

    def __getitem__(self, idx):
        record = self.records[idx]
        prompt = record.get("prompt", "")
        response = record["response"]
        enc = self.tokenizer(
            prompt, response, truncation="only_second", max_length=self.max_length,
            return_offsets_mapping=True,
        )
        seq_ids = enc.sequence_ids()
        offsets = enc["offset_mapping"]
        n_tokens = len(enc["input_ids"])

        targets = torch.zeros(n_tokens, len(data_io.LABEL_ORDER), dtype=torch.float32)
        resp_token_mask = torch.zeros(n_tokens, dtype=torch.bool)

        for i, (seq_id, offset) in enumerate(zip(seq_ids, offsets)):
            if seq_id != 1:
                continue
            a, b = offset
            if a == b:
                continue
            resp_token_mask[i] = True
            for span in record.get("labels", []) or []:
                if a < span["end"] and b > span["start"]:
                    label_idx = LABEL_TO_IDX.get(span["label"])
                    if label_idx is not None:
                        targets[i, label_idx] = max(targets[i, label_idx].item(), float(span["prob"]))

        image_tokens = self.image_cache.get(record["image_name"])
        if image_tokens is None:
            image_tokens = self.default_image_tokens

        return {
            "input_ids": torch.tensor(enc["input_ids"], dtype=torch.long),
            "attention_mask": torch.tensor(enc["attention_mask"], dtype=torch.long),
            "targets": targets,
            "resp_token_mask": resp_token_mask,
            "image_tokens": image_tokens.float(),
            "offsets": offsets,
            "seq_ids": seq_ids,
            "id": record["id"],
            "response": response,
        }


def collate_fn(batch, pad_token_id=1):
    max_len = max(item["input_ids"].shape[0] for item in batch)
    n_labels = batch[0]["targets"].shape[1]
    n_img_tokens = batch[0]["image_tokens"].shape[0]
    img_dim = batch[0]["image_tokens"].shape[1]

    input_ids = torch.full((len(batch), max_len), pad_token_id, dtype=torch.long)
    attention_mask = torch.zeros((len(batch), max_len), dtype=torch.long)
    targets = torch.zeros((len(batch), max_len, n_labels), dtype=torch.float32)
    resp_mask = torch.zeros((len(batch), max_len), dtype=torch.bool)
    image_tokens = torch.zeros((len(batch), n_img_tokens, img_dim), dtype=torch.float32)

    meta = []
    for i, item in enumerate(batch):
        length = item["input_ids"].shape[0]
        input_ids[i, :length] = item["input_ids"]
        attention_mask[i, :length] = item["attention_mask"]
        targets[i, :length] = item["targets"]
        resp_mask[i, :length] = item["resp_token_mask"]
        image_tokens[i] = item["image_tokens"]
        meta.append({
            "id": item["id"], "response": item["response"],
            "offsets": item["offsets"], "seq_ids": item["seq_ids"], "length": length,
        })

    return {
        "input_ids": input_ids, "attention_mask": attention_mask, "targets": targets,
        "resp_token_mask": resp_mask, "image_tokens": image_tokens, "meta": meta,
    }
