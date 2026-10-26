"""Backbone registry for Part 1 (model diversity).

Everything model-specific lives here so train.py / predict.py are identical across
backbones -- the backbone is the only thing that varies between Part 1 runs.

Fixed for every backbone (isolation): same instruction prompt and span-JSON target
(solutions/common/vlm_io.py), same QLoRA recipe (4-bit NF4, r16, alpha16, lr 2e-4,
2 epochs, grad_accum 8), LoRA applied to the *language model's* projection layers
only. What legitimately differs: the model class, processor image-resolution
settings (each model's own native scheme, capped so image+text fits 4096 tokens),
and whether the processor wants images nested per sample.

Every entry must pass the Phase 0 feasibility gate (feasibility.py) before use.
"""
from __future__ import annotations

from dataclasses import dataclass, field

LM_PROJ = r"(q_proj|k_proj|v_proj|o_proj|gate_proj|up_proj|down_proj)"


@dataclass(frozen=True)
class Backbone:
    name: str
    model_id: str
    family: str
    processor_kwargs: dict = field(default_factory=dict)
    nested_images: bool = False          # processor(images=[[img]]) instead of [img]
    lora_target_regex: str = rf".*language_model.*\.{LM_PROJ}"
    trust_remote_code: bool = False
    notes: str = ""


BACKBONES = {
    # Reference: same model as solutions/02, retrained through this pipeline so every
    # backbone shares one code path. Pixel caps identical to solutions/02.
    "qwen3vl8b": Backbone(
        name="qwen3vl8b", model_id="Qwen/Qwen3-VL-8B-Instruct", family="Qwen",
        processor_kwargs={"min_pixels": 256 * 28 * 28, "max_pixels": 1024 * 28 * 28},
    ),
    "gemma3_12b": Backbone(
        name="gemma3_12b", model_id="google/gemma-3-12b-it", family="Gemma",
        processor_kwargs={"do_pan_and_scan": False},   # fixed 896px -> 256 image tokens
        nested_images=True,
        notes="Gated on HF: accept the Gemma license on the account used by nlp2's HF token.",
    ),
    "internvl3_8b": Backbone(
        name="internvl3_8b", model_id="OpenGVLab/InternVL3-8B-hf", family="InternVL",
        processor_kwargs={"max_patches": 6},           # <=6 tiles x 256 tokens
        notes="Language model is Qwen2.5 -> least architecturally distinct of the three.",
    ),
    "pixtral12b": Backbone(
        name="pixtral12b", model_id="mistral-community/pixtral-12b", family="Mistral",
        processor_kwargs={"size": {"longest_edge": 768}},   # caps patch count
        nested_images=True,
    ),
    # Backups if a primary fails the feasibility gate.
    "llama32v11b": Backbone(
        name="llama32v11b", model_id="meta-llama/Llama-3.2-11B-Vision-Instruct", family="Llama",
        nested_images=True,
        notes="Gated on HF. Officially English-only for image+text; multilingual text still works.",
    ),
    "phi4mm": Backbone(
        name="phi4mm", model_id="microsoft/Phi-4-multimodal-instruct", family="Phi",
        trust_remote_code=True,
        lora_target_regex=r".*\.(qkv_proj|o_proj|gate_up_proj|down_proj)",
        notes="Remote code; last-resort backup.",
    ),
}

PRIMARY = ["qwen3vl8b", "gemma3_12b", "internvl3_8b", "pixtral12b"]
BACKUPS = ["llama32v11b", "phi4mm"]


def get(name: str) -> Backbone:
    if name not in BACKBONES:
        raise KeyError(f"unknown backbone {name!r}; known: {sorted(BACKBONES)}")
    return BACKBONES[name]
