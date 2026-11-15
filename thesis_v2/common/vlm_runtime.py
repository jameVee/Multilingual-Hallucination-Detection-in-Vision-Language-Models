"""Model loading / input building shared by Part 1 training, prediction, Part 2
(same code, different data) and Part 3a sampling.

Input building keeps solutions/02's two-step pattern (apply_chat_template with
tokenize=False, then processor(text=, images=)) -- the one-call
apply_chat_template(tokenize=True) path was found to silently drop the image on
some transformers versions. Every call asserts pixel_values is present.
"""
from __future__ import annotations

import torch
from PIL import Image

from thesis_v2.common.backbones import Backbone

# Vision towers / projectors stay in bf16 (not 4-bit) for every backbone: they are
# small, and 4-bit vision encoders are a known source of degraded image features.
SKIP_QUANT = ["visual", "vision_tower", "vision_model", "multi_modal_projector", "mlp1", "lm_head"]


def resolve_device(gpu_index):
    if not torch.cuda.is_available():
        return "cpu"
    if gpu_index is not None:
        torch.cuda.set_device(gpu_index)
        return f"cuda:{gpu_index}"
    return "cuda"


def best_dtype(device):
    if device == "cpu":
        return torch.float32
    idx = int(device.split(":")[1]) if ":" in device else torch.cuda.current_device()
    major, _ = torch.cuda.get_device_capability(idx)
    return torch.bfloat16 if major >= 8 else torch.float16


def load_processor(bb: Backbone, source=None):
    from transformers import AutoProcessor
    source = source or bb.model_id
    try:
        processor = AutoProcessor.from_pretrained(source, trust_remote_code=bb.trust_remote_code, **bb.processor_kwargs)
    except TypeError as exc:
        # Surface it: silently dropping a resolution cap could blow up sequence length.
        raise RuntimeError(f"{bb.name}: processor rejected kwargs {bb.processor_kwargs}: {exc}") from exc
    tok = getattr(processor, "tokenizer", processor)
    if tok.pad_token is None:
        # e.g. Pixtral: no pad token, so padding=True raises. Batches are size 1, so the
        # choice never changes a training example; prefer the vocabulary's own <pad>.
        tok.pad_token = "<pad>" if "<pad>" in tok.get_vocab() else tok.eos_token
    return processor


def load_model(bb: Backbone, device, load_in_4bit=True, attn_implementation=None):
    from transformers import AutoModelForImageTextToText, BitsAndBytesConfig
    extra = {"attn_implementation": attn_implementation} if attn_implementation else {}
    dtype = best_dtype(device)
    quant = None
    if load_in_4bit:
        quant = BitsAndBytesConfig(
            load_in_4bit=True, bnb_4bit_compute_dtype=dtype, bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True, llm_int8_skip_modules=SKIP_QUANT,
        )
    device_map = {"": device} if device != "cpu" else None
    model = AutoModelForImageTextToText.from_pretrained(
        bb.model_id, quantization_config=quant, dtype=dtype, device_map=device_map,
        trust_remote_code=bb.trust_remote_code, **extra,
    )
    return model, dtype


def apply_lora(model, bb: Backbone, r=16, alpha=16, gradient_checkpointing=True, load_in_4bit=True):
    from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
    if load_in_4bit:
        model = prepare_model_for_kbit_training(model, use_gradient_checkpointing=gradient_checkpointing)
    elif gradient_checkpointing:
        model.gradient_checkpointing_enable()
        model.enable_input_require_grads()
    cfg = LoraConfig(r=r, lora_alpha=alpha, lora_dropout=0.0, bias="none",
                     target_modules=bb.lora_target_regex, task_type="CAUSAL_LM")
    model = get_peft_model(model, cfg)
    wrapped = [n for n, m in model.named_modules() if n.endswith("lora_A")]
    if not wrapped:
        raise RuntimeError(f"{bb.name}: LoRA regex {bb.lora_target_regex!r} matched no modules")
    vision_hits = [n for n in wrapped if any(v in n for v in ("visual", "vision_tower", "vision_model"))]
    if vision_hits:
        raise RuntimeError(f"{bb.name}: LoRA wrapped vision modules (e.g. {vision_hits[0]}) -- must be LM only")
    return model, len(wrapped)


def build_inputs(processor, bb: Backbone, messages, image_path, add_generation_prompt):
    image = Image.open(image_path).convert("RGB")
    text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=add_generation_prompt)
    images = [[image]] if bb.nested_images else [image]
    inputs = processor(text=[text], images=images, padding=True, return_tensors="pt")
    if "pixel_values" not in inputs:
        raise RuntimeError(f"{bb.name}: processor returned no pixel_values -- image was dropped")
    return inputs


def build_training_batch(processor, bb: Backbone, conv):
    """Full conversation with the loss masked to the assistant answer only."""
    messages = conv["messages"]
    full = build_inputs(processor, bb, messages, conv["_image_path"], add_generation_prompt=False)
    prompt = build_inputs(processor, bb, [messages[0]], conv["_image_path"], add_generation_prompt=True)
    prompt_len = prompt["input_ids"].shape[1]
    labels = full["input_ids"].clone()
    labels[:, :prompt_len] = -100
    labels[full["attention_mask"] == 0] = -100
    full["labels"] = labels
    return full, prompt_len


def pad_token_id(processor):
    tok = getattr(processor, "tokenizer", processor)
    return tok.pad_token_id if tok.pad_token_id is not None else tok.eos_token_id
