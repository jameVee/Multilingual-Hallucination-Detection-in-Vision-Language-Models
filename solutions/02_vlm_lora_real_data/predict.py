"""Run a LoRA-fine-tuned VLM checkpoint over data/val.jsonl and write
val_predictions.jsonl in the official submission schema.

Usage:
    python predict.py --checkpoint ./checkpoint --gpu_index 0
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch
from PIL import Image

_THIS_DIR = Path(__file__).resolve().parent
_REPO_ROOT = _THIS_DIR.parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from solutions.common import data_io, vlm_io

sys.path.insert(0, str(_THIS_DIR))
from train_lora import DEFAULT_MODEL_ID, best_dtype, build_examples, resolve_model_class


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--val_file", default=str(data_io.VAL_FILE))
    p.add_argument("--checkpoint", default=str(_THIS_DIR / "checkpoint"))
    p.add_argument("--base_model_id", default=DEFAULT_MODEL_ID)
    p.add_argument("--output", default=str(_THIS_DIR / "val_predictions.jsonl"))
    p.add_argument("--max_new_tokens", type=int, default=512)
    p.add_argument("--gpu_index", type=int, default=None,
                    help="8B model (default) fits any of the VM's 5 GPUs in 4-bit. If "
                         "--base_model_id points at the 30B-A3B MoE alternative, restrict to "
                         "the 24GB 3090 Tis (indices 0, 3, 5).")
    p.add_argument("--load_in_4bit", action="store_true", default=True)
    p.add_argument("--no_4bit", dest="load_in_4bit", action="store_false")
    p.add_argument("--max_pixels", type=int, default=1024 * 28 * 28,
                    help="Must match the value used at training time.")
    p.add_argument("--min_pixels", type=int, default=256 * 28 * 28)
    p.add_argument("--smoke_test", action="store_true")
    args = p.parse_args()

    device = "cpu"
    if torch.cuda.is_available():
        device = f"cuda:{args.gpu_index}" if args.gpu_index is not None else "cuda"
        if args.gpu_index is not None:
            torch.cuda.set_device(args.gpu_index)

    records = data_io.load_jsonl(args.val_file)
    if args.smoke_test:
        records = records[:10]
    examples = build_examples(records)
    print(f"Loaded {len(examples)} usable val conversations (device={device})")

    from peft import PeftModel
    from transformers import AutoProcessor, BitsAndBytesConfig

    model_class = resolve_model_class(args.base_model_id)
    dtype = best_dtype(device)
    quant_config = None
    if args.load_in_4bit:
        quant_config = BitsAndBytesConfig(
            load_in_4bit=True, bnb_4bit_compute_dtype=dtype,
            bnb_4bit_quant_type="nf4", bnb_4bit_use_double_quant=True,
        )

    processor = AutoProcessor.from_pretrained(
        str(args.checkpoint), min_pixels=args.min_pixels, max_pixels=args.max_pixels
    )
    device_map = {"": device} if device != "cpu" else None
    print(f"Loading base model {args.base_model_id} via {model_class.__name__} "
          f"(4bit={args.load_in_4bit}, dtype={dtype})...")
    base_model = model_class.from_pretrained(
        args.base_model_id, quantization_config=quant_config, dtype=dtype, device_map=device_map,
    )
    model = PeftModel.from_pretrained(base_model, args.checkpoint)
    model.eval()

    from tqdm.auto import tqdm

    predictions = []
    with torch.no_grad():
        for i, conv in enumerate(tqdm(examples, desc="predicting", unit="ex", dynamic_ncols=True)):
            # Two-step pattern (text via apply_chat_template, image via PIL +
            # processor(text=, images=)), not the model card's one-call
            # apply_chat_template(tokenize=True) convenience flow -- that was
            # verified locally (against Qwen2-VL-2B) to silently drop the
            # image with no pixel_values and no error. See train_lora.py's
            # make_collate_fn docstring for the full explanation.
            messages = [conv["messages"][0]]  # user turn only
            image = Image.open(conv["_image_path"]).convert("RGB")
            text_input = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
            inputs = processor(text=[text_input], images=[image], padding=True, return_tensors="pt").to(device)
            assert "pixel_values" in inputs, (
                "Processor did not return pixel_values -- the image was not actually "
                "encoded. Do not trust these predictions until this is fixed."
            )

            out = model.generate(
                **inputs, max_new_tokens=args.max_new_tokens, do_sample=False,
                pad_token_id=processor.tokenizer.pad_token_id,
            )
            raw = processor.batch_decode(out[:, inputs["input_ids"].shape[1]:], skip_special_tokens=True)[0].strip()
            spans = vlm_io.predict_spans_from_raw_output(conv["_response"], raw)
            predictions.append({"id": conv["_id"], "labels": spans})

    data_io.write_jsonl(args.output, predictions)
    print(f"Wrote {len(predictions)} predictions -> {args.output}")


if __name__ == "__main__":
    main()
