"""LoRA fine-tune of an open VLM (default Qwen3-VL-8B-Instruct, a dense
model -- 8B params / ~9B incl. vision tower, ~18 GB in bf16 / ~4.5 GB in
4-bit, comfortably fits any of the VM's 5 GPUs including the 11 GB 2080
Tis) to directly predict the hallucination span-list JSON for a (image,
prompt, response) triple, using data/train.jsonl only (no synthetic
augmentation -- see solutions 03-06 for the augmented counterparts). This is the
"real data only" reference point for the LoRA-VLM approach, scored via the
official scorer.py through solutions/common/scorer_local.py.

Standard Hugging Face stack (transformers + peft + bitsandbytes +
accelerate), not unsloth, for portability across an arbitrary VM setup this
was not developed on.

Qwen3-VL requires an unreleased transformers build (`pip install
git+https://github.com/huggingface/transformers`) as of this writing. This
code's model-loading/class-selection follows the model card exactly, but
NOT its one-call `apply_chat_template(tokenize=True, return_dict=True)`
convenience example for building multimodal inputs -- that was tested
locally (against Qwen2-VL-2B, the largest model this dev machine could
actually download; a real Qwen3-VL download was impractical over its
~50 KB/s connection) and silently dropped the image, returning no
pixel_values at all with no error. Instead this uses the older, more
foundational two-step pattern (apply_chat_template for text,
processor(text=, images=) for the combined multimodal tensors), verified
locally to reliably include pixel_values -- see make_collate_fn(). Still
run --dry_run then --smoke_test on the VM before a full run, since the
underlying model/processor itself remains unverified.

`resolve_model_class`/`resolve_target_modules` still branch on MoE ("A3B"/
"Moe" in --model_id) vs dense automatically, so passing
`--model_id Qwen/Qwen3-VL-30B-A3B-Instruct` (the original default -- a
sparse MoE model, 31B total params/~3B active per token, ~15.5 GB in
4-bit, needs one of the 24 GB 3090 Tis specifically) still works without
any code changes, just a bigger, slower, less-VRAM-portable run.

Usage:
    python train_lora.py --epochs 2 --grad_accum 8 --gpu_index 0
    python train_lora.py --dry_run                 # data pipeline only, no model download
    python train_lora.py --smoke_test --gpu_index 0  # tiny real run (loads the actual model)
    python train_lora.py --model_id Qwen/Qwen3-VL-30B-A3B-Instruct --gpu_index 0  # bigger MoE alternative
"""
from __future__ import annotations

import argparse
import json
import random
import sys
import time
from pathlib import Path

import torch
from PIL import Image

_THIS_DIR = Path(__file__).resolve().parent
_REPO_ROOT = _THIS_DIR.parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from solutions.common import data_io, vlm_io

DEFAULT_MODEL_ID = "Qwen/Qwen3-VL-8B-Instruct"


def best_dtype(device: str) -> torch.dtype:
    """bf16 on Ampere+ (the VM's RTX 3090 Tis, compute capability 8.6), fp16
    on older cards (the VM's RTX 2080 Tis are Turing, cc 7.5, no native bf16
    tensor cores) -- mirrors the auto-detect pattern used for this exact
    purpose in the prior last_experiment notebooks' `is_bf16_supported()`,
    but decided per-GPU since this VM mixes both generations."""
    if device == "cpu" or not torch.cuda.is_available():
        return torch.float32
    idx = int(device.split(":")[1]) if ":" in device else torch.cuda.current_device()
    major, _ = torch.cuda.get_device_capability(idx)
    return torch.bfloat16 if major >= 8 else torch.float16


def resolve_model_class(model_id: str):
    """Qwen3-VL ships two architectures under different transformers classes
    -- a dense model (`Qwen3VLForConditionalGeneration`) and a sparse MoE
    model (`Qwen3VLMoeForConditionalGeneration`, all A3B/"Moe" checkpoints,
    e.g. Qwen3-VL-30B-A3B-Instruct: 31B total params, ~3B active). Picking
    the wrong one fails to load. Falls back to the generic
    AutoModelForImageTextToText for any other model id (e.g. Qwen2-VL),
    which transformers resolves correctly from the checkpoint's own config.
    """
    from transformers import AutoModelForImageTextToText

    lower = model_id.lower()
    if "qwen3-vl" in lower or "qwen3vl" in lower:
        if "a3b" in lower or "moe" in lower:
            from transformers import Qwen3VLMoeForConditionalGeneration
            return Qwen3VLMoeForConditionalGeneration
        from transformers import Qwen3VLForConditionalGeneration
        return Qwen3VLForConditionalGeneration
    return AutoModelForImageTextToText


def resolve_target_modules(model_id: str, override=None):
    """LoRA target modules.

    An earlier version of this function returned peft's "all-linear"
    catch-all for MoE ("A3B") checkpoints, reasoning that expert-layer
    module names weren't inspectable without downloading the 62 GB
    checkpoint. That was a real mistake, caught by actually computing the
    numbers: Qwen3-VL-30B-A3B-Instruct's config (fetched from its
    Hugging Face config.json) has 128 experts x 48 layers, and peft's
    module-name matching targets every expert's gate/up/down projections
    by name regardless of "all-linear" vs. an explicit list containing
    those same suffixes -- either way works out to LoRA-tuning all 128
    experts per layer. At rank 16 that's ~843M trainable LoRA parameters
    (vs. ~12.6M for attention-only), which alone adds roughly 10 GB of
    gradient + AdamW optimizer-state overhead on top of the ~15.5 GB of
    4-bit base weights -- likely to OOM a 24 GB GPU once activations are
    added, and far more compute than intended for what's meant to be a
    lightweight adapter.

    MoE checkpoints therefore default to attention-only LoRA
    (q/k/v/o projections, dense and shared across all tokens, NOT
    per-expert) -- standard practice for LoRA-tuning MoE models for
    exactly this reason. Dense checkpoints keep the fuller
    attention+FFN list since there's no expert multiplication there.
    """
    if override is not None:
        return override
    lower = model_id.lower()
    if "a3b" in lower or "moe" in lower:
        return ["q_proj", "k_proj", "v_proj", "o_proj"]
    return ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]


def build_examples(records, image_dir=data_io.IMAGE_DIR):
    examples, skipped = [], 0
    for record in records:
        conv = vlm_io.build_conversation(record, image_dir=image_dir, include_target=True)
        if conv is None:
            skipped += 1
            continue
        examples.append(conv)
    if skipped:
        print(f"  skipped {skipped} records (image not resolvable)")
    return examples


class VLMSpanDataset(torch.utils.data.Dataset):
    def __init__(self, examples):
        self.examples = examples

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, idx):
        return self.examples[idx]


def make_collate_fn(processor):
    """Deliberately NOT using the single-call
    `apply_chat_template(..., tokenize=True, return_dict=True)` flow shown
    in Qwen3-VL's model card: verified locally (with Qwen2-VL-2B, the
    largest model this dev machine could actually download) that on the
    currently-released transformers, that call silently returns only
    input_ids/attention_mask with NO pixel_values -- the image is dropped
    without error, which would train a "VLM" that never actually sees the
    picture. Root cause looked version-dependent (a newer transformers
    might auto-fetch correctly), but this was not confirmed against
    Qwen3-VL's own processor (downloading it to test was impractical on
    this connection), so this uses the older, foundational, model-agnostic
    two-step pattern instead: apply_chat_template(tokenize=False) for the
    text, PIL.Image.open() for the picture, processor(text=, images=) to
    combine them -- verified locally to reliably produce pixel_values.
    """
    def collate(batch):
        assert len(batch) == 1, (
            "Trains with per-example forward passes (batch_size=1) + gradient "
            "accumulation -- images/responses vary too much in size to pad "
            "efficiently into a real batch here."
        )
        conv = batch[0]
        messages = conv["messages"]
        user_only = [messages[0]]
        image = Image.open(conv["_image_path"]).convert("RGB")

        full_text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=False)
        prompt_text = processor.apply_chat_template(user_only, tokenize=False, add_generation_prompt=True)

        full_inputs = processor(text=[full_text], images=[image], padding=True, return_tensors="pt")
        prompt_inputs = processor(text=[prompt_text], images=[image], padding=True, return_tensors="pt")
        assert "pixel_values" in full_inputs, (
            "Processor did not return pixel_values -- the image was not actually "
            "encoded. This is the exact silent-image-drop failure mode this code "
            "was written to avoid; do not proceed until this is fixed."
        )
        prompt_len = prompt_inputs["input_ids"].shape[1]

        labels = full_inputs["input_ids"].clone()
        labels[:, :prompt_len] = -100
        labels[full_inputs["attention_mask"] == 0] = -100
        full_inputs["labels"] = labels
        return full_inputs

    return collate


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--train_file", default=str(data_io.TRAIN_FILE))
    p.add_argument("--model_id", default=DEFAULT_MODEL_ID)
    p.add_argument("--output_dir", default=str(_THIS_DIR / "checkpoint"))
    p.add_argument("--epochs", type=int, default=2)
    p.add_argument("--grad_accum", type=int, default=8)
    p.add_argument("--lr", type=float, default=2e-4)
    p.add_argument("--lora_r", type=int, default=16)
    p.add_argument("--lora_alpha", type=int, default=16)
    p.add_argument("--target_modules", default=None,
                    help="Comma-separated module names, or 'all-linear'. Auto-picked "
                         "from --model_id if omitted (see resolve_target_modules).")
    p.add_argument("--warmup_steps", type=int, default=50)
    p.add_argument("--max_seq_len", type=int, default=4096,
                    help="Any training example whose full (image+text) token sequence exceeds "
                         "this length is skipped rather than fed to the model. Guards against a "
                         "rare pathologically long example causing a single-step attention-memory "
                         "spike that OOMs an otherwise-healthy run -- seen in practice on the "
                         "synthetic-augmented dataset (solution 04): a run stable for 726 steps "
                         "suddenly needed to allocate 25GB more for one attention op, almost "
                         "certainly from one unusually long response (e.g. an "
                         "inject_repetition_loop-generated 'other' example, or a long real seed). "
                         "There is no truncation here on purpose -- truncating would risk cutting "
                         "off the target span JSON the model is trained to produce, silently "
                         "corrupting that example's label instead of just skipping it.")
    p.add_argument("--max_pixels", type=int, default=1024 * 28 * 28,
                    help="Caps image resolution (-> visual token count) passed to the "
                         "processor. Some dataset images are large enough to produce 30k+ "
                         "image patches uncapped, which risks OOM even on a 24GB GPU; this "
                         "keeps memory bounded across all 5 of the VM's GPU tiers.")
    p.add_argument("--min_pixels", type=int, default=256 * 28 * 28)
    p.add_argument("--load_in_4bit", action="store_true", default=True)
    p.add_argument("--no_4bit", dest="load_in_4bit", action="store_false",
                    help="Disable 4-bit QLoRA (use plain bf16/fp16 LoRA). The default 8B dense "
                         "model needs ~18GB unquantized -- fits a single 24GB 3090 Ti but not an "
                         "11GB 2080 Ti (keep 4-bit on those). If you switch --model_id to the "
                         "30B-A3B MoE alternative, it needs ~62GB unquantized -- only feasible "
                         "with --no_4bit if you also switch to device_map='auto' multi-GPU "
                         "sharding; keep 4-bit on a single card.")
    p.add_argument("--gpu_index", type=int, default=None,
                    help="8B model (default) fits any of the VM's 5 GPUs in 4-bit, including the "
                         "11GB 2080 Tis (indices 1, 2). If you switch --model_id to the 30B-A3B "
                         "MoE alternative, restrict to the 24GB 3090 Tis (indices 0, 3, 5) -- its "
                         "~15.5GB of 4-bit weights do not fit an 11GB 2080 Ti. Never pass 4 "
                         "(broken on this VM) regardless of model.")
    p.add_argument("--gradient_checkpointing", action="store_true", default=True,
                    help="On by default: trades ~20-30% more compute time for substantially "
                         "lower activation memory, recomputing activations during backward "
                         "instead of storing them. This repo has no measured activation-memory "
                         "profile for a 48-layer/2048-hidden model at this image-token count "
                         "(never ran on real hardware locally) -- keep this on unless you've "
                         "confirmed headroom and want the speed.")
    p.add_argument("--no_gradient_checkpointing", dest="gradient_checkpointing", action="store_false")
    p.add_argument("--smoke_test", action="store_true")
    p.add_argument("--dry_run", action="store_true",
                    help="Build+inspect the data pipeline only; skip loading/training the VLM.")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--save_steps", type=int, default=250,
                    help="Save the LoRA adapter + a small resume-state file every N optimizer "
                         "steps (0 disables periodic saving, only saving once at the very end -- "
                         "the original behavior). Exists because a real run OOMed a Colab A100 "
                         "session's ~24h time limit at 95%% complete with nothing saved anywhere -- "
                         "the previous code only ever saved once, at the end. Point --output_dir at "
                         "a Drive-mounted path (not local /content disk) for this to actually "
                         "survive a session being killed, not just a clean process exit.")
    p.add_argument("--resume", action="store_true",
                    help="If --output_dir already has a saved adapter + train_state.json (from a "
                         "previous --save_steps checkpoint or a completed run), load it and skip "
                         "past the examples it already processed instead of starting fresh from the "
                         "base model. Relies on --seed producing the exact same example order as "
                         "the original run (data is now pre-shuffled once with Python's random, not "
                         "torch's global RNG, specifically so this is reproducible across fresh "
                         "processes). Optimizer momentum is NOT resumed (AdamW state resets) -- "
                         "only model weights and the LR schedule position; a minor, acceptable "
                         "simplification next to losing the whole run.")
    args = p.parse_args()

    torch.manual_seed(args.seed)
    device = "cpu"
    if torch.cuda.is_available():
        device = f"cuda:{args.gpu_index}" if args.gpu_index is not None else "cuda"
        if args.gpu_index is not None:
            torch.cuda.set_device(args.gpu_index)

    records = data_io.load_jsonl(args.train_file)
    if args.smoke_test:
        records = records[:20]
        args.epochs = 1
    print(f"Loaded {len(records)} training records (device={device})")

    examples = build_examples(records)
    print(f"Built {len(examples)} usable conversations")

    # Shuffle once, deterministically, with Python's own RNG rather than relying on
    # torch's global RNG + DataLoader(shuffle=True) -- the latter's exact permutation
    # depends on how many prior draws happened in *this* process, which a resume in a
    # fresh process can't reliably reproduce. random.Random(seed) depends on nothing
    # but the seed, so --resume can correctly recompute "which examples come next"
    # after a restart. The loader below uses shuffle=False since this is already shuffled.
    random.Random(args.seed).shuffle(examples)

    output_dir = Path(args.output_dir)
    resume_state_path = output_dir / "train_state.json"
    examples_done = 0
    resume_global_step = 0
    if args.resume and resume_state_path.exists():
        resume_state = json.loads(resume_state_path.read_text())
        examples_done = resume_state["examples_done"]
        resume_global_step = resume_state["global_step"]
        print(f"--resume: found {examples_done} examples already done, "
              f"global_step={resume_global_step} -- will skip ahead to that point.")

    from transformers import AutoProcessor
    processor = AutoProcessor.from_pretrained(
        args.model_id, min_pixels=args.min_pixels, max_pixels=args.max_pixels
    )
    collate = make_collate_fn(processor)

    if args.dry_run:
        loader = torch.utils.data.DataLoader(VLMSpanDataset(examples[:5]), batch_size=1, collate_fn=collate)
        for batch in loader:
            print({k: (tuple(v.shape) if hasattr(v, "shape") else v) for k, v in batch.items()})
        print("Dry run OK (data pipeline verified, model not loaded).")
        return

    from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
    from transformers import BitsAndBytesConfig

    model_class = resolve_model_class(args.model_id)
    if args.target_modules is None:
        target_override = None
    elif args.target_modules == "all-linear":
        target_override = "all-linear"
    else:
        target_override = args.target_modules.split(",")
    target_modules = resolve_target_modules(args.model_id, override=target_override)

    dtype = best_dtype(device)
    quant_config = None
    if args.load_in_4bit:
        quant_config = BitsAndBytesConfig(
            load_in_4bit=True, bnb_4bit_compute_dtype=dtype,
            bnb_4bit_quant_type="nf4", bnb_4bit_use_double_quant=True,
        )

    print(f"Loading {args.model_id} via {model_class.__name__} (4bit={args.load_in_4bit}, dtype={dtype})...")
    device_map = {"": device} if device != "cpu" else None
    model = model_class.from_pretrained(
        args.model_id, quantization_config=quant_config, dtype=dtype, device_map=device_map,
    )
    if args.load_in_4bit:
        model = prepare_model_for_kbit_training(
            model, use_gradient_checkpointing=args.gradient_checkpointing
        )
    elif args.gradient_checkpointing:
        model.gradient_checkpointing_enable()
        model.enable_input_require_grads()  # needed for grad checkpointing through a frozen base model

    if args.resume and (output_dir / "adapter_config.json").exists():
        from peft import PeftModel
        model = PeftModel.from_pretrained(model, str(output_dir), is_trainable=True)
        print(f"--resume: loaded LoRA adapter weights from {output_dir} (skipping fresh LoRA init).")
    else:
        print(f"LoRA target_modules: {target_modules}")
        lora_config = LoraConfig(
            r=args.lora_r, lora_alpha=args.lora_alpha, lora_dropout=0.0, bias="none",
            target_modules=target_modules,
            task_type="CAUSAL_LM",
        )
        model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()

    loader = torch.utils.data.DataLoader(VLMSpanDataset(examples), batch_size=1, shuffle=False, collate_fn=collate)

    optimizer = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=args.lr, weight_decay=0.01)
    total_steps = max(1, (len(loader) // args.grad_accum) * args.epochs)

    def lr_lambda(step):
        if step < args.warmup_steps:
            return step / max(1, args.warmup_steps)
        progress = (step - args.warmup_steps) / max(1, total_steps - args.warmup_steps)
        return max(0.0, 1.0 - progress)

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
    # AdamW momentum is NOT restored on resume (see --resume help) -- only fast-forward the
    # LR schedule position so the learning rate picks up where it left off, not back at warmup.
    for _ in range(resume_global_step):
        scheduler.step()

    model.train()
    global_step = resume_global_step
    start_time = time.time()
    max_optim_steps = 2 if args.smoke_test else None
    stop = False

    from tqdm.auto import tqdm

    print(f"Total: {args.epochs} epoch(s) x {len(loader)} examples, "
          f"{total_steps} optimizer steps expected. Progress bar below shows "
          f"live it/s and ETA -- multiply by your actual observed rate, not "
          f"any estimate given in chat, for real wall-clock planning.")

    output_dir.mkdir(parents=True, exist_ok=True)

    def save_checkpoint(cumulative_index, mark_complete=False):
        model.save_pretrained(output_dir)
        processor.save_pretrained(output_dir)
        with open(output_dir / "train_config.json", "w") as f:
            json.dump(vars(args), f, indent=2)
        resume_state_path.write_text(json.dumps({
            "examples_done": cumulative_index,
            "global_step": global_step,
            "complete": mark_complete,
        }))

    for epoch in range(args.epochs):
        optimizer.zero_grad()
        pbar = tqdm(loader, desc=f"epoch {epoch}/{args.epochs - 1}", unit="ex", dynamic_ncols=True)
        for step, batch in enumerate(pbar):
            cumulative_index = epoch * len(loader) + step
            if cumulative_index < examples_done:
                # Already processed in a prior (crashed/cut-off) session -- skip. Still iterated
                # (not sliced out upfront) so epoch numbering / len(loader) stay simple; the cost
                # is just re-running collate_fn (CPU-side image/text prep) for skipped items, not
                # any GPU compute, since we continue before the model() call below.
                continue

            seq_len = batch["input_ids"].shape[1]
            if seq_len > args.max_seq_len:
                pbar.write(f"  skipping step {step}: sequence length {seq_len} exceeds "
                           f"--max_seq_len {args.max_seq_len}")
                continue
            batch = {k: (v.to(device) if hasattr(v, "to") else v) for k, v in batch.items()}
            outputs = model(**batch)
            loss = outputs.loss / args.grad_accum
            loss.backward()

            if (step + 1) % args.grad_accum == 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()
                global_step += 1

                if args.save_steps > 0 and global_step % args.save_steps == 0:
                    save_checkpoint(cumulative_index + 1)
                    pbar.write(f"  checkpoint saved at global_step={global_step}, "
                               f"examples_done={cumulative_index + 1} -> {output_dir}")

            elapsed = time.time() - start_time
            pbar.set_postfix({
                "loss": f"{loss.item() * args.grad_accum:.4f}",
                "opt_step": f"{global_step}/{total_steps}",
                "elapsed": f"{elapsed / 60:.1f}m",
            })

            if max_optim_steps is not None and global_step >= max_optim_steps:
                stop = True
                break
        if stop:
            break

    save_checkpoint(len(examples) * args.epochs, mark_complete=True)
    print(f"Saved LoRA adapter -> {output_dir}")


if __name__ == "__main__":
    main()
