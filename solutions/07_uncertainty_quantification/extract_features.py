"""Aim 3, Experiments 1-3: extract frozen-model uncertainty signals for
every response token, from a plain (unmodified) Qwen3-VL-8B-Instruct --
NO fine-tuning, NO LoRA. This is the fine-tuning-free counterpart to
solutions 02-06 (thesis proposal Appendix A):

  Exp 1 (token-level entropy/perplexity): per response token, the entropy
    of the model's own next-token distribution and the negative log-prob
    it assigned the actual observed token, when reading the response as
    its own teacher-forced continuation of (image, question).
  Exp 2 (frozen hidden-state linear probe features): the residual-stream
    hidden state at each response token, at one or more chosen decoder
    layers -- the probe itself is trained separately, in train_probes.py.
  Exp 3 (attention lookback-ratio probe features): per response token,
    per attention head, per layer, the fraction of that token's attention
    mass spent on IMAGE tokens vs. on PRIOR RESPONSE tokens (this turn's
    own previously-generated tokens) -- adapts Lookback Lens's context-
    vs-generation ratio to an image-vs-prior-response ratio, motivated by
    OPERA's finding that multimodal hallucination correlates with
    attention drifting away from image tokens.

Unlike solutions 02-06, the model here is NOT asked to emit a hallucination
span JSON -- it is simply given (image, question) and asked to answer, with
the REAL response substituted in as if it were the model's own answer
(teacher forcing), so the resulting uncertainty signals reflect "how
surprised is a plain VQA model by this exact text", not "how well can this
model list spans".

Memory: attention weights are only ever materialized one layer at a time,
via forward hooks that immediately reduce each layer's raw (1, heads,
seq_len, seq_len) tensor down to a (n_response_tokens, heads) ratio and
then REPLACE the module's returned attention tensor with None before it
propagates to the top-level model call (a hook's return value replaces a
module's output -- see torch.nn.Module.register_forward_hook). Without
this, requesting output_attentions=True across every decoder layer for a
long sequence risks holding all layers' full seq_len^2 tensors
simultaneously, which is exactly the OOM risk flagged in the thesis
proposal's Phase 2 contingency plan. This is that contingency, implemented
up front rather than discovered by an OOM.

**UNVERIFIED ASSUMPTIONS -- run --dry_run on nlp2 before trusting a real
run.** This was written without access to a downloaded Qwen3-VL checkpoint
(no GPU with enough VRAM locally, and this repo's transformers install is
currently broken -- see README.md). Specifically unverified:
  - The image-token id/placeholder detection in `find_image_token_id()`.
  - Every attention submodule actually following the standard HF
    `(attn_output, attn_weights_or_None)` return-tuple convention.
  - That isolated-substring tokenization of `record["response"]` produces
    the exact same token ids as the response region of the full
    multimodal input sequence (`verify_alignment()` checks this per
    record at runtime and falls back to an approximate alignment,
    flagging it, rather than crashing).

Usage:
    python extract_features.py --dry_run --gpu_index 1
    python extract_features.py --split train --shard_id 0 --num_shards 3 --gpu_index 1
    python extract_features.py --split val --gpu_index 1
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
from PIL import Image

_THIS_DIR = Path(__file__).resolve().parent
_REPO_ROOT = _THIS_DIR.parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from solutions.common import data_io

DEFAULT_MODEL_ID = "Qwen/Qwen3-VL-8B-Instruct"
CLEAN_LABEL_ID = -1  # a response token not covered by any gold hallucination span


def best_dtype(device: str) -> torch.dtype:
    """Same rule as solutions/02's train_lora.py: bf16 on Ampere+ (nlp2's
    3090s, compute capability 8.6), fp16 otherwise."""
    if device == "cpu" or not torch.cuda.is_available():
        return torch.float32
    idx = int(device.split(":")[1]) if ":" in device else torch.cuda.current_device()
    major, _ = torch.cuda.get_device_capability(idx)
    return torch.bfloat16 if major >= 8 else torch.float16


def resolve_model_class(model_id: str):
    """Same logic as solutions/02's train_lora.py -- duplicated (not
    imported) since that module also does LoRA-specific setup this script
    doesn't need; both are small enough that duplication beats coupling
    two independently-run solutions together."""
    from transformers import AutoModelForImageTextToText

    lower = model_id.lower()
    if "qwen3-vl" in lower or "qwen3vl" in lower:
        if "a3b" in lower or "moe" in lower:
            from transformers import Qwen3VLMoeForConditionalGeneration
            return Qwen3VLMoeForConditionalGeneration
        from transformers import Qwen3VLForConditionalGeneration
        return Qwen3VLForConditionalGeneration
    return AutoModelForImageTextToText


def find_image_token_id(model, processor):
    """Best-effort discovery of the token id used for expanded image
    patches in input_ids. Qwen2-VL/Qwen2.5-VL/Qwen3-VL's convention (as of
    their model cards/configs at the time this was written) exposes this
    directly as `config.image_token_id`; this also tries the processor and
    a couple of literal special-token strings as fallbacks. Prints what it
    found either way -- ALWAYS check this output under --dry_run; a wrong
    id here silently breaks Experiment 3's image-vs-response distinction
    without erroring.
    """
    for holder, attr in [
        (getattr(model, "config", None), "image_token_id"),
        (getattr(model, "config", None), "image_token_index"),
        (processor, "image_token_id"),
    ]:
        if holder is not None and getattr(holder, attr, None) is not None:
            return int(getattr(holder, attr))
    tok = getattr(processor, "tokenizer", processor)
    for candidate in ("<|image_pad|>", "<|vision_pad|>", "<image>"):
        try:
            tid = tok.convert_tokens_to_ids(candidate)
            if tid is not None and tid != tok.unk_token_id:
                return int(tid)
        except Exception:
            continue
    raise RuntimeError(
        "Could not determine the image-token id from the model/processor. "
        "Inspect `model.config` and `processor` manually on nlp2 and hardcode "
        "it via --image_token_id."
    )


def build_plain_conversation(record: dict, image_dir=data_io.IMAGE_DIR):
    """Plain VQA framing -- NOT the span-finding instruction used by
    solutions 02-06. The model is just asked the original question and
    given the real (possibly hallucinated) response as if it were its own
    answer, so its own uncertainty about that exact text is the signal of
    interest."""
    img_path = data_io.resolve_image_path(record["image_name"], image_dir)
    if img_path is None:
        return None
    messages = [
        {"role": "user", "content": [
            {"type": "image", "image": str(img_path)},
            {"type": "text", "text": record["prompt"]},
        ]},
        {"role": "assistant", "content": [{"type": "text", "text": record["response"]}]},
    ]
    return {"messages": messages, "_image_path": str(img_path), "_record": record}


def token_char_spans_from_ids(tokenizer, token_ids: list, reference_text: str):
    """Recovers per-token character offsets into `reference_text` by
    decoding growing prefixes of `token_ids` -- the ACTUAL ids the model
    was fed (from the real multimodal sequence), not a re-tokenization of
    `reference_text` in isolation.

    `token_ids` is everything from the end of the prompt to the end of the
    tokenized sequence, which is a SUPERSET of the response content: it
    also includes the chat template's end-of-turn closing tokens (e.g.
    `<|im_end|>\\n`), confirmed on a real run against Qwen3-VL-8B-
    Instruct's tokenizer on nlp2 -- decoding the full id list for a short
    Chinese response came back as `"...吉他。<|im_end|>\\n"` when
    `reference_text` was just `"...吉他。"`. An earlier version required
    the full decode to equal `reference_text` exactly, which this trailing
    suffix always fails, and (separately) tokenizing `reference_text` in
    isolation and comparing ITS ids against the real ones, which failed a
    different way (a BPE boundary effect at the very first token). Both
    failures showed up as `approx_alignment=True` on every single sample
    record in real testing.

    This version instead: (1) tolerates the response's first real token
    absorbing a one-character leading-space boundary (same as before), and
    (2) finds exactly how many LEADING tokens are needed to reproduce
    `reference_text` and drops everything after that as closing-tag/
    structural tokens -- they were never part of the model's answer and
    have no business getting an entropy/hidden-state/attention-ratio
    feature row or a hallucination label at all.

    Returns (offsets, None, n_real_tokens) on success -- `offsets` has
    exactly `n_real_tokens` entries, and the caller should trim
    `token_ids` (and every per-token array derived from the same
    positions) down to `token_ids[:n_real_tokens]`. Returns
    (None, reconstructed_text, None) if the decoded text doesn't reconcile
    with `reference_text` at all (caller falls back to an approximate
    proportional split over the FULL token_ids in that case, unchanged).
    """
    if not token_ids:
        return [], None, 0
    full_decoded = tokenizer.decode(token_ids, skip_special_tokens=False)
    shift = 0
    if full_decoded[:len(reference_text)] == reference_text:
        shift = 0
    elif full_decoded[:1].isspace() and full_decoded[1:1 + len(reference_text)] == reference_text:
        shift = 1
    else:
        return None, full_decoded, None

    target_len = shift + len(reference_text)
    offsets = []
    prev_len = 0
    n_real_tokens = len(token_ids)
    for i in range(1, len(token_ids) + 1):
        cur_len_raw = len(tokenizer.decode(token_ids[:i], skip_special_tokens=False))
        cur_len = min(cur_len_raw, target_len) - shift
        offsets.append((max(0, prev_len), max(0, cur_len)))
        prev_len = cur_len
        if cur_len_raw >= target_len:
            n_real_tokens = i
            break
    return offsets, None, n_real_tokens


def label_response_tokens(offsets, labels, label_order=data_io.LABEL_ORDER):
    """offsets: list of (start,end) char spans, one per response token
    (from token_char_spans). labels: gold hallucination spans (char
    offsets into the same response text). Returns a list of ints, one per
    token: CLEAN_LABEL_ID if not covered by any gold span, else the
    label's index in label_order. A token overlapping >=1 char of a gold
    span counts as covered by it; if multiple spans overlap the same
    token (rare -- spans are mostly non-overlapping in this dataset), the
    first one found wins."""
    label_ids = [CLEAN_LABEL_ID] * len(offsets)
    for i, (start, end) in enumerate(offsets):
        if start == end:
            continue
        for lb in labels:
            if start < lb["end"] and end > lb["start"]:
                label_ids[i] = label_order.index(lb["label"])
                break
    return label_ids


class AttentionLookbackCollector:
    """Registers forward hooks on every decoder self-attention module,
    reduces each one's raw attention tensor to a per-response-token,
    per-head image-vs-prior-response ratio, and replaces the module's
    returned attention tensor with None so the top-level model call never
    accumulates the full seq_len^2 tensors across layers.

    Filters to modules whose qualified name contains "language_model" (or,
    if that yields nothing, any name NOT containing "visual"/"vision") --
    the goal is decoder self-attention only, not the vision tower's own
    internal attention. **Verify this filter's module-name list under
    --dry_run before trusting it on nlp2** -- it's a name-based heuristic,
    not something this was able to test against a real Qwen3-VL module
    tree.
    """

    def __init__(self, model, prompt_len: int, image_col_mask: torch.Tensor, resp_len: int, verbose=False):
        self.prompt_len = prompt_len
        self.resp_len = resp_len  # cap query rows to the REAL response tokens, excluding chat-template closing tags
        self.image_col_mask = image_col_mask  # bool tensor, shape (seq_len,), True at image-token columns
        self.per_layer_ratios = {}  # layer_name -> (n_response_tokens, n_heads) float32 CPU tensor
        self.handles = []
        self.verbose = verbose
        self._install(model)

    def _install(self, model):
        candidates = [(n, m) for n, m in model.named_modules()
                      if "attention" in type(m).__name__.lower() and "language_model" in n]
        if not candidates:
            candidates = [(n, m) for n, m in model.named_modules()
                          if "attention" in type(m).__name__.lower()
                          and "visual" not in n and "vision" not in n]
        if self.verbose:
            print(f"  [AttentionLookbackCollector] hooking {len(candidates)} modules:")
            for n, m in candidates:
                print(f"    {n} ({type(m).__name__})")
        for name, module in candidates:
            self.handles.append(module.register_forward_hook(self._make_hook(name)))

    def _make_hook(self, name):
        def hook(module, inputs, output):
            if not (isinstance(output, tuple) and len(output) >= 2 and output[1] is not None):
                return output
            attn_weights = output[1]  # (batch=1, heads, seq_len, seq_len) or (batch=1, seq_len, seq_len)
            if attn_weights.dim() == 3:
                attn_weights = attn_weights.unsqueeze(1)  # -> (1, 1, seq_len, seq_len)
            seq_len = attn_weights.shape[-1]
            row_end = self.prompt_len + self.resp_len  # exclude chat-template closing-tag rows entirely
            resp_rows = attn_weights[0, :, self.prompt_len:row_end, :]  # (heads, n_resp, seq_len)
            image_mask = self.image_col_mask[:seq_len].to(attn_weights.device)

            image_mass = (resp_rows * image_mask).sum(dim=-1)  # (heads, n_resp)
            # "prior response tokens" = response positions strictly before the query itself.
            # Vectorized via a strict lower-triangular mask over the response x response
            # submatrix instead of a per-token Python loop -- the loop version issued one
            # tiny GPU kernel per response token per layer (e.g. ~300 tokens x ~36 layers =
            # ~10,800 launches per record), which is latency- not compute-bound and was the
            # single biggest avoidable cost in this hook.
            n_resp = resp_rows.shape[1]
            resp_cols = resp_rows[:, :, self.prompt_len:self.prompt_len + n_resp]  # (heads, n_resp, n_resp)
            strict_lower = torch.tril(
                torch.ones(n_resp, n_resp, dtype=torch.bool, device=resp_cols.device), diagonal=-1
            )
            resp_prior_mass = (resp_cols * strict_lower).sum(dim=-1)  # (heads, n_resp)
            denom = image_mass + resp_prior_mass
            ratio = torch.where(denom > 1e-8, image_mass / denom, torch.zeros_like(denom))
            self.per_layer_ratios[name] = ratio.transpose(0, 1).to(torch.float32).cpu()  # (n_resp, heads)
            return (output[0], None) + tuple(output[2:])
        return hook

    def stacked_features(self):
        """(n_response_tokens, n_layers * n_heads) float32 array, columns
        ordered by sorted layer name then head index -- column order is
        saved alongside the features (see `extract.py`'s output metadata)
        so train_probes.py doesn't need to guess it."""
        names = sorted(self.per_layer_ratios)
        if not names:
            return np.zeros((0, 0), dtype=np.float32), []
        arrs = [self.per_layer_ratios[n].numpy() for n in names]
        n_resp = arrs[0].shape[0]
        assert all(a.shape[0] == n_resp for a in arrs), "response-token count mismatch across layers"
        stacked = np.concatenate(arrs, axis=1)
        col_names = [f"{n}#head{h}" for n in names for h in range(self.per_layer_ratios[n].shape[1])]
        return stacked, col_names

    def remove(self):
        for h in self.handles:
            h.remove()


class HiddenStateCollector:
    """Hooks a small, explicit set of decoder-layer modules (by index) and
    keeps only the response-token rows of each one's output hidden state,
    instead of requesting output_hidden_states=True for the whole model
    (which retains every layer at once)."""

    def __init__(self, model, layer_indices: list, prompt_len: int, resp_len: int, verbose=False):
        self.prompt_len = prompt_len
        self.resp_len = resp_len  # cap to REAL response tokens, excluding chat-template closing tags
        self.captured = {}  # layer_idx -> (n_resp, hidden) float32 CPU tensor
        self.handles = []
        decoder_layers = [(n, m) for n, m in model.named_modules()
                           if type(m).__name__.endswith("DecoderLayer")]
        if verbose:
            print(f"  [HiddenStateCollector] found {len(decoder_layers)} decoder layers total; "
                  f"hooking indices {layer_indices}")
        for requested_idx in layer_indices:
            # `captured` is keyed by the ORIGINAL requested index (e.g. -1), matching how
            # every caller (process_record, main's hidden_layers/hidden_shard/save_kwargs)
            # refers to it -- resolving negative indices only to look up the right module,
            # NOT to relabel the key, since a prior version reassigned the loop variable
            # itself and silently stored under the resolved positive index (e.g. 35)
            # instead, so every `.get(-1, ...)` lookup missed and fell back to a bare
            # numpy zeros() placeholder that then crashed on the later `.numpy()` call
            # (caught by an actual --dry_run run on nlp2, not by anything testable locally).
            resolved_idx = requested_idx if requested_idx >= 0 else len(decoder_layers) + requested_idx
            name, module = decoder_layers[resolved_idx]
            self.handles.append(module.register_forward_hook(self._make_hook(requested_idx)))

    def _make_hook(self, idx):
        def hook(module, inputs, output):
            hidden = output[0] if isinstance(output, tuple) else output  # (1, seq_len, hidden)
            row_end = self.prompt_len + self.resp_len
            self.captured[idx] = hidden[0, self.prompt_len:row_end, :].detach().to(torch.float32).cpu()
            return output
        return hook

    def remove(self):
        for h in self.handles:
            h.remove()


def process_record(record, model, processor, device, image_token_id, hidden_layers, verbose=False):
    conv = build_plain_conversation(record)
    if conv is None:
        return None
    messages = conv["messages"]
    user_only = [messages[0]]
    image = Image.open(conv["_image_path"]).convert("RGB")

    full_text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=False)
    prompt_text = processor.apply_chat_template(user_only, tokenize=False, add_generation_prompt=True)

    full_inputs = processor(text=[full_text], images=[image], padding=True, return_tensors="pt")
    prompt_inputs = processor(text=[prompt_text], images=[image], padding=True, return_tensors="pt")
    prompt_len = prompt_inputs["input_ids"].shape[1]
    seq_len = full_inputs["input_ids"].shape[1]
    if seq_len <= prompt_len:
        return None  # empty response after tokenization, nothing to score

    input_ids_cpu = full_inputs["input_ids"][0]
    # Superset of the real response: also includes the chat template's trailing
    # end-of-turn closing tokens (e.g. `<|im_end|>\n`) -- token_char_spans_from_ids
    # figures out exactly where the real content ends within this.
    response_ids_superset = input_ids_cpu[prompt_len:].tolist()

    offsets, mismatch_text, n_real_tokens = token_char_spans_from_ids(
        processor.tokenizer, response_ids_superset, record["response"]
    )
    approx_alignment = offsets is None
    if approx_alignment:
        # Degraded fallback: spread offsets evenly across the response length in
        # proportion to token count, over the FULL superset (can't confidently
        # identify the closing-tag boundary here), rather than crashing. Flagged
        # per-record (see `approx_alignment` in the saved shard) so this can be
        # filtered out of probe training later if it turns out to be common.
        if verbose:
            print(f"  [alignment] decode mismatch for {record['id']}: "
                  f"decoded={mismatch_text!r} vs response={record['response']!r}")
        n_real_tokens = len(response_ids_superset)
        resp_char_len = len(record["response"])
        step = resp_char_len / max(1, n_real_tokens)
        offsets = [(int(i * step), int((i + 1) * step)) for i in range(n_real_tokens)]

    label_ids = label_response_tokens(offsets, record.get("labels", []) or [])

    image_col_mask = (input_ids_cpu[:prompt_len] == image_token_id)
    image_col_mask = torch.cat([image_col_mask, torch.zeros(seq_len - prompt_len, dtype=torch.bool)])

    attn_collector = AttentionLookbackCollector(model, prompt_len, image_col_mask, n_real_tokens, verbose=verbose)
    hidden_collector = HiddenStateCollector(model, hidden_layers, prompt_len, n_real_tokens, verbose=verbose)
    try:
        batch = {k: (v.to(device) if hasattr(v, "to") else v) for k, v in full_inputs.items()}
        with torch.no_grad():
            out = model(**batch, output_attentions=True, use_cache=False)
        resp_end = prompt_len + n_real_tokens  # excludes chat-template closing-tag positions
        logits = out.logits[0, prompt_len - 1:resp_end - 1, :].to(torch.float32)  # predicts tokens [prompt_len:resp_end)
        log_probs = torch.log_softmax(logits, dim=-1)
        probs = log_probs.exp()
        entropy = (-(probs * log_probs).sum(dim=-1)).cpu().numpy()
        target_ids = input_ids_cpu[prompt_len:resp_end]
        nll = (-log_probs.gather(1, target_ids.unsqueeze(1).to(log_probs.device))).squeeze(1).cpu().numpy()
        del out, logits, log_probs, probs
    finally:
        attn_collector.remove()
        hidden_collector.remove()

    lookback_feats, lookback_cols = attn_collector.stacked_features()
    n_resp = n_real_tokens

    # Defensive fallback (should not trigger now that HiddenStateCollector keys `captured`
    # by the original requested index -- see its docstring): if a requested layer's hook
    # never fired for some other reason, use an empty-but-correctly-typed torch tensor
    # instead of a bare numpy array, so the `.numpy()` call below can't crash on it either way.
    hidden_feats = {idx: hidden_collector.captured.get(idx, torch.zeros((n_resp, 0), dtype=torch.float32))
                    for idx in hidden_layers}

    return {
        "n_tokens": n_resp,
        "entropy": entropy.astype(np.float32),
        "nll": nll.astype(np.float32),
        "lookback": lookback_feats.astype(np.float32),
        "lookback_columns": lookback_cols,
        "hidden": {str(k): v.numpy() for k, v in hidden_feats.items()},
        "label_id": np.array(label_ids, dtype=np.int8),
        "language": record.get("language", ""),
        "record_id": record["id"],
        "approx_alignment": approx_alignment,
    }


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--split", choices=["train", "val"], default="val")
    p.add_argument("--model_id", default=DEFAULT_MODEL_ID)
    p.add_argument("--gpu_index", type=int, default=None)
    p.add_argument("--shard_id", type=int, default=0)
    p.add_argument("--num_shards", type=int, default=1)
    p.add_argument("--max_examples", type=int, default=None)
    p.add_argument("--hidden_layers", default="-1", help="Comma-separated decoder-layer indices "
                    "(supports negative indexing, e.g. -1 for the last layer) to save hidden states from.")
    p.add_argument("--image_token_id", type=int, default=None,
                    help="Override auto-detection (see find_image_token_id's docstring).")
    p.add_argument("--max_pixels", type=int, default=1024 * 28 * 28)
    p.add_argument("--min_pixels", type=int, default=256 * 28 * 28)
    p.add_argument("--output_dir", default=str(_THIS_DIR / "features"))
    p.add_argument("--dry_run", action="store_true",
                    help="Load the model, run 3 examples, print every unverified-assumption "
                         "diagnostic (image_token_id, hooked module names, alignment check), "
                         "save nothing. Run this on nlp2 before any real extraction job.")
    p.add_argument("--verbose", action="store_true")
    args = p.parse_args()

    hidden_layers = [int(x) for x in args.hidden_layers.split(",")]

    device = "cpu"
    if torch.cuda.is_available():
        device = f"cuda:{args.gpu_index}" if args.gpu_index is not None else "cuda"
        if args.gpu_index is not None:
            torch.cuda.set_device(args.gpu_index)

    from transformers import AutoProcessor
    processor = AutoProcessor.from_pretrained(
        args.model_id, min_pixels=args.min_pixels, max_pixels=args.max_pixels
    )
    model_class = resolve_model_class(args.model_id)
    dtype = best_dtype(device)
    print(f"Loading {args.model_id} via {model_class.__name__} (dtype={dtype}, attn_implementation=eager)...")
    model = model_class.from_pretrained(
        args.model_id, dtype=dtype, device_map={"": device} if device != "cpu" else None,
        attn_implementation="eager",  # required so attention modules actually compute+return weights
    )
    model.eval()

    image_token_id = args.image_token_id or find_image_token_id(model, processor)
    print(f"image_token_id = {image_token_id} "
          f"(decoded: {processor.tokenizer.convert_ids_to_tokens([image_token_id])})")

    path = data_io.TRAIN_FILE if args.split == "train" else data_io.VAL_FILE
    records = data_io.load_jsonl(path)
    records = [r for r in records if r.get("labels")]  # skip clean-only records: no positive tokens possible
    records = records[args.shard_id::args.num_shards]
    if args.max_examples:
        records = records[:args.max_examples]
    if args.dry_run:
        records = records[:3]
    print(f"Processing {len(records)} records from {args.split} (shard {args.shard_id}/{args.num_shards})")

    shard_data = {"entropy": [], "nll": [], "label_id": [], "language": [], "record_id": [],
                  "approx_alignment": [], "lookback": []}
    hidden_shard = {idx: [] for idx in hidden_layers}
    lookback_columns = None

    from tqdm.auto import tqdm
    n_approx = 0
    for record in tqdm(records, desc=f"extract[{args.split}]", unit="ex"):
        result = process_record(record, model, processor, device, image_token_id, hidden_layers,
                                 verbose=args.dry_run)
        if result is None:
            continue
        if args.dry_run:
            print(f"  id={result['record_id']} lang={result['language']} n_tokens={result['n_tokens']} "
                  f"approx_alignment={result['approx_alignment']} "
                  f"entropy[0:3]={result['entropy'][:3]} nll[0:3]={result['nll'][:3]} "
                  f"lookback_shape={result['lookback'].shape} "
                  f"n_positive_tokens={(result['label_id'] != CLEAN_LABEL_ID).sum()}")
            continue

        if lookback_columns is None:
            lookback_columns = result["lookback_columns"]
        n_approx += int(result["approx_alignment"])
        shard_data["entropy"].append(result["entropy"])
        shard_data["nll"].append(result["nll"])
        shard_data["label_id"].append(result["label_id"])
        shard_data["language"].extend([result["language"]] * result["n_tokens"])
        shard_data["record_id"].extend([result["record_id"]] * result["n_tokens"])
        shard_data["approx_alignment"].extend([result["approx_alignment"]] * result["n_tokens"])
        shard_data["lookback"].append(result["lookback"])
        for idx in hidden_layers:
            hidden_shard[idx].append(result["hidden"][str(idx)])

    if args.dry_run:
        print("\n--dry_run complete. Nothing saved. Review the diagnostics above -- "
              "especially image_token_id, hooked module names (rerun with --verbose "
              "if not shown), and whether approx_alignment=True for any record -- "
              "before running a real extraction job.")
        return

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    out_path = output_dir / f"{args.split}_shard{args.shard_id}of{args.num_shards}.npz"
    save_kwargs = {
        "entropy": np.concatenate(shard_data["entropy"]) if shard_data["entropy"] else np.zeros(0, np.float32),
        "nll": np.concatenate(shard_data["nll"]) if shard_data["nll"] else np.zeros(0, np.float32),
        "label_id": np.concatenate(shard_data["label_id"]) if shard_data["label_id"] else np.zeros(0, np.int8),
        "language": np.array(shard_data["language"], dtype=object),
        "record_id": np.array(shard_data["record_id"], dtype=object),
        "approx_alignment": np.array(shard_data["approx_alignment"], dtype=bool),
        "lookback": np.concatenate(shard_data["lookback"], axis=0) if shard_data["lookback"] else np.zeros((0, 0), np.float32),
        "lookback_columns": np.array(lookback_columns or [], dtype=object),
    }
    for idx in hidden_layers:
        save_kwargs[f"hidden_{idx}"] = (np.concatenate(hidden_shard[idx], axis=0)
                                         if hidden_shard[idx] else np.zeros((0, 0), np.float32))
    np.savez_compressed(out_path, **save_kwargs)
    print(f"Saved {save_kwargs['entropy'].shape[0]} response tokens "
          f"({n_approx} records used approx_alignment) -> {out_path}")


if __name__ == "__main__":
    main()
