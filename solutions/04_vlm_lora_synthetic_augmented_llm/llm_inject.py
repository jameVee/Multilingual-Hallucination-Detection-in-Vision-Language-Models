"""Optional LLM-assisted hallucination injection (generate + validate),
adapted from last_experiment/S1b_LLMInjection but fixed to:
  - never hardcode an API key -- reads ANTHROPIC_API_KEY from the
    environment (set that on the VM yourself; never paste a key into chat
    or commit one)
  - target data/train.jsonl's actual clean examples (the old notebook read
    a different, no-longer-present per-language Drive file layout)
  - cache incrementally to disk so a partial/interrupted run doesn't
    re-spend API budget on already-processed seeds

Gated entirely on ANTHROPIC_API_KEY being set: generate_synthetic.py calls
`generate_llm_synthetic` unconditionally, and this module itself no-ops
(zero API calls) if the key is absent -- methods (a) rarity/certainty
upsampling and (b) grounded heuristic injection never depend on this.
"""
from __future__ import annotations

import base64
import json
import os
import re
import sys
import time
from collections import Counter
from pathlib import Path

from tqdm.auto import tqdm

_THIS_DIR = Path(__file__).resolve().parent
_REPO_ROOT = _THIS_DIR.parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from solutions.common import data_io

CLAUDE_MODEL = "claude-sonnet-4-6"
TARGET_CATEGORIES = ["OCR", "other", "miscounting"]

INJECTION_DESC = {
    "OCR": "misread or mistranscribe text that is visibly written in the image "
           "(e.g. swap a letter, add an extra character, or confuse similar-looking characters)",
    "other": "introduce a subtle factual error that does not fit invention, wrong count, "
             "OCR error, or wrong attribute -- e.g. wrong material, wrong action, wrong relationship",
    "miscounting": "state a wrong quantity for something clearly countable in the image "
                   "(change count by exactly 1 or 2)",
}


def _encode_image(img_path: Path):
    ext = img_path.suffix.lower().lstrip(".")
    media_type = {"jpg": "image/jpeg", "jpeg": "image/jpeg", "png": "image/png",
                  "webp": "image/webp", "gif": "image/gif"}.get(ext, "image/jpeg")
    with open(img_path, "rb") as f:
        return base64.standard_b64encode(f.read()).decode(), media_type


def inject_with_llm(client, record: dict, target_label: str, image_dir=data_io.IMAGE_DIR):
    """Step 1: ask Claude to inject exactly one hallucination of
    `target_label` into a clean response. Returns a candidate record dict,
    or None on any failure (unresolvable image, malformed model output,
    span text not found in the modified response)."""
    img_path = data_io.resolve_image_path(record["image_name"], image_dir)
    if img_path is None:
        return None
    try:
        img_b64, media_type = _encode_image(img_path)
    except Exception:
        return None

    prompt = (
        "You are building hallucination-detection training data.\n\n"
        f"Task: Rewrite the VLM response below to inject EXACTLY ONE hallucination of "
        f"type '{target_label}'.\n"
        f"Definition: {INJECTION_DESC[target_label]}\n\n"
        "Rules:\n"
        "1. Make ONLY ONE factual change -- keep everything else identical\n"
        "2. The injected content must be clearly wrong given the image\n"
        "3. Keep the response natural and similar in length\n"
        "4. The 'text' field must be the EXACT substring in modified_response\n"
        "5. Return ONLY valid JSON with no markdown fences\n\n"
        f"Question: {record['prompt']}\n"
        f"Original response: {record['response']}\n\n"
        "Return this JSON:\n"
        '{"modified_response": "...", '
        f'"injected_span": {{"text": "...", "label": "{target_label}", "prob": 1.0}}}}'
    )

    try:
        resp = client.messages.create(
            model=CLAUDE_MODEL, max_tokens=1024,
            messages=[{"role": "user", "content": [
                {"type": "image", "source": {"type": "base64", "media_type": media_type, "data": img_b64}},
                {"type": "text", "text": prompt},
            ]}],
        )
        raw = resp.content[0].text.strip()
        match = re.search(r"\{.*\}", raw, re.DOTALL)
        if not match:
            return None
        data = json.loads(match.group())
        mod_response = str(data.get("modified_response", "")).strip()
        span_info = data.get("injected_span", {})
        span_text = str(span_info.get("text", "")).strip()
        if not mod_response or not span_text:
            return None
        pos = mod_response.find(span_text)
        if pos == -1:
            pos = mod_response.lower().find(span_text.lower())
        if pos == -1:
            return None
        return {
            "id": f"synth_llm_{target_label}_{record['id']}",
            "split": record.get("split", "train"),
            "language": record.get("language", ""),
            "prompt": record["prompt"],
            "image_name": record["image_name"],
            "response": mod_response,
            "labels": [{"start": pos, "end": pos + len(span_text), "prob": 1.0, "label": target_label}],
            "_span_text": span_text,
        }
    except Exception:
        return None


def validate_with_llm(client, candidate: dict, image_dir=data_io.IMAGE_DIR) -> bool:
    """Step 2: ask Claude to confirm the injected span is genuinely wrong
    given the image, catching injections that happen to be accidentally
    correct (keeps synthetic prob=1.0 labels actually trustworthy)."""
    img_path = data_io.resolve_image_path(candidate["image_name"], image_dir)
    if img_path is None:
        return False
    try:
        img_b64, media_type = _encode_image(img_path)
    except Exception:
        return False

    span_text = candidate.get("_span_text", "")
    if not span_text:
        return False

    prompt = (
        "Examine the image carefully.\n\n"
        f'The following claim was made about this image:\n"{span_text}"\n\n'
        "Is this claim FACTUALLY INCORRECT based on what you can actually see in the image?\n"
        "Answer with exactly one word: YES (it is wrong / a hallucination) or NO (it is accurate)."
    )
    try:
        resp = client.messages.create(
            model=CLAUDE_MODEL, max_tokens=5,
            messages=[{"role": "user", "content": [
                {"type": "image", "source": {"type": "base64", "media_type": media_type, "data": img_b64}},
                {"type": "text", "text": prompt},
            ]}],
        )
        return resp.content[0].text.strip().upper().startswith("YES")
    except Exception:
        return False


def generate_llm_synthetic(records, cache_path, max_per_category=100, max_per_language=300,
                            validate=True, rate_limit_sleep=0.5, languages=("en", "fr", "it", "zh")):
    """Generate (or resume generating) LLM-injected synthetic samples for the
    rarest classes. Safe to re-run (skips ids already cached). Returns the
    full list of accepted synthetic records, or an empty list immediately
    (zero API calls) if ANTHROPIC_API_KEY is not set.
    """
    api_key = os.environ.get("ANTHROPIC_API_KEY", "")
    if not api_key:
        print("  ANTHROPIC_API_KEY not set -- skipping LLM-assisted injection (methods a+b still run).")
        return []

    import anthropic

    client = anthropic.Anthropic(api_key=api_key)
    cache_path = Path(cache_path)
    cache = {lang: [] for lang in languages}
    if cache_path.exists():
        cache.update(json.loads(cache_path.read_text(encoding="utf-8")))
        print(f"  Loaded existing LLM-injection cache: {sum(len(v) for v in cache.values())} samples")

    def save_cache():
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        cache_path.write_text(json.dumps(cache, ensure_ascii=False, indent=2), encoding="utf-8")

    by_lang = {lang: [r for r in records if r.get("language") == lang and not r.get("labels")]
               for lang in languages}

    total_calls = 0
    for lang in languages:
        clean_seeds = by_lang.get(lang, [])
        existing_ids = {s["id"] for s in cache.get(lang, [])}
        cat_counts = Counter(s["labels"][0]["label"] for s in cache.get(lang, []) if s.get("labels"))
        print(f"  [{lang}] {len(clean_seeds)} clean seeds, {len(existing_ids)} already cached")

        for target_label in TARGET_CATEGORIES:
            need = max_per_category - cat_counts.get(target_label, 0)
            if need <= 0:
                continue
            seeds = [s for s in clean_seeds if f"synth_llm_{target_label}_{s['id']}" not in existing_ids]
            accepted = 0
            pbar = tqdm(total=need, desc=f"{lang}/{target_label}", unit="accepted", dynamic_ncols=True)
            for seed in seeds:
                if accepted >= need or len(cache.get(lang, [])) >= max_per_language:
                    break
                candidate = inject_with_llm(client, seed, target_label)
                total_calls += 1
                time.sleep(rate_limit_sleep)
                if candidate is None:
                    continue
                if validate:
                    ok = validate_with_llm(client, candidate)
                    total_calls += 1
                    time.sleep(rate_limit_sleep)
                    if not ok:
                        continue
                candidate.pop("_span_text", None)
                cache.setdefault(lang, []).append(candidate)
                accepted += 1
                pbar.update(1)
                pbar.set_postfix({"api_calls": total_calls})
                if accepted % 20 == 0:
                    save_cache()
            pbar.close()
            save_cache()
            print(f"    {target_label}: +{accepted} accepted")

    print(f"  Total API calls: {total_calls}")
    return [rec for lang_records in cache.values() for rec in lang_records]
