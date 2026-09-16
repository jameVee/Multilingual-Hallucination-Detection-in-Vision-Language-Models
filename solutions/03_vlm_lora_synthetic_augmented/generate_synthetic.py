"""Combined synthetic-data generator for solution 03: builds one augmented
train set out of three complementary methods, stacked on top of the real
data/train.jsonl:

  (a) Rarity + certainty aware upsampling (S4_AgreementUpsampling successor)
      -- pure duplication, no model/API, always runs.
  (b) Grounded heuristic injection -- corrupts clean responses using signal
      actually grounded in the paired image (real EasyOCR-detected text for
      the OCR class) or in the response's own existing content (a digit
      already mentioned, a color/size word already present) rather than
      S1a's ungrounded dictionary swaps. Always runs (no model/API).
  (c) Optional LLM-assisted generate+validate injection for the rarest
      classes, gated entirely on ANTHROPIC_API_KEY being set (skips cleanly
      if absent -- see llm_inject.py). This solution's run_pipeline.py
      always passes --skip_llm regardless, so method (c) never actually
      runs from this folder -- see solution 04 for the LLM-assisted variant.

This module is shared verbatim with solution 04 (identical (a)/(b)/(c)
logic and the same r_max=8 default rarity cap) -- the two solutions differ
only in whether run_pipeline.py passes --skip_llm by default: solution 03
always skips method (c) (the missing r_max=8-without-LLM cell of the
augmentation ablation), solution 04 always includes it.

Usage:
    python generate_synthetic.py --output ./train_augmented.jsonl
    python generate_synthetic.py --smoke_test --skip_llm
"""
from __future__ import annotations

import argparse
import random
import re
import sys
from pathlib import Path

_THIS_DIR = Path(__file__).resolve().parent
_REPO_ROOT = _THIS_DIR.parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))
if str(_THIS_DIR) not in sys.path:
    sys.path.insert(0, str(_THIS_DIR))

from solutions.common import data_io
from solutions.common import synth_common as sc
from ocr_extract import extract_ocr_cache, load_ocr_cache
from llm_inject import generate_llm_synthetic

OCR_SENTENCE_TEMPLATES = {
    "en": ' The text visible in the image reads "{}".',
    "fr": ' Le texte visible dans l’image indique « {} ».',
    "it": ' Il testo visibile nell’immagine recita "{}".',
    "zh": ' 图中可见的文字是“{}”。',
}


def make_ocr_injection(record, ocr_cache, rng):
    texts = ocr_cache.get(record["image_name"])
    if not texts:
        return None
    source_text = rng.choice(texts)
    # n_swaps=1 regardless of text length: a single localized misreading, not
    # smeared across the whole string. Measured directly against real gold
    # OCR spans (data/train.jsonl) -- scaling swaps with length compounded
    # confusable_swap's (already alpha-biased) per-swap digit-fallback
    # probability across multiple swaps, pushing "mixed" alpha+digit text
    # well above the real ~10% rate even after that bias was added.
    corrupted = sc.confusable_swap(source_text, rng, n_swaps=1)
    if corrupted is None or corrupted == source_text:
        return None
    template = OCR_SENTENCE_TEMPLATES.get(record.get("language", "en"), OCR_SENTENCE_TEMPLATES["en"])
    appended = template.format(corrupted)
    new_response = record["response"] + appended
    start = len(record["response"]) + appended.index(corrupted)
    end = start + len(corrupted)
    return sc.make_synthetic_record(record, new_response, start, end, "OCR", "ocrgrnd")


def make_miscounting_injection(record, rng):
    lang = record.get("language", "en")
    # Try the spelled-out-word form first: measured directly against real
    # gold miscounting spans (data/train.jsonl, n=2747), 79% contain no
    # digit at all -- they're number words ("two", "trois", "四"), which an
    # earlier digit-only version of this injector could never produce or
    # even fire on for the large majority of realistic seed responses.
    result = sc.shift_number_word(record["response"], lang, rng)
    tag = "cntword"
    if result is None:
        result = sc.shift_number(record["response"], rng)
        tag = "cntdigit"
    if result is None:
        return None
    new_response, start, end = result
    return sc.make_synthetic_record(record, new_response, start, end, "miscounting", tag)


def make_mischaracterization_injection(record, rng):
    lang = record.get("language", "en")
    result = sc.swap_visual_attribute(record["response"], rng, lang=lang)
    if result is None:
        return None
    new_response, start, end = result
    return sc.make_synthetic_record(record, new_response, start, end, "mischaracterization", "attrswap")


def make_other_injection(record, rng):
    lang = record.get("language", "en")
    # Two contributors, not one: measured directly against real gold `other`
    # spans, only ~4% show a clear repeated-chunk pattern (a real but minor
    # generation-degeneration failure mode) -- the majority is a
    # heterogeneous mix with no single recoverable structural signature, so
    # this class doesn't have one dominant grounded method the way OCR/
    # miscounting do. Weighted toward the (translated) state/direction swap
    # since it's the better-understood contributor; repetition-loop adds
    # some of the real minority pattern rather than being ignored entirely.
    if rng.random() < 0.75:
        result = sc.swap_state_word(record["response"], rng, lang=lang)
        tag = "stateswap"
        if result is None:
            result = sc.inject_repetition_loop(record["response"], rng)
            tag = "reployop"
    else:
        result = sc.inject_repetition_loop(record["response"], rng)
        tag = "reployop"
        if result is None:
            result = sc.swap_state_word(record["response"], rng, lang=lang)
            tag = "stateswap"
    if result is None:
        return None
    new_response, start, end = result
    return sc.make_synthetic_record(record, new_response, start, end, "other", tag)


INVENTION_TEMPLATES = {
    "en": [" There also appears to be a small {obj} nearby.", " A {obj} can also be seen in the background."],
    "fr": [" On peut aussi apercevoir {obj} à proximité."],
    "it": [" Si può anche notare {obj} nelle vicinanze."],
    "zh": [" 附近似乎还有{obj}。"],
}
# Per-language object phrases (kept 1:1 across languages by index) so the
# injected addendum stays in a single language -- an earlier version reused
# the English noun list for every language, which produced mixed-language
# sentences (e.g. a Chinese response ending in "...trash can。") that no
# real hallucination would ever look like.
INVENTION_OBJECTS = {
    "en": ["a small bird", "a red bicycle", "a wooden bench", "a street lamp", "a potted plant", "a trash can"],
    "fr": ["un petit oiseau", "un vélo rouge", "un banc en bois", "un lampadaire", "une plante en pot", "une poubelle"],
    "it": ["un piccolo uccello", "una bicicletta rossa", "una panchina di legno", "un lampione", "una pianta in vaso", "un cestino"],
    "zh": ["一只小鸟", "一辆红色自行车", "一张木凳", "一盏路灯", "一盆植物", "一个垃圾桶"],
}


_SENTENCE_BOUNDARY_RE = re.compile(r"(?<=[.!?。！？])\s+")


def _sentence_boundaries(text: str) -> list:
    """Character offsets right after each sentence-ending punctuation mark
    (Latin . ! ? or full-width 。！？), i.e. valid insertion points that
    don't split a sentence mid-thought. Always includes 0 (before anything)
    and len(text) (append) as fallback insertion points."""
    boundaries = [0]
    for m in _SENTENCE_BOUNDARY_RE.finditer(text):
        boundaries.append(m.start())
    boundaries.append(len(text))
    return sorted(set(boundaries))


def make_invention_injection(record, rng):
    # Invention is, by definition, content absent from the image -- there is
    # no real signal to ground it in (unlike OCR/miscounting/
    # mischaracterization above). A short templated addendum is the
    # appropriate synthesis strategy for this one class specifically.
    #
    # Position: measured directly against real gold invention spans
    # (data/train.jsonl, n=14473), only 14% land at/near the response's end
    # -- position is roughly uniform across the response (37%/35%/28% early/
    # mid/late thirds). An earlier version always appended at the very end
    # (100% late by construction), which doesn't match. Now inserted at a
    # randomly chosen sentence boundary instead of only ever the last one.
    lang = record.get("language", "en")
    if lang not in INVENTION_TEMPLATES:
        lang = "en"
    template = rng.choice(INVENTION_TEMPLATES[lang])
    obj = rng.choice(INVENTION_OBJECTS[lang])
    addendum = template.format(obj=obj)

    response = record["response"]
    insert_at = rng.choice(_sentence_boundaries(response))
    new_response = response[:insert_at] + addendum.strip() + " " + response[insert_at:]
    start = insert_at
    end = insert_at + len(addendum.strip())
    return sc.make_synthetic_record(record, new_response, start, end, "invention", "objadd")


INJECTORS = {
    "OCR": lambda rec, ocr_cache, rng: make_ocr_injection(rec, ocr_cache, rng),
    "miscounting": lambda rec, ocr_cache, rng: make_miscounting_injection(rec, rng),
    "mischaracterization": lambda rec, ocr_cache, rng: make_mischaracterization_injection(rec, rng),
    "other": lambda rec, ocr_cache, rng: make_other_injection(rec, rng),
    "invention": lambda rec, ocr_cache, rng: make_invention_injection(rec, rng),
}


def generate_heuristic_injections(records, ocr_cache, rarity_multipliers, base_per_label=150, seed=42):
    """For each label, inject synthetic examples up to roughly
    base_per_label x that label's rarity multiplier, drawing from clean
    seeds. Each injector's corruption primitive requires specific content to
    fire (a detectable OCR string, a number, a color word, ...), so most
    attempts on unrelated seeds simply return None and are skipped --
    throughput is naturally bounded by how much real, groundable signal
    exists, rather than fabricating content to hit a target count."""
    rng = random.Random(seed)
    clean_seeds = [r for r in records if not r.get("labels")]
    rng.shuffle(clean_seeds)

    out = []
    for label in data_io.LABEL_ORDER:
        target_n = base_per_label * rarity_multipliers.get(label, 1)
        injector = INJECTORS[label]
        made = 0
        for seed_record in clean_seeds:
            if made >= target_n:
                break
            synthetic = injector(seed_record, ocr_cache, rng)
            if synthetic is not None:
                out.append(synthetic)
                made += 1
        print(f"  heuristic injection <{label}>: {made}/{target_n} generated")
    return out


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--train_file", default=str(data_io.TRAIN_FILE))
    p.add_argument("--ocr_cache", default=str(_THIS_DIR / "ocr_cache.json"))
    p.add_argument("--llm_cache", default=str(_THIS_DIR / "llm_synth_cache.json"))
    p.add_argument("--output", default=str(_THIS_DIR / "train_augmented.jsonl"))
    p.add_argument("--skip_ocr_extract", action="store_true",
                    help="Reuse an existing ocr_cache.json instead of re-running EasyOCR.")
    p.add_argument("--skip_llm", action="store_true", help="Skip method (c) even if ANTHROPIC_API_KEY is set.")
    p.add_argument("--smoke_test", action="store_true")
    p.add_argument("--seed", type=int, default=42)
    args = p.parse_args()

    records = data_io.load_jsonl(args.train_file)
    if args.smoke_test:
        records = records[:200]
    print(f"Loaded {len(records)} real training records")

    ocr_cache_path = Path(args.ocr_cache)
    if ocr_cache_path.exists():
        print(f"Loading existing OCR cache: {ocr_cache_path}")
        ocr_cache = load_ocr_cache(ocr_cache_path)
    elif not args.skip_ocr_extract:
        print("Running EasyOCR over training images (one-time)...")
        image_names = {r["image_name"] for r in records}
        if args.smoke_test:
            image_names = set(list(image_names)[:15])
        ocr_cache = extract_ocr_cache(image_names, output_path=ocr_cache_path, gpu=True)
    else:
        ocr_cache = {}

    print("\n=== (a) Rarity + certainty aware upsampling ===")
    rarity_multipliers = sc.compute_rarity_multipliers(records)
    print(f"  multipliers: {rarity_multipliers}")
    upsampled = sc.upsample_by_tier_and_rarity(records, rarity_multipliers=rarity_multipliers, seed=args.seed)
    print(f"  {len(records)} -> {len(upsampled)} after upsampling")

    print("\n=== (b) Grounded heuristic injection ===")
    heuristic_synth = generate_heuristic_injections(records, ocr_cache, rarity_multipliers, seed=args.seed)
    print(f"  generated {len(heuristic_synth)} heuristic synthetic examples")

    llm_synth = []
    if not args.skip_llm:
        print("\n=== (c) Optional LLM-assisted injection ===")
        llm_synth = generate_llm_synthetic(records, args.llm_cache)
        print(f"  generated {len(llm_synth)} LLM-assisted synthetic examples")

    combined = upsampled + heuristic_synth + llm_synth
    random.Random(args.seed).shuffle(combined)
    data_io.write_jsonl(args.output, combined)

    before = data_io.class_counts(records)
    after = data_io.class_counts(combined)
    print(f"\nFinal augmented train set: {len(combined)} examples "
          f"(real={len(records)}, +upsampled={len(upsampled) - len(records)}, "
          f"+heuristic={len(heuristic_synth)}, +llm={len(llm_synth)})")
    print(f"  example counts before: {before['example_counts']}")
    print(f"  example counts after : {after['example_counts']}")
    print(f"Saved -> {args.output}")


if __name__ == "__main__":
    main()
