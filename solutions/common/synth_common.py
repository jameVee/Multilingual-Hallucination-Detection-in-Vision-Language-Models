"""Shared synthetic-data-generation primitives, used by
solutions/{03_vlm_lora_synthetic_augmented,04_vlm_lora_synthetic_augmented_llm}/generate_synthetic.py
(solutions 05/06 import that same module rather than duplicating it).

Kept in common/ (rather than duplicated per-solution, as the 6
last_experiment/*.ipynb notebooks did) since these are generic, model- and
language-agnostic building blocks: certainty/rarity-aware upsampling, and
text-corruption primitives that a solution-specific script grounds in real
signal (e.g. actual OCR-detected image text) before applying.
"""
from __future__ import annotations

import random
import re
import sys
from pathlib import Path

_THIS_DIR = Path(__file__).resolve().parent
_REPO_ROOT = _THIS_DIR.parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from solutions.common import data_io

# ---------------------------------------------------------------------------
# Certainty / rarity aware upsampling (S4_AgreementUpsampling successor)
# ---------------------------------------------------------------------------


def certainty_tier(record: dict) -> int:
    """2 = has a unanimous-agreement span (prob>=1.0), 1 = has a majority
    span (prob>=0.667), 0 = only low-agreement spans or none at all."""
    probs = [lb["prob"] for lb in record.get("labels", []) or []]
    if any(p >= 1.0 for p in probs):
        return 2
    if any(p >= 0.667 for p in probs):
        return 1
    return 0


def compute_rarity_multipliers(records: list, label_order=data_io.LABEL_ORDER, max_multiplier: int = 8) -> dict:
    """Per-label multiplier sized off actual current example counts in
    `records`, so the rarest classes (OCR, other) get boosted harder than a
    fixed constant would. Multiplier = round(majority_count / this_count),
    capped at `max_multiplier` to avoid extreme duplication of a handful of
    examples (verbatim duplication saturates rather than helps generalization
    past some point).
    """
    counts = data_io.class_counts(records)["example_counts"]
    if not counts:
        return {label: 1 for label in label_order}
    majority = max(counts.get(label, 0) for label in label_order)
    multipliers = {}
    for label in label_order:
        n = counts.get(label, 0)
        if n == 0:
            multipliers[label] = 1
            continue
        raw = majority / n
        multipliers[label] = max(1, min(max_multiplier, round(raw)))
    return multipliers


def duplicate_record(record: dict, tag: str) -> dict:
    new_record = dict(record)
    new_record["id"] = f"{record['id']}__{tag}"
    return new_record


def upsample_by_tier_and_rarity(records: list, upsample_high: int = 4, upsample_med: int = 2,
                                 rarity_multipliers=None, label_order=data_io.LABEL_ORDER,
                                 max_multiplier: int = 8, seed: int = 42) -> list:
    """Duplicate real examples according to two independent signals:

    1. Annotator-agreement tier (S4's original idea): tier-2 (unanimous)
       examples duplicated `upsample_high`x, tier-1 (majority) `upsample_med`x,
       tier-0/clean kept at 1x.
    2. Per-label rarity (new): an example containing >=1 span of a rare
       class gets an *additional* multiplier on top of its tier multiplier,
       sized by `compute_rarity_multipliers` (or a caller-supplied dict).

    The two multipliers combine multiplicatively per example (using the
    example's rarest present label), then are capped by `max_multiplier` so
    a single tiny-class example can't dominate the resulting dataset. No
    model or API involved -- pure, deterministic, reproducible duplication.
    """
    rng = random.Random(seed)
    if rarity_multipliers is None:
        rarity_multipliers = compute_rarity_multipliers(records, label_order)

    out = list(records)
    for record in records:
        tier = certainty_tier(record)
        tier_mult = {0: 1, 1: upsample_med, 2: upsample_high}[tier]

        labels_present = {lb["label"] for lb in record.get("labels", []) or []}
        rarity_mult = max((rarity_multipliers.get(lbl, 1) for lbl in labels_present), default=1)

        total_mult = min(max_multiplier, tier_mult * rarity_mult)
        for i in range(total_mult - 1):
            out.append(duplicate_record(record, f"up{i}_{rng.randint(0, 1_000_000)}"))
    return out


# ---------------------------------------------------------------------------
# Text-corruption primitives (grounded by the caller in real signal, e.g.
# actual EasyOCR-detected text, before being applied to a clean response)
# ---------------------------------------------------------------------------

OCR_CONFUSABLE_MAP = {
    "O": "0", "0": "O", "o": "0",
    "l": "1", "1": "l", "I": "l", "i": "l",
    "S": "5", "5": "S", "s": "5",
    "Z": "2", "2": "Z",
    "B": "8", "8": "B",
    "G": "6", "6": "G",
    "b": "6", "q": "9", "9": "q",
    "rn": "m", "m": "rn",
    "cl": "d",
    "vv": "w",
}

# Real gold OCR spans, measured directly from data/train.jsonl (n=1284), are
# ~74% alpha-only / 16% numeric-only / 10% mixed -- a VLM misreading text in
# an image usually still outputs something that reads as a different
# plausible WORD ("TAMBA", "logotipo"), not a word visibly contaminated with
# digit lookalikes mid-string. An earlier version of this function picked
# any swappable character uniformly at random, which (measured against a
# real EasyOCR-grounded sample) produced 64% "mixed" alpha+digit text vs the
# real data's 10% -- letter-for-letter confusions that stay alphabetic
# (rn/m, cl/d, vv/w, I/l) are now preferred, falling back to digit-
# introducing swaps (O/0, S/5, ...) only with low probability or when no
# alpha-preserving swap exists in the text, to track that split instead of
# defaulting to "leetspeak"-style corruption.
_ALPHA_PRESERVING_TARGETS = {"I", "l", "m", "rn", "d", "w"}
DIGIT_SWAP_FALLBACK_PROB = 0.08

_NUMBER_RE = re.compile(r"\d+")


def confusable_swap(text: str, rng: random.Random, n_swaps: int = 1):
    """Apply up to `n_swaps` confusable-character substitutions to `text`.
    Returns the corrupted text, or None if no swappable character exists.
    Used to turn a real, correctly-read piece of OCR text into a plausible
    misreading (e.g. grounded in what EasyOCR actually detected in the
    paired image, so the "correct" text anchor is real, not invented).
    """
    chars = list(text)
    swappable = [i for i, c in enumerate(chars) if c in OCR_CONFUSABLE_MAP]
    if not swappable:
        return None

    alpha_positions = [i for i in swappable if OCR_CONFUSABLE_MAP[chars[i]] in _ALPHA_PRESERVING_TARGETS]
    digit_positions = [i for i in swappable if i not in set(alpha_positions)]

    ordered = []
    while alpha_positions or digit_positions:
        use_alpha = alpha_positions and (not digit_positions or rng.random() > DIGIT_SWAP_FALLBACK_PROB)
        pool = alpha_positions if use_alpha else digit_positions
        ordered.append(pool.pop(rng.randrange(len(pool))))

    changed = False
    for idx in ordered[:n_swaps]:
        chars[idx] = OCR_CONFUSABLE_MAP[chars[idx]]
        changed = True
    return "".join(chars) if changed else None


def shift_number(text: str, rng: random.Random):
    """Find a numeric-digit mention in `text` and shift it by a small amount.

    Offset magnitude is drawn from a fixed prior favoring small errors (+/-1
    most common, +/-2 less common) -- the released data marks *which* spans
    are hallucinated but not the original correct value, so this is a
    reasonable modeling assumption about typical counting-error magnitude,
    not something empirically fit from the gold spans themselves.

    Digit-only: measured directly against real gold miscounting spans
    (data/train.jsonl, n=2747), only 21% contain any digit at all -- 79%
    are spelled-out number words ("two", "trois", "四"). This function is
    the minority-case handler; see `shift_number_word` for the majority
    case, and prefer trying that one first.
    Returns (corrupted_text, match_start, match_end) or None.
    """
    matches = list(_NUMBER_RE.finditer(text))
    if not matches:
        return None
    match = rng.choice(matches)
    original = int(match.group())
    offset = rng.choices([-2, -1, 1, 2], weights=[1, 3, 3, 1])[0]
    new_value = max(0, original + offset)
    if new_value == original:
        new_value = original + 1
    new_text = text[:match.start()] + str(new_value) + text[match.end():]
    return new_text, match.start(), match.start() + len(str(new_value))


# Spelled-out number words, 0-20 (covers the vast majority of real counting
# mentions -- "how many X" answers rarely go higher). zh only covers 0-20 via
# the simple/compound Han numerals actually used for counting (not the
# formal/financial numeral variants) plus 两 as the common alternate "two"
# used before a measure word.
NUMBER_WORDS = {
    "en": ["zero", "one", "two", "three", "four", "five", "six", "seven", "eight", "nine", "ten",
           "eleven", "twelve", "thirteen", "fourteen", "fifteen", "sixteen", "seventeen", "eighteen",
           "nineteen", "twenty"],
    "fr": ["zéro", "un", "deux", "trois", "quatre", "cinq", "six", "sept", "huit", "neuf", "dix",
           "onze", "douze", "treize", "quatorze", "quinze", "seize", "dix-sept", "dix-huit",
           "dix-neuf", "vingt"],
    "it": ["zero", "uno", "due", "tre", "quattro", "cinque", "sei", "sette", "otto", "nove", "dieci",
           "undici", "dodici", "tredici", "quattordici", "quindici", "sedici", "diciassette",
           "diciotto", "diciannove", "venti"],
    "zh": ["零", "一", "二", "三", "四", "五", "六", "七", "八", "九", "十",
           "十一", "十二", "十三", "十四", "十五", "十六", "十七", "十八", "十九", "二十"],
}
NUMBER_WORD_ALIASES = {
    "fr": {"une": 1},
    "it": {"una": 1},
    "zh": {"两": 2},
}


def _number_word_lookup(lang: str) -> dict:
    words = NUMBER_WORDS.get(lang, NUMBER_WORDS["en"])
    lookup = {w: i for i, w in enumerate(words)}
    lookup.update(NUMBER_WORD_ALIASES.get(lang, {}))
    return lookup


def shift_number_word(text: str, lang: str, rng: random.Random):
    """Find a spelled-out number word (0-20) in `text`, in the response's own
    language, and shift it to a different number word in the same language.
    This is the majority-case counterpart to `shift_number` -- try this
    first, fall back to the digit version only if no word-number is found.
    Returns (corrupted_text, start, end) or None.
    """
    lookup = _number_word_lookup(lang)
    if not lookup:
        return None

    if lang == "zh":
        # No word boundaries in Chinese; match by substring, longest key
        # first so "十一" isn't mistakenly matched as "十" + "一".
        candidates = []
        for word in sorted(lookup.keys(), key=len, reverse=True):
            start = 0
            while True:
                idx = text.find(word, start)
                if idx == -1:
                    break
                candidates.append((idx, idx + len(word), word))
                start = idx + 1
        if not candidates:
            return None
        start, end, word = rng.choice(candidates)
        original = lookup[word]
    else:
        matches = list(re.finditer(r"[A-Za-zÀ-ÖØ-öø-ÿ\-]+", text))
        candidates = [m for m in matches if m.group().lower() in lookup]
        if not candidates:
            return None
        match = rng.choice(candidates)
        start, end = match.start(), match.end()
        original = lookup[match.group().lower()]

    words_for_lang = NUMBER_WORDS.get(lang, NUMBER_WORDS["en"])
    offset = rng.choices([-2, -1, 1, 2], weights=[1, 3, 3, 1])[0]
    new_value = max(0, min(len(words_for_lang) - 1, original + offset))
    if new_value == original:
        new_value = min(len(words_for_lang) - 1, original + 1)
    replacement = words_for_lang[new_value]
    if lang != "zh" and text[start:end][0].isupper():
        replacement = replacement.capitalize()

    new_text = text[:start] + replacement + text[end:]
    return new_text, start, start + len(replacement)


# Two separate antonym maps, kept apart because they ground different gold
# label categories: a wrong color/size/material/texture on something that IS
# visible is mischaracterization, while a wrong state/direction/relation is
# closest to the catch-all "other" category (mirrors how docs/2026.md itself
# separates "incorrect description of visible content" from "doesn't fit
# classes A-D").
#
# Per-language, not just English: an earlier version had one English-only
# dict applied via an `[A-Za-z]+` word regex to all 4 languages' responses.
# Measured against the actual clean-seed pool (data/train.jsonl), that fired
# on 14.8% of English clean seeds but only 3-6% of fr/it/zh ones (occasional
# English loanwords) -- i.e. this injector barely ever ran for 3 of the 4
# task languages. zh entries are matched by substring (no word boundaries),
# same approach as `shift_number_word`.
VISUAL_ATTRIBUTE_ANTONYMS = {
    "en": {
        "red": "blue", "blue": "red", "green": "purple", "yellow": "orange",
        "black": "white", "white": "black", "large": "small", "small": "large",
        "big": "tiny", "tiny": "big", "round": "square", "square": "round",
        "wooden": "metal", "metal": "wooden", "wet": "dry", "dry": "wet",
    },
    "fr": {
        "rouge": "bleu", "bleu": "rouge", "vert": "violet", "jaune": "orange",
        "noir": "blanc", "blanc": "noir", "grand": "petit", "petit": "grand",
        "rond": "carré", "carré": "rond", "mouillé": "sec", "sec": "mouillé",
    },
    "it": {
        "rosso": "blu", "blu": "rosso", "verde": "viola", "giallo": "arancione",
        "nero": "bianco", "bianco": "nero", "grande": "piccolo", "piccolo": "grande",
        "rotondo": "quadrato", "quadrato": "rotondo", "bagnato": "asciutto", "asciutto": "bagnato",
    },
    "zh": {
        "红": "蓝", "蓝": "红", "绿": "紫", "黄": "橙", "黑": "白", "白": "黑",
        "大": "小", "小": "大", "圆": "方", "方": "圆", "湿": "干", "干": "湿",
    },
}

STATE_DIRECTION_ANTONYMS = {
    "en": {
        "left": "right", "right": "left", "open": "closed", "closed": "open",
        "old": "new", "new": "old", "empty": "full", "full": "empty",
        "before": "after", "after": "before", "front": "back", "back": "front",
    },
    "fr": {
        "gauche": "droite", "droite": "gauche", "ouvert": "fermé", "fermé": "ouvert",
        "vieux": "neuf", "neuf": "vieux", "vide": "plein", "plein": "vide",
        "avant": "après", "après": "avant",
    },
    "it": {
        "sinistra": "destra", "destra": "sinistra", "aperto": "chiuso", "chiuso": "aperto",
        "vecchio": "nuovo", "nuovo": "vecchio", "vuoto": "pieno", "pieno": "vuoto",
        "prima": "dopo", "dopo": "prima",
    },
    "zh": {
        "左": "右", "右": "左", "开": "关", "关": "开", "旧": "新", "新": "旧",
        "空": "满", "满": "空", "前": "后", "后": "前",
    },
}


def _swap_word(text: str, mapping_by_lang: dict, rng: random.Random, lang: str = "en"):
    mapping = mapping_by_lang.get(lang) or mapping_by_lang.get("en", {})
    if not mapping:
        return None

    if lang == "zh":
        candidates = []
        for key in sorted(mapping.keys(), key=len, reverse=True):
            start = 0
            while True:
                idx = text.find(key, start)
                if idx == -1:
                    break
                candidates.append((idx, idx + len(key), key))
                start = idx + 1
        if not candidates:
            return None
        start, end, key = rng.choice(candidates)
        replacement = mapping[key]
        new_text = text[:start] + replacement + text[end:]
        return new_text, start, start + len(replacement)

    words = list(re.finditer(r"[A-Za-zÀ-ÖØ-öø-ÿ]+", text))
    candidates = [m for m in words if m.group().lower() in mapping]
    if not candidates:
        return None
    match = rng.choice(candidates)
    replacement = mapping[match.group().lower()]
    if match.group()[0].isupper():
        replacement = replacement.capitalize()
    new_text = text[:match.start()] + replacement + text[match.end():]
    return new_text, match.start(), match.start() + len(replacement)


def swap_visual_attribute(text: str, rng: random.Random, lang: str = "en"):
    """Find a color/size/material/texture word (in the response's own
    language) and replace it with a plausible-but-wrong antonym -> grounds a
    synthetic `mischaracterization` span. Returns (corrupted_text, start, end)
    or None."""
    return _swap_word(text, VISUAL_ATTRIBUTE_ANTONYMS, rng, lang)


def swap_state_word(text: str, rng: random.Random, lang: str = "en"):
    """Find a state/direction/relation word (in the response's own language)
    and replace it with a plausible-but-wrong antonym -> grounds a synthetic
    `other` span (doesn't fit invention/mischaracterization/OCR/miscounting).
    Returns (corrupted_text, start, end) or None."""
    return _swap_word(text, STATE_DIRECTION_ANTONYMS, rng, lang)


def inject_repetition_loop(text: str, rng: random.Random, min_repeats: int = 6, max_repeats: int = 40):
    """Corrupts `text` by picking a short chunk (1-3 words) and repeating it
    many times, appended near the end of the response.

    Grounded in a real, quantified (not just eyeballed) pattern in gold
    `other` spans: checking data/train.jsonl directly, only ~4% of real
    `other` spans show a clear repeated-chunk pattern (e.g. "25 ans d'age, "
    repeated 50+ times, "1647" repeated dozens of times) -- a real
    generation-degeneration failure mode, but a minority one, not the
    dominant pattern (the majority of real `other` spans are a heterogeneous
    mix -- truncated word fragments, odd word choices, foreign loanwords --
    with no single recoverable structural signature). This is offered as
    *one* contributor to `other` synthetic data alongside
    `swap_state_word`, not a replacement for it, and should not be expected
    to characterize most of the class.
    Returns (corrupted_text, start, end) or None.
    """
    words = [w for w in text.split(" ") if w]
    if len(words) < 4:
        return None
    chunk_len = rng.choice([1, 1, 2, 2, 3])
    start_word = rng.randrange(0, max(1, len(words) - chunk_len))
    chunk = " ".join(words[start_word:start_word + chunk_len]).strip()
    if not chunk:
        return None
    n_repeats = rng.randint(min_repeats, max_repeats)
    repeated_blob = (chunk + " ") * n_repeats
    new_text = text.rstrip() + " " + repeated_blob.strip()
    start = len(text.rstrip()) + 1
    end = len(new_text)
    return new_text, start, end


def make_synthetic_record(base_record: dict, new_response: str, span_start: int, span_end: int,
                           label: str, tag: str, prob: float = 1.0) -> dict:
    """Build a full synthetic record (same schema as data/train.jsonl) from a
    corrupted response and the single injected span's char offsets."""
    return {
        "id": f"synth_{tag}_{base_record['id']}",
        "split": base_record.get("split", "train"),
        "language": base_record.get("language", ""),
        "prompt": base_record["prompt"],
        "image_name": base_record["image_name"],
        "response": new_response,
        "labels": [{"start": span_start, "end": span_end, "prob": prob, "label": label}],
    }
