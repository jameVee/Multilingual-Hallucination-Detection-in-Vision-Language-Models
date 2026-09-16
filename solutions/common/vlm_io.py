"""Shared VLM conversation-building and output-parsing utilities.

Used by 02_vlm_lora_real_data and every synthetic-augmentation solution
(03-06) so the instruction prompt, target format, and span-text ->
char-offset realignment logic stay identical between the "real data only"
run and all the "+ synthetic" variants -- the only thing that should differ
between those solutions is the training data, not the task framing.

The parsing/realignment logic here is a cleaned-up, model-agnostic version of
the boilerplate that was duplicated across all 6 last_experiment/*.ipynb
notebooks (resolve_image_path/build_conversation/parse_model_output/
find_span_pos/map_spans_to_positions), consolidated into one module instead
of being copy-pasted per notebook.
"""
from __future__ import annotations

import difflib
import json
import re
import sys
from pathlib import Path

_THIS_DIR = Path(__file__).resolve().parent
_REPO_ROOT = _THIS_DIR.parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from solutions.common import data_io

INSTRUCTION = """Identify all hallucinated text spans in the vision model's response.
A hallucination is text that is factually incorrect compared to what is actually visible in the image.

Categories:
- invention: entities or facts completely absent from the image
- mischaracterization: visible content described incorrectly
- OCR: text visible in the image is misread or mistranscribed
- miscounting: wrong quantity of visible items
- other: hallucination not fitting the above

List spans in the order they appear in the response, left to right.

Return ONLY a JSON array, no markdown fences, no extra text:
[{"text": "<exact text copied verbatim from the response>", "prob": <0.0-1.0>, "label": "<category>"}]
If there are no hallucinations, return: []"""

LABEL_NORM = {
    "mischaracterisation": "mischaracterization",
    "ocr_problem": "OCR",
    "ocr problem": "OCR",
    "ocr": "OCR",
    "fabrication": "invention",
    "invented": "invention",
    "miscount": "miscounting",
}


def normalize_label(raw_label: str) -> str:
    key = str(raw_label).strip()
    if key in data_io.VALID_LABELS:
        return key
    normalized = LABEL_NORM.get(key.lower().strip())
    if normalized is not None:
        return normalized
    return "other"


def build_target_spans(record: dict) -> list:
    """Gold `labels` (char offsets) -> the text-span JSON the model should emit."""
    response = record["response"]
    spans = []
    for lb in sorted(record.get("labels", []) or [], key=lambda x: x["start"]):
        text = response[lb["start"]:lb["end"]]
        if text.strip():
            spans.append({"text": text, "prob": round(float(lb["prob"]), 4), "label": lb["label"]})
    return spans


def build_conversation(record: dict, image_dir=data_io.IMAGE_DIR, include_target=True):
    """Build a chat-format conversation dict for one record.

    Returns None if the referenced image can't be resolved on disk.
    `messages` uses the generic {"role", "content": [{"type": "image"/"text", ...}]}
    shape that transformers' processor.apply_chat_template expects for
    Qwen2-VL-style models; adapt in the model-loading code if a different
    VLM's chat template needs different keys.
    """
    img_path = data_io.resolve_image_path(record["image_name"], image_dir)
    if img_path is None:
        return None
    response = record["response"]
    user_text = f"{INSTRUCTION}\n\nQuestion asked: {record['prompt']}\n\nVLM response to analyse:\n{response}"
    messages = [
        {"role": "user", "content": [
            {"type": "image", "image": str(img_path)},
            {"type": "text", "text": user_text},
        ]}
    ]
    if include_target:
        target_json = json.dumps(build_target_spans(record), ensure_ascii=False)
        messages.append({"role": "assistant", "content": [{"type": "text", "text": target_json}]})
    return {
        "messages": messages,
        "_id": record["id"],
        "_lang": record.get("language", ""),
        "_response": response,
        "_labels": record.get("labels", []) or [],
        "_image_path": str(img_path),
    }


def parse_model_output(raw_text: str) -> list:
    """Model's raw generated text -> list of {text, prob, label} dicts (unaligned)."""
    match = re.search(r"\[.*\]", raw_text, re.DOTALL)
    if not match:
        return []
    try:
        spans = json.loads(match.group())
    except (json.JSONDecodeError, ValueError):
        return []
    if not isinstance(spans, list):
        return []
    out = []
    for sp in spans:
        if not isinstance(sp, dict) or "text" not in sp:
            continue
        text = str(sp["text"]).strip()
        if not text:
            continue
        label = normalize_label(sp.get("label", "other"))
        try:
            prob = float(sp.get("prob", 0.8))
        except (TypeError, ValueError):
            prob = 0.8
        prob = max(0.0, min(1.0, prob))
        out.append({"text": text, "prob": prob, "label": label})
    return out


def find_span_pos(response: str, text: str, search_from: int = 0):
    """Locate `text` inside `response`, tolerating minor generation drift.

    `search_from` biases the search towards matches at or after that cursor
    position first (falling back to a full-text search) -- this matters a
    lot for short/common spans (a bare number, "her", "the door", ...),
    which are exactly the kind of span the minority classes (OCR,
    miscounting) tend to produce: a naive first-occurrence `.find()` will
    happily latch onto an earlier, unrelated occurrence of the same text
    elsewhere in a long response. Callers processing spans in left-to-right
    order should pass the previous span's end as `search_from`.

    Tries, in order: exact substring, case-insensitive substring, markdown-
    stripped substring, then a sliding-window fuzzy match (difflib) for
    longer spans. Returns (start, end) or None.
    """
    if not text:
        return None

    def _find_biased(haystack: str, needle: str):
        pos = haystack.find(needle, search_from)
        if pos == -1:
            pos = haystack.find(needle)
        return pos

    pos = _find_biased(response, text)
    if pos != -1:
        return pos, pos + len(text)
    resp_lower, text_lower = response.lower(), text.lower()
    pos = _find_biased(resp_lower, text_lower)
    if pos != -1:
        return pos, pos + len(text)
    stripped = re.sub(r"[\*\_`]+", "", text).strip()
    if stripped and stripped != text:
        pos = _find_biased(resp_lower, stripped.lower())
        if pos != -1:
            return pos, pos + len(stripped)
    if len(text) >= 8:
        text_lower, resp_lower, tlen = text.lower(), response.lower(), len(text)
        step = max(1, tlen // 4)
        best_ratio, best_pos = 0.0, 0
        for i in range(0, max(1, len(response) - tlen + 1), step):
            ratio = difflib.SequenceMatcher(None, resp_lower[i:i + tlen], text_lower).ratio()
            if ratio > best_ratio:
                best_ratio, best_pos = ratio, i
        for i in range(max(0, best_pos - step), min(len(response) - tlen + 1, best_pos + step + 1)):
            ratio = difflib.SequenceMatcher(None, resp_lower[i:i + tlen], text_lower).ratio()
            if ratio > best_ratio:
                best_ratio, best_pos = ratio, i
        if best_ratio >= 0.80:
            return best_pos, best_pos + tlen
    return None


def map_spans_to_positions(response: str, parsed_spans: list) -> list:
    """Unaligned {text, prob, label} spans -> char-offset {start, end, prob, label}.

    Spans are assumed to be emitted in left-to-right order (see INSTRUCTION)
    and located with a forward-moving cursor so repeated/common substrings
    resolve to their correct, later occurrence instead of always the first
    one in the response. Falls back to an unbiased full-text search if the
    cursor-biased match collides with an already-placed span.
    """
    result, used = [], set()
    cursor = 0
    for sp in parsed_spans:
        pos = find_span_pos(response, sp["text"], search_from=cursor)
        if pos is not None and len(set(range(*pos)) & used) > (pos[1] - pos[0]) * 0.5:
            pos = find_span_pos(response, sp["text"], search_from=0)
        if pos is None:
            continue
        start, end = pos
        if len(set(range(start, end)) & used) > (end - start) * 0.5:
            continue
        result.append({"start": start, "end": end, "prob": sp["prob"], "label": sp["label"]})
        used.update(range(start, end))
        cursor = end
    return sorted(result, key=lambda x: x["start"])


def predict_spans_from_raw_output(response: str, raw_model_output: str) -> list:
    """Full decode path: raw generated text -> submission-format span list."""
    return map_spans_to_positions(response, parse_model_output(raw_model_output))
