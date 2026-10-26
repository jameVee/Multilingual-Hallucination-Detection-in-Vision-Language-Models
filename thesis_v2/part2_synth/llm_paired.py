"""C3: LLM-written content for the SAME edit sites as C2 (paired design).

For every site in sites.jsonl (built by build_conditions --stage sites) Claude sees
the image, the question and the clean response with the site marked, and writes
only the replacement text for that site. Everything else -- which response, where,
span length target, prob, the noun/verb grammar fixes of count sites -- is fixed by
the site and shared with C2, so C2 vs C3 isolates who writes the content.

Per site kind:
  OCR (in_quote / unquoted)  a misreading of the marked text, as a VLM might produce
  count_*                    only the wrong number (>= 2, same form as the original:
                             digits or a number word); the pipeline appends the
                             counted noun exactly as in C2
  freetext                   a rewrite of the marked phrase that states something
                             wrong about the image (not an object, count or text error)
  refusal                    a short unjustified refusal / inability statement
  leak, repetition           decoding artifacts with no content to write: C2's
                             template is reused (no API call)
Outputs are validated in code (changed, length range, number form, language) and
retried with the error message up to MAX_ATTEMPTS times.

Runs on the laptop (API + local images, no GPU). The API key is read from the
environment (ANTHROPIC_API_KEY or an `ant auth login` profile) -- never from a file.
Results are cached incrementally in thesis_v2/runs/part2/llm_paired_cache.json, so
an interrupted run resumes where it stopped.

Usage:
    python -m thesis_v2.part2_synth.llm_paired --dry_run          # prompts + call count, no API
    python -m thesis_v2.part2_synth.llm_paired --limit 20         # smoke test
    python -m thesis_v2.part2_synth.llm_paired                    # all sites -> C3_content.jsonl
"""
from __future__ import annotations

import argparse
import base64
import io
import json
import threading
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed

from thesis_v2.common.paths import PART2_DATA_DIR, RUNS_DIR
from thesis_v2.part2_synth import retrieval_inject
from thesis_v2.part2_synth.build_conditions import count_value, count_tail
from solutions.common import data_io

MODEL = "claude-opus-5-5"
MAX_ATTEMPTS = 3
CACHE = RUNS_DIR / "part2" / "llm_paired_cache.json"
OUT = PART2_DATA_DIR / "C3_content.jsonl"
SHARED_KINDS = {"leak", "repetition"}
LANG_NAME = {"en": "English", "fr": "French", "it": "Italian", "zh": "Chinese"}
SCHEMA = {"type": "object", "properties": {"content": {"type": "string"}},
          "required": ["content"], "additionalProperties": False}

SYSTEM = (
    "You help build training data for a detector of hallucinations in vision-language "
    "model answers. You are shown an image, a question about it, and a model answer that "
    "is correct. One location in the answer is marked with ⟦ and ⟧. You write the text "
    "that replaces the marked location so that the answer contains one realistic "
    "hallucination of the requested type, the way a vision-language model would "
    "actually get it wrong: fluent, plausible, in the answer's language, fitting the "
    "surrounding sentence exactly. Return only the replacement text in the JSON field "
    "`content` -- no quotes around it, no explanation."
)


def marked(t, site):
    s, e = site["start"], site["end"]
    return t[:s] + "⟦" + t[s:e] + "⟧" + t[e:]


def instruction(rec, site):
    t, lang, kind = rec["response"], rec["language"], site["kind"]
    seg = t[site["start"]:site["end"]]
    lname = LANG_NAME[lang]
    if kind in ("in_quote", "unquoted"):
        lo, hi = length_range(site, rec)
        return (f"Type: OCR error. The marked text ⟦{seg}⟧ is text the answer reads from the image. "
                f"Replace it with a misreading of that same text -- wrong, missing, extra or swapped "
                f"characters, as an imperfect text reader would produce. Keep it {lo}-{hi} characters "
                f"long and keep the surrounding quotes/formatting outside the marked part unchanged.")
    if kind.startswith("count"):
        num = t[site["start"]:site["num_end"]]
        form = "digits" if num.isdigit() else f"one {lname} number word"
        return (f"Type: counting error. The marked part starts with the count «{num}». Write ONLY a "
                f"different, wrong count of the same objects -- an integer of at least 2, written as "
                f"{form}, the way a model miscounting the objects in this image would. Output just the "
                f"number" + (" (use 两 for 2)." if lang == "zh" else "."))
    if kind == "freetext":
        lo, hi = length_range(site, rec)
        return (f"Type: other error (not an object, count or text-reading error). Rewrite the marked "
                f"phrase ⟦{seg}⟧ so that it states something wrong about the image -- e.g. a wrong "
                f"position, direction, state or relation -- while still fitting the sentence. "
                f"{lname}, {lo}-{hi} characters.")
    if kind == "refusal":
        lo, hi = length_range(site, rec)
        return (f"Type: other error. The marked location is an empty insertion point. Write a short, "
                f"unjustified refusal or inability statement (the image is perfectly answerable) that "
                f"a vision-language model might wrongly insert here. {lname}, {lo}-{hi} characters.")
    raise ValueError(kind)


def length_range(site, rec):
    t = rec["response"]
    if site["kind"] == "refusal":
        L = max(15, site["target_len"])
        return max(10, int(L * 0.5)), int(L * 1.5) + 5
    L = site["end"] - site["start"]
    if site["kind"] in ("in_quote", "unquoted"):
        d = max(2, round(0.3 * L))
        return max(1, L - d), L + d
    return max(1, int(L * 0.6)), int(L * 1.5) + 2   # freetext


def validate(rec, site, out):
    """-> (content for apply_edit, None) or (None, error message for the retry)."""
    t, lang, kind = rec["response"], rec["language"], site["kind"]
    seg = t[site["start"]:site["end"]]
    out = out.strip().strip("⟦⟧").strip()
    if not out:
        return None, "empty output"
    if kind.startswith("count"):
        num = t[site["start"]:site["num_end"]]
        if lang == "zh" and out == "二":
            out = "两"
        v, v0 = count_value(out, lang), count_value(num, lang)
        if v is None or (out.isdigit() != num.isdigit()):
            return None, f"'{out}' is not a single {'number in digits' if num.isdigit() else 'number word'}"
        if v < 2 or v == v0:
            return None, f"the count must be at least 2 and differ from {num}"
        if num[:1].isupper() and lang != "zh":
            out = out.capitalize()
        return out + count_tail(t, site), None
    lo, hi = length_range(site, rec)
    if not lo <= len(out) <= hi:
        return None, f"length {len(out)} is outside {lo}-{hi} characters"
    if out == seg:
        return None, "the output is identical to the marked text"
    if kind in ("freetext", "refusal") and len(out) >= 12 and not retrieval_inject.lang_ok(out, lang):
        return None, f"the output is not in {LANG_NAME[lang]}"
    return out, None


def encode_image(rec, max_side=1568):
    from PIL import Image
    path = data_io.resolve_image_path(rec["image_name"])
    if path is None:
        return None
    img = Image.open(path).convert("RGB")
    img.thumbnail((max_side, max_side))
    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=90)
    return base64.standard_b64encode(buf.getvalue()).decode()


def generate(client, rec, site):
    import anthropic
    img = encode_image(rec)
    if img is None:
        return {"content": None, "error": "image not found", "attempts": 0}
    content = [
        {"type": "image", "source": {"type": "base64", "media_type": "image/jpeg", "data": img}},
        {"type": "text", "text": f"Question: {rec['prompt']}\n\nAnswer with the marked location:\n"
                                 f"{marked(rec['response'], site)}\n\n{instruction(rec, site)}"},
    ]
    messages = [{"role": "user", "content": content}]
    err = None
    for attempt in range(1, MAX_ATTEMPTS + 1):
        try:
            resp = client.beta.messages.create(
                model=MODEL, max_tokens=2000, system=SYSTEM, messages=messages,
                output_config={"effort": "low", "format": {"type": "json_schema", "schema": SCHEMA}},
                betas=["server-side-fallback-2026-07-01"], fallbacks="default",
            )
        except anthropic.BadRequestError as e:
            return {"content": None, "error": f"bad request: {e.message}"[:300], "attempts": attempt}
        if resp.stop_reason == "refusal":
            return {"content": None, "error": "refusal", "attempts": attempt}
        text = next((b.text for b in resp.content if b.type == "text"), "")
        try:
            raw = json.loads(text)["content"]
        except (json.JSONDecodeError, KeyError, TypeError):
            raw = ""
        out, err = validate(rec, site, raw)
        if out is not None:
            return {"content": out, "raw": raw, "attempts": attempt}
        # append-only retry: the model sees its answer and what was wrong with it
        messages = messages + [{"role": "assistant", "content": text or "{}"},
                               {"role": "user", "content": f"Rejected: {err}. Try again."}]
    return {"content": None, "error": err, "attempts": MAX_ATTEMPTS}


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--dry_run", action="store_true")
    p.add_argument("--limit", type=int, default=None)
    p.add_argument("--workers", type=int, default=4)
    args = p.parse_args()

    by_id = {r["id"]: r for r in data_io.load_jsonl(data_io.TRAIN_FILE)}
    sites = data_io.load_jsonl(PART2_DATA_DIR / "sites.jsonl")
    cache = json.loads(CACHE.read_text(encoding="utf-8")) if CACHE.exists() else {}
    todo = [s for s in sites if s["kind"] not in SHARED_KINDS and s["site_id"] not in cache]
    if args.limit:
        todo = todo[:args.limit]
    print(f"sites {len(sites)}  cached {len(cache)}  to call {len(todo)}  "
          f"kinds {dict(Counter(s['kind'] for s in todo))}")

    if args.dry_run:
        seen = set()
        for s in todo:
            if s["kind"] in seen:
                continue
            seen.add(s["kind"])
            rec = by_id[s["base_id"]]
            print(f"\n--- {s['site_id']} ({s['kind']}) C2: {s['c2_content']!r}\n{instruction(rec, s)}")
        print(f"\n~{len(todo)} calls x (~1.6k image + ~0.8k text input, ~0.3k output tokens) at "
              f"$4/$20 per MTok = ~${len(todo) * (2.4e3 * 4 + 0.3e3 * 20) / 1e6:.0f} (+ retries)")
        return

    import anthropic
    client = anthropic.Anthropic()   # credentials from the environment only
    lock = threading.Lock()
    CACHE.parent.mkdir(parents=True, exist_ok=True)

    def save():
        tmp = CACHE.with_suffix(".tmp")
        tmp.write_text(json.dumps(cache, ensure_ascii=False), encoding="utf-8")
        tmp.replace(CACHE)

    done = 0
    with ThreadPoolExecutor(max_workers=args.workers) as ex:
        futs = {ex.submit(generate, client, by_id[s["base_id"]], s): s for s in todo}
        for f in as_completed(futs):
            s = futs[f]
            try:
                res = f.result()
            except anthropic.APIError as e:   # rate limit / server error after SDK retries: leave uncached
                print(f"{s['site_id']}: {type(e).__name__} -- will retry on the next run")
                continue
            with lock:
                cache[s["site_id"]] = res
                done += 1
                if done % 20 == 0:
                    save()
                    ok = sum(1 for v in cache.values() if v.get("content"))
                    print(f"{done}/{len(todo)}  ok so far {ok}/{len(cache)}")
    save()

    rows = []
    for s in sites:
        if s["kind"] in SHARED_KINDS:
            rows.append({"site_id": s["site_id"], "content": s["c2_content"], "shared_with_c2": True})
        elif cache.get(s["site_id"], {}).get("content"):
            rows.append({"site_id": s["site_id"], "content": cache[s["site_id"]]["content"]})
    OUT.parent.mkdir(parents=True, exist_ok=True)
    with open(OUT, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    fails = Counter(v.get("error", "")[:40] for v in cache.values() if not v.get("content"))
    print(f"wrote {len(rows)} -> {OUT}   failures: {dict(fails)}")


if __name__ == "__main__":
    main()
