"""Runs EasyOCR once over every image referenced in data/train.jsonl and
caches the detected text per image to ocr_cache.json. Used by
generate_synthetic.py to ground OCR-class synthetic corruption in text that
is actually present in the image, rather than swapping characters inside
arbitrary dictionary words the way S1a's approach did.

EasyOCR is pure-pip (bundles its own detection/recognition models, no system
Tesseract binary needed) and covers the 4 languages in this dataset (en, fr,
it, zh via 'ch_sim') -- but NOT in a single reader. An earlier version of
this file tried `easyocr.Reader(["en","fr","it","ch_sim"])`, which fails
immediately: EasyOCR only allows 'ch_sim' combined with 'en', not with other
Latin-script languages in the same reader (raises "Chinese_sim is only
compatible with English" -- verified locally, this is a hard error, not a
warning). Fixed by running two separate readers per image and merging their
detections: one for en/fr/it, one for ch_sim/en.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

_THIS_DIR = Path(__file__).resolve().parent
_REPO_ROOT = _THIS_DIR.parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from solutions.common import data_io

LATIN_LANGS = ["en", "fr", "it"]
CJK_LANGS = ["ch_sim", "en"]


def extract_ocr_cache(image_names, image_dir=data_io.IMAGE_DIR, output_path=None,
                       gpu=True, min_confidence=0.4):
    import easyocr
    from tqdm.auto import tqdm

    reader_latin = easyocr.Reader(LATIN_LANGS, gpu=gpu)
    reader_cjk = easyocr.Reader(CJK_LANGS, gpu=gpu)
    cache = {}
    names = sorted(set(image_names))
    pbar = tqdm(names, desc="EasyOCR", unit="img", dynamic_ncols=True)
    for name in pbar:
        path = data_io.resolve_image_path(name, image_dir)
        if path is None:
            continue
        texts = []
        for reader in (reader_latin, reader_cjk):
            try:
                results = reader.readtext(str(path))
            except Exception as exc:
                pbar.write(f"  OCR failed on {name}: {exc}")
                continue
            texts.extend(text for (_, text, conf) in results if conf >= min_confidence and text.strip())
        # dedupe while preserving order (both readers often agree on plain Latin text)
        seen = set()
        deduped = [t for t in texts if not (t in seen or seen.add(t))]
        if deduped:
            cache[name] = deduped
        pbar.set_postfix({"with_text": len(cache)})

    if output_path is not None:
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(json.dumps(cache, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"Saved OCR cache ({len(cache)} images with text) -> {output_path}")
    return cache


def load_ocr_cache(path):
    if not Path(path).exists():
        return {}
    return json.loads(Path(path).read_text(encoding="utf-8"))


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--output", default=str(_THIS_DIR / "ocr_cache.json"))
    p.add_argument("--limit", type=int, default=None)
    p.add_argument("--cpu", action="store_true", help="Force CPU (EasyOCR defaults to GPU if available).")
    args = p.parse_args()

    train = data_io.load_jsonl(data_io.TRAIN_FILE)
    names = sorted({r["image_name"] for r in train})
    if args.limit:
        names = names[: args.limit]
    print(f"Running EasyOCR over {len(names)} unique training images...")
    extract_ocr_cache(names, output_path=args.output, gpu=not args.cpu)
