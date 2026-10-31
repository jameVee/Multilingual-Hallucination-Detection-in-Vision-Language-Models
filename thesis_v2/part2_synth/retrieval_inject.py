"""C4: retrieval-grounded injection.

Train has 12,082 responses over only 2,054 images, so a clean response usually has
"sibling" responses about the same image (same language). When a sibling carries a
real, annotator-labeled hallucination of the target category, C4 takes the
sentence containing it and REPLACES the most similar sentence of the clean
response. Every injected span is real annotated hallucination text about this
very image, with its real annotator-agreement prob.

Placement, in order of preference:
  1. replace -- the clean response has a sentence similar to the donor sentence
     (token Jaccard >= 0.25): it is replaced, so the response still answers its question.
  2. insert_interior -- otherwise (siblings often answer a different question about
     the same image), the donor sentence is inserted at the sentence boundary that
     puts the span closest to a position sampled from the real position
     distribution of that category. The end is a boundary too, but is chosen only as
     often as real spans sit there -- the old heuristic appended unconditionally
     (79% of its OCR spans at the very end vs 14% real).
  3. insert_start -- only for 'other' leak / refusal / repetition spans, which real
     data places at the start of a response.
An automated check rejects any placement that leaves a contradicting count or a
near-duplicate quoted text elsewhere in the response (e.g. "There are 7 glasses"
next to "There are five glasses").
"""
from __future__ import annotations

import difflib
import random
import re
from collections import Counter, defaultdict

SIM_THRESHOLD = 0.25
MAX_DONOR_REUSE = 2
BOUND = re.compile(r"(?<=[.!?])\s+|(?<=[。！？])|\n+")
QUOTED = re.compile(r'["“«「]([^"”»」\n]{2,60})["”»」]|\*\*([^*\n]{2,60})\*\*')
COUNT = re.compile(r"(\d+|zero|one|two|three|four|five|six|seven|eight|nine|ten|eleven|twelve|"
                   r"un|une|deux|trois|quatre|cinq|sept|huit|neuf|dix|uno|una|due|tre|quattro|cinque|"
                   r"sei|sette|otto|nove|dieci)[*\s]+(\w+)", re.I)   # [*\s]: "**6** portes"
ZH_COUNT = re.compile(r"([零一二两三四五六七八九十\d]+)[*\s]*"
                      r"[个只位条张辆名件头匹棵台本盏把块片艘架根座份扇颗栋群种对双套层枚支朵尾部间道幅面杯瓶盘碗粒人]")
LEAK_RE = re.compile(r"(Human|Assistant|User|USER|System)\s*[:：]|<\|[^|>]*\|>|</?[a-z_]+>|\bfim_\w+", re.I)
REFUSE_RE = re.compile(r"unable to|can't|cannot|not able to|ne (peux|suis) pas|non (posso|sono in grado)|无法|不能|抱歉", re.I)


def other_type(span_text, context):
    """Real 'other' spans split into: freetext 84%, repetition 10%, leak 4%, refusal 1% (train)."""
    if LEAK_RE.search(context):
        return "leak"
    if REFUSE_RE.search(span_text):
        return "refusal"
    w = span_text.split()
    if (len(w) >= 3 and max(Counter(w).values()) >= 3) or re.search(r"(.{1,6})\1{3,}", span_text):
        return "repetition"
    return "freetext"


def sentences(text):
    spans, last = [], 0
    for m in BOUND.finditer(text):
        if m.start() > last:
            spans.append((last, m.start()))
        last = m.end()
    if last < len(text):
        spans.append((last, len(text)))
    return [(s, e) for s, e in spans if text[s:e].strip()]


def tokens(text, lang):
    if lang == "zh":
        t = re.sub(r"\s+", "", text)
        return {t[i:i + 2] for i in range(len(t) - 1)}
    return set(re.findall(r"\w+", text.lower()))


def jaccard(a, b):
    return len(a & b) / len(a | b) if a and b else 0.0


CJK = re.compile(r"[一-鿿]")
LATIN = re.compile(r"[A-Za-zÀ-ÖØ-öø-ÿ]")
STOPWORDS = {
    "en": {"the", "and", "is", "of", "a", "in", "to", "with", "this", "are", "it", "on"},
    "fr": {"le", "la", "les", "et", "est", "des", "un", "une", "dans", "avec", "du", "il", "sur", "qui"},
    "it": {"il", "la", "le", "e", "è", "di", "un", "una", "nel", "con", "sono", "della", "che", "per"},
}


def lang_ok(text, lang):
    """Reject donor sentences written in a different language than the target
    (real responses sometimes code-switch; transplanting that would put a language
    switch exactly at the injected span -- an easy shortcut)."""
    n_cjk, n_lat = len(CJK.findall(text)), len(LATIN.findall(text))
    if lang == "zh":
        return n_cjk >= 0.5 * max(1, n_cjk + n_lat)
    if n_cjk:
        return False
    words = re.findall(r"\w+", text.lower())
    scores = {lg: sum(w in sw for w in words) for lg, sw in STOPWORDS.items()}
    return scores[lang] > 0 and scores[lang] >= max(scores.values())


def _counts(text, lang):
    if lang == "zh":
        return {(m.group(1), "") for m in ZH_COUNT.finditer(text)}
    return {(m.group(1).lower(), m.group(2).lower()) for m in COUNT.finditer(text)}


CLAIM_CATEGORIES = {"invention", "mischaracterization"}
NEGATION = {"en": {"no", "not", "none", "without", "never", "cannot", "nothing"},
            "fr": {"ne", "pas", "aucun", "aucune", "sans", "jamais", "rien"},
            "it": {"non", "nessun", "nessuna", "nessuno", "senza", "mai", "niente"}}
ZH_NEGATION = "不没无未非"
GENERIC = {"en": {"image", "photo", "picture", "visible", "shown", "appears", "seen", "there", "that", "which"},
           "fr": {"image", "photo", "visible", "semble", "cette", "cela", "pour", "plus"},
           "it": {"immagine", "foto", "visibile", "sembra", "questa", "questo", "sono", "dell"}}
ZH_GENERIC = set("的了是在有和与及或图片中这那一个它们其也都很上下左右看到可以从")


def _negated(text, lang):
    if lang == "zh":
        return any(c in text for c in ZH_NEGATION)
    low = text.lower()
    return bool(tokens(low, lang) & NEGATION[lang]) or "n't" in low or (lang == "fr" and bool(re.search(r"\bn['’]", low)))


def _content(text, lang):
    if lang == "zh":
        return set(CJK.findall(text)) - ZH_GENERIC
    return {w for w in tokens(text, lang) if len(w) > 2} - STOPWORDS[lang] - NEGATION[lang] - GENERIC[lang]


def polarity_conflict(new_sentence, rest, lang):
    """A claim next to a sentence about the same thing with the opposite polarity: "there are
    five fish" + "no fish can be seen". The content-word overlap is a text-only proxy, so
    it over-rejects; that only costs pool size, and it avoids teaching the shortcut "flag
    whatever the rest of the answer contradicts"."""
    need = 2 if lang == "zh" else 1
    new, neg = _content(new_sentence, lang), _negated(new_sentence, lang)
    for s, e in sentences(rest):
        if _negated(rest[s:e], lang) != neg and len(new & _content(rest[s:e], lang)) >= need:
            return True
    return False


def contradicts(new_sentence, rest, category, lang):
    if category in CLAIM_CATEGORIES:
        return polarity_conflict(new_sentence, rest, lang)
    if category == "miscounting":
        new = _counts(new_sentence, lang)
        old = _counts(rest, lang)
        for num, noun in new:
            if any(noun2 == noun and num2 != num for num2, noun2 in old):
                return True
        if lang == "zh" and new and old and new != old:
            return True
    if category == "OCR":
        newq = [a or b for a, b in QUOTED.findall(new_sentence)]
        oldq = [a or b for a, b in QUOTED.findall(rest)]
        for x in newq:
            for y in oldq:
                if x != y and difflib.SequenceMatcher(None, x.lower(), y.lower()).ratio() >= 0.6:
                    return True
    return False


def build_candidates(train, categories, seed):
    """-> {(category, language): [synthetic record, ...]} in deterministic order."""
    by_img = defaultdict(list)
    for r in train:
        by_img[(r["image_name"], r["language"])].append(r)
    rng = random.Random(seed)
    # some records are labeled e.g. "it" but written in English; a donor sentence in the
    # labeled language would then be a language switch exactly at the injected span
    targets = [r for r in train if not r.get("labels") and lang_ok(r["response"], r["language"])]
    rng.shuffle(targets)

    # real relative span positions per category (for 'other': free-text spans only),
    # used to place interior insertions where real spans of that category sit
    pos_dist = defaultdict(list)
    for r in train:
        t = r["response"]
        for s in r.get("labels", []) or []:
            if s["label"] not in categories:
                continue
            if s["label"] == "other" and other_type(t[s["start"]:s["end"]],
                                                    t[max(0, s["start"] - 3):s["end"] + 3]) != "freetext":
                continue
            pos_dist[s["label"]].append(s["start"] / max(1, len(t)))

    out = defaultdict(list)
    used_targets = set()
    donor_uses = defaultdict(int)
    for cat in categories:
        for t in targets:
            lang = t["language"]
            if (t["id"], cat) in used_targets:
                continue
            donors = sorted((d for d in by_img[(t["image_name"], lang)]
                             if d.get("labels") and any(s["label"] == cat for s in d["labels"])),
                            key=lambda d: (d["prompt"] != t["prompt"], d["id"]))  # same-question siblings first
            rec = None
            for d in donors:
                span = next(s for s in sorted(d["labels"], key=lambda s: s["start"]) if s["label"] == cat)
                key = (d["id"], span["start"], span["end"])
                if donor_uses[key] >= MAX_DONOR_REUSE:
                    continue
                rec = _inject(t, d, span, cat, lang, rng, pos_dist)
                if rec is not None:
                    donor_uses[key] += 1
                    break
            if rec is not None:
                used_targets.add((t["id"], cat))
                out[(cat, lang)].append(rec)
    return out


def _inject(target, donor, span, cat, lang, rng, pos_dist):
    dt, tt = donor["response"], target["response"]
    dsent = next(((s, e) for s, e in sentences(dt) if s <= span["start"] < e), None)
    if dsent is None:
        return None
    ds, de = dsent
    dtext = dt[ds:de]
    sep = "" if lang == "zh" else " "
    typ_probe = cat == "other" and other_type(dt[span["start"]:span["end"]],
                                              dt[max(0, span["start"] - 3):span["end"] + 3]) != "freetext"
    if not typ_probe and not lang_ok(dtext, lang):
        return None   # leak/repetition spans (e.g. "Human:", "600, 600, ...") have no language to check
    typ = other_type(dt[span["start"]:span["end"]], dt[max(0, span["start"] - 3):span["end"] + 3]) \
        if cat == "other" else None
    tsents = sentences(tt)
    # the donor's own whitespace around its sentence ("\n\n", "\n- ", " "), so inserted
    # list items / paragraphs keep their formatting
    lead = re.search(r"\s*$", dt[:ds]).group() or sep
    trail = re.match(r"\s*", dt[de:]).group()
    if typ not in ("leak", "refusal", "repetition") and not trail and not re.search(r"[.!?。！？]\W*$", dtext):
        return None   # unterminated fragment (donor response cut off mid-sentence)
    trail = trail or sep
    if typ in ("leak", "refusal", "repetition"):
        mode = "insert_start"
        new = dtext + sep + tt
        offset, rest = -ds, tt
    else:
        dtok = tokens(dtext, lang)
        best = max(((jaccard(dtok, tokens(tt[s:e], lang)), s, e) for s, e in tsents), default=None)
        if best is not None and best[0] >= SIM_THRESHOLD:
            mode = "replace"
            _, ts, te = best
            new = tt[:ts] + dtext + tt[te:]
            offset, rest = ts - ds, tt[:ts] + tt[te:]
        else:
            # every sentence boundary, including before the first and after the last.
            # The boundary is chosen to put the SPAN itself (not the sentence start)
            # closest to a position sampled from the real distribution, so the end is
            # used only as often as real spans sit there (OCR: 14%) -- unlike the old
            # heuristic, which appended unconditionally (79%).
            bounds = [s for s, _ in tsents]
            if not bounds:
                return None
            p = rng.choice(pos_dist[cat])
            in_sent = span["start"] - ds
            end = len(tt.rstrip())
            new_len = len(tt) + len(dtext) + len(trail)
            b = min(bounds + [end], key=lambda x: abs((x + in_sent + (len(lead) if x == end else 0)) / new_len - p))
            mode = "insert_interior"
            if b == end:
                new = tt[:end] + lead + dtext + tt[end:]
                offset = end + len(lead) - ds
            else:
                new = tt[:b] + dtext + trail + tt[b:]
                offset = b - ds
            rest = tt
        if contradicts(dtext, rest, cat, lang):
            return None
    labels = []
    for s in donor["labels"]:
        if s["start"] >= ds and s["end"] <= de:
            labels.append({"start": s["start"] + offset, "end": s["end"] + offset,
                           "prob": s["prob"], "label": s["label"]})
    if not any(l["label"] == cat for l in labels):
        return None
    for l in labels:  # offsets must still point at the same text
        assert new[l["start"]:l["end"]] == dt[l["start"] - offset:l["end"] - offset]
    return {
        "id": f"synth_c4{cat}_{target['id']}", "split": "train", "language": lang,
        "prompt": target["prompt"], "image_name": target["image_name"], "response": new,
        "labels": sorted(labels, key=lambda l: l["start"]),
        "_meta": {"method": "C4", "category": cat, "base_id": target["id"], "donor_id": donor["id"],
                  "mode": mode},
    }
