"""Build Part 2's isolated training conditions (C1-C4) from train data only.

Stages (run in order):
  python -m thesis_v2.part2_synth.build_conditions --stage c1c4      # C1 file, C4 candidates, volume per cell
  python -m thesis_v2.part2_synth.build_conditions --stage sites     # paired edit sites + C2 content
  python -m thesis_v2.part2_synth.llm_paired                         # C3 content via Claude API (laptop)
  python -m thesis_v2.part2_synth.build_conditions --stage finalize  # identical counts, realism gate
  python -m thesis_v2.part2_synth.build_conditions --stage preview --method C2 --n 10

Design (see the approved plan):
- C2 and C3 edit the SAME site in the SAME clean response; only who writes the
  replacement content differs (heuristic vs LLM). Sites are sampled so their
  kind/length/position follow the real per-category distributions measured on
  train (e.g. OCR: 74% inside quoted text; miscounting: 65% bare number vs 35%
  number+noun; other: 84% free-text / 10% repetition / 4% leak / 1% refusal).
- prob is sampled from the real (category, language) distribution, once per site,
  and shared by C2 and C3 (synthetic spans were always prob=1.0 before; real
  rare-category spans are mostly 0.33).
- C4 (retrieval_inject.py) has its own pool and is matched on counts per
  (category, language); all three must pass the realism gate (realism.py).
"""
from __future__ import annotations

import argparse
import json
import random
import re
from collections import Counter, defaultdict

from thesis_v2.common.paths import OCR_CACHE, PART2_DATA_DIR
from thesis_v2.part2_synth import realism, retrieval_inject
from solutions.common import data_io
from solutions.common import synth_common as sc

DATA_SEED = 7
CATEGORIES = ["OCR", "miscounting", "other"]
LANGS = ["en", "fr", "it", "zh"]
TARGET_PER_CELL = 100
OVERSAMPLE = 2.0
PILOT_CLAIMS_PER_CELL = 230

QUOTED_SPAN = re.compile(r'["“«「]\s*([^"”»」\n]{2,60}?)\s*["”»」]|\*\*([^*\n]{2,60})\*\*')
EXCL_NEXT = {"%", "percent", "pour", "per", "cm", "mm", "m", "km", "kg", "g", "lb", "lbs", "mph", "h", "hours",
             "hour", "heures", "heure", "ore", "ora", "minutes", "min", "pm", "am", "years", "year", "ans", "an",
             "anni", "anno", "century", "centuries", "siècle", "siècles", "secolo", "secoli", "th", "st", "nd",
             "rd", "ème", "e", "o", "a", "times", "fois", "volte", "x", "pounds", "pound", "grams", "gram",
             "grammes", "grammi", "inches", "inch", "feet", "foot", "ft", "meters", "metres", "mètres", "metri",
             "miles", "litres", "liters", "litri", "ml", "l", "seconds", "secondes", "secondi", "days", "jours",
             "giorni", "weeks", "months", "mois", "mesi", "degrees", "degrés", "gradi", "dollars", "euros", "euro",
             "km/h", "kilometers", "kilomètres", "chilometri", "centimeters", "centimètres", "centimetri",
             # dates and verbs right after a number ("4 juin", "10 n'est") are not counts
             "january", "february", "march", "april", "may", "june", "july", "august", "september", "october",
             "november", "december", "janvier", "février", "mars", "avril", "mai", "juin", "juillet", "août",
             "septembre", "octobre", "novembre", "décembre", "gennaio", "febbraio", "marzo", "aprile", "maggio",
             "giugno", "luglio", "agosto", "settembre", "ottobre", "dicembre", "is", "are", "was", "were",
             "est", "sont", "n'est", "è", "sono", "or", "ou", "o", "to", "à"}
EXCL_PREV = {"page", "no", "number", "numéro", "numero", "#", "$", "€", "£", "chapter", "version", "v"}
# Count 1 -> 2/3 needs the noun in the plural and sometimes the verb ("There is one dog"
# -> "There are two dogs", "c'è un cane" -> "ci sono due cani"). Sources of value 1 are used
# only where the noun is clearly the head (followed by punctuation or a function word),
# so the pluralized word is not an adjective.
ONE_WORDS = {"en": {"one"}, "fr": {"un", "une"}, "it": {"uno", "una"}}
HEAD_NEXT = {
    "en": {"in", "on", "at", "with", "is", "was", "that", "which", "and", "near", "next", "under", "behind",
           "visible", "sitting", "standing", "of", "from", "by", "here", "there"},
    "fr": {"dans", "sur", "avec", "est", "qui", "et", "près", "à", "au", "sous", "derrière", "visible", "de", "du"},
    "it": {"nel", "nella", "sul", "sulla", "con", "è", "che", "e", "vicino", "a", "sotto", "dietro", "visibile",
           "di", "del", "della", "in"},
}
VERB_BEFORE = {"en": {"is": "are", "was": "were", "there's": "there are"},
               "fr": {}, "it": {"c'è": "ci sono", "è": "sono", "c’è": "ci sono"}}
VERB_AFTER = {"en": {"is": "are", "was": "were", "has": "have"}, "fr": {"est": "sont", "a": "ont"},
              "it": {"è": "sono", "ha": "hanno"}}
EN_IRREGULAR = {"person": "people", "man": "men", "woman": "women", "child": "children", "foot": "feet",
                "tooth": "teeth", "mouse": "mice", "goose": "geese", "sheep": "sheep", "fish": "fish",
                "deer": "deer", "knife": "knives", "leaf": "leaves", "shelf": "shelves", "wolf": "wolves"}


def pluralize(noun, lang):
    low = noun.lower()
    if lang == "en":
        if low in EN_IRREGULAR:
            p = EN_IRREGULAR[low]
        elif low.endswith(("s", "x", "z", "ch", "sh")):
            p = noun + "es"
        elif low.endswith("y") and len(low) > 1 and low[-2] not in "aeiou":
            p = noun[:-1] + "ies"
        else:
            p = noun + "s"
    elif lang == "fr":
        if low.endswith(("s", "x", "z")):
            p = noun
        elif low.endswith("al"):
            p = noun[:-2] + "aux"
        elif low.endswith(("eau", "au", "eu")):
            p = noun + "x"
        else:
            p = noun + "s"
    else:   # it
        if low.endswith(("ca", "ga")):
            p = noun[:-1] + "he"
        elif low.endswith(("co", "go")):
            p = noun[:-1] + "hi"
        elif low.endswith("a"):
            p = noun[:-1] + "e"
        elif low.endswith(("o", "e")):
            p = noun[:-1] + "i"
        else:
            p = noun   # invariable (città, bar)
    return p.capitalize() if noun[:1].isupper() and not p[:1].isupper() else p


def one_fixes(t, m, lang):
    """Edits that keep a 1 -> 2/3 count grammatical, or None if the noun is not clearly
    the head. -> (noun (start, end, plural), [verb (start, end, replacement), ...])"""
    after = t[m.end(2):m.end(2) + 30]
    nxt = re.match(r"\s*([^\s.,;:!?]*)", after).group(1).lower()
    if lang == "en" and m.group(2).lower().endswith("s"):
        return None   # "indoors", "fits": already plural or not a noun
    if not (re.match(r"\s*[.,;:!?)]", after) or not after.strip() or nxt in HEAD_NEXT[lang]):
        return None
    noun = (m.start(2), m.end(2), pluralize(m.group(2), lang))
    verbs = []
    vm = re.match(r"(\s+)(\S+)", after)
    if vm and vm.group(2).lower() in VERB_AFTER[lang]:
        s = m.end(2) + vm.start(2)
        verbs.append((s, s + len(vm.group(2)), match_case(VERB_AFTER[lang][vm.group(2).lower()], vm.group(2))))
    before = t[max(0, m.start(1) - 12):m.start(1)]
    for w, rep in VERB_BEFORE[lang].items():
        bm = re.search(rf"(?i)(?<![\w'’]){re.escape(w)}\s+$", before)
        if bm:
            s = m.start(1) - len(before) + bm.start()
            verbs.append((s, s + len(w), match_case(rep, before[bm.start():bm.start() + len(w)])))
            break
    if lang == "fr" and re.search(r"(?i)\b(c'|c’)?est\s+$", before):
        return None   # "c'est un chien" -> "ce sont deux chiens": not handled
    el = re.search(r"(?i)\b(d|qu|l)['’]$", before)
    if el:   # elision: "d'un" -> "de deux", "qu'une" -> "que deux"; "l'un" is not a count
        if el.group(1).lower() == "l":
            return None
        s = m.start(1) - len(before) + el.start()
        verbs.append((s, m.start(1), "de " if el.group(1).lower() == "d" else "que "))
    return noun, verbs
ZH_MEASURE = "个只位条张辆名件头匹棵台本盏把块片艘架层扇根"
# zh state words as whole words: single-character swaps broke compounds (展开 -> 展关,
# 关于 -> 开于). Direction words swap one character, verbs/adjectives the whole word.
ZH_SWAP = {"左侧": "右侧", "右侧": "左侧", "左边": "右边", "右边": "左边", "左上": "右上", "右上": "左上",
           "左下": "右下", "右下": "左下", "左手": "右手", "右手": "左手", "左方": "右方", "右方": "左方",
           "前面": "后面", "后面": "前面", "前方": "后方", "后方": "前方", "前排": "后排", "后排": "前排",
           "前腿": "后腿", "后腿": "前腿", "上方": "下方", "下方": "上方", "上面": "下面", "下面": "上面",
           "打开": "关闭", "关闭": "打开", "开着": "关着", "关着": "开着", "空的": "满的", "满的": "空的",
           "新的": "旧的", "旧的": "新的", "崭新": "陈旧", "陈旧": "崭新", "明亮": "昏暗", "昏暗": "明亮",
           "站着": "坐着", "坐着": "站着", "干燥": "潮湿", "潮湿": "干燥"}
ZH_SWAP_RE = re.compile("|".join(sorted(ZH_SWAP, key=len, reverse=True)))


def match_case(word, like):
    return word.capitalize() if like[:1].isupper() else word


REFUSALS = {
    "en": ["I'm unable to provide details about this image.", "I cannot see the image clearly enough to answer."],
    "fr": ["Je ne peux pas fournir de détails sur cette image.", "Je ne suis pas en mesure de voir l'image."],
    "it": ["Non posso fornire dettagli su questa immagine.", "Non sono in grado di vedere l'immagine."],
    "zh": ["我无法提供关于这张图片的详细信息。", "抱歉，我无法看清这张图片。"],
}


# ---------------------------------------------------------------------------
# Real reference distributions (train only)
# ---------------------------------------------------------------------------

other_type = retrieval_inject.other_type


COUNT_START = re.compile(r"^\W*(\d+|[零一二两三四五六七八九十]+)|^\W*\w+", re.I)


def count_patterns(r, ref):
    """Real non-number miscount spans: 'split' = the noun / measure word labeled as its own
    span right after a labeled number ("four« legs»", "四«个»"); 'phrase' = one span
    starting with the number ("quatre côtés")."""
    t, lang = r["response"], r["language"]
    spans = [s for s in r.get("labels", []) or [] if s["label"] == "miscounting"]
    bare = [s for s in spans if realism.NUMBER_ONLY.match(t[s["start"]:s["end"]].strip())]
    for s in spans:
        if s in bare:
            continue
        txt = t[s["start"]:s["end"]]
        if any(0 <= s["start"] - b["end"] <= 1 for b in bare):
            ref["count_pattern_n"]["split"] += 1
            ref["count_split_len"][lang].append(len(txt))
        else:
            m = COUNT_START.match(txt)
            if m and count_value(m.group(0).strip(" *\"«“"), lang) is not None:
                ref["count_pattern_n"]["phrase"] += 1
                ref["count_phrase_len"][lang].append(len(txt))


def reference(train):
    ref = {"prob": defaultdict(list), "ocr_len": [], "other": defaultdict(lambda: {"len": [], "pos": []}),
           "freetext_len": defaultdict(list), "count_phrase_len": defaultdict(list), "leak_text": [],
           "count_values": defaultdict(Counter), "count_split_len": defaultdict(list),
           "count_pattern_n": Counter()}
    ocr_quoted = ocr_n = cnt_num = cnt_n = 0
    for r in train:
        t = r["response"]
        for s in r.get("labels", []) or []:
            ref["prob"][(s["label"], r["language"])].append(s["prob"])
            txt = t[s["start"]:s["end"]]
            if s["label"] == "OCR":
                ref["ocr_len"].append(len(txt))
                ocr_n += 1
                ocr_quoted += bool(realism.QUOTE_BEFORE.search(t[max(0, s["start"] - 15):s["start"]]))
            elif s["label"] == "miscounting":
                cnt_n += 1
                bare = bool(realism.NUMBER_ONLY.match(txt.strip()))
                cnt_num += bare
                if bare and (v := count_value(txt.strip(" *"), r["language"])) is not None:
                    ref["count_values"][r["language"]][v] += 1
            elif s["label"] == "other":
                typ = other_type(txt, t[max(0, s["start"] - 3):s["end"] + 3])
                ref["other"][typ]["len"].append(len(txt))
                ref["other"][typ]["pos"].append(s["start"] / max(1, len(t)))
                if typ == "freetext":
                    ref["freetext_len"][r["language"]].append(len(txt))
                if typ == "leak" and s["start"] == 0 and len(txt) <= 12:
                    ref["leak_text"].append(txt)   # mostly "Human" (followed by ": ...")
        count_patterns(r, ref)
    ref["ocr_quoted_rate"] = ocr_quoted / ocr_n
    ref["count_bare_rate"] = cnt_num / cnt_n
    # site shares that reproduce the real span-level mix: count_split sites yield one
    # bare + one non-bare span; non-bare spans are split vs phrase in the real ratio
    # (real non-bare spans of neither pattern -- "Yes", "une seule" -- are not modeled)
    ns = 1 - ref["count_bare_rate"]
    S, P = ref["count_pattern_n"]["split"], ref["count_pattern_n"]["phrase"]
    x_split, x_phrase = ns * S / (S + P), ns * P / (S + P)
    x = {"count_num": ref["count_bare_rate"] - x_split, "count_split": x_split, "count_phrase": x_phrase}
    ref["count_mix"] = {k: v / sum(x.values()) for k, v in x.items()}
    tot = sum(len(v["len"]) for v in ref["other"].values())
    ref["other_mix"] = {k: len(v["len"]) / tot for k, v in ref["other"].items()}
    return ref


# ---------------------------------------------------------------------------
# Site finders: where in a clean response an edit goes (shared by C2 and C3)
# ---------------------------------------------------------------------------

def kind_mix(ref, category):
    """Real share of each site kind per category (train)."""
    if category == "OCR":
        return {"in_quote": ref["ocr_quoted_rate"], "unquoted": 1 - ref["ocr_quoted_rate"]}
    if category == "miscounting":
        return dict(ref["count_mix"])
    return dict(ref["other_mix"])


def ocr_site(r, kind, ocr_cache, ref, rng):
    t = r["response"]
    dets = [d for d in ocr_cache.get(r["image_name"], []) if len(d) >= 3]
    quoted = []
    for m in QUOTED_SPAN.finditer(t):
        g = 1 if m.group(1) else 2
        if g == 2:   # **bold** is often plain emphasis ("**almeno 6 veicoli**"): only if EasyOCR saw it
            txt = m.group(2).lower()
            if not any(d.lower() in txt or txt in d.lower() for d in dets):
                continue
        quoted.append((m.start(g), m.end(g)))
    inside = lambda i: any(a <= i < b for a, b in quoted)
    unquoted = []
    for det in dets:
        for m in re.finditer(re.escape(det), t, re.I):
            if not inside(m.start()):
                unquoted.append((m.start(), m.end()))
    pool = quoted if kind == "in_quote" else unquoted
    if not pool:
        return None
    L = max(1, rng.choice(ref["ocr_len"]))
    # a string at least L long if there is one (it is cropped to ~L below)
    s, e = min(pool, key=lambda x: (max(0, L - (x[1] - x[0])), rng.random()))
    if r["language"] == "zh" or " " not in t[s:e]:
        if e - s > L:
            off = rng.randrange(0, e - s - L + 1)
            s, e = s + off, s + off + L
    else:
        words = [(s + m.start(), s + m.end()) for m in re.finditer(r"\S+", t[s:e])]
        best = min(((abs((words[j][1] - words[i][0]) - L), words[i][0], words[j][1])
                    for i in range(len(words)) for j in range(i, len(words))), default=None)
        if best:
            _, s, e = best
    return {"kind": kind, "start": s, "end": e, "insert": False}


NUM_WORDS = {lg: sorted(set(sc.NUMBER_WORDS[lg]) | set(sc.NUMBER_WORD_ALIASES.get(lg, {})), key=len, reverse=True)
             for lg in LANGS}


def count_value(num_text, lang):
    if num_text.isdigit():
        return int(num_text)
    lookup = {w: i for i, w in enumerate(sc.NUMBER_WORDS[lang])} | sc.NUMBER_WORD_ALIASES.get(lang, {})
    return lookup.get(num_text.lower())


def count_cands(r):
    """Count expressions in a response: (start, end, phrase_end, value, is_digit, fixes);
    fixes = grammar edits for a value-1 source (see one_fixes), None otherwise."""
    t, lang = r["response"], r["language"]
    cands = []
    if lang == "zh":   # no agreement in zh: 一个 -> 两个 needs no other edit
        for m in re.finditer(rf"([零一二两三四五六七八九十]+|\d{{1,3}})([{ZH_MEASURE}])", t):
            v = count_value(m.group(1), lang)
            if v is not None and v >= 1:
                cands.append((m.start(1), m.end(1), m.end(2), v, m.group(1).isdigit(), (None, []) if v == 1 else None))
    else:
        words = "|".join(re.escape(w) for w in NUM_WORDS[lang])
        for m in re.finditer(rf"\b(\d{{1,3}}|{words})\b[ \t]+([^\W\d_][\w'-]*)", t, re.I):
            nxt, prev = m.group(2).lower(), t[max(0, m.start() - 12):m.start()].lower().split()
            if nxt in EXCL_NEXT or (prev and prev[-1].strip(".:") in EXCL_PREV):
                continue
            if lang == "en" and nxt.endswith(("ing", "ed", "ly")):
                continue   # "10 walking", "two closed": not the counted noun
            if m.start() > 0 and t[m.start() - 1] in "-/":
                continue   # "5-in-1", "1/2"
            if re.match(r"[%:./,]\d", t[m.end(1):m.end(1) + 2]):
                continue
            v = count_value(m.group(1), lang)
            if v is not None and v >= 2:
                cands.append((m.start(1), m.end(1), m.end(2), v, m.group(1).isdigit(), None))
            elif v == 1 and (m.group(1).isdigit() or m.group(1).lower() in ONE_WORDS[lang]):
                fixes = one_fixes(t, m, lang)
                if fixes is not None:
                    cands.append((m.start(1), m.end(1), m.end(2), v, m.group(1).isdigit(), fixes))
    return cands


def reachable(c, w, lang):
    """Can count candidate `c` be shifted to wrong value `w` (+-1 or +-2, never to 0/1, as a
    word if it is one)?"""
    return w >= 2 and 1 <= abs(c[3] - w) <= 2 and (c[4] or w < len(sc.NUMBER_WORDS[lang]))


def count_reach_rates(clean, ref):
    """P(a clean response can reach wrong value w), per language -- for importance weighting."""
    rates = {}
    for lg in LANGS:
        cs = [count_cands(r) for r in clean if r["language"] == lg]
        rates[lg] = {w: sum(any(reachable(c, w, lg) for c in cc) for cc in cs) / max(1, len(cs))
                     for w in ref["count_values"][lg] if w >= 2}
    return rates


def count_site(r, kind, ref, rng):
    """Counts >= 2 only: shifting 'one'/'a'/'une' breaks noun agreement ("deux chemise")
    and produced the unnatural "zero car" of the first version. The wrong value w is
    sampled among the values this response's counts can reach, with weight
    p_real(w) / P(reachable | w): over the pool, wrong values then follow the real
    distribution (mostly "two") instead of what clean responses happen to allow
    (their counts are mostly 2, so plain sampling gives mostly "three"/"four")."""
    lang = r["language"]
    freq, rate = ref["count_values"][lang], ref["count_reach"][lang]
    all_cands = count_cands(r)
    vals = [w for w in rate if rate[w] > 0 and any(reachable(c, w, lang) for c in all_cands)]
    if not vals:
        return None
    wrong = rng.choices(vals, weights=[freq[w] / rate[w] for w in vals])[0]
    cands = [c for c in all_cands if reachable(c, wrong, lang)]
    if kind == "count_phrase":   # the phrase whose length is closest to a real one
        L = rng.choice(ref["count_phrase_len"][lang])
        c = min(cands, key=lambda c: (abs(c[2] - c[0] - L), rng.random()))
    elif kind == "count_split":  # the noun / measure word whose length is closest to a real one
        L = rng.choice(ref["count_split_len"][lang])
        c = min(cands, key=lambda c: (abs(c[2] - c[1] - L), rng.random()))
    else:
        c = rng.choice(cands)
    ns, ne, phrase_end, _, _, fixes = c
    site = {"kind": kind, "start": ns, "end": ne if kind == "count_num" else phrase_end, "num_end": ne,
            "wrong_value": wrong, "insert": False}
    if fixes is not None:   # value-1 source: noun to plural, verb agreement
        noun, verbs = fixes
        site["verb_fixes"] = verbs
        if noun is not None:
            site["noun_fix"] = noun
    return site


def other_site(r, kind, ref, rng):
    t, lang = r["response"], r["language"]
    dist = ref["other"][kind]
    if kind == "freetext":
        if lang == "zh":
            hits = [(m.start(), m.end(), ZH_SWAP[m.group()]) for m in ZH_SWAP_RE.finditer(t)]
        else:
            mapping = sc.STATE_DIRECTION_ANTONYMS[lang]
            hits = [(m.start(), m.end(), match_case(mapping[m.group().lower()], m.group()))
                    for m in re.finditer(r"[A-Za-zÀ-ÖØ-öø-ÿ]+", t) if m.group().lower() in mapping]
        if not hits:
            return None
        s, e, repl = rng.choice(hits)
        word = (s, e, repl)
        L = rng.choice(ref["freetext_len"][lang])
        # extend over the following words (zh: characters) up to the sampled real
        # length, staying inside the sentence
        if lang == "zh":
            stop = re.search(r"[。！？\n]", t[e:])
            e = min(max(e, s + L), e + (stop.start() if stop else len(t) - e))
        else:
            ends = [e]
            for m in re.finditer(r"\s+[^\s.!?]+", t[e:]):
                if "\n" in m.group() or e + m.end() - s > 2 * L:
                    break
                ends.append(e + m.end())
            e = min(ends, key=lambda x: abs(x - s - L))   # word boundary nearest the real length
            e = s + len(t[s:e].rstrip(",;:"))
        return {"kind": "freetext", "start": s, "end": e, "word": word, "insert": False}
    pos = rng.choice(dist["pos"])
    bounds = [0] + [m.end() for m in re.finditer(r"(?<=[.!?。！？])\s*", t)] + [len(t)]
    p = min(bounds, key=lambda b: abs(b / max(1, len(t)) - pos))
    site = {"kind": kind, "start": p, "end": p, "insert": True, "target_len": rng.choice(dist["len"])}
    if kind == "leak":
        # real leaks are a template token at the start, "Human" + ": ..." (only "Human"
        # labeled); a template has no "content" to write, so C2 and C3 share it
        site["leak_text"] = rng.choice(ref["leak_text"])
        site["tail"] = " " if site["leak_text"].endswith(":") else ": "
    return site


# ---------------------------------------------------------------------------
# Heuristic (C2) content for a site, and the shared edit-application function
# ---------------------------------------------------------------------------

SIMILAR = {"a": "o", "o": "a", "e": "c", "c": "e", "n": "h", "h": "n", "u": "v", "v": "u", "t": "f", "f": "t",
           "E": "F", "F": "E", "P": "R", "R": "P", "C": "G", "G": "C", "U": "V", "V": "U"}


def misread(text, rng):
    out = sc.confusable_swap(text, rng, n_swaps=rng.choice([1, 1, 2]))
    if out and out != text:
        return out
    chars = list(text)
    idx = [i for i, c in enumerate(chars) if c.isalnum()]
    if not idx:
        return None
    i = rng.choice(idx)
    c = chars[i]
    if c.isdigit():
        chars[i] = str((int(c) + rng.choice([1, 2, 8, 9])) % 10)
    else:
        op = rng.choice(["sub", "drop", "dup", "swap"])
        if op == "sub" and c in SIMILAR:
            chars[i] = SIMILAR[c]
        elif op == "drop" and len(idx) > 2:
            del chars[i]
        elif op == "swap" and i + 1 < len(chars) and chars[i + 1].isalnum() and chars[i + 1] != c:
            chars[i], chars[i + 1] = chars[i + 1], chars[i]
        else:
            chars.insert(i, c)
    new = "".join(chars)
    return new if new != text else None


def write_count(num_text, nv, lang):
    """`nv` in the same form as the original number (digits or a word)."""
    words = sc.NUMBER_WORDS[lang]
    if num_text.isdigit():
        return str(nv)
    w = "两" if lang == "zh" and nv == 2 else words[nv]   # 两个, never 二个, before a measure word
    return w.capitalize() if num_text[:1].isupper() and lang != "zh" else w


def heuristic_content(base, site, ref, rng):
    t, lang, kind = base["response"], base["language"], site["kind"]
    seg = t[site["start"]:site["end"]]
    if kind in ("in_quote", "unquoted"):
        return misread(seg, rng)
    if kind in ("count_num", "count_phrase", "count_split"):
        num = t[site["start"]:site["num_end"]]
        return write_count(num, site["wrong_value"], lang) + count_tail(t, site)
    if kind == "freetext":
        ws, we, repl = site["word"]
        return seg[:ws - site["start"]] + repl + seg[we - site["start"]:]
    if kind == "leak":
        return site["leak_text"]
    if kind == "refusal":
        return rng.choice(REFUSALS[lang])
    if kind == "repetition":
        words = [w for w in re.split(r"\s+", t) if w] if lang != "zh" else list(re.sub(r"\s+", "", t))
        if len(words) < 3:
            return None
        k = rng.choice([1, 2, 3])
        i = rng.randrange(0, max(1, len(words) - k))
        sep = " " if lang != "zh" else ""
        chunk = sep.join(words[i:i + k])
        reps = max(3, site["target_len"] // max(1, len(chunk) + len(sep)))
        return sep.join([chunk] * reps)
    raise ValueError(kind)


def count_tail(t, site):
    """Text between the number and the end of a count span, with the noun in the plural
    when the source count was 1 ("" for a bare-number span)."""
    if site["end"] == site["num_end"]:
        return ""
    nf = site.get("noun_fix")
    return t[site["num_end"]:nf[0]] + nf[2] if nf else t[site["num_end"]:site["end"]]


def outside_edits(site):
    """Grammar edits outside the labeled span (verb agreement; the noun for bare-number spans)."""
    edits = list(site.get("verb_fixes", []))
    if site.get("noun_fix") and site["end"] == site["num_end"]:
        edits.append(tuple(site["noun_fix"]))
    return edits


def apply_edit(base, site, content, category, method, prob):
    """Same function for C2 and C3, so position handling is identical by construction."""
    t = base["response"]
    if site.get("verb_fixes") or site.get("noun_fix"):
        # labeled replacement + unlabeled grammar edits, applied left to right
        edits = sorted([(site["start"], site["end"], content, True)] +
                       [(a, b, r, False) for a, b, r in outside_edits(site)])
        new, last, s = "", 0, None
        for a, b, rep, labeled in edits:
            assert a >= last, "overlapping edits"
            new += t[last:a]
            if labeled:
                s = len(new)
            new += rep
            last = b
        new += t[last:]
        e = s + len(content)
    elif site["insert"]:
        p = site["start"]
        sep = "" if base["language"] == "zh" else " "
        tail = site.get("tail", sep if p < len(t) and not content.endswith(sep) else "")
        new = t[:p] + content + tail + t[p:]
        s, e = p, p + len(content)
    else:
        new = t[:site["start"]] + content + t[site["end"]:]
        s, e = site["start"], site["start"] + len(content)
    rec = sc.make_synthetic_record(base, new, s, e, category, f"{method.lower()}{category}", prob)
    if site["kind"] == "count_split":
        # number and noun as two adjacent spans, like real annotations ("four« legs»");
        # the noun is the original text after the number, so content must end with it
        tail = count_tail(t, site)
        if not content.endswith(tail) or len(content) == len(tail):
            return None
        mid = e - len(tail)
        rec["labels"] = [{"start": s, "end": mid, "prob": prob, "label": category},
                         {"start": mid, "end": e, "prob": prob, "label": category}]
    rec["_meta"] = {"method": method, "category": category, "base_id": base["id"], "site_id": site["site_id"],
                    "kind": site["kind"]}
    return rec


# ---------------------------------------------------------------------------
# Stages
# ---------------------------------------------------------------------------

def write_jsonl(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


def stage_c1c4(train):
    mult = sc.compute_rarity_multipliers(train, max_multiplier=3)
    ups = sc.upsample_by_tier_and_rarity(train, rarity_multipliers=mult, max_multiplier=3, seed=DATA_SEED)
    dups = ups[len(train):]
    write_jsonl(PART2_DATA_DIR / "C1_upsample.jsonl", dups)
    print(f"C1: {len(dups)} duplicated real records (rarity multipliers {mult})")

    cands = retrieval_inject.build_candidates(train, CATEGORIES, DATA_SEED)
    rows = [r for cat in CATEGORIES for lg in LANGS for r in cands[(cat, lg)]]
    write_jsonl(PART2_DATA_DIR / "C4_candidates.jsonl", rows)
    volume = {f"{cat}/{lg}": min(TARGET_PER_CELL, len(cands[(cat, lg)])) for cat in CATEGORIES for lg in LANGS}
    (PART2_DATA_DIR / "volume.json").write_text(json.dumps(volume, indent=2))
    modes = Counter(r["_meta"]["mode"] for r in rows)
    print(f"C4 candidates per cell: { {k: len(v) for k, v in sorted(cands.items())} }  modes={dict(modes)}")
    print(f"volume (target per cell, capped by C4): {volume}")


def stage_sites(train):
    ref = reference(train)
    ref["count_reach"] = count_reach_rates([r for r in train if not r.get("labels")], ref)
    ocr_cache = json.loads(OCR_CACHE.read_text(encoding="utf-8"))
    volume = json.loads((PART2_DATA_DIR / "volume.json").read_text())
    rng = random.Random(DATA_SEED)
    # same base filter as C4: records labeled e.g. "it" but written in Portuguese/English are skipped
    clean = [r for r in train if not r.get("labels") and retrieval_inject.lang_ok(r["response"], r["language"])]
    rng.shuffle(clean)
    finders = {"OCR": lambda r, k: ocr_site(r, k, ocr_cache, ref, rng),
               "miscounting": lambda r, k: count_site(r, k, ref, rng),
               "other": lambda r, k: other_site(r, k, ref, rng)}
    by_id = {r["id"]: r for r in clean}
    sites, c2_rows, shortfall = [], [], {}
    for cat in CATEGORIES:
        mix = kind_mix(ref, cat)
        for lg in LANGS:
            want = int(volume[f"{cat}/{lg}"] * OVERSAMPLE)
            # a quota per kind: falling back to whichever kind is easy to find skews the
            # mix (e.g. quoted OCR text is far easier to find than unquoted)
            # one pass over the bases (one site per base per cell); each base goes to the
            # kind furthest below its quota, so a scarce pool is shared in the real ratio
            per_kind = {kind: ([], max(1, round(want * share))) for kind, share in mix.items()}
            for base in clean:
                open_kinds = sorted((len(f) / q, kind) for kind, (f, q) in per_kind.items() if len(f) < q)
                if not open_kinds:
                    break
                if base["language"] != lg:
                    continue
                for _, kind in open_kinds:
                    site = finders[cat](base, kind)
                    if site is None:
                        continue
                    content = heuristic_content(base, site, ref, rng)
                    if content is None:
                        continue
                    site |= {"base_id": base["id"], "c2_content": content}
                    per_kind[kind][0].append(site)
                    break
            for kind, (found, quota) in per_kind.items():
                if len(found) < quota:
                    shortfall[f"{cat}/{lg}/{kind}"] = f"{len(found)}/{quota}"
            # interleave kinds by quota so any prefix finalize takes keeps the real mix
            order = sorted(((i + 0.5) / q, kind, s) for kind, (found, q) in per_kind.items()
                           for i, s in enumerate(found))
            for k, (_, _, site) in enumerate(order):
                base = by_id[site["base_id"]]
                site |= {"site_id": f"{cat}_{lg}_{k:03d}", "category": cat, "language": lg,
                         "prob": rng.choice(ref["prob"][(cat, lg)])}
                sites.append(site)
                c2_rows.append(apply_edit(base, site, site["c2_content"], cat, "C2", site["prob"]))
    write_jsonl(PART2_DATA_DIR / "sites.jsonl", sites)
    write_jsonl(PART2_DATA_DIR / "C2_candidates.jsonl", c2_rows)
    kinds = Counter((s["category"], s["kind"]) for s in sites)
    print(f"sites: {len(sites)}  kinds: {dict(kinds)}")
    print(f"kind quota shortfalls (found/quota): {shortfall or 'none'}")


def first_span_len(rec, category):
    s = next(l for l in rec["labels"] if l["label"] == category)
    return s["end"] - s["start"]


def stage_finalize(train):
    by_id = {r["id"]: r for r in train}
    volume = json.loads((PART2_DATA_DIR / "volume.json").read_text())
    sites = data_io.load_jsonl(PART2_DATA_DIR / "sites.jsonl")
    c3_path = PART2_DATA_DIR / "C3_content.jsonl"
    if not c3_path.exists():
        raise SystemExit("C3_content.jsonl missing -- run llm_paired first")
    c3 = {r["site_id"]: r["content"] for r in data_io.load_jsonl(c3_path) if r.get("content")}
    c4 = defaultdict(list)
    for r in data_io.load_jsonl(PART2_DATA_DIR / "C4_candidates.jsonl"):
        c4[(r["_meta"]["category"], r["language"])].append(r)

    out = {"C2": [], "C3": [], "C4": []}
    final_volume = {}
    for cat in CATEGORIES:
        for lg in LANGS:
            paired = []
            for s in sites:
                if s["category"] != cat or s["language"] != lg or s["site_id"] not in c3:
                    continue
                base = by_id[s["base_id"]]
                r2 = apply_edit(base, s, s["c2_content"], cat, "C2", s["prob"])
                r3 = apply_edit(base, s, c3[s["site_id"]], cat, "C3", s["prob"])
                if r2 is not None and r3 is not None:
                    paired.append((r2, r3))
            n = min(volume[f"{cat}/{lg}"], len(paired), len(c4[(cat, lg)]))
            final_volume[f"{cat}/{lg}"] = n
            # stratified selection (same procedure for every method): the n kept per cell
            # follow the real span-length distribution of that (category, language).
            # C2/C3 pairs are ordered by the C2 span so both keep identical sites.
            ref_len = realism.real_lengths(train, cat, lg)
            seed = DATA_SEED + CATEGORIES.index(cat) * 10 + LANGS.index(lg)
            paired = realism.matched_order(paired, [first_span_len(r2, cat) for r2, _ in paired], ref_len, seed)
            c4_cell = realism.matched_order(c4[(cat, lg)], [first_span_len(r, cat) for r in c4[(cat, lg)]],
                                            ref_len, seed)
            for r2, r3 in paired[:n]:
                out["C2"].append(r2)
                out["C3"].append(r3)
            out["C4"].extend(c4_cell[:n])

    # identical counts and (for C2/C3) identical base records + sites, asserted not assumed
    cnt = {m: Counter((r["_meta"]["category"], r["language"]) for r in rows) for m, rows in out.items()}
    assert cnt["C2"] == cnt["C3"] == cnt["C4"], cnt
    assert [r["_meta"]["site_id"] for r in out["C2"]] == [r["_meta"]["site_id"] for r in out["C3"]]
    for m, name in [("C2", "C2_heuristic"), ("C3", "C3_llm"), ("C4", "C4_retrieval")]:
        write_jsonl(PART2_DATA_DIR / f"{name}.jsonl", out[m])
    (PART2_DATA_DIR / "volume_final.json").write_text(json.dumps(final_volume, indent=2))
    print(f"final volume per cell: {final_volume}  (total {sum(final_volume.values())} per method)")
    rep = realism.gate({m: PART2_DATA_DIR / f"{n}.jsonl" for m, n in
                        [("C2", "C2_heuristic"), ("C3", "C3_llm"), ("C4", "C4_retrieval")]})
    realism.print_report(rep)


def stage_pilot(train):
    """Stage 1 pilot: invention + mischaracterization records for "C4 extended" (see
    results/preregistration.md, R2). Trained together with C4_candidates.jsonl, so the
    extended condition differs from C4 by exactly these records."""
    cats = ["invention", "mischaracterization"]
    cands = retrieval_inject.build_candidates(train, cats, DATA_SEED)
    rows = []
    for cat in cats:
        for lg in LANGS:
            pool = cands[(cat, lg)]
            seed = DATA_SEED + 100 + cats.index(cat) * 10 + LANGS.index(lg)
            ordered = realism.matched_order(pool, [first_span_len(r, cat) for r in pool],
                                            realism.real_lengths(train, cat, lg), seed)
            rows.extend(ordered[:PILOT_CLAIMS_PER_CELL])
    write_jsonl(PART2_DATA_DIR / "C4x_claims.jsonl", rows)
    print(f"pool {({f'{k[0][:5]}/{k[1]}': len(v) for k, v in sorted(cands.items())})}")
    print(f"kept {len(rows)}  modes {dict(Counter(r['_meta']['mode'] for r in rows))}")
    extended = data_io.load_jsonl(PART2_DATA_DIR / "C4_candidates.jsonl") + rows
    rep = {c: realism.compare(train, extended, c) for c in data_io.LABEL_ORDER}
    (realism.RESULTS_DIR / "part2_pilot_realism.json").write_text(json.dumps(rep, indent=2))
    for c, x in rep.items():
        print(f"  {c:20} n={x['n_synth']:5} KS(len)={x['ks_len']:.3f} KS(pos)={x['ks_pos']:.3f} "
              f"TV(prob)={x['tv_prob']:.3f} -> {'ok' if x['pass'] else 'FAIL'}")


def stage_preview(method, n, seed=0):
    name = {"C2": "C2_candidates", "C3": "C3_llm", "C4": "C4_candidates"}[method]
    path = PART2_DATA_DIR / f"{name}.jsonl"
    rows = data_io.load_jsonl(path)
    for r in random.Random(seed).sample(rows, min(n, len(rows))):
        t = r["response"]
        for s in r["labels"]:
            print(f"[{method} {s['label']} p={s['prob']:.2f} {r['_meta'].get('kind', r['_meta'].get('mode'))}] "
                  f"...{t[max(0, s['start'] - 70):s['start']]}«{t[s['start']:s['end']][:120]}»{t[s['end']:s['end'] + 50]}...")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--stage", required=True, choices=["c1c4", "sites", "finalize", "preview", "pilot"])
    p.add_argument("--method", default="C2")
    p.add_argument("--n", type=int, default=10)
    args = p.parse_args()
    train = data_io.load_jsonl(data_io.TRAIN_FILE)
    {"c1c4": lambda: stage_c1c4(train), "sites": lambda: stage_sites(train),
     "finalize": lambda: stage_finalize(train), "pilot": lambda: stage_pilot(train), "preview": lambda: stage_preview(args.method, args.n)}[args.stage]()


if __name__ == "__main__":
    main()
