"""Part 3b: learned neural probe with jointly-trained span-boundary decoding.

Token features of the frozen judge (features.py: entropy, nll, lookback, hidden_-1;
standardized on train) -> Linear 512 -> BiGRU 256 (+ language embedding) -> one
2-state linear-chain CRF per category (O / inside). Each CRF is trained jointly with
the encoder on the exact sequence likelihood, so span boundaries are decoded, not
thresholded. One fixed hyperparameter set; early stopping on val_select LOSS (not on
the metric); no threshold search.

Two target/decoding modes (results/preregistration.md, R3):
  default       the judge's single token label (the first gold span covering the token
                wins, so a token never has two labels); a token predicted by several
                categories keeps the one with the highest marginal. -> crf_seed{seed}
  --multilabel  every gold span covering a token sets its category, so text that gold
                labels both invention and mischaracterization trains both heads; each
                category's Viterbi path becomes spans independently. -> crfml_seed{seed}
Consecutive tokens of one category -> one span (char offsets from the tokenizer
offsets); span prob = mean marginal. Records the judge could not process get an
explicit empty prediction (features.py lists them).

The five CRFs are one batched module (parameters per category), checked against brute
force in --selftest.

Usage (nlp2):
    python -m thesis_v2.part3_uncert.crf_probe --selftest
    python -m thesis_v2.part3_uncert.crf_probe --seed 42 --gpu_index 0 [--multilabel]
"""
from __future__ import annotations

import argparse
import itertools
import json

import numpy as np
import torch
import torch.nn as nn

from thesis_v2.common import evaluate as ev
from thesis_v2.common.paths import LANGUAGES, RUNS_DIR
from thesis_v2.common.splits import load_split
from thesis_v2.part3_uncert import features
from solutions.common import data_io

LABELS = data_io.LABEL_ORDER
HP = {"proj": 512, "gru": 256, "lang_dim": 8, "dropout": 0.2, "lr": 1e-3, "batch": 16,
      "max_epochs": 20, "patience": 3, "max_len": 1024}


class BinaryCRFs(nn.Module):
    """C independent 2-state linear-chain CRFs over emissions (B, T, C, 2), mask (B, T)."""

    def __init__(self, n):
        super().__init__()
        self.trans = nn.Parameter(torch.zeros(n, 2, 2))
        self.start = nn.Parameter(torch.zeros(n, 2))
        self.end = nn.Parameter(torch.zeros(n, 2))

    def _forward_alpha(self, em, mask):
        alpha = self.start + em[:, 0]                                    # (B, C, 2)
        alphas = [alpha]
        for t in range(1, em.shape[1]):
            nxt = torch.logsumexp(alpha.unsqueeze(3) + self.trans + em[:, t].unsqueeze(2), dim=2)
            alpha = torch.where(mask[:, t, None, None], nxt, alpha)
            alphas.append(alpha)
        return alpha, alphas

    def nll(self, em, tags, mask):
        """Negative log-likelihood per sequence and category, (B, C). tags: (B, T, C)."""
        C = em.shape[2]
        c = torch.arange(C, device=em.device).expand(em.shape[0], C)

        def emit(t):
            return em[:, t].gather(2, tags[:, t].unsqueeze(2)).squeeze(2)
        score = self.start[c, tags[:, 0]] + emit(0)
        for t in range(1, em.shape[1]):
            step = self.trans[c, tags[:, t - 1], tags[:, t]] + emit(t)
            score = score + step * mask[:, t, None]
        last_t = (mask.sum(1) - 1).long().view(-1, 1, 1).expand(-1, 1, C)
        score = score + self.end[c, tags.gather(1, last_t).squeeze(1)]
        alpha, _ = self._forward_alpha(em, mask)
        return torch.logsumexp(alpha + self.end, dim=2) - score

    def marginals(self, em, mask):
        """P(tag = 1) per token and category, (B, T, C), by forward-backward."""
        _, alphas = self._forward_alpha(em, mask)
        T = em.shape[1]
        is_last_at = (mask.sum(1).long() - 1)
        end = self.end.expand_as(alphas[0])
        beta = end.clone()
        betas = [None] * T
        for t in range(T - 1, -1, -1):
            if t < T - 1:
                nxt = torch.logsumexp(self.trans + (em[:, t + 1] + beta).unsqueeze(2), dim=3)
                beta = torch.where(mask[:, t + 1, None, None], nxt, beta)
            beta = torch.where((is_last_at == t)[:, None, None], end, beta)
            betas[t] = beta
        logZ = torch.logsumexp(alphas[-1] + self.end, dim=2, keepdim=True)
        post = torch.stack([alphas[t] + betas[t] - logZ for t in range(T)], dim=1)
        return post.exp()[..., 1]

    def viterbi(self, em, mask):
        """Best tag path per category, (B, T, C)."""
        score = self.start + em[:, 0]
        back = []
        keep = torch.arange(2, device=em.device).expand_as(score)
        for t in range(1, em.shape[1]):
            best, idx = (score.unsqueeze(3) + self.trans).max(dim=2)
            on = mask[:, t, None, None]
            score = torch.where(on, best + em[:, t], score)
            back.append(torch.where(on, idx, keep))
        tag = (score + self.end).argmax(2)
        path = [tag]
        for idx in reversed(back):
            tag = idx.gather(2, tag.unsqueeze(2)).squeeze(2)
            path.append(tag)
        return torch.stack(path[::-1], dim=1)


class Probe(nn.Module):
    def __init__(self, d_in):
        super().__init__()
        self.proj = nn.Sequential(nn.Linear(d_in, HP["proj"]), nn.ReLU(), nn.Dropout(HP["dropout"]))
        self.lang = nn.Embedding(len(LANGUAGES), HP["lang_dim"])
        self.gru = nn.GRU(HP["proj"] + HP["lang_dim"], HP["gru"], batch_first=True, bidirectional=True)
        self.heads = nn.Linear(2 * HP["gru"], 2 * len(LABELS))
        self.crfs = BinaryCRFs(len(LABELS))

    def emissions(self, x, lang):
        h = self.proj(x)
        h = torch.cat([h, self.lang(lang).unsqueeze(1).expand(-1, h.shape[1], -1)], dim=-1)
        h, _ = self.gru(h)
        return self.heads(h).view(x.shape[0], x.shape[1], len(LABELS), 2)

    def loss(self, x, lang, tags, mask):
        return self.crfs.nll(self.emissions(x, lang), tags, mask).sum(1).mean()


def multilabel_targets(data, split):
    """(n_tokens, C) 0/1: category c is set if ANY gold span of label c covers the token
    (same overlap rule as the judge's single label). Checked to contain the single label."""
    gold = {r["id"]: r.get("labels") or [] for r in data_io.load_jsonl(features.SPLIT_FILES[split])}
    ptr, os_, oe = data["rec_ptr"], data["off_start"], data["off_end"]
    Y = np.zeros((len(data["y"]), len(LABELS)), np.int8)
    for j, rid in enumerate(data["record_id"]):
        a, b = ptr[j], ptr[j + 1]
        s, e = os_[a:b], oe[a:b]
        for lb in gold[rid]:
            Y[a:b, LABELS.index(lb["label"])] |= ((s < lb["end"]) & (e > lb["start"]) & (s != e))
    y = data["y"].astype(np.int64)
    pos = y >= 0
    if not (Y[np.flatnonzero(pos), y[pos]] == 1).all() or Y[~pos].any():
        raise AssertionError(f"{split}: multi-label targets disagree with the judge's single labels")
    n_multi = int((Y.sum(1) > 1).sum())
    print(f"{split}: {int(pos.sum())} labeled tokens, {n_multi} carry more than one category")
    return Y


def batches(data, idx, mean, std, device, lang_ids, shuffle_rng=None):
    idx = list(idx)
    if shuffle_rng is not None:
        shuffle_rng.shuffle(idx)
    ptr = data["rec_ptr"]
    C = len(LABELS)
    for i in range(0, len(idx), HP["batch"]):
        chunk = idx[i:i + HP["batch"]]
        lens = [min(HP["max_len"], ptr[j + 1] - ptr[j]) for j in chunk]
        T = max(lens)
        x = np.zeros((len(chunk), T, data["x"].shape[1]), np.float32)
        tags = np.zeros((len(chunk), T, C), np.int64)
        m = np.zeros((len(chunk), T), bool)
        for b, (j, L) in enumerate(zip(chunk, lens)):
            x[b, :L] = (data["x"][ptr[j]:ptr[j] + L].astype(np.float32) - mean) / std
            if "Y" in data:
                tags[b, :L] = data["Y"][ptr[j]:ptr[j] + L]
            else:
                y = data["y"][ptr[j]:ptr[j] + L].astype(np.int64)
                tags[b, :L] = y[:, None] == np.arange(C)
            m[b, :L] = True
        lang = torch.tensor([lang_ids[data["language"][j]] for j in chunk], device=device)
        yield chunk, lens, torch.from_numpy(x).to(device), lang, torch.from_numpy(tags).to(device), \
            torch.from_numpy(m).to(device)


@torch.no_grad()
def predict(model, data, mean, std, device, lang_ids, multilabel=False):
    model.eval()
    out = {}
    for chunk, lens, x, lang, _, m in batches(data, range(len(data["record_id"])), mean, std, device, lang_ids):
        em = model.emissions(x, lang)
        paths = model.crfs.viterbi(em, m)
        marg = model.crfs.marginals(em, m)
        if multilabel:
            on_c = paths.bool()                                           # (B, T, C)
        else:
            score = torch.where(paths.bool(), marg, torch.full_like(marg, -1.0))
            on_c = torch.zeros_like(paths, dtype=torch.bool)
            on_c.scatter_(2, score.argmax(-1, keepdim=True), (score.max(-1, keepdim=True).values > 0))
        on_c, marg = on_c.cpu().numpy(), marg.cpu().numpy()
        for b, j in enumerate(chunk):
            p0 = data["rec_ptr"][j]
            spans = []
            for c in range(len(LABELS)):
                t = 0
                while t < lens[b]:
                    if not on_c[b, t, c]:
                        t += 1
                        continue
                    u = t
                    while u < lens[b] and on_c[b, u, c]:
                        u += 1
                    spans.append({"start": int(data["off_start"][p0 + t]), "end": int(data["off_end"][p0 + u - 1]),
                                  "prob": round(float(marg[b, t:u, c].mean()), 4), "label": LABELS[c]})
                    t = u
            out[data["record_id"][j]] = sorted(spans, key=lambda s: (s["start"], s["label"]))
    return out


def to_records(pred, data, split):
    """Full prediction file: every record of the split, explicit empty for skipped ones."""
    recs = data_io.load_jsonl(features.SPLIT_FILES[split])
    skipped = set(data["skipped_ids"].tolist())
    rows = []
    for r in recs:
        if r["id"] not in pred and r["id"] not in skipped:
            raise ev.CoverageError(f"{r['id']} has neither features nor an explicit skip")
        rows.append({"id": r["id"], "labels": pred.get(r["id"], [])})
    return rows, [i for i in pred]


def selftest():
    """Viterbi == brute force and marginals sum correctly on random toy sequences, per category."""
    torch.manual_seed(0)
    C = 3
    crf = BinaryCRFs(C)
    with torch.no_grad():
        crf.trans.normal_(); crf.start.normal_(); crf.end.normal_()
    for T in (1, 2, 5):
        em = torch.randn(3, T, C, 2)
        mask = torch.ones(3, T, dtype=torch.bool)
        paths = list(itertools.product([0, 1], repeat=T))
        vit, marg = crf.viterbi(em, mask), crf.marginals(em, mask)

        def path_score(b, c, pth):
            s = crf.start[c, pth[0]] + em[b, 0, c, pth[0]]
            for t in range(1, T):
                s = s + crf.trans[c, pth[t - 1], pth[t]] + em[b, t, c, pth[t]]
            return s + crf.end[c, pth[-1]]
        tags = torch.tensor([[[paths[(c + 1) % len(paths)][t] for c in range(C)] for t in range(T)]] * 3)
        nll = crf.nll(em, tags, mask)
        for b in range(3):
            for c in range(C):
                scores = torch.stack([path_score(b, c, p) for p in paths])
                assert list(paths[int(scores.argmax())]) == vit[b, :, c].tolist()
                logZ = torch.logsumexp(scores, 0)
                assert torch.allclose(nll[b, c], logZ - scores[(c + 1) % len(paths)], atol=1e-4)
                probs = (scores - logZ).exp()
                m1 = torch.stack([sum(probs[i] for i, p in enumerate(paths) if p[t] == 1) for t in range(T)])
                assert torch.allclose(marg[b, :, c], m1, atol=1e-4)
    # padded sequence: masked steps must not change the result
    em = torch.randn(1, 6, C, 2)
    m_short = torch.tensor([[1, 1, 1, 0, 0, 0]], dtype=torch.bool)
    assert crf.viterbi(em, m_short)[0, :3].tolist() == crf.viterbi(em[:, :3], m_short[:, :3])[0].tolist()
    assert torch.allclose(crf.marginals(em, m_short)[0, :3], crf.marginals(em[:, :3], m_short[:, :3])[0], atol=1e-5)
    # the whole probe must overfit a tiny batch, including two categories on the same tokens
    model = Probe(10)
    x, lang = torch.randn(4, 12, 10), torch.tensor([0, 1, 2, 3])
    tags = torch.zeros(4, 12, len(LABELS), dtype=torch.long)
    tags[0, 3:6, 2] = 1; tags[1, 0:2, 4] = 1; tags[2, 7:12, 0] = 1; tags[2, 9:12, 1] = 1; tags[3, 5, 3] = 1
    mask = torch.ones(4, 12, dtype=torch.bool)
    mask[3, 8:] = False
    opt = torch.optim.Adam(model.parameters(), lr=1e-2)
    first = model.loss(x, lang, tags, mask).item()
    for _ in range(150):
        opt.zero_grad()
        loss = model.loss(x, lang, tags, mask)
        loss.backward()
        opt.step()
    model.eval()
    pred = model.crfs.viterbi(model.emissions(x, lang), mask)
    assert (pred[mask] == tags[mask]).all(), "probe failed to overfit a tiny batch"
    print(f"CRF selftest passed (Viterbi, NLL, marginals vs brute force per category; padding; "
          f"tiny-batch overfit incl. dual labels {first:.2f} -> {loss.item():.4f})")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--gpu_index", type=int, default=None)
    p.add_argument("--multilabel", action="store_true")
    p.add_argument("--selftest", action="store_true")
    args = p.parse_args()
    if args.selftest:
        return selftest()

    torch.manual_seed(args.seed)
    rng = np.random.default_rng(args.seed)
    device = f"cuda:{args.gpu_index}" if torch.cuda.is_available() and args.gpu_index is not None else \
        ("cuda" if torch.cuda.is_available() else "cpu")
    train, val, test = features.load("train"), features.load("val"), features.load("test")
    if args.multilabel:
        train["Y"], val["Y"] = multilabel_targets(train, "train"), multilabel_targets(val, "val")
    lang_ids = {lg: i for i, lg in enumerate(LANGUAGES)}
    xs = train["x"][rng.choice(len(train["x"]), min(500_000, len(train["x"])), replace=False)].astype(np.float32)
    mean, std = xs.mean(0), xs.std(0) + 1e-6

    sel = load_split("val_select")
    sel_idx = [j for j, rid in enumerate(val["record_id"]) if rid in sel]
    model = Probe(train["x"].shape[1]).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=HP["lr"])
    tag = f"crf{'ml' if args.multilabel else ''}_seed{args.seed}"
    out_dir = RUNS_DIR / "part3" / tag
    out_dir.mkdir(parents=True, exist_ok=True)
    best, bad, log = float("inf"), 0, []
    for epoch in range(HP["max_epochs"]):
        model.train()
        tr = []
        for _, _, x, lang, y, m in batches(train, range(len(train["record_id"])), mean, std, device, lang_ids,
                                           shuffle_rng=rng):
            loss = model.loss(x, lang, y, m)
            opt.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            opt.step()
            tr.append(loss.item())
        model.eval()
        with torch.no_grad():
            vl = [model.loss(x, lang, y, m).item()
                  for _, _, x, lang, y, m in batches(val, sel_idx, mean, std, device, lang_ids)]
        log.append({"epoch": epoch, "train_loss": float(np.mean(tr)), "val_select_loss": float(np.mean(vl))})
        print(log[-1])
        if log[-1]["val_select_loss"] < best:
            best, bad = log[-1]["val_select_loss"], 0
            torch.save(model.state_dict(), out_dir / "model.pt")
        else:
            bad += 1
            if bad >= HP["patience"]:
                break
    model.load_state_dict(torch.load(out_dir / "model.pt", map_location=device))
    (out_dir / "train_log.json").write_text(json.dumps({"hp": HP, "seed": args.seed,
                                                        "multilabel": args.multilabel, "log": log}, indent=2))

    name = f"3b_{tag}"
    for split, data in (("val", val), ("test", test)):
        rows, fids = to_records(predict(model, data, mean, std, device, lang_ids, args.multilabel), data, split)
        with open(out_dir / f"{split}_predictions.jsonl", "w", encoding="utf-8") as f:
            for r in rows:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
        if split == "val":
            covered = fids + data["skipped_ids"].tolist()   # explicit skips count as covered
            for s in ("val_select", "val_confirm"):
                res, _ = ev.evaluate(rows, s, feature_ids=covered)
                ev.save_scores(name, res)
                print(ev.format_result(name, res))
    print(f"-> {out_dir}")


if __name__ == "__main__":
    main()
