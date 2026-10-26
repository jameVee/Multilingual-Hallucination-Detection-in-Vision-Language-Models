"""Part 3b: learned neural probe with jointly-trained span-boundary decoding.

Token features of the frozen judge (features.py: entropy, nll, lookback, hidden_-1;
standardized on train) -> Linear 512 -> BiGRU 256 (+ language embedding) -> one
2-state linear-chain CRF per category (O / inside). Each CRF is trained jointly with
the encoder on the exact sequence likelihood, so span boundaries are decoded, not
thresholded. One fixed hyperparameter set; early stopping on val_select LOSS (not on
the metric); no threshold search.

Prediction: Viterbi path per category; a token predicted by several categories keeps
the one with the highest marginal; consecutive tokens of one category -> one span
(char offsets from the tokenizer offsets); span prob = mean marginal. Records the
judge could not process get an explicit empty prediction (features.py lists them).

Usage (nlp2):
    python -m thesis_v2.part3_uncert.crf_probe --selftest
    python -m thesis_v2.part3_uncert.crf_probe --seed 42 --gpu_index 0
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


class BinaryCRF(nn.Module):
    """2-state linear-chain CRF over emissions (B, T, 2) with a mask (B, T)."""

    def __init__(self):
        super().__init__()
        self.trans = nn.Parameter(torch.zeros(2, 2))
        self.start = nn.Parameter(torch.zeros(2))
        self.end = nn.Parameter(torch.zeros(2))

    def _forward_alpha(self, em, mask):
        alpha = self.start + em[:, 0]
        alphas = [alpha]
        for t in range(1, em.shape[1]):
            nxt = torch.logsumexp(alpha.unsqueeze(2) + self.trans + em[:, t].unsqueeze(1), dim=1)
            alpha = torch.where(mask[:, t:t + 1], nxt, alpha)
            alphas.append(alpha)
        return alpha, alphas

    def nll(self, em, tags, mask):
        """Negative log-likelihood per sequence."""
        b = torch.arange(em.shape[0], device=em.device)
        score = self.start[tags[:, 0]] + em[b, 0, tags[:, 0]]
        for t in range(1, em.shape[1]):
            step = self.trans[tags[:, t - 1], tags[:, t]] + em[b, t, tags[:, t]]
            score = score + step * mask[:, t]
        last = tags.gather(1, (mask.sum(1, keepdim=True) - 1).long()).squeeze(1)
        score = score + self.end[last]
        alpha, _ = self._forward_alpha(em, mask)
        return torch.logsumexp(alpha + self.end, dim=1) - score

    def marginals(self, em, mask):
        """P(tag_t = 1) per token, by forward-backward."""
        _, alphas = self._forward_alpha(em, mask)
        T = em.shape[1]
        lengths = mask.sum(1).long()
        beta = torch.zeros_like(alphas[0])
        betas = [None] * T
        for t in range(T - 1, -1, -1):
            is_last = (lengths - 1 == t).unsqueeze(1)
            if t == T - 1:
                beta = self.end.expand_as(beta).clone()
            else:
                nxt = torch.logsumexp(self.trans + (em[:, t + 1] + beta).unsqueeze(1), dim=2)
                beta = torch.where(mask[:, t + 1:t + 2], nxt, beta)
            beta = torch.where(is_last, self.end.expand_as(beta), beta)
            betas[t] = beta
        logZ = torch.logsumexp(alphas[-1] + self.end, dim=1, keepdim=True)
        post = torch.stack([alphas[t] + betas[t] - logZ for t in range(T)], dim=1)
        return post.exp()[..., 1]

    def viterbi(self, em, mask):
        score = self.start + em[:, 0]
        back = []
        for t in range(1, em.shape[1]):
            cand = score.unsqueeze(2) + self.trans
            best, idx = cand.max(dim=1)
            nxt = best + em[:, t]
            score = torch.where(mask[:, t:t + 1], nxt, score)
            back.append(torch.where(mask[:, t:t + 1], idx, torch.arange(2, device=em.device).expand_as(idx)))
        score = score + self.end
        tag = score.argmax(1)
        path = [tag]
        for idx in reversed(back):
            tag = idx.gather(1, tag.unsqueeze(1)).squeeze(1)
            path.append(tag)
        return torch.stack(path[::-1], dim=1)


class Probe(nn.Module):
    def __init__(self, d_in):
        super().__init__()
        self.proj = nn.Sequential(nn.Linear(d_in, HP["proj"]), nn.ReLU(), nn.Dropout(HP["dropout"]))
        self.lang = nn.Embedding(len(LANGUAGES), HP["lang_dim"])
        self.gru = nn.GRU(HP["proj"] + HP["lang_dim"], HP["gru"], batch_first=True, bidirectional=True)
        self.heads = nn.Linear(2 * HP["gru"], 2 * len(LABELS))
        self.crfs = nn.ModuleList(BinaryCRF() for _ in LABELS)

    def emissions(self, x, lang):
        h = self.proj(x)
        h = torch.cat([h, self.lang(lang).unsqueeze(1).expand(-1, h.shape[1], -1)], dim=-1)
        h, _ = self.gru(h)
        return self.heads(h).view(x.shape[0], x.shape[1], len(LABELS), 2)

    def loss(self, x, lang, y, mask):
        em = self.emissions(x, lang)
        total = 0
        for c, crf in enumerate(self.crfs):
            total = total + crf.nll(em[:, :, c], (y == c).long(), mask)
        return total.mean()


def batches(data, idx, mean, std, device, lang_ids, shuffle_rng=None):
    idx = list(idx)
    if shuffle_rng is not None:
        shuffle_rng.shuffle(idx)
    ptr = data["rec_ptr"]
    for i in range(0, len(idx), HP["batch"]):
        chunk = idx[i:i + HP["batch"]]
        lens = [min(HP["max_len"], ptr[j + 1] - ptr[j]) for j in chunk]
        T = max(lens)
        x = np.zeros((len(chunk), T, data["x"].shape[1]), np.float32)
        y = np.full((len(chunk), T), -1, np.int64)
        m = np.zeros((len(chunk), T), bool)
        for b, (j, L) in enumerate(zip(chunk, lens)):
            x[b, :L] = (data["x"][ptr[j]:ptr[j] + L].astype(np.float32) - mean) / std
            y[b, :L] = data["y"][ptr[j]:ptr[j] + L]
            m[b, :L] = True
        lang = torch.tensor([lang_ids[data["language"][j]] for j in chunk], device=device)
        yield chunk, lens, torch.from_numpy(x).to(device), lang, torch.from_numpy(y).to(device), \
            torch.from_numpy(m).to(device)


@torch.no_grad()
def predict(model, data, mean, std, device, lang_ids):
    model.eval()
    out = {}
    for chunk, lens, x, lang, _, m in batches(data, range(len(data["record_id"])), mean, std, device, lang_ids):
        em = model.emissions(x, lang)
        paths = torch.stack([crf.viterbi(em[:, :, c], m) for c, crf in enumerate(model.crfs)], -1)
        marg = torch.stack([crf.marginals(em[:, :, c], m) for c, crf in enumerate(model.crfs)], -1)
        score = torch.where(paths.bool(), marg, torch.full_like(marg, -1.0))
        best = score.argmax(-1)
        on = score.max(-1).values > 0
        for b, j in enumerate(chunk):
            p0 = data["rec_ptr"][j]
            spans, t = [], 0
            while t < lens[b]:
                if not on[b, t]:
                    t += 1
                    continue
                c, u = int(best[b, t]), t
                while u < lens[b] and on[b, u] and int(best[b, u]) == c:
                    u += 1
                spans.append({"start": int(data["off_start"][p0 + t]), "end": int(data["off_end"][p0 + u - 1]),
                              "prob": round(float(marg[b, t:u, c].mean()), 4), "label": LABELS[c]})
                t = u
            out[data["record_id"][j]] = spans
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
    """Viterbi == brute force and marginals sum correctly on random toy sequences."""
    torch.manual_seed(0)
    crf = BinaryCRF()
    with torch.no_grad():
        crf.trans.normal_(); crf.start.normal_(); crf.end.normal_()
    for T in (1, 2, 5):
        em = torch.randn(3, T, 2)
        mask = torch.ones(3, T, dtype=torch.bool)
        paths = list(itertools.product([0, 1], repeat=T))

        def path_score(b, pth):
            s = crf.start[pth[0]] + em[b, 0, pth[0]]
            for t in range(1, T):
                s = s + crf.trans[pth[t - 1], pth[t]] + em[b, t, pth[t]]
            return s + crf.end[pth[-1]]
        for b in range(3):
            scores = torch.stack([path_score(b, p) for p in paths])
            assert list(paths[int(scores.argmax())]) == crf.viterbi(em, mask)[b].tolist()
            logZ = torch.logsumexp(scores, 0)
            nll = crf.nll(em, torch.tensor([paths[0]] * 3), mask)[b]
            assert torch.allclose(nll, logZ - scores[0], atol=1e-4)
            probs = (scores - logZ).exp()
            m1 = torch.stack([sum(probs[i] for i, p in enumerate(paths) if p[t] == 1) for t in range(T)])
            assert torch.allclose(crf.marginals(em, mask)[b], m1, atol=1e-4)
    # padded sequence: masked steps must not change the result
    em = torch.randn(1, 6, 2)
    m_short = torch.tensor([[1, 1, 1, 0, 0, 0]], dtype=torch.bool)
    assert crf.viterbi(em, m_short)[0, :3].tolist() == crf.viterbi(em[:, :3], m_short[:, :3])[0].tolist()
    assert torch.allclose(crf.marginals(em, m_short)[0, :3], crf.marginals(em[:, :3], m_short[:, :3])[0], atol=1e-5)
    # the whole probe must overfit a tiny batch (catches wiring/masking bugs)
    model = Probe(10)
    x, lang = torch.randn(4, 12, 10), torch.tensor([0, 1, 2, 3])
    y = torch.full((4, 12), -1)
    y[0, 3:6], y[1, 0:2], y[2, 7:12], y[3, 5] = 2, 4, 0, 3
    mask = torch.ones(4, 12, dtype=torch.bool)
    mask[3, 8:] = False
    opt = torch.optim.Adam(model.parameters(), lr=1e-2)
    first = model.loss(x, lang, y, mask).item()
    for _ in range(150):
        opt.zero_grad()
        loss = model.loss(x, lang, y, mask)
        loss.backward()
        opt.step()
    model.eval()
    em = model.emissions(x, lang)
    pred = torch.stack([crf.viterbi(em[:, :, c], mask) for c, crf in enumerate(model.crfs)], -1)
    gold = torch.stack([(y == c).long() for c in range(len(LABELS))], -1)
    assert (pred[mask] == gold[mask]).all(), "probe failed to overfit a tiny batch"
    print(f"CRF selftest passed (Viterbi, NLL, marginals vs brute force; padding; "
          f"tiny-batch overfit {first:.2f} -> {loss.item():.4f})")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--gpu_index", type=int, default=None)
    p.add_argument("--selftest", action="store_true")
    args = p.parse_args()
    if args.selftest:
        return selftest()

    torch.manual_seed(args.seed)
    rng = np.random.default_rng(args.seed)
    device = f"cuda:{args.gpu_index}" if torch.cuda.is_available() and args.gpu_index is not None else \
        ("cuda" if torch.cuda.is_available() else "cpu")
    train, val, test = features.load("train"), features.load("val"), features.load("test")
    lang_ids = {lg: i for i, lg in enumerate(LANGUAGES)}
    xs = train["x"][rng.choice(len(train["x"]), min(500_000, len(train["x"])), replace=False)].astype(np.float32)
    mean, std = xs.mean(0), xs.std(0) + 1e-6

    sel = load_split("val_select")
    sel_idx = [j for j, rid in enumerate(val["record_id"]) if rid in sel]
    model = Probe(train["x"].shape[1]).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=HP["lr"])
    out_dir = RUNS_DIR / "part3" / f"crf_seed{args.seed}"
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
    (out_dir / "train_log.json").write_text(json.dumps({"hp": HP, "seed": args.seed, "log": log}, indent=2))

    for split, data in (("val", val), ("test", test)):
        rows, fids = to_records(predict(model, data, mean, std, device, lang_ids), data, split)
        with open(out_dir / f"{split}_predictions.jsonl", "w", encoding="utf-8") as f:
            for r in rows:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
        if split == "val":
            covered = fids + data["skipped_ids"].tolist()   # explicit skips count as covered
            for s in ("val_select", "val_confirm"):
                res, _ = ev.evaluate(rows, s, feature_ids=covered)
                ev.save_scores(f"3b_crf_seed{args.seed}", res)
                print(ev.format_result(f"3b_crf_seed{args.seed}", res))
    print(f"-> {out_dir}")


if __name__ == "__main__":
    main()
