"""Audit trail for every validation-based choice in thesis_v2.

Each decision records its candidates, their val_select scores, the choice, and
(once checked) the val_confirm score of the choice. The search budget -- at most
MAX_CANDIDATES per decision -- is enforced here, so an oversized search cannot
happen silently. The log (results/decision_log.jsonl) is committed and is the
thesis's evidence that no result came from an unbounded search.
"""
from __future__ import annotations

import datetime as _dt
import json

from thesis_v2.common.paths import RESULTS_DIR

LOG_FILE = RESULTS_DIR / "decision_log.jsonl"
MAX_CANDIDATES = 5


class SearchBudgetError(RuntimeError):
    pass


def _read():
    if not LOG_FILE.exists():
        return []
    return [json.loads(l) for l in LOG_FILE.read_text(encoding="utf-8").splitlines() if l.strip()]


def record_decision(decision_id, description, candidates, metric="primary_cor_lbl (val_select)",
                    chosen=None, rule=None):
    """candidates: dict {candidate_name: val_select score}. Chooses the max unless
    `chosen` is given (e.g. a pre-registered tie rule). Refuses >MAX_CANDIDATES and
    refuses to re-open an existing decision id."""
    if len(candidates) > MAX_CANDIDATES:
        raise SearchBudgetError(f"{decision_id}: {len(candidates)} candidates > budget {MAX_CANDIDATES}")
    if any(d["decision_id"] == decision_id for d in _read()):
        raise SearchBudgetError(f"{decision_id} already decided -- decisions are not re-opened")
    if chosen is None:
        chosen = max(candidates, key=candidates.get)
    entry = {
        "decision_id": decision_id, "description": description, "metric": metric,
        "candidates": candidates, "chosen": chosen, "rule": rule,
        "confirm": None, "timestamp": _dt.datetime.now().isoformat(timespec="seconds"),
    }
    LOG_FILE.parent.mkdir(parents=True, exist_ok=True)
    with open(LOG_FILE, "a", encoding="utf-8") as f:
        f.write(json.dumps(entry) + "\n")
    return chosen


def record_confirmation(decision_id, confirm_result):
    """Attach the val_confirm outcome to an already-made decision."""
    entries = _read()
    for e in entries:
        if e["decision_id"] == decision_id:
            if e["confirm"] is not None:
                raise SearchBudgetError(f"{decision_id} already confirmed")
            e["confirm"] = confirm_result
            break
    else:
        raise KeyError(decision_id)
    LOG_FILE.write_text("".join(json.dumps(e) + "\n" for e in entries), encoding="utf-8")


def show():
    for e in _read():
        print(f"{e['decision_id']}: chose {e['chosen']} from {len(e['candidates'])} candidates "
              f"({e['description']}); confirm={e['confirm']}")


if __name__ == "__main__":
    show()
