import json

import numpy as np
import pandas as pd

from config import CUTOFFS, EVENT_NAMES, RESULTS, SLATE_SIZE

GROUPS = {"all": None, "view": 0, "addtocart": 1, "purchase": 2}


def topk_from_scores(scores, k):
    scores = scores.copy()
    scores[:, 0] = -np.inf
    idx = np.argpartition(-scores, k, axis=1)[:, :k]
    order = np.argsort(-np.take_along_axis(scores, idx, axis=1), axis=1)
    return np.take_along_axis(idx, order, axis=1)


def rank_metrics(topk, target, event, reward):
    """topk: (n, K) recommended item ids, best first."""
    kmax = topk.shape[1]
    hit_pos = np.full(len(target), kmax, dtype=np.int32)
    match = topk == target[:, None]
    has = match.any(axis=1)
    hit_pos[has] = match[has].argmax(axis=1)

    out = {}
    for name, ev in GROUPS.items():
        mask = np.ones(len(target), bool) if ev is None else event == ev
        if not mask.any():
            continue
        p = hit_pos[mask]
        row = {"n": int(mask.sum())}
        for k in CUTOFFS:
            hit = p < k
            row[f"HR@{k}"] = float(hit.mean())
            row[f"NDCG@{k}"] = float(np.where(hit, 1.0 / np.log2(p + 2), 0.0).mean())
        out[name] = row
    in_slate = hit_pos < SLATE_SIZE
    out["reward@10"] = float((reward * in_slate).sum() / len(target))
    return out


def evaluate(model, data, n_items, batch=2048, name=None):
    kmax = max(CUTOFFS)
    tops = []
    for i in range(0, len(data["target"]), batch):
        sl = slice(i, i + batch)
        tops.append(model.recommend(data["state"][sl], data["state_ev"][sl], kmax))
    topk = np.concatenate(tops)
    res = rank_metrics(topk, data["target"], data["event"], data["reward"])
    res["coverage@10"] = float(len(np.unique(topk[:, :SLATE_SIZE])) / n_items)
    if name:
        res["model"] = name
    return res, topk


def to_table(results):
    rows = []
    for r in results:
        row = {"model": r["model"]}
        for g in ("all", "view", "addtocart", "purchase"):
            for m in ("HR@10", "NDCG@10"):
                row[f"{g} {m}"] = r[g][m]
        row["reward@10"] = r["reward@10"]
        row["coverage@10"] = r["coverage@10"]
        rows.append(row)
    return pd.DataFrame(rows)


def save(results, stem):
    (RESULTS / f"{stem}.json").write_text(json.dumps(results, indent=2))
    table = to_table(results)
    table.to_csv(RESULTS / f"{stem}.csv", index=False)
    return table
