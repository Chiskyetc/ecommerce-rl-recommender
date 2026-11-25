"""Pick real test sessions that illustrate the main result patterns, and measure how common each pattern is.

Selection is rule-based and seeded, so the same examples come out every run.
"""
import json

import numpy as np
import pandas as pd

import evaluate as ev
import steps
from bandit import BanditRecommender, FeatureBuilder, LinearBandit
from baselines import ItemKNN, load_train
from compare import load_net
from config import MODELS, PROCESSED, RESULTS, SEED
from models import TorchRecommender

EVENT = {0: "viewed", 1: "added to cart", 2: "purchased"}


def ranks(topk, target):
    match = topk == target[:, None]
    return np.where(match.any(1), match.argmax(1) + 1, 0)  # 0 = not in top 10


def main():
    n = steps.n_items()
    test = steps.build("test")
    train = load_train()
    knn = ItemKNN(n, decay=0.8, include_seen=True, seen_boost=2.0).fit(train)
    gru = TorchRecommender(load_net(MODELS / "gru.pt")[0], "ce")
    models = {
        "Item-KNN": knn,
        "LinUCB": BanditRecommender(LinearBandit.from_theta(np.load(MODELS / "linucb_theta.npy")),
                                    FeatureBuilder(knn, train, n)),
        "GRU": gru,
        "LinUCB + GRU": BanditRecommender(LinearBandit.from_theta(np.load(MODELS / "linucb_hybrid_theta.npy")),
                                          FeatureBuilder(knn, train, n, scorer=gru)),
    }
    tops, rk = {}, {}
    for name, m in models.items():
        _, topk = ev.evaluate(m, test, n)
        tops[name] = topk[:, :10]
        rk[name] = ranks(tops[name], test["target"])
        print(name, "done", flush=True)

    cands, _ = (None, None)
    shortlist_hit = np.concatenate([
        (knn.candidates(test["state"][i:i + 2048], 50)[0] == test["target"][i:i + 2048, None]).any(1)
        for i in range(0, len(test["target"]), 2048)])

    target, event = test["target"], test["event"]
    in_state = (test["state"] == target[:, None]).any(1)
    carted = ((test["state"] == target[:, None]) & (test["state_ev"] == 2)).any(1)
    length = (test["state"] > 0).sum(1)
    short = (length >= 2) & (length <= 5)

    patterns = {
        "A": ("Buying a product already in the cart", (event == 2) & carted),
        "B": ("Moving on to a new product", (event == 0) & ~in_state & shortlist_hit),
        "C": ("Returning to an earlier product", (event == 0) & in_state),
        "D": ("A next product no model can reach", ~shortlist_hit),
    }
    rules = {
        "A": (rk["LinUCB"] == 1) & (rk["Item-KNN"] > 1),
        "B": (rk["LinUCB + GRU"] >= 1) & (rk["LinUCB + GRU"] <= 3) & ((rk["Item-KNN"] == 0) | (rk["Item-KNN"] > 3)),
        "C": (rk["Item-KNN"] >= 1) & (rk["Item-KNN"] <= 3) & (rk["GRU"] >= 1),
        "D": np.ones(len(target), bool),
    }

    rng = np.random.default_rng(SEED)
    items = pd.read_parquet(PROCESSED / "items.parquet").set_index("item")
    label = lambda it: f"#{int(items.loc[it, 'itemid'])}"
    cat = lambda it: "unknown" if items.loc[it, "category"] < 0 else str(int(items.loc[it, "category"]))

    out = {"selection": "rule-based, seed 42, sessions of 2-5 prior events", "patterns": {}}
    for key, (title, mask) in patterns.items():
        stats = {"title": title, "share_of_test": float(mask.mean()), "count": int(mask.sum()),
                 "hit_rate_top10": {m: float((rk[m][mask] >= 1).mean()) for m in models},
                 "ranked_first": {m: float((rk[m][mask] == 1).mean()) for m in models}}
        pool = np.flatnonzero(mask & rules[key] & short)
        row = int(rng.choice(pool))
        st, se = test["state"][row], test["state_ev"][row]
        history = [{"product": label(it), "category": cat(it), "action": EVENT[e - 1]}
                   for it, e in zip(st, se) if it > 0]
        example = {
            "history": history,
            "next": {"product": label(target[row]), "category": cat(target[row]), "action": EVENT[int(event[row])]},
            "models": {m: {"rank": int(rk[m][row]),
                           "top3": [f"{label(i)} (cat {cat(i)})" for i in tops[m][row][:3]]} for m in models},
            "pool_size": int(len(pool)),
        }
        out["patterns"][key] = {**stats, "example": example}
    (RESULTS / "session_examples.json").write_text(json.dumps(out, indent=2))
    print(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
