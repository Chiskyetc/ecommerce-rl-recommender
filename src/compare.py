"""Evaluate every trained model on the test set and produce the final tables and charts."""
import json

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch

import evaluate as ev
import plotstyle
import steps
from config import FIGURES, MODELS, RESULTS
from baselines import ItemKNN, load_train
from models import SessionGRU, TorchRecommender


def load_net(path):
    ckpt = torch.load(path, weights_only=False)
    net = SessionGRU(ckpt["n_items"], emb=ckpt["dim"], hidden=ckpt["dim"],
                     q_head=ckpt["q_head"], repeat=ckpt.get("repeat", True))
    net.load_state_dict(ckpt["state_dict"])
    return net.eval(), ckpt


def model_label(ckpt):
    if not ckpt["q_head"]:
        return "GRU (supervised)"
    a = ckpt["args"]
    return f"{'SNQN' if a.get('neg') else 'SQN'} gamma={a['gamma']}"


def tune_beta(net, val, n, betas=(0.5, 1, 2, 5, 10)):
    scores = {}
    for b in betas:
        r, _ = ev.evaluate(TorchRecommender(net, "mix", beta=b), val, n)
        scores[b] = r["reward@10"]
    return max(scores, key=scores.get), scores


class ShortlistReranker:
    """Score only the Item-KNN top-N with a trained network's head."""

    def __init__(self, knn, rec, n_cand=50):
        self.knn, self.rec, self.n_cand = knn, rec, n_cand

    def recommend(self, state, state_ev, k):
        cands, _ = self.knn.candidates(state, self.n_cand)
        s = self.rec.score_items(state, state_ev, cands)
        order = np.argsort(-s, axis=1)[:, :k]
        return np.take_along_axis(cands, order, axis=1)


def neural_results(val, test, n, knn):
    results, betas = [], {}
    for path in sorted(MODELS.glob("*.pt")):
        net, ckpt = load_net(path)
        label = model_label(ckpt)
        if not ckpt["q_head"]:
            r, _ = ev.evaluate(TorchRecommender(net, "ce"), test, n, name=label)
            results.append(r)
            r, _ = ev.evaluate(ShortlistReranker(knn, TorchRecommender(net, "ce")), test, n,
                               name=f"{label}, re-ranking KNN top 50")
            results.append(r)
            continue
        beta, sweep = tune_beta(net, val, n)
        betas[label] = {"chosen": beta, "val_reward@10": sweep}
        for head, suffix in (("ce", "supervised head"), ("q", "Q head"),
                             ("mix", f"combined, beta={beta}")):
            r, _ = ev.evaluate(TorchRecommender(net, head, beta=beta), test, n,
                               name=f"{label} ({suffix})")
            results.append(r)
        r, _ = ev.evaluate(ShortlistReranker(knn, TorchRecommender(net, "q")), test, n,
                           name=f"{label} (Q head), re-ranking KNN top 50")
        results.append(r)
    (RESULTS / "beta_tuning.json").write_text(json.dumps(betas, indent=2))
    return results


def bar_chart(table, metric, title, path):
    t = table.sort_values(metric)
    fig, ax = plt.subplots(figsize=(8, 0.42 * len(t) + 1.2))
    ax.barh(t["model"], t[metric], color=plotstyle.SERIES[0], height=0.6)
    for i, v in enumerate(t[metric]):
        ax.text(v + t[metric].max() * 0.01, i, f"{v:.3f}", va="center", fontsize=8,
                color=plotstyle.INK)
    ax.grid(axis="y", visible=False)
    ax.grid(axis="x")
    ax.set_xlim(0, t[metric].max() * 1.12)
    ax.set_title(title)
    plotstyle.save(fig, path)


def training_curves():
    hist = sorted(RESULTS.glob("history_*.json"))
    if not hist:
        return
    fig, ax = plt.subplots(figsize=(7, 3.4))
    for i, h in enumerate(hist):
        rows = json.loads(h.read_text())
        label = (h.stem.replace("history_", "").replace("snqn_g", "SNQN gamma=")
                 .replace("sqn_g", "SQN gamma=").replace("gru", "GRU"))
        ax.plot([r["epoch"] for r in rows], [r["val_ce_NDCG@10"] for r in rows],
                marker="o", markersize=4, label=label, color=plotstyle.SERIES[i])
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Validation NDCG@10")
    ax.legend()
    ax.set_title("Validation accuracy during training")
    plotstyle.save(fig, FIGURES / "09_training_curves.png")


def main(reuse=False):
    plotstyle.apply()
    n = steps.n_items()
    if reuse and (RESULTS / "final_test.json").exists():
        neural = [r for r in json.loads((RESULTS / "final_test.json").read_text())
                  if r["model"].startswith(("GRU", "SQN", "SNQN"))]
    else:
        val, test = steps.build("val"), steps.build("test")
        knn = ItemKNN(n, decay=0.8, include_seen=True, seen_boost=2.0).fit(load_train())
        neural = neural_results(val, test, n, knn)
    results = json.loads((RESULTS / "baselines_test.json").read_text())
    results += json.loads((RESULTS / "bandit_test.json").read_text())
    if (RESULTS / "bandit_hybrid_test.json").exists():
        results += json.loads((RESULTS / "bandit_hybrid_test.json").read_text())
    results += neural
    table = ev.save(results, "final_test")

    bar_chart(table, "all NDCG@10", "Overall ranking quality on the test period (NDCG@10)",
              FIGURES / "07_ndcg_all_models.png")
    bar_chart(table, "reward@10", "Average reward per recommendation slate (test period)",
              FIGURES / "08_reward_all_models.png")
    training_curves()

    cols = ["model", "all HR@10", "all NDCG@10", "addtocart HR@10", "purchase HR@10",
            "purchase NDCG@10", "reward@10", "coverage@10"]
    print(table[cols].round(4).to_string(index=False))
    (RESULTS / "final_test.md").write_text(table[cols].round(4).to_markdown(index=False))


if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("--reuse", action="store_true",
                   help="reuse saved neural-model results and only refresh tables and charts")
    main(p.parse_args().reuse)
