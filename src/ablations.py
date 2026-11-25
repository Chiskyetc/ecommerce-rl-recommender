"""Ablation studies: what each design choice contributes.

    python ablations.py bandit     # LinUCB trained with four reward schemes (minutes)
    python ablations.py evaluate   # score every ablation on test, with paired bootstrap CIs

The neural ablations are trained first with train_sqn.py --ablation (see its docstring).
Training rewards change between variants, but every model is scored with the standard
rewards (0.2, 0.6, 1.0), so the results are comparable.
"""
import json
import sys

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import evaluate as ev
import plotstyle
import steps
from bandit import BanditRecommender, FeatureBuilder, LinearBandit
from baselines import ItemKNN, load_train
from bootstrap_ci import B, METRICS, per_transition, session_sums
from compare import load_net, tune_beta
from config import FIGURES, MODELS, RESULTS, SEED
from models import TorchRecommender
from run_bandit import run_online, stream_order

OUT_MODELS = MODELS / "ablations"
OUT = RESULTS / "ablations"
OUT_MODELS.mkdir(exist_ok=True)
OUT.mkdir(exist_ok=True)

SCHEMES = {
    "default": ("Default rewards (0.2, 0.6, 1.0)", {0: 0.2, 1: 0.6, 2: 1.0}),
    "equal": ("Equal rewards (1, 1, 1)", {0: 1.0, 1: 1.0, 2: 1.0}),
    "steep": ("Steeper rewards (0.1, 0.5, 2.0)", {0: 0.1, 1: 0.5, 2: 2.0}),
    "purchase": ("Purchases only (0, 0, 1)", {0: 0.0, 1: 0.0, 2: 1.0}),
}


def knn_and_features(n, train):
    knn = ItemKNN(n, decay=0.8, include_seen=True, seen_boost=2.0).fit(train)
    return knn, FeatureBuilder(knn, train, n)


def bandit_rewards():
    n = steps.n_items()
    train = load_train()
    val = steps.build("val")
    _, fb = knn_and_features(n, train)
    order = stream_order(val)
    for key, (label, rewards) in SCHEMES.items():
        pol = LinearBandit(len(fb.names), "ucb", alpha=0.5, seed=SEED)
        run_online({"LinUCB": pol}, fb, val, order, rewards=rewards)
        np.save(OUT_MODELS / f"linucb_{key}_theta.npy", pol.theta)
        print(f"{label}: weights saved", flush=True)


def models_to_score(n, val):
    train = load_train()
    knn, fb = knn_and_features(n, train)
    betas = json.loads((RESULTS / "beta_tuning.json").read_text())
    m = {}

    m["Item-KNN"] = knn
    m["Item-KNN, new products only"] = ItemKNN(n, decay=0.8, include_seen=False).fit(train)

    m["GRU"] = TorchRecommender(load_net(MODELS / "gru.pt")[0], "ce")
    m["GRU without repeat term"] = TorchRecommender(load_net(OUT_MODELS / "gru_norepeat.pt")[0], "ce")

    sqn_files = {"SQN γ=0.5": (MODELS / "sqn_g0.5.pt", betas["SQN gamma=0.5"]["chosen"]),
                 "SQN γ=0": (OUT_MODELS / "sqn_g0.0.pt", None),
                 "SQN γ=0.9": (OUT_MODELS / "sqn_g0.9.pt", None),
                 "SQN, equal rewards": (OUT_MODELS / "sqn_equal_rewards.pt", None)}
    chosen = {}
    for name, (path, beta) in sqn_files.items():
        net = load_net(path)[0]
        if beta is None:
            beta, _ = tune_beta(net, val, n)
        chosen[name] = beta
        m[f"{name} (supervised head)"] = TorchRecommender(net, "ce")
        m[f"{name} (combined)"] = TorchRecommender(net, "mix", beta=beta)

    for key, (label, _) in SCHEMES.items():
        theta = np.load(OUT_MODELS / f"linucb_{key}_theta.npy")
        m[f"LinUCB, {label[0].lower() + label[1:]}"] = BanditRecommender(LinearBandit.from_theta(theta), fb)
    return m, chosen


COMPARISONS = [
    ("Seen products", "Item-KNN, new products only", "Item-KNN"),
    ("Repeat term", "GRU without repeat term", "GRU"),
    ("Discount factor", "SQN γ=0 (supervised head)", "SQN γ=0.5 (supervised head)"),
    ("Discount factor", "SQN γ=0.9 (supervised head)", "SQN γ=0.5 (supervised head)"),
    ("Discount factor", "SQN γ=0 (combined)", "SQN γ=0.5 (combined)"),
    ("Discount factor", "SQN γ=0.9 (combined)", "SQN γ=0.5 (combined)"),
    ("Reward weights", "SQN, equal rewards (supervised head)", "SQN γ=0.5 (supervised head)"),
    ("Reward weights", "SQN, equal rewards (combined)", "SQN γ=0.5 (combined)"),
    ("Reward weights", "LinUCB, equal rewards (1, 1, 1)", "LinUCB, default rewards (0.2, 0.6, 1.0)"),
    ("Reward weights", "LinUCB, steeper rewards (0.1, 0.5, 2.0)", "LinUCB, default rewards (0.2, 0.6, 1.0)"),
    ("Reward weights", "LinUCB, purchases only (0, 0, 1)", "LinUCB, default rewards (0.2, 0.6, 1.0)"),
]


def evaluate_all():
    plotstyle.apply()
    n = steps.n_items()
    val, test = steps.build("val"), steps.build("test")
    codes = np.unique(test["sid"], return_inverse=True)[1]
    n_sess = codes.max() + 1
    purchase = test["event"] == 2

    models, betas = models_to_score(n, val)
    sums, point = {}, {}
    for name, model in models.items():
        _, topk = ev.evaluate(model, test, n)
        hit, ndcg, reward = per_transition(topk, test)
        sums[name] = {
            "HR@10": session_sums(hit, codes, n_sess),
            "NDCG@10": session_sums(ndcg, codes, n_sess),
            "Reward@10": session_sums(reward, codes, n_sess),
            "Purchase HR@10": session_sums(hit, codes, n_sess, purchase),
            "Purchase NDCG@10": session_sums(ndcg, codes, n_sess, purchase),
        }
        point[name] = {k: num.sum() / den.sum() for k, (num, den) in sums[name].items()}
        print(f"{name}: NDCG@10 {point[name]['NDCG@10']:.4f}", flush=True)

    rng = np.random.default_rng(SEED)
    weights = np.stack([np.bincount(rng.integers(0, n_sess, n_sess), minlength=n_sess)
                        for _ in range(B)]).astype(float)
    boot = {name: {k: (weights @ num) / (weights @ den) for k, (num, den) in m.items()}
            for name, m in sums.items()}

    rows = []
    for name in models:
        row = {"model": name}
        for metric in METRICS:
            lo, hi = np.percentile(boot[name][metric], [2.5, 97.5])
            row.update({metric: point[name][metric], f"{metric} low": lo, f"{metric} high": hi})
        rows.append(row)
    table = pd.DataFrame(rows)
    table.to_csv(OUT / "ablation_models.csv", index=False)

    diffs = []
    for group, a, b in COMPARISONS:
        for metric in METRICS:
            d = boot[a][metric] - boot[b][metric]
            lo, hi = np.percentile(d, [2.5, 97.5])
            p = min(1.0, 2 * min((d <= 0).mean(), (d >= 0).mean()))
            diffs.append({"ablation": group, "model": a, "reference": b, "metric": metric,
                          "difference": point[a][metric] - point[b][metric], "low": lo, "high": hi,
                          "p": p, "verdict": "higher" if lo > 0 else "lower" if hi < 0 else "no clear difference"})
    dtable = pd.DataFrame(diffs)
    dtable.to_csv(OUT / "ablation_differences.csv", index=False)
    (OUT / "ablation_betas.json").write_text(json.dumps(betas, indent=2))
    write_markdown(table, dtable)
    chart(dtable)


def write_markdown(table, dtable):
    lines = ["# Ablation results (test period)", "",
             f"95% confidence intervals from a paired bootstrap over test sessions, {B} resamples. "
             "All models are scored with the standard rewards (0.2, 0.6, 1.0).", "",
             "## Models", "", "| Model | " + " | ".join(METRICS) + " |",
             "| --- | " + " | ".join("---" for _ in METRICS) + " |"]
    for _, r in table.iterrows():
        lines.append(f"| {r['model']} | " + " | ".join(
            f"{r[m]:.3f} ({r[m + ' low']:.3f} to {r[m + ' high']:.3f})" for m in METRICS) + " |")
    lines += ["", "## Paired differences (variant minus reference)", "",
              "| Ablation | Variant | Reference | Metric | Difference (95% CI) | p | Verdict |",
              "| --- | --- | --- | --- | --- | --- | --- |"]
    for _, r in dtable.iterrows():
        p = "<0.001" if r["p"] < 0.001 else f"{r['p']:.3f}"
        lines.append(f"| {r['ablation']} | {r['model']} | {r['reference']} | {r['metric']} | "
                     f"{r['difference']:+.4f} ({r['low']:+.4f} to {r['high']:+.4f}) | {p} | {r['verdict']} |")
    (OUT / "ablations.md").write_text("\n".join(lines) + "\n")


MAIN_ROWS = [
    ("Item-KNN, new products only", "Item-KNN", "Item-KNN: block seen products"),
    ("GRU without repeat term", "GRU", "GRU: remove repeat term"),
    ("LinUCB, equal rewards (1, 1, 1)", "LinUCB, default rewards (0.2, 0.6, 1.0)", "LinUCB: equal rewards"),
    ("LinUCB, steeper rewards (0.1, 0.5, 2.0)", "LinUCB, default rewards (0.2, 0.6, 1.0)", "LinUCB: steeper rewards"),
    ("LinUCB, purchases only (0, 0, 1)", "LinUCB, default rewards (0.2, 0.6, 1.0)", "LinUCB: purchases only"),
]
SQN_ROWS = [
    ("SQN \u03b3=0 (supervised head)", "SQN \u03b3=0.5 (supervised head)", "\u03b3 = 0, supervised head"),
    ("SQN \u03b3=0 (combined)", "SQN \u03b3=0.5 (combined)", "\u03b3 = 0, combined"),
    ("SQN \u03b3=0.9 (supervised head)", "SQN \u03b3=0.5 (supervised head)", "\u03b3 = 0.9, supervised head"),
    ("SQN \u03b3=0.9 (combined)", "SQN \u03b3=0.5 (combined)", "\u03b3 = 0.9, combined"),
    ("SQN, equal rewards (supervised head)", "SQN \u03b3=0.5 (supervised head)", "Equal rewards, supervised head"),
    ("SQN, equal rewards (combined)", "SQN \u03b3=0.5 (combined)", "Equal rewards, combined"),
]


def chart(dtable):
    draw(dtable, MAIN_ROWS, [2.5], "Effect of key design choices (change from the standard model, 95% CI)",
         FIGURES / "13_ablations_design.png")
    draw(dtable, SQN_ROWS, [3.5, 1.5], "SQN settings: change from \u03b3 = 0.5 with default rewards (95% CI)",
         FIGURES / "14_ablations_sqn.png")


def draw(dtable, rows, splits, title, path):
    metrics = ["NDCG@10", "Reward@10", "Purchase NDCG@10"]
    fig, axes = plt.subplots(1, 3, figsize=(11, 0.55 * len(rows) + 1.9), sharey=True)
    y = np.arange(len(rows))[::-1]
    for ax, metric in zip(axes, metrics):
        for yi, (a, b, _) in zip(y, rows):
            r = dtable[(dtable.model == a) & (dtable.reference == b) & (dtable.metric == metric)].iloc[0]
            clear = r["verdict"] != "no clear difference"
            c = plotstyle.SERIES[0] if clear else "#a3a29e"
            ax.hlines(yi, r["low"], r["high"], color=c, linewidth=2)
            ax.plot(r["difference"], yi, "o", color=c, markersize=6, markerfacecolor=c if clear else "white")
        ax.axvline(0, color=plotstyle.INK_2, linewidth=1, linestyle="--")
        ax.set_title(metric, fontsize=11)
        ax.grid(axis="y", visible=False)
        ax.grid(axis="x")
        ax.tick_params(axis="x", labelsize=8)
        ax.xaxis.set_major_locator(plt.MaxNLocator(4))
        for split in splits:
            ax.axhline(split, color=plotstyle.GRID, linewidth=1)
    axes[0].set_yticks(y, [label for _, _, label in rows])
    fig.suptitle(title + "; right of the dashed line = better", x=0.01, ha="left", fontweight="bold", fontsize=11.5)
    filled = plt.Line2D([], [], marker="o", color=plotstyle.SERIES[0], linestyle="-", label="interval excludes 0")
    hollow = plt.Line2D([], [], marker="o", color="#a3a29e", markerfacecolor="white", linestyle="-",
                        label="interval includes 0 (no clear difference)")
    fig.legend(handles=[filled, hollow], loc="lower center", ncol=2, frameon=False, fontsize=9,
               bbox_to_anchor=(0.5, -0.02))
    fig.tight_layout(rect=(0, 0.06, 1, 0.92))
    plotstyle.save(fig, path)


if __name__ == "__main__":
    cmd = sys.argv[1] if len(sys.argv) > 1 else "evaluate"
    if cmd == "bandit":
        bandit_rewards()
    elif cmd == "charts":
        plotstyle.apply()
        chart(pd.read_csv(OUT / "ablation_differences.csv"))
    else:
        evaluate_all()
