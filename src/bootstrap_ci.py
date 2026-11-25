"""95% confidence intervals for the test results, by paired bootstrap over test sessions.

Transitions from the same session are not independent, so whole sessions are
resampled (with replacement, same draw for every model). Each metric is
recomputed on each resample as total over resampled transitions divided by
their number, and the 2.5th and 97.5th percentiles give the interval. Because
every model is scored on the same resample, differences between two models
get their own interval, which is what decides whether one model really beats
another.
"""
import json

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import evaluate as ev
import plotstyle
import steps
from bandit import BanditRecommender, FeatureBuilder, LinearBandit
from baselines import ItemKNN, Popularity, load_train
from compare import ShortlistReranker, load_net
from config import FIGURES, MODELS, REWARDS, RESULTS, SEED, SLATE_SIZE
from models import TorchRecommender

B = 1000
METRICS = ["HR@10", "NDCG@10", "Reward@10", "Purchase HR@10", "Purchase NDCG@10"]

COMPARISONS = [
    # (hypothesis, model, reference)
    ("H1", "Bandit: LinUCB", "Item-KNN"),
    ("H1", "Bandit: Thompson sampling", "Item-KNN"),
    ("H1", "Bandit: Epsilon-greedy", "Item-KNN"),
    ("H1", "Bandit + GRU: LinUCB", "Item-KNN"),
    ("H1", "Bandit + GRU: Thompson sampling", "Item-KNN"),
    ("H1", "Bandit + GRU: Epsilon-greedy", "Item-KNN"),
    ("H2", "SQN (supervised head)", "GRU (supervised)"),
    ("H2", "SQN (combined)", "GRU (supervised)"),
    ("H2", "SNQN (supervised head)", "GRU (supervised)"),
    ("H2", "SNQN (combined)", "GRU (supervised)"),
    ("Other", "GRU re-ranking KNN top 50", "Item-KNN"),
    ("Other", "Bandit + GRU: LinUCB", "GRU re-ranking KNN top 50"),
    ("H3", "Item-KNN", "Popularity"),
    ("H3", "GRU (supervised)", "Popularity"),
    ("H3", "Bandit: LinUCB", "Popularity"),
    ("H3", "SNQN (combined)", "Popularity"),
]


def build_models(n):
    train = load_train()
    knn = ItemKNN(n, decay=0.8, include_seen=True, seen_boost=2.0).fit(train)
    fb = FeatureBuilder(knn, train, n)
    gru = TorchRecommender(load_net(MODELS / "gru.pt")[0], "ce")
    fb_h = FeatureBuilder(knn, train, n, scorer=gru)
    betas = json.loads((RESULTS / "beta_tuning.json").read_text())
    sqn, snqn = load_net(MODELS / "sqn_g0.5.pt")[0], load_net(MODELS / "snqn_g0.5.pt")[0]
    bandit = lambda f, feats: BanditRecommender(LinearBandit.from_theta(np.load(MODELS / f)), feats)
    return {
        "Popularity": Popularity(n).fit(train),
        "Item-KNN": knn,
        "Bandit: LinUCB": bandit("linucb_theta.npy", fb),
        "Bandit: Thompson sampling": bandit("ts_theta.npy", fb),
        "Bandit: Epsilon-greedy": bandit("egreedy_theta.npy", fb),
        "Bandit + GRU: LinUCB": bandit("linucb_hybrid_theta.npy", fb_h),
        "Bandit + GRU: Thompson sampling": bandit("ts_hybrid_theta.npy", fb_h),
        "Bandit + GRU: Epsilon-greedy": bandit("egreedy_hybrid_theta.npy", fb_h),
        "GRU (supervised)": gru,
        "GRU re-ranking KNN top 50": ShortlistReranker(knn, gru),
        "SQN (supervised head)": TorchRecommender(sqn, "ce"),
        "SQN (combined)": TorchRecommender(sqn, "mix", beta=betas["SQN gamma=0.5"]["chosen"]),
        "SNQN (supervised head)": TorchRecommender(snqn, "ce"),
        "SNQN (combined)": TorchRecommender(snqn, "mix", beta=betas["SNQN gamma=0.5"]["chosen"]),
    }


def per_transition(topk, test):
    match = topk[:, :SLATE_SIZE] == test["target"][:, None]
    hit = match.any(1)
    pos = np.where(hit, match.argmax(1), SLATE_SIZE)
    ndcg = np.where(hit, 1 / np.log2(pos + 2), 0.0)
    reward = hit * np.vectorize(REWARDS.get)(test["event"])
    return hit.astype(float), ndcg, reward


def session_sums(values, codes, n_sessions, mask=None):
    w = np.ones_like(values) if mask is None else mask.astype(float)
    return (np.bincount(codes, values * w, n_sessions), np.bincount(codes, w, n_sessions))


def main():
    plotstyle.apply()
    n = steps.n_items()
    test = steps.build("test")
    codes = np.unique(test["sid"], return_inverse=True)[1]
    n_sess = codes.max() + 1
    purchase = test["event"] == 2

    models = build_models(n)
    sums = {}
    for name, model in models.items():
        res, topk = ev.evaluate(model, test, n)
        hit, ndcg, reward = per_transition(topk, test)
        # the per-transition arrays must reproduce the evaluation code's numbers
        assert abs(hit.mean() - res["all"]["HR@10"]) < 1e-9, name
        assert abs(ndcg.mean() - res["all"]["NDCG@10"]) < 1e-9, name
        sums[name] = {
            "HR@10": session_sums(hit, codes, n_sess),
            "NDCG@10": session_sums(ndcg, codes, n_sess),
            "Reward@10": session_sums(reward, codes, n_sess),
            "Purchase HR@10": session_sums(hit, codes, n_sess, purchase),
            "Purchase NDCG@10": session_sums(ndcg, codes, n_sess, purchase),
        }
        print(f"{name}: NDCG@10 {ndcg.mean():.4f}", flush=True)

    rng = np.random.default_rng(SEED)
    weights = np.stack([np.bincount(rng.integers(0, n_sess, n_sess), minlength=n_sess)
                        for _ in range(B)]).astype(float)

    boot, point = {}, {}
    for name, m in sums.items():
        boot[name], point[name] = {}, {}
        for metric, (num, den) in m.items():
            point[name][metric] = num.sum() / den.sum()
            boot[name][metric] = (weights @ num) / (weights @ den)

    rows = []
    for name in models:
        row = {"model": name}
        for metric in METRICS:
            lo, hi = np.percentile(boot[name][metric], [2.5, 97.5])
            row[metric] = point[name][metric]
            row[f"{metric} low"], row[f"{metric} high"] = lo, hi
        rows.append(row)
    table = pd.DataFrame(rows)
    table.to_csv(RESULTS / "bootstrap_ci_models.csv", index=False)

    diffs = []
    for hyp, a, b in COMPARISONS:
        for metric in METRICS:
            d = boot[a][metric] - boot[b][metric]
            lo, hi = np.percentile(d, [2.5, 97.5])
            p = min(1.0, 2 * min((d <= 0).mean(), (d >= 0).mean()))
            diffs.append({"hypothesis": hyp, "model": a, "reference": b, "metric": metric,
                          "difference": point[a][metric] - point[b][metric],
                          "low": lo, "high": hi, "p": p,
                          "verdict": "higher" if lo > 0 else "lower" if hi < 0 else "no clear difference"})
    dtable = pd.DataFrame(diffs)
    dtable.to_csv(RESULTS / "bootstrap_ci_differences.csv", index=False)

    write_markdown(table, dtable)
    ci_chart(table, "NDCG@10", "Overall ranking quality with 95% confidence intervals (NDCG@10)",
             FIGURES / "10_ci_ndcg.png")
    ci_chart(table, "Purchase NDCG@10",
             "Purchase ranking quality with 95% confidence intervals (purchase NDCG@10)",
             FIGURES / "11_ci_purchase_ndcg.png")
    diff_chart(dtable)
    (RESULTS / "bootstrap_ci.json").write_text(json.dumps(
        {"resamples": B, "unit": "test session", "sessions": int(n_sess),
         "transitions": int(len(codes)), "purchase_transitions": int(purchase.sum()),
         "seed": SEED}, indent=2))


def fmt(v, lo, hi):
    return f"{v:.3f} ({lo:.3f} to {hi:.3f})"


def write_markdown(table, dtable):
    lines = ["# Bootstrap 95% confidence intervals (test period)", "",
             f"Paired bootstrap over test sessions, {B} resamples.", "",
             "## Models", "",
             "| Model | " + " | ".join(METRICS) + " |",
             "| --- | " + " | ".join("---" for _ in METRICS) + " |"]
    for _, r in table.iterrows():
        cells = [fmt(r[m], r[f"{m} low"], r[f"{m} high"]) for m in METRICS]
        lines.append(f"| {r['model']} | " + " | ".join(cells) + " |")
    lines += ["", "## Paired differences (model minus reference)", "",
              "| Hypothesis | Model | Reference | Metric | Difference (95% CI) | p | Verdict |",
              "| --- | --- | --- | --- | --- | --- | --- |"]
    for _, r in dtable.iterrows():
        p = "<0.001" if r["p"] < 0.001 else f"{r['p']:.3f}"
        lines.append(f"| {r['hypothesis']} | {r['model']} | {r['reference']} | {r['metric']} | "
                     f"{r['difference']:+.4f} ({r['low']:+.4f} to {r['high']:+.4f}) | {p} | {r['verdict']} |")
    (RESULTS / "bootstrap_ci.md").write_text("\n".join(lines) + "\n")


def ci_chart(table, metric, title, path):
    t = table.sort_values(metric)
    fig, ax = plt.subplots(figsize=(8, 0.4 * len(t) + 1.3))
    y = np.arange(len(t))
    ax.hlines(y, t[f"{metric} low"], t[f"{metric} high"], color=plotstyle.SERIES[0], linewidth=2)
    ax.plot(t[metric], y, "o", color=plotstyle.SERIES[0], markersize=6)
    for yi, (_, r) in zip(y, t.iterrows()):
        ax.text(r[f"{metric} high"] + 0.004, yi, f"{r[metric]:.3f}", va="center", fontsize=8,
                color=plotstyle.INK)
    ax.set_yticks(y, t["model"])
    ax.grid(axis="y", visible=False)
    ax.grid(axis="x")
    ax.set_xlim(0, t[f"{metric} high"].max() * 1.1)
    ax.set_title(title)
    ax.set_xlabel(f"{metric} (dot = test value, line = 95% confidence interval)")
    plotstyle.save(fig, path)


DIFF_ROWS = [
    ("Bandit: LinUCB", "Item-KNN", "LinUCB vs Item-KNN"),
    ("Bandit: Thompson sampling", "Item-KNN", "Thompson sampling vs Item-KNN"),
    ("Bandit: Epsilon-greedy", "Item-KNN", "Epsilon-greedy vs Item-KNN"),
    ("Bandit + GRU: LinUCB", "Item-KNN", "LinUCB + GRU vs Item-KNN"),
    ("SQN (supervised head)", "GRU (supervised)", "SQN vs GRU"),
    ("SQN (combined)", "GRU (supervised)", "SQN combined vs GRU"),
    ("SNQN (supervised head)", "GRU (supervised)", "SNQN vs GRU"),
    ("SNQN (combined)", "GRU (supervised)", "SNQN combined vs GRU"),
]


def diff_chart(dtable):
    metrics = ["NDCG@10", "Reward@10", "Purchase NDCG@10"]
    fig, axes = plt.subplots(1, 3, figsize=(11, 4.4), sharey=True)
    y = np.arange(len(DIFF_ROWS))[::-1]
    for ax, metric in zip(axes, metrics):
        for yi, (a, b, _) in zip(y, DIFF_ROWS):
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
        ax.axhline(3.5, color=plotstyle.GRID, linewidth=1)
    axes[0].set_yticks(y, [label for _, _, label in DIFF_ROWS])
    fig.suptitle("Paired differences with 95% confidence intervals (right of the dashed line = better)",
                 x=0.01, ha="left", fontweight="bold", fontsize=12)
    filled = plt.Line2D([], [], marker="o", color=plotstyle.SERIES[0], linestyle="-", label="interval excludes 0")
    hollow = plt.Line2D([], [], marker="o", color="#a3a29e", markerfacecolor="white", linestyle="-",
                        label="interval includes 0 (no clear difference)")
    fig.legend(handles=[filled, hollow], loc="lower center", ncol=2, frameon=False, fontsize=9,
               bbox_to_anchor=(0.5, -0.04))
    fig.tight_layout(rect=(0, 0.04, 1, 0.95))
    plotstyle.save(fig, FIGURES / "12_paired_differences.png")


if __name__ == "__main__":
    import sys
    if "--charts" in sys.argv:
        plotstyle.apply()
        diff_chart(pd.read_csv(RESULTS / "bootstrap_ci_differences.csv"))
    else:
        main()
