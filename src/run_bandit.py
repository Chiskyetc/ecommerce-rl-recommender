"""Learn bandit policies online over the validation period, then evaluate them frozen on test.

The validation period plays the role of a live deployment: the bandit starts
knowing nothing and learns only from feedback on the slates it shows.
"""
import argparse
import json

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import evaluate as ev
import plotstyle
import steps
from bandit import BanditRecommender, FeatureBuilder, LinearBandit
from baselines import ItemKNN, load_train
from config import FIGURES, MODELS, PROCESSED, RESULTS, REWARDS, SEED, SLATE_SIZE
from env import SessionReplayEnv

N_CAND = 50
CHUNK = 2048


def stream_order(data):
    sessions = pd.read_parquet(PROCESSED / "sessions.parquet", columns=["sid", "timestamp"])
    start = sessions.groupby("sid")["timestamp"].min()
    return np.lexsort((data["pos"], start.reindex(data["sid"]).to_numpy()))


def run_online(policies, fb, data, order, rewards=None):
    log = {name: np.zeros(len(order), np.float32) for name in policies}
    rng = np.random.default_rng(SEED)
    rewards_all = np.vectorize((rewards or REWARDS).get)(data["event"])
    for c in range(0, len(order), CHUNK):
        rows = order[c:c + CHUNK]
        cands, x = fb.build(data["state"][rows], data["state_ev"][rows], N_CAND)
        for j, row in enumerate(rows):
            hit_vec = (cands[j] == data["target"][row]) * rewards_all[row]
            for name, pol in policies.items():
                if name == "Item-KNN (fixed)":
                    idx = np.arange(SLATE_SIZE)
                elif name == "Random from shortlist":
                    idx = rng.choice(N_CAND, SLATE_SIZE, replace=False)
                else:
                    idx = pol.select(x[j])
                    pol.update(x[j, idx], hit_vec[idx])
                log[name][c + j] = hit_vec[idx].sum()
    return log


def plot_curves(log, path):
    fig, ax = plt.subplots(figsize=(8, 4))
    start = 200  # the running average is meaningless over the first handful of steps
    for i, (name, r) in enumerate(log.items()):
        steps_ = np.arange(1, len(r) + 1)
        cum = np.cumsum(r) / steps_
        ax.plot(steps_[start:], cum[start:], label=name, color=plotstyle.SERIES[i])
    ax.set_xlabel("Recommendations served (validation period, in time order)")
    ax.set_ylabel("Running average reward per step")
    ax.set_xscale("log")
    ax.set_ylim(0, None)
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.2), ncol=3)
    ax.set_title("Online learning in the replay environment")
    plotstyle.save(fig, path)


def check_env(bandit, fb, data, n_items, episodes=300):
    """Play episodes through the gym interface to confirm it agrees with the fast loop."""
    def cand_fn(state, n):
        c, x = fb.build(state, np.zeros_like(state, dtype=np.int8), n)
        cand_fn.x = x
        return c, None

    env = SessionReplayEnv(data, n_items, cand_fn, N_CAND, shuffle=False, seed=SEED)
    total, n_steps = 0.0, 0
    for _ in range(episodes):
        obs, _ = env.reset()
        done = False
        while not done:
            # rebuild with true event context for the policy
            _, x = fb.build(obs["state"][None], obs["state_ev"][None], N_CAND)
            action = bandit.select(x[0], explore=False)
            obs, r, done, _, _ = env.step(action)
            total += r
            n_steps += 1
    return total / n_steps


def main(hybrid=False):
    plotstyle.apply()
    n = steps.n_items()
    train = load_train()
    val, test = steps.build("val"), steps.build("test")
    knn = ItemKNN(n, decay=0.8, include_seen=True, seen_boost=2.0).fit(train)
    scorer = None
    if hybrid:
        from compare import load_net
        from models import TorchRecommender
        scorer = TorchRecommender(load_net(MODELS / "gru.pt")[0], "ce")
    fb = FeatureBuilder(knn, train, n, scorer=scorer)
    suffix, prefix = ("_hybrid", "Bandit + GRU score") if hybrid else ("", "Bandit")

    d = len(fb.names)
    policies = {
        "LinUCB": LinearBandit(d, "ucb", alpha=0.5, seed=SEED),
        "Linear Thompson sampling": LinearBandit(d, "ts", v=0.25, seed=SEED),
        "Epsilon-greedy (0.1)": LinearBandit(d, "egreedy", epsilon=0.1, seed=SEED),
        "Item-KNN (fixed)": None,
        "Random from shortlist": None,
    }
    log = run_online(policies, fb, val, stream_order(val))
    plot_curves(log, FIGURES / f"06_bandit_learning_curves{suffix}.png")
    online = {k: float(v.mean()) for k, v in log.items()}
    print("online avg reward per step:", json.dumps(online, indent=2))

    hits = []
    for c in range(0, len(test["target"]), CHUNK):
        cands, _ = knn.candidates(test["state"][c:c + CHUNK], N_CAND)
        hits.append((cands == test["target"][c:c + CHUNK, None]).any(1))
    cand_recall = float(np.concatenate(hits).mean())

    results = []
    for name in ("LinUCB", "Linear Thompson sampling", "Epsilon-greedy (0.1)"):
        r, _ = ev.evaluate(BanditRecommender(policies[name], fb, N_CAND), test, n,
                           name=f"{prefix}: {name}")
        results.append(r)
    ev.save(results, f"bandit{suffix}_test")

    weights = {name: dict(zip(fb.names, map(float, policies[name].theta)))
               for name in ("LinUCB", "Linear Thompson sampling", "Epsilon-greedy (0.1)")}
    env_reward = check_env(policies["LinUCB"], fb, test, n)
    summary = {"online_val_reward_per_step": online, "test_candidate_recall@50": cand_recall,
               "feature_weights": weights, "env_check_linucb_reward_per_step": env_reward}
    (RESULTS / f"bandit{suffix}_summary.json").write_text(json.dumps(summary, indent=2))
    np.save(MODELS / f"linucb{suffix}_theta.npy", policies["LinUCB"].theta)
    np.save(MODELS / f"ts{suffix}_theta.npy", policies["Linear Thompson sampling"].theta)
    np.save(MODELS / f"egreedy{suffix}_theta.npy", policies["Epsilon-greedy (0.1)"].theta)
    print(json.dumps({k: v for k, v in summary.items() if k != "feature_weights"}, indent=2))
    print(ev.to_table(results).round(4).to_string(index=False))


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--hybrid", action="store_true", help="add the trained GRU's scores as features")
    main(p.parse_args().hybrid)
