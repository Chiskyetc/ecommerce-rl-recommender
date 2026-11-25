"""Contextual bandits that re-rank the Item-KNN shortlist."""
import numpy as np
import pandas as pd

from config import PROCESSED, SLATE_SIZE, STATE_LEN

FEATURES = [
    "bias", "knn_score", "knn_rank", "in_session", "recency", "carted_in_session",
    "bought_in_session", "log_popularity", "conversion_rate", "same_category",
    "same_parent", "in_session_x_carted",
]
NEURAL_FEATURES = ["gru_prob", "gru_rank"]


class FeatureBuilder:
    def __init__(self, knn, train, n_items, scorer=None):
        self.knn = knn
        self.scorer = scorer  # optional trained network, adds its view of each candidate
        self.names = FEATURES + (NEURAL_FEATURES if scorer is not None else [])
        cats = pd.read_parquet(PROCESSED / "items.parquet")
        self.cat = np.full(n_items + 1, -2, np.int32)
        self.parent = np.full(n_items + 1, -2, np.int32)
        self.cat[cats["item"]] = cats["category"]
        self.parent[cats["item"]] = cats["parent"]
        self.cat[self.cat == -1] = -3  # unknown category should never match
        counts = np.bincount(train["item"], minlength=n_items + 1).astype(np.float32)
        conv = np.bincount(train["item"], weights=(train["event"] > 0), minlength=n_items + 1)
        self.log_pop = np.log1p(counts) / np.log1p(counts.max())
        self.conv = (conv + 0.05 * 2) / (counts + 2)  # smoothed toward the global rate

    def build(self, state, state_ev, n_cand):
        cands, scores = self.knn.candidates(state, n_cand)
        B = len(state)
        x = np.zeros((B, n_cand, len(self.names)), np.float32)
        x[..., 0] = 1
        finite = np.where(np.isfinite(scores), scores, 0)
        top = finite.max(axis=1, keepdims=True)
        top[top == 0] = 1
        x[..., 1] = finite / top
        x[..., 2] = 1 - np.arange(n_cand) / n_cand

        match = cands[:, :, None] == state[:, None, :]  # (B, C, L)
        in_sess = match.any(-1)
        pos = np.where(match, np.arange(STATE_LEN)[None, None, :], -1).max(-1)
        ev = np.where(match, state_ev[:, None, :], 0).max(-1)  # 1 view, 2 cart, 3 purchase
        x[..., 3] = in_sess
        x[..., 4] = np.where(in_sess, (pos + 1) / STATE_LEN, 0)
        x[..., 5] = ev == 2
        x[..., 6] = ev == 3
        x[..., 7] = self.log_pop[cands]
        x[..., 8] = self.conv[cands]
        last = state[:, -1:]
        x[..., 9] = self.cat[cands] == self.cat[last]
        x[..., 10] = self.parent[cands] == self.parent[last]
        x[..., 11] = x[..., 3] * x[..., 5]
        if self.scorer is not None:
            s = self.scorer.score_items(state, state_ev, cands)
            s = np.exp(s - s.max(axis=1, keepdims=True))
            x[..., 12] = s / s.sum(axis=1, keepdims=True)
            rank = np.argsort(np.argsort(-s, axis=1), axis=1)
            x[..., 13] = 1 - rank / n_cand
        return cands, x


class LinearBandit:
    """Shared-parameter linear bandit with semi-bandit (per-slot) feedback.

    policy: "ucb" (LinUCB), "ts" (linear Thompson sampling) or "egreedy".
    """

    def __init__(self, d, policy="ucb", alpha=0.5, lam=1.0, epsilon=0.1, v=0.25, seed=0):
        self.policy, self.alpha, self.epsilon, self.v = policy, alpha, epsilon, v
        self.A_inv = np.eye(d) / lam
        self.b = np.zeros(d)
        self.rng = np.random.default_rng(seed)

    @classmethod
    def from_theta(cls, theta):
        b = cls(len(theta))
        b.b = np.asarray(theta, dtype=float).copy()  # A_inv stays identity, so theta == b
        return b

    @property
    def theta(self):
        return self.A_inv @ self.b

    def scores(self, x, explore=True):
        theta = self.theta
        if not explore:
            return x @ theta
        if self.policy == "ucb":
            width = np.sqrt(np.einsum("cd,de,ce->c", x, self.A_inv, x))
            return x @ theta + self.alpha * width
        if self.policy == "ts":
            sample = self.rng.multivariate_normal(theta, self.v ** 2 * self.A_inv)
            return x @ sample
        s = x @ theta
        if self.rng.random() < self.epsilon:
            s = self.rng.random(len(s))
        return s

    def select(self, x, k=SLATE_SIZE, explore=True):
        return np.argsort(-self.scores(x, explore))[:k]

    def update(self, x_shown, rewards):
        for x, r in zip(x_shown, rewards):
            Ax = self.A_inv @ x
            self.A_inv -= np.outer(Ax, Ax) / (1 + x @ Ax)
            self.b += r * x


class BanditRecommender:
    """Frozen bandit policy for offline evaluation."""

    def __init__(self, bandit, features, n_cand=50):
        self.bandit, self.features, self.n_cand = bandit, features, n_cand

    def recommend(self, state, state_ev, k):
        cands, x = self.features.build(state, state_ev, self.n_cand)
        s = x @ self.bandit.theta
        order = np.argsort(-s, axis=1)[:, :k]
        return np.take_along_axis(cands, order, axis=1)
