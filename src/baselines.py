import numpy as np
import pandas as pd
import scipy.sparse as sp

from config import PROCESSED, STATE_LEN
from evaluate import topk_from_scores


class Popularity:
    def __init__(self, n_items):
        self.n_items = n_items

    def fit(self, sessions):
        counts = np.bincount(sessions["item"], minlength=self.n_items + 1).astype(np.float32)
        counts[0] = 0
        self.scores = counts
        self.top = np.argsort(-counts)
        return self

    def recommend(self, state, state_ev, k):
        return np.tile(self.top[:k], (len(state), 1))


class ItemKNN:
    """Session co-occurrence with cosine normalisation and recency-weighted aggregation.

    `include_seen` controls whether items already in the session can be recommended.
    Shoppers often return to something they just looked at (to cart or buy it),
    so ruling those out is a real design choice, not a detail.
    """

    def __init__(self, n_items, neighbours=200, decay=0.7, include_seen=True, seen_boost=1.0):
        self.n_items = n_items
        self.neighbours = neighbours
        self.decay = decay
        self.include_seen = include_seen
        self.seen_boost = seen_boost

    def fit(self, sessions):
        codes, _ = pd.factorize(sessions["sid"])
        x = sp.csr_matrix(
            (np.ones(len(codes), np.float32), (codes, sessions["item"].to_numpy())),
            shape=(codes.max() + 1, self.n_items + 1),
        )
        x.data[:] = 1.0
        x.sum_duplicates()
        x.data[:] = 1.0
        co = (x.T @ x).tocsr().astype(np.float32)
        norm = np.sqrt(co.diagonal())
        norm[norm == 0] = 1
        co = sp.diags(1 / norm) @ co @ sp.diags(1 / norm)
        co.setdiag(0)
        co.eliminate_zeros()
        self.sim = self._prune(co.tocsr())
        return self

    def _prune(self, m):
        rows, cols, vals = [], [], []
        for i in range(m.shape[0]):
            s, e = m.indptr[i], m.indptr[i + 1]
            if e == s:
                continue
            d, c = m.data[s:e], m.indices[s:e]
            if len(d) > self.neighbours:
                keep = np.argpartition(-d, self.neighbours)[:self.neighbours]
                d, c = d[keep], c[keep]
            rows.append(np.full(len(d), i))
            cols.append(c)
            vals.append(d)
        return sp.csr_matrix(
            (np.concatenate(vals), (np.concatenate(rows), np.concatenate(cols))), shape=m.shape
        )

    def scores(self, state):
        n = len(state)
        weights = self.decay ** np.arange(STATE_LEN - 1, -1, -1, dtype=np.float32)
        w = np.where(state > 0, weights[None, :], 0).astype(np.float32)
        rows = np.repeat(np.arange(n), STATE_LEN)
        q = sp.csr_matrix((w.ravel(), (rows, state.ravel())), shape=(n, self.n_items + 1))
        s = np.asarray((q @ self.sim).todense())
        if self.include_seen:
            # the latest item gets the largest boost
            s[rows, state.ravel()] += self.seen_boost * w.ravel()
        else:
            s[rows, state.ravel()] = -np.inf
        s[:, 0] = -np.inf
        return s

    def recommend(self, state, state_ev, k):
        return topk_from_scores(self.scores(state), k)

    def candidates(self, state, k):
        s = self.scores(state)
        top = topk_from_scores(s, k)
        return top, np.take_along_axis(s, top, axis=1)


def load_train():
    df = pd.read_parquet(PROCESSED / "sessions.parquet")
    return df[df["split"] == "train"]
