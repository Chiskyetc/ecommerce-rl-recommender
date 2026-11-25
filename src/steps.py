"""Unroll sessions into (state, action, reward, next_state) transitions."""
import json

import numpy as np
import pandas as pd

from config import PROCESSED, REWARDS, STATE_LEN


def _pad_left(seq, n):
    seq = seq[-n:]
    out = np.zeros(n, dtype=seq.dtype)
    if len(seq):
        out[-len(seq):] = seq
    return out


def build(split, sessions=None):
    path = PROCESSED / f"steps_{split}.npz"
    if path.exists():
        return dict(np.load(path))
    if sessions is None:
        sessions = pd.read_parquet(PROCESSED / "sessions.parquet")
    part = sessions[sessions["split"] == split].sort_values(["sid", "timestamp"], kind="mergesort")

    s_items, s_events, ns_items, ns_events = [], [], [], []
    target, t_event, done, sid, pos = [], [], [], [], []
    for session_id, g in part.groupby("sid", sort=False):
        items = g["item"].to_numpy(np.int32)
        events = g["event"].to_numpy(np.int8)
        # events are shifted by one so 0 can stay as padding
        ev = events + 1
        n = len(items)
        for t in range(1, n):
            s_items.append(_pad_left(items[:t], STATE_LEN))
            s_events.append(_pad_left(ev[:t], STATE_LEN))
            ns_items.append(_pad_left(items[:t + 1], STATE_LEN))
            ns_events.append(_pad_left(ev[:t + 1], STATE_LEN))
            target.append(items[t])
            t_event.append(events[t])
            done.append(t == n - 1)
            sid.append(session_id)
            pos.append(t)

    t_event = np.array(t_event, dtype=np.int8)
    data = {
        "state": np.stack(s_items),
        "state_ev": np.stack(s_events).astype(np.int8),
        "next_state": np.stack(ns_items),
        "next_state_ev": np.stack(ns_events).astype(np.int8),
        "target": np.array(target, dtype=np.int32),
        "event": t_event,
        "reward": np.vectorize(REWARDS.get)(t_event).astype(np.float32),
        "done": np.array(done, dtype=bool),
        "sid": np.array(sid, dtype=np.int64),
        "pos": np.array(pos, dtype=np.int16),
    }
    np.savez(path, **data)
    return data


def n_items():
    return json.loads((PROCESSED / "stats.json").read_text())["items"]


if __name__ == "__main__":
    sessions = pd.read_parquet(PROCESSED / "sessions.parquet")
    for split in ("train", "val", "test"):
        d = build(split, sessions)
        in_state = np.mean([t in s for t, s in zip(d["target"], d["state"])])
        print(f"{split}: {len(d['target']):,} steps, target already in state {in_state:.1%}, "
              f"events {np.bincount(d['event'], minlength=3).tolist()}")
