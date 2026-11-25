"""Session replay environment.

Each episode replays one logged session. At every step the agent sees the
session so far plus a shortlist of candidate products, and shows a slate of
SLATE_SIZE of them. If the shopper's real next product is in the slate the
agent earns the reward for that event (view 0.2, cart 0.6, purchase 1.0).

Transitions follow the log, not the agent's choices, so this measures how well
a policy anticipates real behaviour. It cannot show how a shopper would have
reacted to products they were never shown.
"""
import gymnasium as gym
import numpy as np
from gymnasium import spaces

from config import REWARDS, SLATE_SIZE, STATE_LEN


class SessionReplayEnv(gym.Env):
    metadata = {"render_modes": []}

    def __init__(self, data, n_items, candidate_fn, n_candidates=50, shuffle=True, seed=None):
        super().__init__()
        self.d = data
        self.candidate_fn = candidate_fn
        self.n_candidates = n_candidates
        self.shuffle = shuffle
        starts = np.flatnonzero(np.r_[True, data["sid"][1:] != data["sid"][:-1]])
        self.episodes = np.c_[starts, np.r_[starts[1:], len(data["sid"])]]
        self.rng = np.random.default_rng(seed)
        self.order = np.arange(len(self.episodes))
        self.ep_ptr = -1
        self.observation_space = spaces.Dict({
            "state": spaces.Box(0, n_items, (STATE_LEN,), np.int32),
            "state_ev": spaces.Box(0, 3, (STATE_LEN,), np.int8),
            "candidates": spaces.Box(0, n_items, (n_candidates,), np.int32),
        })
        self.action_space = spaces.MultiDiscrete([n_candidates] * SLATE_SIZE)

    def _obs(self):
        i = self.row
        state, state_ev = self.d["state"][i], self.d["state_ev"][i]
        cands, _ = self.candidate_fn(state[None], n=self.n_candidates)
        self.cands = cands[0]
        return {"state": state, "state_ev": state_ev, "candidates": self.cands}

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.ep_ptr += 1
        if self.ep_ptr >= len(self.order):
            self.ep_ptr = 0
            if self.shuffle:
                self.rng.shuffle(self.order)
        self.row, self.end = self.episodes[self.order[self.ep_ptr]]
        return self._obs(), {}

    def step(self, action):
        slate = self.cands[np.asarray(action)[:SLATE_SIZE]]
        target, event = self.d["target"][self.row], int(self.d["event"][self.row])
        hit = np.flatnonzero(slate == target)
        reward = REWARDS[event] if len(hit) else 0.0
        info = {"target": int(target), "event": event,
                "hit_rank": int(hit[0]) if len(hit) else -1, "slate": slate}
        self.row += 1
        terminated = self.row >= self.end
        obs = None if terminated else self._obs()
        return obs, reward, terminated, False, info
