import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class SessionGRU(nn.Module):
    """GRU session encoder with a supervised head and an optional Q-value head.

    With q_head=False this is a plain next-item model (GRU4Rec-style).
    With q_head=True the same encoder is also trained by double Q-learning on
    event rewards, following the SQN set-up of Xin et al. (2020).
    """

    def __init__(self, n_items, emb=64, hidden=64, ev_emb=8, dropout=0.1, q_head=True,
                 repeat=True):
        super().__init__()
        self.item_emb = nn.Embedding(n_items + 1, emb, padding_idx=0)
        self.ev_emb = nn.Embedding(4, ev_emb, padding_idx=0)
        self.gru = nn.GRU(emb + ev_emb, hidden, batch_first=True)
        self.drop = nn.Dropout(dropout)
        self.ce_head = nn.Linear(hidden, n_items + 1)
        self.q_head = nn.Linear(hidden, n_items + 1) if q_head else None
        self.repeat = repeat
        if repeat:
            # per-position score for items already in the session, one output per head
            self.repeat_mlp = nn.Sequential(
                nn.Linear(hidden + hidden + ev_emb + 1, hidden), nn.Tanh(),
                nn.Linear(hidden, 2 if q_head else 1),
            )

    def forward(self, items, events):
        x = torch.cat([self.item_emb(items), self.ev_emb(events)], dim=-1)
        out, _ = self.gru(self.drop(x))
        h = out[:, -1]
        logits = self.ce_head(self.drop(h))
        q = self.q_head(h) if self.q_head is not None else None
        if self.repeat:
            L = items.shape[1]
            recency = torch.linspace(1, 0, L, device=items.device).flip(0)
            feats = torch.cat([
                h[:, None, :].expand(-1, L, -1), out, self.ev_emb(events),
                recency[None, :, None].expand(items.shape[0], -1, -1),
            ], dim=-1)
            rep = self.repeat_mlp(feats) * (items > 0).unsqueeze(-1)
            logits = logits.scatter_add(1, items, rep[..., 0])
            if q is not None:
                q = q.scatter_add(1, items, rep[..., 1])
        return logits, q


class TorchRecommender:
    """Adapter so trained networks plug into the shared evaluation code."""

    def __init__(self, net, head="ce", beta=1.0, device="cpu"):
        self.net = net.eval()
        self.head = head
        self.beta = beta
        self.device = device

    @torch.no_grad()
    def scores(self, state, state_ev):
        items = torch.as_tensor(state, dtype=torch.long, device=self.device)
        events = torch.as_tensor(state_ev, dtype=torch.long, device=self.device)
        logits, q = self.net(items, events)
        if self.head == "ce":
            s = logits
        elif self.head == "q":
            s = q
        else:
            # supervised log-probability plus the expected reward the Q head predicts
            s = F.log_softmax(logits, -1) + self.beta * q
        s[:, 0] = -float("inf")
        return s

    def recommend(self, state, state_ev, k):
        return self.scores(state, state_ev).topk(k, dim=1).indices.cpu().numpy()

    def score_items(self, state, state_ev, cands):
        s = self.scores(state, state_ev)
        return torch.gather(s, 1, torch.as_tensor(cands, dtype=torch.long)).numpy()


def set_seed(seed):
    np.random.seed(seed)
    torch.manual_seed(seed)
