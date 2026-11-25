"""Train the supervised GRU or the SQN variant.

    python train_sqn.py --mode gru
    python train_sqn.py --mode sqn --gamma 0.5
    python train_sqn.py --mode sqn --gamma 0.5 --neg 10   (SNQN)

Ablations (saved under models/ablations and results/ablations):
    python train_sqn.py --mode gru --no_repeat --ablation --tag gru_norepeat
    python train_sqn.py --mode sqn --gamma 0.9 --ablation
    python train_sqn.py --mode sqn --rewards 1,1,1 --ablation --tag sqn_equal_rewards
"""
import argparse
import copy
import json
import time

import numpy as np
import torch
import torch.nn.functional as F

import evaluate as ev
import steps
from config import MODELS, RESULTS, SEED
from models import SessionGRU, TorchRecommender, set_seed


def tensors(d, idx):
    t = lambda k: torch.as_tensor(d[k][idx], dtype=torch.long)
    return (t("state"), t("state_ev"), t("next_state"), t("next_state_ev"), t("target"),
            torch.as_tensor(d["reward"][idx]), torch.as_tensor(d["done"][idx], dtype=torch.float32))


def train(args):
    set_seed(args.seed)
    torch.set_num_threads(args.threads)
    n = steps.n_items()
    tr, va = steps.build("train"), steps.build("val")
    if args.rewards:
        # only the training signal changes; validation keeps the standard rewards
        weights = np.array([float(x) for x in args.rewards.split(",")], dtype=np.float32)
        tr = dict(tr)
        tr["reward"] = weights[tr["event"]]
    model_dir, result_dir = MODELS, RESULTS
    if args.ablation:
        model_dir, result_dir = MODELS / "ablations", RESULTS / "ablations"
        model_dir.mkdir(exist_ok=True)
        result_dir.mkdir(exist_ok=True)
    use_q = args.mode == "sqn"
    net = SessionGRU(n, emb=args.dim, hidden=args.dim, q_head=use_q, repeat=not args.no_repeat)
    target_net = copy.deepcopy(net).eval() if use_q else None
    opt = torch.optim.Adam(net.parameters(), lr=args.lr)

    if args.tag:
        tag = args.tag
    elif use_q:
        tag = f"{'snqn' if args.neg else 'sqn'}_g{args.gamma}"
    else:
        tag = "gru"

    def checkpoint(state):
        torch.save({"state_dict": state, "n_items": n, "dim": args.dim, "q_head": use_q,
                    "repeat": not args.no_repeat, "args": vars(args)}, model_dir / f"{tag}.pt")

    history, best, best_state, bad = [], -1, None, 0
    n_steps = len(tr["target"])
    updates = 0
    for epoch in range(1, args.epochs + 1):
        net.train()
        perm = np.random.permutation(n_steps)
        t0, tot_ce, tot_q = time.time(), 0.0, 0.0
        for b in range(0, n_steps, args.batch):
            s, se, ns, nse, a, r, done = tensors(tr, perm[b:b + args.batch])
            logits, q = net(s, se)
            loss_ce = F.cross_entropy(logits, a)
            loss = loss_ce
            if use_q:
                with torch.no_grad():
                    _, q_next_online = net(ns, nse)
                    q_next_online[:, 0] = -float("inf")
                    a_star = q_next_online.argmax(1, keepdim=True)
                    _, q_next_target = target_net(ns, nse)
                    y = r + args.gamma * (1 - done) * q_next_target.gather(1, a_star).squeeze(1)
                q_sa = q.gather(1, a[:, None]).squeeze(1)
                loss_q = F.mse_loss(q_sa, y)
                if args.neg:
                    # SNQN: unchosen products earn nothing and the shopper stays in the
                    # current state, so their target is gamma * V(s)
                    neg = torch.randint(1, n + 1, (len(a), args.neg))
                    with torch.no_grad():
                        q_cur = q.detach().clone()
                        q_cur[:, 0] = -float("inf")
                        a_cur = q_cur.argmax(1, keepdim=True)
                        _, q_cur_target = target_net(s, se)
                        y_neg = args.gamma * q_cur_target.gather(1, a_cur)
                    q_neg = q.gather(1, neg)
                    loss_q = loss_q + F.mse_loss(q_neg, y_neg.expand_as(q_neg))
                loss = loss + args.q_weight * loss_q
                tot_q += loss_q.item()
            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(net.parameters(), 5.0)
            opt.step()
            tot_ce += loss_ce.item()
            updates += 1
            if use_q and updates % args.target_every == 0:
                target_net.load_state_dict(net.state_dict())

        nb = int(np.ceil(n_steps / args.batch))
        row = {"epoch": epoch, "ce_loss": tot_ce / nb, "q_loss": tot_q / nb if use_q else None,
               "secs": round(time.time() - t0, 1)}
        heads = ["ce", "q", "mix"] if use_q else ["ce"]
        for head in heads:
            r, _ = ev.evaluate(TorchRecommender(net, head), va, n)
            row[f"val_{head}_NDCG@10"] = r["all"]["NDCG@10"]
            row[f"val_{head}_HR@10"] = r["all"]["HR@10"]
            row[f"val_{head}_reward@10"] = r["reward@10"]
            row[f"val_{head}_purchase_HR@10"] = r["purchase"]["HR@10"]
        history.append(row)
        print(json.dumps(row), flush=True)

        score = max(row[f"val_{h}_NDCG@10"] for h in heads)
        (result_dir / f"history_{tag}.json").write_text(json.dumps(history, indent=2))
        if score > best:
            best, best_state, bad = score, copy.deepcopy(net.state_dict()), 0
            checkpoint(best_state)
        else:
            bad += 1
            if bad >= args.patience:
                break

    net.load_state_dict(best_state)
    return net, tag


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--mode", choices=["gru", "sqn"], default="sqn")
    p.add_argument("--gamma", type=float, default=0.5)
    p.add_argument("--q_weight", type=float, default=1.0)
    p.add_argument("--epochs", type=int, default=12)
    p.add_argument("--patience", type=int, default=2)
    p.add_argument("--batch", type=int, default=256)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--dim", type=int, default=64)
    p.add_argument("--target_every", type=int, default=1000)
    p.add_argument("--threads", type=int, default=2)
    p.add_argument("--seed", type=int, default=SEED)
    p.add_argument("--tag", default=None)
    p.add_argument("--no_repeat", action="store_true")
    p.add_argument("--neg", type=int, default=0, help="negative samples per step (SNQN)")
    p.add_argument("--rewards", default=None, help="training rewards for view,cart,purchase, e.g. 1,1,1")
    p.add_argument("--ablation", action="store_true", help="save under models/ablations and results/ablations")
    train(p.parse_args())
