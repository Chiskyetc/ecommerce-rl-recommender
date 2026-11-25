# Reinforcement Learning Recommender System for E-Commerce

A session-based product recommender trained and evaluated on the RetailRocket e-commerce dataset. It compares classical baselines against two reinforcement learning approaches: contextual bandits that learn online from simulated feedback, and a deep Q-learning model (SQN) that learns to favour recommendations leading to add-to-carts and purchases.

## End result

By the end of the week the project delivers:

1. A reproducible pipeline from raw data to final results (`run_all.sh`)
2. Five models evaluated on the same held-out test period, with HR, NDCG, reward and coverage reported separately for views, add-to-carts and purchases
3. A Gymnasium replay environment that any RL agent can be plugged into
4. A Streamlit demo that replays real shopping sessions, shows each model's recommendations side by side, and collects user feedback
5. Figures and tables ready for the report

## Project structure

```
recsys_rl/
├── PLAN.md               problem framing, schedule, risks
├── SURVEY.md             user evaluation questionnaire
├── requirements.txt
├── run_all.sh            full pipeline, raw data to final tables
├── src/
│   ├── config.py         paths, cleaning thresholds, reward weights
│   ├── download_data.py  fetches RetailRocket from Kaggle
│   ├── data.py           cleaning, sessions, time split, item index
│   ├── eda.py            exploratory figures
│   ├── steps.py          sessions -> (state, action, reward, next state)
│   ├── evaluate.py       HR@k, NDCG@k, reward, coverage
│   ├── baselines.py      Popularity, Item-KNN
│   ├── env.py            session replay environment (Gymnasium)
│   ├── bandit.py         features, LinUCB / Thompson / epsilon-greedy
│   ├── run_bandit.py     online learning on validation, test evaluation
│   ├── models.py         GRU encoder with supervised and Q-value heads
│   ├── train_sqn.py      GRU, SQN and SNQN training
│   ├── compare.py        final tables and charts
│   ├── bootstrap_ci.py   bootstrap confidence intervals
│   ├── ablations.py      ablation studies
│   └── plotstyle.py
├── app/app.py            Streamlit demo
├── figures/              all charts
├── results/              metrics as JSON, CSV and Markdown
└── models/               trained weights
```

## Setup

Python 3.10 or newer.

```bash
python -m venv .venv
.venv\Scripts\activate          # Windows
# source .venv/bin/activate     # macOS / Linux
pip install -r requirements.txt
```

## Running

The full pipeline:

```bash
bash run_all.sh
```

Or step by step from `src/`:

```bash
python download_data.py     # ~300 MB
python data.py              # cleaned sessions -> data/processed
python eda.py               # figures 01-05
python steps.py             # transition arrays
python run_baselines.py     # Popularity, Item-KNN
python run_bandit.py        # bandits, figure 06
python train_sqn.py --mode gru
python train_sqn.py --mode sqn --gamma 0.5
python train_sqn.py --mode sqn --gamma 0.5 --neg 10
python run_bandit.py --hybrid
python compare.py           # final table, figures 07-09
python bootstrap_ci.py      # confidence intervals, figures 10-12
python ablations.py bandit  # bandit reward-weight ablation
python ablations.py evaluate  # all ablations, figures 13-14 (train the ablation networks first, see train_sqn.py)
```

The trained models and the processed test data are included, so the demo works straight after installing the requirements, with no retraining. From the project root:

```bash
streamlit run app/app.py
```

## Method summary

**Sessions.** RetailRocket has no session ID, so a session is a run of events from one visitor with no gap over 30 minutes. Cleaning steps:

- Remove repeated identical events and visitors with over 1,000 events (bots).
- Remove products seen fewer than 5 times.
- Keep sessions of 3 to 100 events.

**Split.** Sessions are ordered by start time: 80% train, 10% validation, 10% test.

**MDP.**

- **State:** the last 10 products and what the shopper did with each.
- **Action:** a slate of 10 products.
- **Reward:** 0.2, 0.6 or 1.0 if the real next product is in the slate as a view, add-to-cart or purchase.

**Repeat-aware scoring.** About 32% of next products are ones already seen earlier in the session (a view, then the cart, then the purchase). The GRU models therefore score in-session products directly as well as through the full catalogue output, an idea taken from RepeatNet (Ren et al., 2019).

**SQN and SNQN.** The GRU encoder feeds two heads:

- A cross-entropy head for next-item prediction.
- A Q-value head trained by double Q-learning on the event rewards, with a target network refreshed every 1,000 updates.

Both losses train the shared encoder. This follows Xin et al. (2020), *Self-Supervised Reinforcement Learning for Sequential Recommender Systems*, SIGIR.

SNQN adds 10 randomly sampled products per step as zero-reward actions, following Xin et al. (2022). The Q head then also learns which products not to recommend.

At test time each network can rank by its supervised head, by its Q head, or by both combined (log-probability plus beta times Q, with beta tuned on validation).

**Bandits.** The bandits re-rank the top 50 Item-KNN candidates using 12 features: similarity, rank, in-session flags, recency, cart and purchase flags, popularity, conversion rate and category match. They start with no knowledge and learn online across the validation period, seeing feedback only for the products they show. They are then frozen and scored on test. A hybrid variant adds two more features: the trained GRU's probability for each candidate and its rank within the shortlist.

## Results

All numbers are on the test period (the last 10% of sessions, which no model saw during training or tuning). The full table, with every model and metric, is in [results/final_test.md](results/final_test.md).

| Model | HR@10 | NDCG@10 | Cart HR@10 | Purchase HR@10 | Purchase NDCG@10 | Reward per slate |
| --- | --- | --- | --- | --- | --- | --- |
| Popularity | 0.013 | 0.007 | 0.015 | 0.020 | 0.016 | 0.004 |
| Item-KNN | **0.469** | 0.299 | **0.800** | 0.871 | 0.754 | **0.147** |
| GRU (supervised) | 0.337 | 0.265 | 0.674 | 0.830 | 0.740 | 0.115 |
| GRU re-ranking KNN top 50 | 0.443 | 0.312 | 0.769 | **0.873** | 0.767 | 0.141 |
| SQN, supervised head | 0.343 | 0.267 | 0.663 | 0.819 | 0.736 | 0.116 |
| SNQN, supervised head | 0.342 | 0.266 | 0.661 | 0.802 | 0.725 | 0.115 |
| LinUCB bandit | 0.468 | 0.296 | 0.789 | 0.862 | **0.808** | 0.147 |
| LinUCB bandit + GRU score | 0.452 | **0.313** | 0.781 | 0.857 | 0.804 | 0.143 |

Figures: `figures/07_ndcg_all_models.png`, `08_reward_all_models.png`, `09_training_curves.png`, and the bandit learning curves `06_*.png`.

### Statistical reliability

95% confidence intervals come from a paired bootstrap over test sessions (1,000 resamples of the 10,591 sessions; `src/bootstrap_ci.py`). Whole sessions are resampled because steps within a session are not independent, and every model is scored on the same resample, so differences between models get their own intervals. Full tables: [results/bootstrap_ci.md](results/bootstrap_ci.md); charts: `figures/10_ci_ndcg.png`, `11_ci_purchase_ndcg.png`, `12_paired_differences.png`. 80 comparisons are tested (16 model pairs, 5 metrics), so a single difference with p near 0.05 should be read with caution.

### Findings

1. **Item-KNN is a strong baseline.** About 32% of next events involve a product already seen in the session, and most purchases are of a product already in the cart. A model that can recommend in-session products is essential: the same Item-KNN restricted to new products scores 0.107 NDCG@10 instead of 0.299.
2. **The contextual bandits learn a competitive policy from scratch, and rank purchases clearly better.** Starting with no knowledge and seeing feedback only for what they show, they come within 1% of a hand-tuned Item-KNN on NDCG@10 and reward. That gap is small but statistically real (LinUCB: NDCG@10 −0.003, 95% CI −0.005 to −0.002). On purchases they are clearly better: purchase NDCG@10 0.808 vs 0.754 (+0.054, 95% CI +0.045 to +0.064). The learned weights put most value on recency and on whether the product was already carted.
3. **Adding the GRU's score to the bandit gives the best overall NDCG@10 (0.313).** It beats Item-KNN by +0.014 (95% CI +0.012 to +0.017) and keeps the purchase advantage, but its HR@10 and reward are lower than Item-KNN's: it ranks the right product higher when it finds it, but finds it slightly less often.
4. **Deep Q-learning (SQN, SNQN) did not improve on supervised learning in this offline setting.** Against the plain GRU, their supervised heads gain about 0.005 HR@10 (significant) but show no clear gain in reward, and SNQN ranks purchases worse (purchase NDCG@10 −0.015, 95% CI −0.025 to −0.005). The Q heads on their own could not rank the catalogue. Negative sampling (SNQN) roughly doubled the Q head's accuracy when re-ranking the shortlist, but that is still far below the supervised head. The likely cause is that offline replay only ever rewards the logged action, so the Q head has little signal about the other 30,000 products.
5. **Overall:** in offline replay, the RL approach that worked was the bandit re-ranker, which learns online, needs no hand-tuning, and favours purchases. Deep Q-learning needs either a richer simulator or live interaction to show its value.

### Ablations

Each ablation changes one design choice and compares against the standard model on the test period, with paired bootstrap 95% confidence intervals. Training rewards change between variants, but every model is scored with the standard rewards. Full tables: [results/ablations/ablations.md](results/ablations/ablations.md); charts: `figures/13_ablations_design.png`, `figures/14_ablations_sqn.png`.

| Ablation | Change | NDCG@10 | Purchase NDCG@10 | Verdict |
| --- | --- | --- | --- | --- |
| Products already seen | Item-KNN may not recommend them | 0.299 → 0.106 | 0.754 → 0.006 | Essential |
| Repeat-aware term | GRU trained without it | 0.265 → 0.185 | 0.740 → 0.363 | Essential |
| Discount factor | SQN with γ = 0 instead of 0.5 | 0.263 → 0.268 | 0.729 → 0.741 | γ = 0 slightly better |
| Discount factor | SQN with γ = 0.9 instead of 0.5 | 0.263 → 0.254 | 0.729 → 0.718 | Worse |
| Reward weights | SQN with equal rewards | 0.263 → 0.260 | 0.729 → 0.717 | Slightly worse |
| Reward weights | LinUCB with equal rewards | 0.296 → 0.302 | 0.808 → 0.805 (no clear difference) | Slightly better overall |
| Reward weights | LinUCB with steeper rewards (0.1, 0.5, 2.0) | 0.296 → 0.219 | 0.808 → 0.802 | Much worse |
| Reward weights | LinUCB rewarded for purchases only | 0.296 → 0.170 | 0.808 → 0.814 (no clear difference) | Much worse |

SQN rows use the combined ranking (supervised head plus Q-values); the supervised-head results point the same way.

What the ablations show:

1. **Letting models recommend products already in the session is the single most important design choice.** Without it, purchase ranking almost disappears, because most purchases are of a product the shopper has already carted.
2. **Looking further ahead does not help in offline replay.** SQN with γ = 0, which values only the immediate reward, slightly beats γ = 0.5, and γ = 0.9 is worse. Because replayed transitions do not depend on what the agent shows, the estimated future value adds noise rather than information.
3. **Weighting the reward toward purchases does not improve purchase ranking.** Equal rewards do as well or slightly better, and steeper or purchase-only rewards are much worse overall, without a clear purchase gain: they make the learning signal sparser. The bandits' advantage on purchases comes from what they learn about features such as "already carted" and recency, not from the reward weights.

## Limitations

- Offline replay: rewards come from logged behaviour, so the environment cannot show how a shopper would have reacted to a product they were never shown. This applies to all offline RL recommender studies, and is the main reason to back the metrics with the user survey.
- Product names are hashed in the source data, so the demo shows IDs and categories.
- One dataset from a single retailer in 2015.

## References

- Ren, P. et al. (2019) RepeatNet: A repeat aware neural recommendation machine for session-based recommendation. *AAAI*.
- Hidasi, B. et al. (2016) Session-based recommendations with recurrent neural networks. *ICLR*.
- Li, L. et al. (2010) A contextual-bandit approach to personalized news article recommendation. *WWW*.
- Xin, X. et al. (2020) Self-supervised reinforcement learning for sequential recommender systems. *SIGIR*.
- Xin, X. et al. (2022) Supervised advantage actor-critic for recommender systems. *WSDM*.
- RetailRocket (2016) Retailrocket recommender system dataset. Kaggle.
