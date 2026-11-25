#!/usr/bin/env bash
# Full pipeline from raw data to final results. Training takes about 5 hours on a laptop CPU.
set -e
cd "$(dirname "$0")/src"
python download_data.py
python data.py
python eda.py
python steps.py
python run_baselines.py
python run_bandit.py
python -u train_sqn.py --mode gru --epochs 10
python -u train_sqn.py --mode sqn --gamma 0.5 --epochs 6
python -u train_sqn.py --mode sqn --gamma 0.5 --neg 10 --epochs 6
python run_bandit.py --hybrid
python compare.py
python bootstrap_ci.py
python -u train_sqn.py --mode gru --no_repeat --epochs 10 --ablation --tag gru_norepeat
python -u train_sqn.py --mode sqn --gamma 0.0 --epochs 6 --ablation
python -u train_sqn.py --mode sqn --gamma 0.9 --epochs 6 --ablation
python -u train_sqn.py --mode sqn --gamma 0.5 --rewards 1,1,1 --epochs 6 --ablation --tag sqn_equal_rewards
python ablations.py bandit
python ablations.py evaluate
