# Bootstrap 95% confidence intervals (test period)

Paired bootstrap over test sessions, 1000 resamples.

## Models

| Model | HR@10 | NDCG@10 | Reward@10 | Purchase HR@10 | Purchase NDCG@10 |
| --- | --- | --- | --- | --- | --- |
| Popularity | 0.012 (0.011 to 0.014) | 0.007 (0.006 to 0.008) | 0.004 (0.003 to 0.004) | 0.020 (0.013 to 0.027) | 0.016 (0.010 to 0.023) |
| Item-KNN | 0.469 (0.461 to 0.476) | 0.299 (0.294 to 0.304) | 0.147 (0.144 to 0.151) | 0.870 (0.828 to 0.907) | 0.754 (0.713 to 0.792) |
| Bandit: LinUCB | 0.468 (0.460 to 0.475) | 0.296 (0.290 to 0.301) | 0.146 (0.143 to 0.150) | 0.862 (0.824 to 0.899) | 0.808 (0.766 to 0.847) |
| Bandit: Thompson sampling | 0.463 (0.455 to 0.470) | 0.293 (0.288 to 0.298) | 0.145 (0.142 to 0.149) | 0.859 (0.820 to 0.895) | 0.807 (0.764 to 0.846) |
| Bandit: Epsilon-greedy | 0.467 (0.459 to 0.474) | 0.296 (0.290 to 0.301) | 0.146 (0.143 to 0.150) | 0.862 (0.824 to 0.898) | 0.808 (0.766 to 0.847) |
| Bandit + GRU: LinUCB | 0.452 (0.444 to 0.460) | 0.313 (0.307 to 0.319) | 0.143 (0.140 to 0.147) | 0.857 (0.816 to 0.895) | 0.804 (0.761 to 0.842) |
| Bandit + GRU: Thompson sampling | 0.437 (0.430 to 0.445) | 0.306 (0.300 to 0.312) | 0.139 (0.136 to 0.143) | 0.841 (0.800 to 0.879) | 0.795 (0.752 to 0.835) |
| Bandit + GRU: Epsilon-greedy | 0.444 (0.437 to 0.452) | 0.310 (0.304 to 0.316) | 0.141 (0.138 to 0.145) | 0.853 (0.814 to 0.889) | 0.803 (0.761 to 0.841) |
| GRU (supervised) | 0.337 (0.330 to 0.344) | 0.265 (0.259 to 0.271) | 0.115 (0.112 to 0.119) | 0.830 (0.788 to 0.869) | 0.740 (0.700 to 0.777) |
| GRU re-ranking KNN top 50 | 0.443 (0.435 to 0.450) | 0.312 (0.306 to 0.318) | 0.141 (0.138 to 0.145) | 0.873 (0.832 to 0.908) | 0.767 (0.726 to 0.802) |
| SQN (supervised head) | 0.343 (0.335 to 0.350) | 0.267 (0.261 to 0.272) | 0.116 (0.113 to 0.119) | 0.819 (0.778 to 0.858) | 0.736 (0.695 to 0.773) |
| SQN (combined) | 0.338 (0.331 to 0.346) | 0.263 (0.257 to 0.269) | 0.114 (0.111 to 0.117) | 0.806 (0.765 to 0.843) | 0.729 (0.689 to 0.766) |
| SNQN (supervised head) | 0.342 (0.334 to 0.349) | 0.266 (0.260 to 0.272) | 0.115 (0.112 to 0.118) | 0.802 (0.758 to 0.839) | 0.725 (0.683 to 0.762) |
| SNQN (combined) | 0.343 (0.335 to 0.350) | 0.266 (0.260 to 0.272) | 0.115 (0.112 to 0.118) | 0.797 (0.753 to 0.834) | 0.723 (0.681 to 0.761) |

## Paired differences (model minus reference)

| Hypothesis | Model | Reference | Metric | Difference (95% CI) | p | Verdict |
| --- | --- | --- | --- | --- | --- | --- |
| H1 | Bandit: LinUCB | Item-KNN | HR@10 | -0.0013 (-0.0046 to +0.0021) | 0.440 | no clear difference |
| H1 | Bandit: LinUCB | Item-KNN | NDCG@10 | -0.0032 (-0.0048 to -0.0017) | <0.001 | lower |
| H1 | Bandit: LinUCB | Item-KNN | Reward@10 | -0.0009 (-0.0018 to -0.0001) | 0.040 | lower |
| H1 | Bandit: LinUCB | Item-KNN | Purchase HR@10 | -0.0082 (-0.0193 to +0.0027) | 0.168 | no clear difference |
| H1 | Bandit: LinUCB | Item-KNN | Purchase NDCG@10 | +0.0538 (+0.0446 to +0.0636) | <0.001 | higher |
| H1 | Bandit: Thompson sampling | Item-KNN | HR@10 | -0.0062 (-0.0096 to -0.0027) | <0.001 | lower |
| H1 | Bandit: Thompson sampling | Item-KNN | NDCG@10 | -0.0059 (-0.0076 to -0.0042) | <0.001 | lower |
| H1 | Bandit: Thompson sampling | Item-KNN | Reward@10 | -0.0021 (-0.0030 to -0.0012) | <0.001 | lower |
| H1 | Bandit: Thompson sampling | Item-KNN | Purchase HR@10 | -0.0116 (-0.0229 to -0.0013) | 0.030 | lower |
| H1 | Bandit: Thompson sampling | Item-KNN | Purchase NDCG@10 | +0.0523 (+0.0439 to +0.0613) | <0.001 | higher |
| H1 | Bandit: Epsilon-greedy | Item-KNN | HR@10 | -0.0017 (-0.0052 to +0.0017) | 0.318 | no clear difference |
| H1 | Bandit: Epsilon-greedy | Item-KNN | NDCG@10 | -0.0032 (-0.0049 to -0.0017) | 0.002 | lower |
| H1 | Bandit: Epsilon-greedy | Item-KNN | Reward@10 | -0.0010 (-0.0019 to -0.0001) | 0.026 | lower |
| H1 | Bandit: Epsilon-greedy | Item-KNN | Purchase HR@10 | -0.0089 (-0.0200 to +0.0025) | 0.128 | no clear difference |
| H1 | Bandit: Epsilon-greedy | Item-KNN | Purchase NDCG@10 | +0.0542 (+0.0452 to +0.0639) | <0.001 | higher |
| H1 | Bandit + GRU: LinUCB | Item-KNN | HR@10 | -0.0168 (-0.0208 to -0.0127) | <0.001 | lower |
| H1 | Bandit + GRU: LinUCB | Item-KNN | NDCG@10 | +0.0144 (+0.0120 to +0.0168) | <0.001 | higher |
| H1 | Bandit + GRU: LinUCB | Item-KNN | Reward@10 | -0.0045 (-0.0055 to -0.0035) | <0.001 | lower |
| H1 | Bandit + GRU: LinUCB | Item-KNN | Purchase HR@10 | -0.0136 (-0.0259 to -0.0014) | 0.034 | lower |
| H1 | Bandit + GRU: LinUCB | Item-KNN | Purchase NDCG@10 | +0.0493 (+0.0401 to +0.0589) | <0.001 | higher |
| H1 | Bandit + GRU: Thompson sampling | Item-KNN | HR@10 | -0.0315 (-0.0358 to -0.0271) | <0.001 | lower |
| H1 | Bandit + GRU: Thompson sampling | Item-KNN | NDCG@10 | +0.0070 (+0.0043 to +0.0094) | <0.001 | higher |
| H1 | Bandit + GRU: Thompson sampling | Item-KNN | Reward@10 | -0.0080 (-0.0091 to -0.0070) | <0.001 | lower |
| H1 | Bandit + GRU: Thompson sampling | Item-KNN | Purchase HR@10 | -0.0293 (-0.0409 to -0.0179) | <0.001 | lower |
| H1 | Bandit + GRU: Thompson sampling | Item-KNN | Purchase NDCG@10 | +0.0411 (+0.0323 to +0.0496) | <0.001 | higher |
| H1 | Bandit + GRU: Epsilon-greedy | Item-KNN | HR@10 | -0.0247 (-0.0291 to -0.0204) | <0.001 | lower |
| H1 | Bandit + GRU: Epsilon-greedy | Item-KNN | NDCG@10 | +0.0113 (+0.0087 to +0.0137) | <0.001 | higher |
| H1 | Bandit + GRU: Epsilon-greedy | Item-KNN | Reward@10 | -0.0063 (-0.0073 to -0.0052) | <0.001 | lower |
| H1 | Bandit + GRU: Epsilon-greedy | Item-KNN | Purchase HR@10 | -0.0177 (-0.0297 to -0.0053) | 0.012 | lower |
| H1 | Bandit + GRU: Epsilon-greedy | Item-KNN | Purchase NDCG@10 | +0.0483 (+0.0397 to +0.0578) | <0.001 | higher |
| H2 | SQN (supervised head) | GRU (supervised) | HR@10 | +0.0053 (+0.0032 to +0.0074) | <0.001 | higher |
| H2 | SQN (supervised head) | GRU (supervised) | NDCG@10 | +0.0016 (+0.0001 to +0.0031) | 0.044 | higher |
| H2 | SQN (supervised head) | GRU (supervised) | Reward@10 | +0.0003 (-0.0004 to +0.0011) | 0.346 | no clear difference |
| H2 | SQN (supervised head) | GRU (supervised) | Purchase HR@10 | -0.0109 (-0.0221 to +0.0013) | 0.084 | no clear difference |
| H2 | SQN (supervised head) | GRU (supervised) | Purchase NDCG@10 | -0.0042 (-0.0133 to +0.0055) | 0.402 | no clear difference |
| H2 | SQN (combined) | GRU (supervised) | HR@10 | +0.0010 (-0.0012 to +0.0031) | 0.348 | no clear difference |
| H2 | SQN (combined) | GRU (supervised) | NDCG@10 | -0.0018 (-0.0034 to -0.0003) | 0.022 | lower |
| H2 | SQN (combined) | GRU (supervised) | Reward@10 | -0.0013 (-0.0022 to -0.0005) | 0.002 | lower |
| H2 | SQN (combined) | GRU (supervised) | Purchase HR@10 | -0.0239 (-0.0385 to -0.0105) | <0.001 | lower |
| H2 | SQN (combined) | GRU (supervised) | Purchase NDCG@10 | -0.0110 (-0.0222 to +0.0001) | 0.054 | no clear difference |
| H2 | SNQN (supervised head) | GRU (supervised) | HR@10 | +0.0047 (+0.0024 to +0.0067) | <0.001 | higher |
| H2 | SNQN (supervised head) | GRU (supervised) | NDCG@10 | +0.0011 (-0.0004 to +0.0025) | 0.170 | no clear difference |
| H2 | SNQN (supervised head) | GRU (supervised) | Reward@10 | -0.0004 (-0.0012 to +0.0004) | 0.398 | no clear difference |
| H2 | SNQN (supervised head) | GRU (supervised) | Purchase HR@10 | -0.0286 (-0.0416 to -0.0153) | <0.001 | lower |
| H2 | SNQN (supervised head) | GRU (supervised) | Purchase NDCG@10 | -0.0151 (-0.0254 to -0.0052) | 0.004 | lower |
| H2 | SNQN (combined) | GRU (supervised) | HR@10 | +0.0053 (+0.0031 to +0.0074) | <0.001 | higher |
| H2 | SNQN (combined) | GRU (supervised) | NDCG@10 | +0.0010 (-0.0006 to +0.0024) | 0.226 | no clear difference |
| H2 | SNQN (combined) | GRU (supervised) | Reward@10 | -0.0002 (-0.0010 to +0.0006) | 0.664 | no clear difference |
| H2 | SNQN (combined) | GRU (supervised) | Purchase HR@10 | -0.0334 (-0.0475 to -0.0198) | <0.001 | lower |
| H2 | SNQN (combined) | GRU (supervised) | Purchase NDCG@10 | -0.0166 (-0.0273 to -0.0058) | 0.002 | lower |
| Other | GRU re-ranking KNN top 50 | Item-KNN | HR@10 | -0.0263 (-0.0294 to -0.0231) | <0.001 | lower |
| Other | GRU re-ranking KNN top 50 | Item-KNN | NDCG@10 | +0.0129 (+0.0105 to +0.0155) | <0.001 | higher |
| Other | GRU re-ranking KNN top 50 | Item-KNN | Reward@10 | -0.0063 (-0.0071 to -0.0054) | <0.001 | lower |
| Other | GRU re-ranking KNN top 50 | Item-KNN | Purchase HR@10 | +0.0027 (-0.0058 to +0.0121) | 0.628 | no clear difference |
| Other | GRU re-ranking KNN top 50 | Item-KNN | Purchase NDCG@10 | +0.0126 (+0.0012 to +0.0232) | 0.024 | higher |
| Other | Bandit + GRU: LinUCB | GRU re-ranking KNN top 50 | HR@10 | +0.0095 (+0.0058 to +0.0131) | <0.001 | higher |
| Other | Bandit + GRU: LinUCB | GRU re-ranking KNN top 50 | NDCG@10 | +0.0015 (-0.0007 to +0.0036) | 0.192 | no clear difference |
| Other | Bandit + GRU: LinUCB | GRU re-ranking KNN top 50 | Reward@10 | +0.0018 (+0.0009 to +0.0029) | 0.002 | higher |
| Other | Bandit + GRU: LinUCB | GRU re-ranking KNN top 50 | Purchase HR@10 | -0.0164 (-0.0287 to -0.0051) | <0.001 | lower |
| Other | Bandit + GRU: LinUCB | GRU re-ranking KNN top 50 | Purchase NDCG@10 | +0.0367 (+0.0271 to +0.0463) | <0.001 | higher |
| H3 | Item-KNN | Popularity | HR@10 | +0.4564 (+0.4484 to +0.4638) | <0.001 | higher |
| H3 | Item-KNN | Popularity | NDCG@10 | +0.2920 (+0.2867 to +0.2970) | <0.001 | higher |
| H3 | Item-KNN | Popularity | Reward@10 | +0.1438 (+0.1407 to +0.1474) | <0.001 | higher |
| H3 | Item-KNN | Popularity | Purchase HR@10 | +0.8507 (+0.8095 to +0.8876) | <0.001 | higher |
| H3 | Item-KNN | Popularity | Purchase NDCG@10 | +0.7382 (+0.6966 to +0.7767) | <0.001 | higher |
| H3 | GRU (supervised) | Popularity | HR@10 | +0.3248 (+0.3180 to +0.3315) | <0.001 | higher |
| H3 | GRU (supervised) | Popularity | NDCG@10 | +0.2582 (+0.2525 to +0.2636) | <0.001 | higher |
| H3 | GRU (supervised) | Popularity | Reward@10 | +0.1118 (+0.1089 to +0.1152) | <0.001 | higher |
| H3 | GRU (supervised) | Popularity | Purchase HR@10 | +0.8105 (+0.7680 to +0.8489) | <0.001 | higher |
| H3 | GRU (supervised) | Popularity | Purchase NDCG@10 | +0.7237 (+0.6839 to +0.7602) | <0.001 | higher |
| H3 | Bandit: LinUCB | Popularity | HR@10 | +0.4550 (+0.4476 to +0.4625) | <0.001 | higher |
| H3 | Bandit: LinUCB | Popularity | NDCG@10 | +0.2888 (+0.2832 to +0.2940) | <0.001 | higher |
| H3 | Bandit: LinUCB | Popularity | Reward@10 | +0.1429 (+0.1398 to +0.1465) | <0.001 | higher |
| H3 | Bandit: LinUCB | Popularity | Purchase HR@10 | +0.8425 (+0.8036 to +0.8796) | <0.001 | higher |
| H3 | Bandit: LinUCB | Popularity | Purchase NDCG@10 | +0.7920 (+0.7493 to +0.8319) | <0.001 | higher |
| H3 | SNQN (combined) | Popularity | HR@10 | +0.3301 (+0.3231 to +0.3373) | <0.001 | higher |
| H3 | SNQN (combined) | Popularity | NDCG@10 | +0.2591 (+0.2532 to +0.2647) | <0.001 | higher |
| H3 | SNQN (combined) | Popularity | Reward@10 | +0.1117 (+0.1087 to +0.1147) | <0.001 | higher |
| H3 | SNQN (combined) | Popularity | Purchase HR@10 | +0.7771 (+0.7337 to +0.8151) | <0.001 | higher |
| H3 | SNQN (combined) | Popularity | Purchase NDCG@10 | +0.7070 (+0.6673 to +0.7453) | <0.001 | higher |
