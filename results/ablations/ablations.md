# Ablation results (test period)

95% confidence intervals from a paired bootstrap over test sessions, 1000 resamples. All models are scored with the standard rewards (0.2, 0.6, 1.0).

## Models

| Model | HR@10 | NDCG@10 | Reward@10 | Purchase HR@10 | Purchase NDCG@10 |
| --- | --- | --- | --- | --- | --- |
| Item-KNN | 0.469 (0.461 to 0.476) | 0.299 (0.294 to 0.304) | 0.147 (0.144 to 0.151) | 0.870 (0.828 to 0.907) | 0.754 (0.713 to 0.792) |
| Item-KNN, new products only | 0.182 (0.176 to 0.187) | 0.106 (0.102 to 0.109) | 0.038 (0.037 to 0.039) | 0.010 (0.004 to 0.016) | 0.006 (0.002 to 0.010) |
| GRU | 0.337 (0.330 to 0.344) | 0.265 (0.259 to 0.271) | 0.115 (0.112 to 0.119) | 0.830 (0.788 to 0.869) | 0.740 (0.700 to 0.777) |
| GRU without repeat term | 0.287 (0.280 to 0.295) | 0.185 (0.179 to 0.190) | 0.085 (0.083 to 0.088) | 0.450 (0.415 to 0.486) | 0.363 (0.332 to 0.394) |
| SQN γ=0.5 (supervised head) | 0.343 (0.335 to 0.350) | 0.267 (0.261 to 0.272) | 0.116 (0.113 to 0.119) | 0.819 (0.778 to 0.858) | 0.736 (0.695 to 0.773) |
| SQN γ=0.5 (combined) | 0.338 (0.331 to 0.346) | 0.263 (0.257 to 0.269) | 0.114 (0.111 to 0.117) | 0.806 (0.765 to 0.843) | 0.729 (0.689 to 0.766) |
| SQN γ=0 (supervised head) | 0.345 (0.338 to 0.352) | 0.268 (0.262 to 0.274) | 0.116 (0.113 to 0.120) | 0.826 (0.784 to 0.864) | 0.741 (0.702 to 0.777) |
| SQN γ=0 (combined) | 0.345 (0.337 to 0.352) | 0.268 (0.262 to 0.274) | 0.116 (0.113 to 0.120) | 0.819 (0.777 to 0.857) | 0.741 (0.701 to 0.777) |
| SQN γ=0.9 (supervised head) | 0.327 (0.319 to 0.334) | 0.260 (0.254 to 0.266) | 0.112 (0.109 to 0.115) | 0.817 (0.774 to 0.854) | 0.735 (0.694 to 0.772) |
| SQN γ=0.9 (combined) | 0.318 (0.311 to 0.324) | 0.254 (0.248 to 0.260) | 0.109 (0.106 to 0.112) | 0.793 (0.751 to 0.831) | 0.718 (0.678 to 0.756) |
| SQN, equal rewards (supervised head) | 0.339 (0.332 to 0.347) | 0.266 (0.260 to 0.272) | 0.116 (0.113 to 0.119) | 0.817 (0.772 to 0.855) | 0.733 (0.692 to 0.771) |
| SQN, equal rewards (combined) | 0.331 (0.324 to 0.338) | 0.260 (0.254 to 0.265) | 0.112 (0.110 to 0.116) | 0.800 (0.757 to 0.838) | 0.716 (0.675 to 0.755) |
| LinUCB, default rewards (0.2, 0.6, 1.0) | 0.468 (0.460 to 0.475) | 0.296 (0.290 to 0.301) | 0.146 (0.143 to 0.150) | 0.862 (0.824 to 0.899) | 0.808 (0.766 to 0.847) |
| LinUCB, equal rewards (1, 1, 1) | 0.482 (0.474 to 0.489) | 0.302 (0.297 to 0.307) | 0.150 (0.147 to 0.154) | 0.871 (0.828 to 0.910) | 0.805 (0.762 to 0.844) |
| LinUCB, steeper rewards (0.1, 0.5, 2.0) | 0.339 (0.333 to 0.346) | 0.219 (0.214 to 0.224) | 0.119 (0.116 to 0.122) | 0.845 (0.805 to 0.882) | 0.802 (0.759 to 0.843) |
| LinUCB, purchases only (0, 0, 1) | 0.225 (0.219 to 0.231) | 0.170 (0.165 to 0.175) | 0.096 (0.092 to 0.099) | 0.856 (0.817 to 0.892) | 0.814 (0.773 to 0.853) |

## Paired differences (variant minus reference)

| Ablation | Variant | Reference | Metric | Difference (95% CI) | p | Verdict |
| --- | --- | --- | --- | --- | --- | --- |
| Seen products | Item-KNN, new products only | Item-KNN | HR@10 | -0.2872 (-0.2940 to -0.2803) | <0.001 | lower |
| Seen products | Item-KNN, new products only | Item-KNN | NDCG@10 | -0.1935 (-0.1991 to -0.1882) | <0.001 | lower |
| Seen products | Item-KNN, new products only | Item-KNN | Reward@10 | -0.1092 (-0.1128 to -0.1058) | <0.001 | lower |
| Seen products | Item-KNN, new products only | Item-KNN | Purchase HR@10 | -0.8609 (-0.8994 to -0.8166) | <0.001 | lower |
| Seen products | Item-KNN, new products only | Item-KNN | Purchase NDCG@10 | -0.7484 (-0.7886 to -0.7050) | <0.001 | lower |
| Repeat term | GRU without repeat term | GRU | HR@10 | -0.0500 (-0.0556 to -0.0441) | <0.001 | lower |
| Repeat term | GRU without repeat term | GRU | NDCG@10 | -0.0804 (-0.0846 to -0.0762) | <0.001 | lower |
| Repeat term | GRU without repeat term | GRU | Reward@10 | -0.0301 (-0.0325 to -0.0277) | <0.001 | lower |
| Repeat term | GRU without repeat term | GRU | Purchase HR@10 | -0.3804 (-0.4109 to -0.3502) | <0.001 | lower |
| Repeat term | GRU without repeat term | GRU | Purchase NDCG@10 | -0.3772 (-0.4065 to -0.3510) | <0.001 | lower |
| Discount factor | SQN γ=0 (supervised head) | SQN γ=0.5 (supervised head) | HR@10 | +0.0025 (+0.0010 to +0.0041) | 0.002 | higher |
| Discount factor | SQN γ=0 (supervised head) | SQN γ=0.5 (supervised head) | NDCG@10 | +0.0015 (+0.0005 to +0.0024) | <0.001 | higher |
| Discount factor | SQN γ=0 (supervised head) | SQN γ=0.5 (supervised head) | Reward@10 | +0.0007 (+0.0001 to +0.0013) | 0.024 | higher |
| Discount factor | SQN γ=0 (supervised head) | SQN γ=0.5 (supervised head) | Purchase HR@10 | +0.0068 (-0.0035 to +0.0174) | 0.236 | no clear difference |
| Discount factor | SQN γ=0 (supervised head) | SQN γ=0.5 (supervised head) | Purchase NDCG@10 | +0.0059 (-0.0008 to +0.0127) | 0.092 | no clear difference |
| Discount factor | SQN γ=0.9 (supervised head) | SQN γ=0.5 (supervised head) | HR@10 | -0.0157 (-0.0178 to -0.0138) | <0.001 | lower |
| Discount factor | SQN γ=0.9 (supervised head) | SQN γ=0.5 (supervised head) | NDCG@10 | -0.0065 (-0.0077 to -0.0053) | <0.001 | lower |
| Discount factor | SQN γ=0.9 (supervised head) | SQN γ=0.5 (supervised head) | Reward@10 | -0.0036 (-0.0043 to -0.0029) | <0.001 | lower |
| Discount factor | SQN γ=0.9 (supervised head) | SQN γ=0.5 (supervised head) | Purchase HR@10 | -0.0027 (-0.0141 to +0.0074) | 0.662 | no clear difference |
| Discount factor | SQN γ=0.9 (supervised head) | SQN γ=0.5 (supervised head) | Purchase NDCG@10 | -0.0005 (-0.0080 to +0.0068) | 0.938 | no clear difference |
| Discount factor | SQN γ=0 (combined) | SQN γ=0.5 (combined) | HR@10 | +0.0064 (+0.0046 to +0.0082) | <0.001 | higher |
| Discount factor | SQN γ=0 (combined) | SQN γ=0.5 (combined) | NDCG@10 | +0.0045 (+0.0034 to +0.0056) | <0.001 | higher |
| Discount factor | SQN γ=0 (combined) | SQN γ=0.5 (combined) | Reward@10 | +0.0023 (+0.0016 to +0.0029) | <0.001 | higher |
| Discount factor | SQN γ=0 (combined) | SQN γ=0.5 (combined) | Purchase HR@10 | +0.0130 (+0.0028 to +0.0249) | 0.010 | higher |
| Discount factor | SQN γ=0 (combined) | SQN γ=0.5 (combined) | Purchase NDCG@10 | +0.0119 (+0.0044 to +0.0192) | 0.002 | higher |
| Discount factor | SQN γ=0.9 (combined) | SQN γ=0.5 (combined) | HR@10 | -0.0203 (-0.0225 to -0.0182) | <0.001 | lower |
| Discount factor | SQN γ=0.9 (combined) | SQN γ=0.5 (combined) | NDCG@10 | -0.0091 (-0.0104 to -0.0077) | <0.001 | lower |
| Discount factor | SQN γ=0.9 (combined) | SQN γ=0.5 (combined) | Reward@10 | -0.0050 (-0.0058 to -0.0042) | <0.001 | lower |
| Discount factor | SQN γ=0.9 (combined) | SQN γ=0.5 (combined) | Purchase HR@10 | -0.0130 (-0.0244 to -0.0021) | 0.030 | lower |
| Discount factor | SQN γ=0.9 (combined) | SQN γ=0.5 (combined) | Purchase NDCG@10 | -0.0111 (-0.0193 to -0.0023) | 0.020 | lower |
| Reward weights | SQN, equal rewards (supervised head) | SQN γ=0.5 (supervised head) | HR@10 | -0.0033 (-0.0053 to -0.0012) | 0.002 | lower |
| Reward weights | SQN, equal rewards (supervised head) | SQN γ=0.5 (supervised head) | NDCG@10 | -0.0009 (-0.0023 to +0.0004) | 0.180 | no clear difference |
| Reward weights | SQN, equal rewards (supervised head) | SQN γ=0.5 (supervised head) | Reward@10 | -0.0001 (-0.0009 to +0.0006) | 0.756 | no clear difference |
| Reward weights | SQN, equal rewards (supervised head) | SQN γ=0.5 (supervised head) | Purchase HR@10 | -0.0020 (-0.0149 to +0.0100) | 0.736 | no clear difference |
| Reward weights | SQN, equal rewards (supervised head) | SQN γ=0.5 (supervised head) | Purchase NDCG@10 | -0.0023 (-0.0126 to +0.0079) | 0.666 | no clear difference |
| Reward weights | SQN, equal rewards (combined) | SQN γ=0.5 (combined) | HR@10 | -0.0074 (-0.0093 to -0.0052) | <0.001 | lower |
| Reward weights | SQN, equal rewards (combined) | SQN γ=0.5 (combined) | NDCG@10 | -0.0038 (-0.0052 to -0.0024) | <0.001 | lower |
| Reward weights | SQN, equal rewards (combined) | SQN γ=0.5 (combined) | Reward@10 | -0.0017 (-0.0024 to -0.0009) | <0.001 | lower |
| Reward weights | SQN, equal rewards (combined) | SQN γ=0.5 (combined) | Purchase HR@10 | -0.0061 (-0.0187 to +0.0071) | 0.364 | no clear difference |
| Reward weights | SQN, equal rewards (combined) | SQN γ=0.5 (combined) | Purchase NDCG@10 | -0.0122 (-0.0227 to -0.0012) | 0.026 | lower |
| Reward weights | LinUCB, equal rewards (1, 1, 1) | LinUCB, default rewards (0.2, 0.6, 1.0) | HR@10 | +0.0141 (+0.0117 to +0.0164) | <0.001 | higher |
| Reward weights | LinUCB, equal rewards (1, 1, 1) | LinUCB, default rewards (0.2, 0.6, 1.0) | NDCG@10 | +0.0065 (+0.0055 to +0.0074) | <0.001 | higher |
| Reward weights | LinUCB, equal rewards (1, 1, 1) | LinUCB, default rewards (0.2, 0.6, 1.0) | Reward@10 | +0.0035 (+0.0029 to +0.0040) | <0.001 | higher |
| Reward weights | LinUCB, equal rewards (1, 1, 1) | LinUCB, default rewards (0.2, 0.6, 1.0) | Purchase HR@10 | +0.0089 (+0.0000 to +0.0173) | 0.058 | no clear difference |
| Reward weights | LinUCB, equal rewards (1, 1, 1) | LinUCB, default rewards (0.2, 0.6, 1.0) | Purchase NDCG@10 | -0.0032 (-0.0071 to +0.0007) | 0.104 | no clear difference |
| Reward weights | LinUCB, steeper rewards (0.1, 0.5, 2.0) | LinUCB, default rewards (0.2, 0.6, 1.0) | HR@10 | -0.1284 (-0.1330 to -0.1243) | <0.001 | lower |
| Reward weights | LinUCB, steeper rewards (0.1, 0.5, 2.0) | LinUCB, default rewards (0.2, 0.6, 1.0) | NDCG@10 | -0.0766 (-0.0793 to -0.0740) | <0.001 | lower |
| Reward weights | LinUCB, steeper rewards (0.1, 0.5, 2.0) | LinUCB, default rewards (0.2, 0.6, 1.0) | Reward@10 | -0.0274 (-0.0284 to -0.0264) | <0.001 | lower |
| Reward weights | LinUCB, steeper rewards (0.1, 0.5, 2.0) | LinUCB, default rewards (0.2, 0.6, 1.0) | Purchase HR@10 | -0.0177 (-0.0256 to -0.0097) | <0.001 | lower |
| Reward weights | LinUCB, steeper rewards (0.1, 0.5, 2.0) | LinUCB, default rewards (0.2, 0.6, 1.0) | Purchase NDCG@10 | -0.0062 (-0.0109 to -0.0017) | 0.002 | lower |
| Reward weights | LinUCB, purchases only (0, 0, 1) | LinUCB, default rewards (0.2, 0.6, 1.0) | HR@10 | -0.2423 (-0.2486 to -0.2363) | <0.001 | lower |
| Reward weights | LinUCB, purchases only (0, 0, 1) | LinUCB, default rewards (0.2, 0.6, 1.0) | NDCG@10 | -0.1258 (-0.1294 to -0.1226) | <0.001 | lower |
| Reward weights | LinUCB, purchases only (0, 0, 1) | LinUCB, default rewards (0.2, 0.6, 1.0) | Reward@10 | -0.0507 (-0.0520 to -0.0494) | <0.001 | lower |
| Reward weights | LinUCB, purchases only (0, 0, 1) | LinUCB, default rewards (0.2, 0.6, 1.0) | Purchase HR@10 | -0.0061 (-0.0142 to +0.0019) | 0.166 | no clear difference |
| Reward weights | LinUCB, purchases only (0, 0, 1) | LinUCB, default rewards (0.2, 0.6, 1.0) | Purchase NDCG@10 | +0.0056 (-0.0001 to +0.0117) | 0.056 | no clear difference |
