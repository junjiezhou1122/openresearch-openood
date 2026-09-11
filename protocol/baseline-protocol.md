# FML-bench-Lite OpenOOD Baseline Protocol (frozen)

Protocol ID: `fml-lite-openood-baseline-v1`
Frozen at: 2026-09-11
Status: frozen BEFORE any baseline measurement was observed.

## 1. Upstream provenance

| Item | Value |
|---|---|
| Upstream repository | https://github.com/Jingkang50/OpenOOD.git |
| Upstream snapshot commit | `3c35632ee91b54b09d1f085d04f94744cece7d0b` |
| FML-bench source | https://github.com/qrzou/FML-bench.git |
| FML-bench commit | `d336651ebea50c622c256f02ded82b68b4451fdc` |
| FML task | `ml_tasks/Robustness_openood` |
| FML adapter files | `algorithm.py`, `split_config.json` |
| FML evaluator script | `train_eval_baseline.py`, sha256 `daa13119eb4e1e5aff6b86b5a1aac2c12d21b246b842a66d7bfa13cd2fb7b92c` |

## 2. Data

| Item | Value |
|---|---|
| In-Distribution (ID) | CIFAR-10 (50K train + 10K test) |
| Validation OOD | CIFAR-100 test set (10K) |
| Test OOD | SVHN test set (5K sampled) |
| Split | 53K visible pool (50K train + 3K test), 7K hidden ID test |
| Default split_config | `val_ratio=0.057`, `val_seed=42` |

## 3. Baseline algorithm & model

- Model: ResNet-18 (standard CIFAR variant)
- Method: MSP (Maximum Softmax Probability)
- Training: 100 epochs, batch size 128, lr 0.1 (decay at 50, 75, 90 by 0.1)

## 4. Metrics

- Primary metric: `auroc_mean` (AUROC distinguishing ID from OOD, higher is better).
- Secondary metric: `fpr95_mean` (FPR at 95% TPR, lower is better).
- Reference FML baseline: AUROC ~0.8758, FPR95 ~0.6049 on val.

## 5. Worker edit surface

- `algorithm.py`
- `split_config.json`
