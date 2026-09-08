# Robustness / OpenOOD

This is an imported FML-bench-Lite research starter, not a certified baseline.

## Goal

You are working with MSP (Maximum Softmax Probability) as the baseline OOD detection method on CIFAR-10. The model is a ResNet-18 trained on CIFAR-10 for 100 epochs. OOD detection is evaluated by measuring how well the model distinguishes CIFAR-10 (in-distribution) from OOD inputs. The baseline simply uses the maximum softmax probability as the OOD score.

Your goal is to improve the AUROC for OOD detection. You may modify the OOD scoring function (Energy score, ODIN, ReAct, ASH, Mahalanobis distance, KNN, etc.), modify the model architecture, add feature-space methods, modify the training procedure, or propose entirely new detection approaches.

You are evaluated on a validation set during development (CIFAR-10 vs CIFAR-100 as OOD). Your final performance will be measured on a separate test set (CIFAR-10 vs SVHN as OOD).

The algorithm.py must export get_model(num_classes), get_training_config(), and compute_ood_score(model, x) where x is a batch of images and the return is a score tensor where HIGHER = more likely in-distribution.

You may also modify split_config.json to control how visible data is split between training and validation. Fields:
- val_ratio (default: 0.057): fraction of visible data reserved for validation. Lower values mean more training data but noisier validation metrics.
- val_seed (default: 42): random seed for the train/val split.
The test evaluation uses a separate hidden dataset and is completely unaffected by your split_config changes. If you do not modify split_config.json, the default split is used.

## Metrics

- `auroc_mean`: higher
- `fpr95_mean`: lower

## Worker edit surface

- `algorithm.py`
- `split_config.json`

The worker may commit candidate changes, but it may not create acceptance tags or verdicts. The OpenResearch control plane checks out the exact candidate commit on the runner and an independent evaluator owns final acceptance.
