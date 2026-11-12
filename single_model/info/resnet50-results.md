# ResNet50 NIH experiment: results and cost

The original result archive is extracted to [`../output/resnet50_nih_official/`](../output/resnet50_nih_official/). It contains the original checkpoints, prediction arrays, thresholds, figures, and `results.json`. The source notebook is [`resnet50.ipynb`](./resnet50.ipynb).

## Test performance

Metrics below use the archived 25,592 test images and validation-selected thresholds. AUROC and AUPRC are threshold-free; the other thresholded metrics use the saved per-class thresholds.

| Metric | Result |
|---|---:|
| Validation macro AUROC (best) | 0.7745 |
| Test macro AUROC | 0.7281 |
| Test macro AUPRC | 0.1676 |
| Test macro precision / recall / F1 | 0.1634 / 0.4011 / 0.2190 |
| Test micro precision / recall / F1 | 0.2079 / 0.5487 / 0.3015 |
| Hamming accuracy | 0.8070 |
| Exact-match (subset) accuracy | 0.1648 |
| Validation-fitted temperature | 1.0108 |

Per-class test AUROC ranges from 0.6435 (Pneumonia) to 0.8456 (Hernia). Hernia has only 86 positive test examples, so its point estimate is especially uncertain. The archive has no patient-level bootstrap confidence intervals.

## Computational cost

| Measure | Result |
|---|---:|
| Logged training duration | 9,602 s (2 h 40 min 2 s) across 10 epochs |
| Warm-up / fine-tuning time | 4,062 s / 5,540 s |
| Parameters | 23,536,719 (as logged by the notebook) |
| Calibrated checkpoint size | 94,477,343 bytes (about 90.1 MiB) |
| Training setup | CUDA, mixed precision, batch size 32, input 224 x 224 |

The notebook log identifies CUDA but not the GPU model. The archive contains no measured inference latency, throughput, or cloud billing data; therefore no wall-clock inference cost or dollar cost can be reported. A standard ResNet50 at 224 x 224 is roughly 4.1 GFLOPs per image as an architecture-level estimate, not a measurement from this run.

## Protocol and comparison caveats

This is a separate ResNet50 experiment, not a reproduction of the registered M08/M09 protocol:

- The notebook filtered 16 implausible-age records, leaving 78,221 train, 8,291 validation, and 25,592 test images; the registered test split contains 25,596.
- Validation uses a 10% patient fraction rather than the registered 15%.
- The run uses a weighted sampler and `BCEWithLogitsLoss` with capped positive weights, rather than the registered asymmetric loss.
- The archived notebook uses a validation threshold grid and temperature scaling; no test-based threshold selection is indicated.

For context, the registered ConvNeXt-Tiny test macro AUROC is 0.8263 and RAD-DINO is 0.8364, but those values use the registered split and protocol. The ResNet50 value of 0.7281 is not an apples-to-apples comparison, and the archive does not provide confidence intervals.

The archived prediction arrays were independently checked: recomputed macro AUROC is 0.7280811, matching `results.json`; macro AUPRC and thresholded metrics were computed from `test_labels.npy`, `test_probs.npy`, and `tuned_thresholds.npy`.
