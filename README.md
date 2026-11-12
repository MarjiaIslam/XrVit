# XrVit: NIH ChestX-ray14 single-model baselines and distillation

This research project evaluates 14-label classification on the NIH ChestX-ray14 official test split. Its active workflows are independent single-model runs and teacher-to-student distillation. The registered comparison uses one training seed (42), no test-time augmentation, validation-fitted calibration/thresholds, and patient-level bootstrap confidence intervals. This is research code, not a clinical tool.

## Results

| System | Macro AUROC [95% CI] | Macro F1 | Cost vs RAD-DINO (batch 32) |
|---|---:|---:|---:|
| CNN alone | 0.8263 [0.8200, 0.8317] | 0.3539 | 0.18 |
| **Distilled CNN** | **0.8339 [0.8283, 0.8387]** | **0.3702** | **0.18** |
| RAD-DINO alone | 0.8364 [0.8303, 0.8415] | 0.3788 | 1.00 |
| Average of both models | 0.8428 [0.8373, 0.8478] | 0.3818 | 1.18 |

The distilled CNN improved over the CNN baseline but did not meet the registered rule for matching RAD-DINO. These values use the official test set (25,596 images), one training seed, and no test-time augmentation.

The separate ResNet50 experiment scored 0.7281 test macro AUROC with 2 h 40 min of logged training time. It used a different validation fraction and loss, and filtered four official test images; see the [ResNet50 result analysis](single_model/info/resnet50-results.md). It is not a direct comparison with the registered rows above.

## Repository layout

| Path | Contents |
|---|---|
| [`single_model/info/`](single_model/info/README.md) | Single-model notebooks, configurations, and run instructions |
| [`single_model/output/`](single_model/output/) | Retained M08/M09 baseline artifacts and other single-model outputs |
| [`distill/info/`](distill/info/README.md) | Distillation notebooks, dependency requirements, and run instructions |
| [`distill/output/`](distill/output/) | Distillation results, logs, metrics, and checkpoints |
| [`evaluation_protocol.yaml`](evaluation_protocol.yaml) | Active single-model and distillation rules |
| [`exploratory/`](exploratory/) | Legacy path for retained post-hoc analysis sources |

Large outputs are local and ignored by Git. Original root notebook paths and remaining baseline/result folders are kept where possible for compatibility; the `single_model/` and `distill/` copies are the organized paths to use going forward.

## Run order

1. Attach NIH ChestX-ray14 to Kaggle and enable a GPU. Read [`single_model/info/README.md`](single_model/info/README.md) for the single-model environment and retained teacher artifacts.
2. For the registered M10 run, attach the exact seed-42 M08/M09 bundle from `single_model/output/results_s42/` as a Kaggle input dataset. The pinned checkpoint hashes matter; a newly trained teacher is a different experiment.
3. Follow D1 and D2 in [`distill/info/README.md`](distill/info/README.md). Notebooks create their `/kaggle/working/.../output` folders before writing.
4. Download the run artifacts into the matching workflow's `output/` directory to preserve them after Kaggle sessions expire.

The current distillation notebook's D3 section is a placeholder; archived D3 results are retained in `distill/output/`. Do not report new final test metrics from a fresh D2 run until the approved D3 evaluation code is available.

The fixed-value plotting utility is [`distill/info/generate_plots.py`](distill/info/generate_plots.py); it writes figures to `distill/output/figures/` and does not recalculate metrics.

## Reproducibility and limitations

The active protocol fixes the official split and its hashes, patient separation, seed 42, model settings, and validation-only selection. Do not tune on test labels or combine artifacts from different splits/checkpoints. The results use one dataset and one training seed; labels are text-mined and noisy, and no external test set is included.
