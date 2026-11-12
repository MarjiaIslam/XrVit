# XrVit onboarding

XrVit contains two active workflows: single-model NIH ChestX-ray14 runs and distillation. The project performs research-only 14-label classification and is not a clinical product.

## 1. Single-model workflow

Start with [`single_model/info/README.md`](single_model/info/README.md). Kaggle is the supported environment: attach the NIH dataset, enable a GPU, and retain each model's config, checkpoint, logits, and logs in the configured output folder. The official M08/M09 seed-42 teacher bundle is already retained under `single_model/output/results_s42/`.

The registered distillation uses pinned teacher checkpoint hashes. Re-running a teacher creates a different checkpoint and does not replace the registered input bundle.

## 2. Distillation workflow

Follow [`distill/info/README.md`](distill/info/README.md). Attach the exact `results_s42/` teacher bundle as a Kaggle input dataset, run D1 to verify the checkpoints and build train-only soft labels, then run D2 to train M10. Each notebook creates its output directory under `/kaggle/working` (override with `CXR_OUT_DIR` for distillation). Download outputs into `distill/output/`.

The checked-in distillation notebook currently has a D3 placeholder. Previously generated D3 metrics and tables are retained in `distill/output/`; fresh final metrics require the approved evaluation implementation.

## Dataset and protocol checks

Attach the NIH ChestX-ray14 metadata, official train/validation and test lists, and image files. Before training, verify the official split hashes and patient separation. Use seed 42, preserve the registered model/preprocessing settings, fit calibration and thresholds on validation only, and do not tune against test labels. See [`evaluation_protocol.yaml`](evaluation_protocol.yaml).

## Output retention

Kaggle's `/kaggle/working` is temporary. Download the generated files before a session expires. Large local outputs under `single_model/output/` and `distill/output/` are excluded from Git; keep the run configuration and related logs with each result bundle.
