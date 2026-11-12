# Post-hoc distillation analyses

These analyses operate on saved predictions and do not train or modify model weights. They are descriptive and are not part of the registered evaluation.

The expected result bundle is `single_model/output/results_s42/` plus the files and M10 run folder under `distill/output/`.

- [`distill_analysis.py`](./distill_analysis.py): error counts and student/teacher agreement.
- [`perfinding_ci.py`](./perfinding_ci.py): per-finding AUROC comparisons with patient-level bootstrap intervals.
- [`qualitative_attribution_kaggle.py`](./qualitative_attribution_kaggle.py): optional Kaggle attribution visualization; see [`QUALITATIVE_ATTRIBUTION_KAGGLE.md`](./QUALITATIVE_ATTRIBUTION_KAGGLE.md).
