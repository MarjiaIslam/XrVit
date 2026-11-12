# Distillation workflow

This workflow trains a student from teacher predictions. Its active protocol is the [`evaluation_protocol.yaml`](../../evaluation_protocol.yaml); the registered M10 methodology is unchanged.

## Environment and output

Use Kaggle with the NIH ChestX-ray14 dataset attached, GPU enabled, and Internet enabled when pretrained weights or packages are needed. The notebooks create `/kaggle/working/distill/output` before writing. Set `CXR_OUT_DIR` to override that destination. Download the completed folder into [`../output/`](../output/) to retain results between Kaggle sessions.

## Registered ConvNeXt-Tiny student (M10)[text](../../4-distillation-ipynb.ipynb) [text](../../5-distillation-densenet121-teacher-student.ipynb) [text](../../5-distillation-densenet121.ipynb) [text](../../6-distillation-densenet121-biovilt.ipynb) [text](../../generate_plots.py) [text](../../evaluation_protocol.yaml) [text](../../Green281%29_aligned.tex) [text](../../gradcam.ipynb) [text](../../onboard.md) [text](../../rad-dino_official-split.ipynb) [text](../../resnet50_nih_official_split.ipynb) [text](../../resnet50-nih-official-updated.ipynb)

Run [`4-distillation-ipynb.ipynb`](./4-distillation-ipynb.ipynb) in order:

1. Attach the exact `results_s42/` teacher bundle described in [`single_model/`](../../single_model/info/README.md), keeping the directory structure.
2. Run D1 (`RUN_TEACHER=True`). It verifies M08/M09 checkpoint hashes, rebuilds teacher predictions, checks saved validation logits, writes train-only soft labels, and applies the go/no-go rule.
3. Run D2 (`RUN_STUDENT=True`) only after D1 passes. The student uses the registered 0.5 hard-label/0.5 soft-label target and writes its checkpoint, logs, and validation/test logits.
4. The archived results in [`../output/`](../output/) include D3 tables and metrics. The current notebook ends with a D3 placeholder rather than executable D3 evaluation code; do not claim new final metrics from a fresh D2 run until that evaluation step is restored from its approved source.

Expected teacher inputs include `checkpoint_best.pt`, `logits_val.npz`, `calibration.json`, and `split_NIH_OFFICIAL_PV.csv`. D1 writes `teacher/M08_CONVNEXT_TINY_ASL_384_CR_logits_train.npz`, `teacher/M09_RAD_DINO_518_CR_logits_train.npz`, validation rebuilds, `soft_labels_train.npz`, `distill_go_no_go.json`, and QC reports. The M10 run folder contains `config.json`, `checkpoint_best.pt`, `train_log.csv`, `logits_val.npz`, and `logits_test.npz`.

## Other distillation notebooks

The DenseNet/BioViL notebooks are retained as separate experimental variants:

- [`5-distillation-densenet121-teacher-student.ipynb`](./5-distillation-densenet121-teacher-student.ipynb)
- [`5-distillation-densenet121.ipynb`](./5-distillation-densenet121.ipynb)
- [`6-distillation-densenet121-biovilt.ipynb`](./6-distillation-densenet121-biovilt.ipynb)

Each variant needs its own matching teacher checkpoints and configuration. Do not substitute its outputs for M10 or combine outputs from different variants.

[`analysis/`](./analysis/) contains optional post-hoc analyses; these are descriptive and not part of the registered evaluation.

## Figures

Run `python distill/info/generate_plots.py` from the repository root to regenerate the presentation figures from the fixed values in the script. It writes them to `distill/output/figures/`; it does not recompute metrics.
