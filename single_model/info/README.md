# Single-model workflow

This workflow contains independent NIH ChestX-ray14 model runs. The active evaluation rules and fixed dataset split are in [`evaluation_protocol.yaml`](./evaluation_protocol.yaml).

## Environment and data

Run notebooks on Kaggle with the NIH ChestX-ray14 dataset attached and a GPU enabled. The notebooks expect the metadata CSV, official `train_val_list.txt` and `test_list.txt`, and referenced PNG images under `/kaggle/input`. Enable Internet for pretrained weights and dependencies when required.

Single-model notebooks create their output directories under `/kaggle/working/single_model/output` (or use the notebook's output environment variable). Download the resulting folder from Kaggle to [`../output/`](../output/) after the run. Keep each run's configuration, checkpoint, validation/test logits, and logs together.

## Notebooks

- [`../cnn/README.md`](../cnn/README.md): standalone Kaggle notebooks and smoke test for hard-label ConvNeXt-Tiny, DenseNet-121, and ResNet-50 runs. These are independent from distillation.
- [`m08-m09-teacher-training.ipynb`](./m08-m09-teacher-training.ipynb): registered M08/M09 training and optional forward-pass benchmarking. Outputs the split and teacher runs under `results_s42/`.
- [`rad-dino_official-split.ipynb`](./rad-dino_official-split.ipynb): standalone RAD-DINO training/evaluation.
- [`resnet50.ipynb`](./resnet50.ipynb): separate ResNet50 experiment; its archived outputs and analysis are in [`resnet50-results.md`](./resnet50-results.md). Do not mix its outputs with the registered M08/M09 teacher runs.
- [`gradcam.ipynb`](./gradcam.ipynb): optional attribution analysis.

## Distillation teacher artifacts

The registered M10 distillation run requires the exact seed-42 teacher files retained under [`../output/results_s42/`](../output/results_s42/):

- `calibration.json` (per-class Platt parameters for `cnn` and `vit`);
- `split_NIH_OFFICIAL_PV.csv` and the official split/hash metadata;
- `runs/*M08*/checkpoint_best.pt` and `runs/*M08*/logits_val.npz`;
- `runs/*M09*/checkpoint_best.pt` and `runs/*M09*/logits_val.npz`;
- `timing_M08_CONVNEXT_TINY_ASL_384_CR.json` and `timing_M09_RAD_DINO_518_CR.json`.

The protocol pins the checkpoint hashes. Preserve those files as a matched bundle; a newly trained checkpoint is not a drop-in replacement for the registered M10 experiment. The dedicated M08/M09 trainer can regenerate runs using the original configuration, but new checkpoints have to be treated as a new experiment before using them as distillation teachers.

To run the registered distillation notebook on Kaggle, publish or upload the `results_s42/` folder as a Kaggle input dataset without changing its internal paths. Do not recompute or mix only part of the teacher bundle.

## Reproducibility

Use the official split, seed 42, original model configurations, validation-only calibration/threshold selection, and no test-time augmentation. Do not tune against test results. Confirm the split hash and patient separation before training.
