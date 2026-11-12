# Independent NIH single-CNN notebooks

These are standalone **hard-label** CNN experiments on NIH ChestX-ray14. They
do not load the RAD-DINO teacher, teacher logits, soft labels, or a distilled
checkpoint. Each model gets its own run folder and calibration/threshold files.

## Notebooks and protocol

| Model | Initial weights | Notebook |
|---|---|---|
| Smoke test | Random initialization (no downloads) | [`cnn_smoke_test.ipynb`](./cnn_smoke_test.ipynb) |
| ConvNeXt-Tiny | `convnext_tiny.fb_in22k_ft_in1k` | [`train_convnext_tiny.ipynb`](./train_convnext_tiny.ipynb) |
| DenseNet-121 | `densenet121.tv_in1k` | [`train_densenet121.ipynb`](./train_densenet121.ipynb) |
| ResNet-50 | `resnet50.tv_in1k` | [`train_resnet50.ipynb`](./train_resnet50.ipynb) |

All three runs use the official `NIH_OFFICIAL_PV` patient split (seed 42), 14
labels in the order in `evaluation_protocol.yaml`, 384 x 384 grayscale images
replicated to three channels, ImageNet normalization, ASL
(`gamma_neg=4`, `gamma_pos=1`, `clip=0.05`), AdamW, head-only warm-up, then full
fine-tuning, and no test-time augmentation.

The ConvNeXt settings reproduce the single-CNN recipe recorded in the protocol:
batch 32, one warm-up epoch, up to eight fine-tuning epochs, head/backbone LR
`1e-4`, weight decay `0.05`, drop path `0.1`, and early stopping after two
non-improving fine-tuning epochs. ResNet-50 uses the same drop-path setting.
The timm DenseNet-121 implementation has no `drop_path_rate` option, so that
architecture uses its native default (no stochastic depth); this is recorded
as `null` in its run config. All three are new hard-label experiments, not
hash-pinned teachers or the archived ResNet50 result.

Each notebook is self-contained and checks the official list-file hashes,
split counts, patient separation, and exact split hash. Validation logits
determine the best checkpoint, per-class Platt calibration, and F1 thresholds.
Test predictions are made only after those choices are frozen. Each run exports
raw validation/test logits, metrics, and 95% patient-bootstrap intervals (1,000
resamples).

## Run on Kaggle

1. Import the desired `.ipynb` into Kaggle (or upload it as a notebook), add the
   NIH ChestX-ray14 dataset containing `Data_Entry_2017.csv`,
   `train_val_list.txt`, `test_list.txt`, and the image directories, and enable
   a GPU (a T4 is the protocol reference).
2. Enable Internet for package installation and pretrained weights. Run the
   notebook's setup cell, which installs `scikit-learn==1.6.1` and
   `timm==1.0.26` without replacing Kaggle's CUDA-enabled `torch` and
   `torchvision`.
3. In the paths/options cell, set `DATA_ROOT` to the directory containing the
   NIH metadata and official list files. The default is
   `/kaggle/input/datasets/nih-chest-xrays/data`; change it if your Kaggle dataset has a
   different slug. Keep `OUTPUT_DIR` under `/kaggle/working`.
4. Run all cells in [`cnn_smoke_test.ipynb`](./cnn_smoke_test.ipynb) first. It
   validates the attached split/data and runs one real-image train step for
   each model with random initialization. It does not download pretrained
   weights or train full models. Fix any reported data, dependency, or GPU
   issue before proceeding.
5. Import and run one of the three training notebooks. Each notebook trains
   only its named model. Runs take substantially longer than the smoke test.
   Save a Kaggle Notebook version or download `/kaggle/working` outputs before
   the session expires.

## Outputs

Each model writes to
`runs/NIH_CXR14__NIH_OFFICIAL_PV__<model-id>__s42/`:

- `config.json`, `train_log.csv`, and `checkpoint_best.pt`;
- `logits_val.npz` and `logits_test.npz` with image/patient IDs, raw float32
  logits, uint8 labels, label order, and split hash;
- validation-only `calibration.json` and `thresholds.json`;
- `test_metrics.json` with aggregate/per-class metrics, dataset
  characteristics, and patient-bootstrap confidence intervals.

The output root also contains `split_NIH_OFFICIAL_PV.csv`. Do not mix files
between model runs. The test set is for final reporting only; do not use it to
select checkpoints, calibrate probabilities, or choose thresholds.
