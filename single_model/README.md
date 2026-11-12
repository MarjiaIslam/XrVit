# Standalone NIH ChestX-ray14 models

The model folders contain independent, hard-label experiments on the official
NIH ChestX-ray14 patient split. They do not use teacher predictions, a
distillation checkpoint, or another model's outputs. The shared evaluation and
training rules are registered in
[`info/evaluation_protocol.yaml`](./info/evaluation_protocol.yaml).

## Transformer notebooks

Files in [`transformer/`](./transformer/):

| Model | Training notebook | Protocol ID |
|---|---|---|
| RAD-DINO | [`train_rad_dino.ipynb`](./transformer/train_rad_dino.ipynb) | `M09_RAD_DINO_518_CR` |
| BioViL-T | [`train_biovil_t.ipynb`](./transformer/train_biovil_t.ipynb) | `M11_BIOVIL_T_448_CR` |
| CXformer-B | [`train_cxformer_b.ipynb`](./transformer/train_cxformer_b.ipynb) | `M12_CXFORMER_BASE_518_CR` |
| Kaggle preflight | [`transformer_smoke_test.ipynb`](./transformer/transformer_smoke_test.ipynb) | All three |

Each training notebook is self-contained. The shared implementation is also
kept in [`transformer_workflow.py`](./transformer/transformer_workflow.py) for
review and local reuse.

## Run on Kaggle

1. Import the smoke-test notebook into Kaggle. Attach an **extracted** NIH
   ChestX-ray14 dataset containing `Data_Entry_2017.csv`,
   `train_val_list.txt`, `test_list.txt`, and the PNG files.
2. Enable **Internet** and select a GPU accelerator. A Tesla T4 is the protocol
   reference. Internet is needed to install dependencies and retrieve model
   weights.
3. Run all cells in `transformer_smoke_test.ipynb` first. It validates the
   official file hashes, split counts/hash, patient separation, image coverage,
   model downloads/processors, and one real-image forward/backward/optimizer
   step per model. If the initial fp16 scale overflows, the smoke test retries
   with GradScaler's reduced scale; it fails if no finite update is produced.
   The first run downloads pretrained weights.
4. If the smoke test passes, import and run **one** desired model notebook.
   Leave `DATA_ROOT = None` for discovery under `/kaggle/input`, or set it to
   the folder containing the NIH metadata and official lists.
5. Save a Kaggle Notebook version or download
   `/kaggle/working/single_model/transformer` before the Kaggle session expires.

Full runs use seed 42, validation-only checkpoint selection/calibration/F1
thresholds, no test-time augmentation, and 1,000 patient-level bootstrap
resamples. They also measure inference latency with the protocol's batch-1 and
batch-32 CUDA-event procedure. Test metrics are produced only after
validation-based choices are frozen. A full training run can take many hours.

## Interpretation and limitations

- The test split has a known shift from train/validation in findings per image,
  images per patient, PA-view fraction, and label prevalence. The exported
  metrics include dataset characteristics.
- The RAD-DINO model card lists NIH-CXR8 images among its pretraining data. The
  CXformer-B model card reports pretraining on official NIH training splits.
  Treat these as pretrained-model evaluations, not automatically independent
  tests; disclose any known image overlap when reporting results.
- BioViL-T is a temporal vision-language hybrid. The NIH experiment supplies
  one image at a time through the encoder's documented single-image path; it
  does not measure temporal comparison.
- BioViL-T and RAD-DINO are research models. CXformer-B is distributed under
  CC BY-NC 4.0, research/non-commercial use only. None of these experiments is
  clinical validation or a basis for clinical deployment.
