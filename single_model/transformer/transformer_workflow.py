import os

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

import copy
import csv
import gc
import hashlib
import json
import math
import random
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torchvision.transforms as T
from PIL import Image
from sklearn.model_selection import GroupShuffleSplit
from torch.utils.data import DataLoader, Dataset


LABELS = [
    "Atelectasis",
    "Cardiomegaly",
    "Effusion",
    "Infiltration",
    "Mass",
    "Nodule",
    "Pneumonia",
    "Pneumothorax",
    "Consolidation",
    "Edema",
    "Emphysema",
    "Fibrosis",
    "Pleural_Thickening",
    "Hernia",
]
LABEL_ARRAY = np.asarray(LABELS)
EXPECTED_COUNTS = {"train": 73916, "val": 12608, "test": 25596}
EXPECTED_LIST_COUNTS = {"train_val_list.txt": 86524, "test_list.txt": 25596}
EXPECTED_LIST_HASHES = {
    "train_val_list.txt": "61fbe896321c1c1c8b75f3e4f3a08e4fef6486d95ef8a667c31d4d60dca6cb81",
    "test_list.txt": "38ca5ef7f756092946f57c1a59faca882ed589a1ab1f72590b45dc06c6d5e1cc",
}
EXPECTED_SPLIT_HASH = "b9f2fbc9aa061dafa76994494b92779651dc3ddd2ed3e7b82c70bbf5cf7071dd"
SEED = 42
BOOTSTRAP_SEED = 20260922
INPUT_DIR = "/kaggle/input"
DATA_ROOT = None
OUTPUT_DIR = "/kaggle/working/single_model/transformer"
NUM_WORKERS = 2
BOOTSTRAP_RESAMPLES = 1000
MODEL_CONFIGS = {
    "rad_dino": {
        "model_id": "M09_RAD_DINO_518_CR",
        "hub_id": "microsoft/rad-dino",
        "revision": "110cbc18d5133582e320b43d53bf5c44e410c936",
        "input_size": 518,
        "batch_size": 12,
        "warmup_epochs": 2,
        "finetune_epochs": 4,
        "lr_backbone": 2e-5,
        "lr_head": 2e-4,
        "lr_head_warmup": 1e-3,
        "ema_decay": 0.999,
        "pooling": "CLS concatenated with mean of patch tokens",
        "preprocess": "HF processor: 518, mean 0.5307, std 0.2583",
    },
    "biovil_t": {
        "model_id": "M11_BIOVIL_T_448_CR",
        "hub_id": "microsoft/BiomedVLP-BioViL-T",
        "revision": "v1.0",
        "input_size": 448,
        "batch_size": 12,
        "warmup_epochs": 2,
        "finetune_epochs": 4,
        "lr_backbone": 2e-5,
        "lr_head": 2e-4,
        "lr_head_warmup": 1e-3,
        "ema_decay": None,
        "pooling": "official BioViL-T single-image pooled embedding",
        "preprocess": "official resize 512, center crop 448, [0,1], RGB-replicated grayscale",
    },
    "cxformer_b": {
        "model_id": "M12_CXFORMER_BASE_518_CR",
        "hub_id": "m42-health/CXformer-base",
        "revision": "f778e2ac267fdd651205e07b6fd1113754b146bb",
        "input_size": 518,
        "batch_size": 12,
        "warmup_epochs": 2,
        "finetune_epochs": 4,
        "lr_backbone": 2e-5,
        "lr_head": 2e-4,
        "lr_head_warmup": 1e-3,
        "ema_decay": None,
        "pooling": "CLS concatenated with mean of patch tokens; exclude four register tokens",
        "preprocess": "HF processor: 518, histogram equalization, ImageNet mean/std",
    },
}


def _find_files(root, wanted):
    found = {name: [] for name in wanted}
    for current, _, files in os.walk(root):
        for name in wanted:
            if name in files:
                found[name].append(Path(current) / name)
        if all(found.values()):
            break
    return found


def _locate_data(input_dir, data_root=None):
    search_root = Path(input_dir).expanduser().resolve()
    if not search_root.exists():
        raise FileNotFoundError(f"Input directory does not exist: {search_root}")

    root = Path(data_root).expanduser().resolve() if data_root else search_root
    if not root.exists():
        raise FileNotFoundError(f"Data root does not exist: {root}")
    files = _find_files(root, {"Data_Entry_2017.csv", "train_val_list.txt", "test_list.txt"})
    missing = [name for name, paths in files.items() if not paths]
    if missing:
        raise FileNotFoundError(
            f"Missing NIH files under {root}: {', '.join(missing)}. "
            "Attach the extracted NIH ChestX-ray14 dataset or set DATA_ROOT."
        )
    metadata_paths = files["Data_Entry_2017.csv"]
    if len(metadata_paths) != 1:
        raise ValueError(f"Found {len(metadata_paths)} metadata CSV files under {root}; set DATA_ROOT.")
    metadata = metadata_paths[0]
    list_paths = {}
    for name in ("train_val_list.txt", "test_list.txt"):
        siblings = [path for path in files[name] if path.parent == metadata.parent]
        if siblings:
            list_paths[name] = siblings[0]
        elif len(files[name]) == 1:
            list_paths[name] = files[name][0]
        else:
            raise ValueError(f"Could not uniquely select {name} under {root}; set DATA_ROOT.")
    return metadata.parent, metadata, list_paths


def _read_ids(path):
    ids = Path(path).read_text(encoding="utf-8").split()
    if len(ids) != len(set(ids)):
        raise ValueError(f"Duplicate image IDs in {path}")
    return ids


def _sha256(data):
    return hashlib.sha256(data).hexdigest()


def load_protocol_data(input_dir, data_root=None):
    root, metadata_path, list_paths = _locate_data(input_dir, data_root)
    df = pd.read_csv(metadata_path)
    required = {"Image Index", "Patient ID", "Finding Labels"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Metadata is missing columns: {sorted(missing)}")
    if len(df) != 112120:
        raise ValueError(f"Expected 112120 metadata rows, found {len(df)}")

    df["Image Index"] = df["Image Index"].astype(str)
    df = df.sort_values("Image Index").reset_index(drop=True)
    findings = df["Finding Labels"].fillna("").str.split("|")
    for label in LABELS:
        df[label] = findings.map(lambda row, name=label: float(name in row))

    train_val_ids = _read_ids(list_paths["train_val_list.txt"])
    test_ids = _read_ids(list_paths["test_list.txt"])
    for name, ids in (
        ("train_val_list.txt", train_val_ids),
        ("test_list.txt", test_ids),
    ):
        if len(ids) != EXPECTED_LIST_COUNTS[name]:
            raise ValueError(f"QC-01: unexpected {name} row count: {len(ids)}")
        digest = _sha256("\n".join(sorted(ids)).encode("utf-8"))
        if digest != EXPECTED_LIST_HASHES[name]:
            raise ValueError(f"QC-01: {name} hash mismatch: {digest}")

    all_ids = set(df["Image Index"])
    if set(train_val_ids) & set(test_ids):
        raise ValueError("QC-01: official train/validation and test lists overlap")
    if set(train_val_ids) | set(test_ids) != all_ids:
        raise ValueError("QC-01: official lists do not exactly cover metadata images")

    split = np.where(df["Image Index"].isin(test_ids), "test", "trainval")
    tv_frame = df.loc[split == "trainval"]
    _, val_indices = next(
        GroupShuffleSplit(n_splits=1, test_size=0.15, random_state=SEED).split(
            tv_frame, groups=tv_frame["Patient ID"]
        )
    )
    partition = np.asarray(split, dtype=object)
    partition[split == "trainval"] = "train"
    partition[tv_frame.index[val_indices]] = "val"
    df["partition"] = partition

    split_frame = (
        df[["Image Index", "Patient ID", "partition"]]
        .rename(columns={"Image Index": "image_index", "Patient ID": "patient_id"})
        .sort_values("image_index")
    )
    split_hash = _sha256(split_frame.to_csv(index=False, lineterminator="\n").encode("utf-8"))
    if split_hash != EXPECTED_SPLIT_HASH:
        raise ValueError(
            "QC-03: official split hash mismatch. Check scikit-learn version; "
            f"expected {EXPECTED_SPLIT_HASH}, received {split_hash}."
        )
    counts = df["partition"].value_counts().to_dict()
    if counts != EXPECTED_COUNTS:
        raise ValueError(f"QC-01: split counts differ from protocol: {counts}")
    patients = {
        part: set(df.loc[df["partition"] == part, "Patient ID"])
        for part in ("train", "val", "test")
    }
    if (
        patients["train"] & patients["val"]
        or patients["train"] & patients["test"]
        or patients["val"] & patients["test"]
    ):
        raise ValueError("QC-02: a patient appears in multiple partitions")

    image_paths = {}
    for current, _, names in os.walk(root):
        for name in names:
            if name.endswith(".png"):
                image_paths.setdefault(name, str(Path(current) / name))
    df["path"] = df["Image Index"].map(image_paths)
    if df["path"].isna().any():
        raise FileNotFoundError(
            f"QC-04: {int(df['path'].isna().sum())} metadata images have no PNG under {root}"
        )
    return df, split_hash, split_frame


class CXRDataset(Dataset):
    def __init__(self, frame, model_key, processor=None, training=False):
        self.frame = frame.reset_index(drop=True)
        self.model_key = model_key
        self.processor = processor
        self.labels = self.frame[LABELS].to_numpy(dtype=np.float32)
        size = MODEL_CONFIGS[model_key]["input_size"]
        self.augmentation = (
            T.Compose(
                [
                    T.RandomResizedCrop(
                        size,
                        scale=(0.85, 1.0),
                        ratio=(0.9, 1.1),
                        interpolation=T.InterpolationMode.BILINEAR,
                    ),
                    T.RandomHorizontalFlip(p=0.5),
                    T.RandomApply([T.RandomRotation(7)], p=0.5),
                    T.ColorJitter(brightness=0.1, contrast=0.1),
                ]
            )
            if training
            else None
        )
        self.training = training
        if model_key == "biovil_t":
            self.eval_transform = T.Compose(
                [
                    T.Resize(512, interpolation=T.InterpolationMode.BILINEAR),
                    T.CenterCrop(448),
                    T.ToTensor(),
                ]
            )
            self.train_to_tensor = T.ToTensor()

    def __len__(self):
        return len(self.frame)

    def __getitem__(self, index):
        row = self.frame.iloc[index]
        with Image.open(row["path"]) as source:
            image = source.convert("L")
            if self.model_key != "biovil_t":
                image = image.convert("RGB")
            if self.training:
                image = self.augmentation(image)
            if self.model_key == "biovil_t":
                tensor = (
                    self.train_to_tensor(image)
                    if self.training
                    else self.eval_transform(image)
                ).repeat(3, 1, 1)
            else:
                processor = self.processor[1 if self.training else 0]
                tensor = processor(images=image, return_tensors="pt")["pixel_values"][0]
        return tensor, self.labels[index]


class AsymmetricLoss(nn.Module):
    def __init__(self, gamma_neg=4, gamma_pos=1, clip=0.05, eps=1e-8):
        super().__init__()
        self.gamma_neg = gamma_neg
        self.gamma_pos = gamma_pos
        self.clip = clip
        self.eps = eps

    def forward(self, logits, targets):
        probabilities = torch.sigmoid(logits.float())
        negative = (
            (probabilities - self.clip).clamp(min=0)
            if self.clip > 0
            else probabilities
        )
        positive_loss = (
            targets
            * torch.log(probabilities.clamp(min=self.eps))
            * (1 - probabilities).pow(self.gamma_pos)
        )
        negative_loss = (
            (1 - targets)
            * torch.log((1 - negative).clamp(min=self.eps))
            * negative.pow(self.gamma_neg)
        )
        return -(positive_loss + negative_loss).sum(dim=1).mean()


class ImageClassifier(nn.Module):
    def __init__(self, model_key, backbone, hidden_size, register_tokens=0):
        super().__init__()
        self.model_key = model_key
        self.backbone = backbone
        self.register_tokens = register_tokens
        feature_size = 512 if model_key == "biovil_t" else hidden_size * 2
        self.head = nn.Sequential(nn.Dropout(p=0.2), nn.Linear(feature_size, len(LABELS)))

    def forward(self, images):
        if self.model_key == "biovil_t":
            features = self.backbone(images).img_embedding
        else:
            tokens = self.backbone(pixel_values=images).last_hidden_state
            patches = tokens[:, 1 + self.register_tokens :, :]
            features = torch.cat((tokens[:, 0, :], patches.mean(dim=1)), dim=1)
        return self.head(features)


def build_model(model_key):
    cfg = MODEL_CONFIGS[model_key]
    if model_key == "biovil_t":
        from health_multimodal.image.model.pretrained import get_biovil_t_image_encoder

        model = ImageClassifier(model_key, get_biovil_t_image_encoder(), 0)
        return model, None

    from transformers import AutoConfig, AutoImageProcessor, AutoModel

    kwargs = {"revision": cfg["revision"]}
    if model_key == "cxformer_b":
        processor = AutoImageProcessor.from_pretrained(
            cfg["hub_id"], trust_remote_code=True, **kwargs
        )
        if not getattr(processor, "do_hist_equalization", False):
            raise RuntimeError("The published CXformer-B processor must enable histogram equalization.")
    else:
        processor = AutoImageProcessor.from_pretrained(cfg["hub_id"], **kwargs)
    hf_config = AutoConfig.from_pretrained(cfg["hub_id"], **kwargs)
    hf_config.drop_path_rate = 0.1
    backbone = AutoModel.from_pretrained(cfg["hub_id"], config=hf_config, **kwargs)

    train_processor = copy.deepcopy(processor)
    train_processor.do_resize = False
    train_processor.do_center_crop = False
    register_tokens = 4 if model_key == "cxformer_b" else 0
    backbone.gradient_checkpointing_enable(
        gradient_checkpointing_kwargs={"use_reentrant": False}
    )
    model = ImageClassifier(
        model_key,
        backbone,
        backbone.config.hidden_size,
        register_tokens=register_tokens,
    )
    return model, (processor, train_processor)


def seed_everything(seed=SEED):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.use_deterministic_algorithms(True, warn_only=True)


def _worker_seed(_worker_id):
    seed = torch.initial_seed() % (2**32)
    np.random.seed(seed)
    random.seed(seed)


def _loader(frame, model_key, processors, training, workers, batch_size=None):
    generator = torch.Generator().manual_seed(SEED)
    return DataLoader(
        CXRDataset(frame, model_key, processors, training=training),
        batch_size=batch_size or MODEL_CONFIGS[model_key]["batch_size"],
        shuffle=training,
        num_workers=workers,
        pin_memory=torch.cuda.is_available(),
        persistent_workers=workers > 0,
        drop_last=False,
        worker_init_fn=_worker_seed,
        generator=generator,
    )


def _macro_auroc(labels, logits):
    from sklearn.metrics import roc_auc_score

    values = []
    for index in range(len(LABELS)):
        target = labels[:, index]
        if target.min() != target.max():
            values.append(roc_auc_score(target, logits[:, index]))
    if not values:
        raise ValueError("Validation labels have no class with both positive and negative cases")
    return float(np.mean(values))


def _trainable_backbone_groups(model, model_key):
    if model_key == "biovil_t":
        return [{"params": list(model.backbone.encoder.parameters()), "lr": 2e-5}]

    layers = model.backbone.encoder.layer
    selected = list(enumerate(layers[-8:], start=len(layers) - 8))
    groups = []
    for index, layer in selected:
        distance_from_top = len(layers) - 1 - index
        groups.append(
            {
                "params": list(layer.parameters()),
                "lr": 2e-5 * (0.8 ** distance_from_top),
            }
        )
    groups.append(
        {
            "params": list(model.backbone.layernorm.parameters()),
            "lr": 2e-5,
        }
    )
    return groups


def _phase_optimizer(model, phase):
    for parameter in model.parameters():
        parameter.requires_grad = False
    for parameter in model.head.parameters():
        parameter.requires_grad = True
    head_lr = 1e-3 if phase == "warmup" else 2e-4
    groups = [{"params": list(model.head.parameters()), "lr": head_lr}]
    if phase == "finetune":
        if model.model_key == "biovil_t":
            for parameter in model.backbone.encoder.parameters():
                parameter.requires_grad = True
        else:
            for layer in model.backbone.encoder.layer[-8:]:
                for parameter in layer.parameters():
                    parameter.requires_grad = True
            for parameter in model.backbone.layernorm.parameters():
                parameter.requires_grad = True
        groups.extend(_trainable_backbone_groups(model, model.model_key))
    return torch.optim.AdamW(groups, weight_decay=0.05)


def _schedule(optimizer, total_steps):
    warmup_steps = max(1, int(total_steps * 0.03))

    def scale(step):
        if step < warmup_steps:
            return (step + 1) / warmup_steps
        return 0.5 * (
            1 + math.cos(math.pi * (step - warmup_steps) / max(1, total_steps - warmup_steps))
        )

    return torch.optim.lr_scheduler.LambdaLR(optimizer, scale)


class ModelEMA:
    def __init__(self, model, decay):
        self.model = copy.deepcopy(model).eval()
        self.decay = decay
        for parameter in self.model.parameters():
            parameter.requires_grad_(False)

    @torch.no_grad()
    def update(self, model):
        ema_state = self.model.state_dict()
        model_state = model.state_dict()
        for name, ema_value in ema_state.items():
            current = model_state[name].detach()
            if torch.is_floating_point(ema_value):
                ema_value.mul_(self.decay).add_(current, alpha=1 - self.decay)
            else:
                ema_value.copy_(current)


def _predict(model, loader, device):
    model.eval()
    logits = []
    labels = []
    with torch.inference_mode():
        for images, targets in loader:
            images = images.to(device, non_blocking=True)
            with torch.autocast(
                device_type=device.type,
                dtype=torch.float16,
                enabled=device.type == "cuda",
            ):
                output = model(images)
            logits.append(output.float().cpu().numpy())
            labels.append(targets.numpy())
    return np.concatenate(logits), np.concatenate(labels)


def _save_logits(path, frame, logits, labels, split_hash):
    np.savez_compressed(
        path,
        image_index=frame["Image Index"].to_numpy(dtype=str),
        patient_id=frame["Patient ID"].to_numpy(),
        logits=logits.astype(np.float32),
        labels=labels.astype(np.uint8),
        label_order=LABEL_ARRAY,
        split_sha256=np.asarray(split_hash),
    )


def _fit_platt(labels, logits):
    from scipy.optimize import minimize
    from scipy.special import expit

    parameters = np.zeros((len(LABELS), 2), dtype=np.float64)
    for index in range(len(LABELS)):
        target = labels[:, index].astype(np.float64)
        raw = logits[:, index].astype(np.float64)
        if target.min() == target.max():
            raise ValueError(f"Cannot calibrate {LABELS[index]}: validation labels are constant")

        def objective(coef):
            calibrated = coef[0] * raw + coef[1]
            probabilities = expit(calibrated)
            loss = np.logaddexp(0, calibrated) - target * calibrated
            residual = probabilities - target
            gradient = np.asarray([np.dot(residual, raw), residual.sum()])
            return float(loss.sum()), gradient

        result = minimize(objective, np.asarray([1.0, 0.0]), method="L-BFGS-B", jac=True)
        if not result.success or not np.isfinite(result.fun):
            raise RuntimeError(f"Platt calibration failed for {LABELS[index]}: {result.message}")
        parameters[index] = result.x
    return parameters


def _calibrate(logits, parameters):
    from scipy.special import expit

    return expit(logits * parameters[:, 0] + parameters[:, 1])


def _fit_thresholds(labels, probabilities):
    thresholds = np.empty(len(LABELS), dtype=np.float64)
    fallback = []
    for index, name in enumerate(LABELS):
        target = labels[:, index].astype(np.int64)
        scores = probabilities[:, index]
        if target.sum() == 0:
            thresholds[index] = 0.5
            fallback.append(name)
            continue
        order = np.argsort(scores, kind="mergesort")[::-1]
        sorted_scores = scores[order]
        cumulative_tp = np.cumsum(target[order])
        ends = np.r_[np.flatnonzero(sorted_scores[:-1] != sorted_scores[1:]), len(scores) - 1]
        tp = cumulative_tp[ends]
        predicted = ends + 1
        fp = predicted - tp
        fn = target.sum() - tp
        denominator = 2 * tp + fp + fn
        f1 = np.divide(
            2 * tp, denominator, out=np.zeros_like(tp, dtype=float), where=denominator > 0
        )
        best = np.flatnonzero(f1 == f1.max())
        thresholds[index] = sorted_scores[ends[best[-1]]]
        if f1.max() == 0:
            thresholds[index] = 0.5
            fallback.append(name)
    return thresholds, fallback


def _metrics(labels, probabilities, thresholds, sample_weight=None):
    from sklearn.metrics import average_precision_score, roc_auc_score

    binary = probabilities >= thresholds[None, :]
    weights = (
        np.ones(len(labels), dtype=np.float64)
        if sample_weight is None
        else sample_weight.astype(np.float64)
    )
    weighted_labels = labels * weights[:, None]
    tp = np.sum(weighted_labels * binary, axis=0)
    fp = np.sum((weights[:, None] - weighted_labels) * binary, axis=0)
    fn = np.sum(weighted_labels * (~binary), axis=0)
    precision = np.divide(tp, tp + fp, out=np.zeros_like(tp), where=tp + fp > 0)
    recall = np.divide(tp, tp + fn, out=np.full_like(tp, np.nan), where=tp + fn > 0)
    f1 = np.divide(
        2 * tp,
        2 * tp + fp + fn,
        out=np.zeros_like(tp),
        where=2 * tp + fp + fn > 0,
    )
    auroc = np.full(len(LABELS), np.nan, dtype=np.float64)
    auprc = np.full(len(LABELS), np.nan, dtype=np.float64)
    for index in range(len(LABELS)):
        target = labels[:, index]
        if np.unique(target[weights > 0]).size == 2:
            auroc[index] = roc_auc_score(
                target, probabilities[:, index], sample_weight=weights
            )
            auprc[index] = average_precision_score(
                target, probabilities[:, index], sample_weight=weights
            )
    total_tp, total_fp, total_fn = tp.sum(), fp.sum(), fn.sum()
    micro_f1 = (
        2 * total_tp / (2 * total_tp + total_fp + total_fn)
        if 2 * total_tp + total_fp + total_fn
        else 0.0
    )
    aggregates = {
        "macro_auroc": float(np.nanmean(auroc)),
        "macro_auprc": float(np.nanmean(auprc)),
        "macro_f1": float(np.mean(f1)),
        "micro_f1": float(micro_f1),
        "macro_precision": float(np.mean(precision)),
        "macro_recall": float(np.nanmean(recall)),
    }
    per_class = {
        name: {
            "auroc": float(auroc[index]),
            "auprc": float(auprc[index]),
            "precision": float(precision[index]),
            "recall": float(recall[index]),
            "f1": float(f1[index]),
            "support": int(labels[:, index].sum()),
        }
        for index, name in enumerate(LABELS)
    }
    return aggregates, per_class


def _patient_bootstrap(labels, probabilities, thresholds, patient_ids, resamples=1000):
    unique_patients, patient_index = np.unique(patient_ids, return_inverse=True)
    rng = np.random.default_rng(BOOTSTRAP_SEED)
    names = (
        "macro_auroc",
        "macro_auprc",
        "macro_f1",
        "micro_f1",
        "macro_precision",
        "macro_recall",
    )
    samples = {name: np.empty(resamples, dtype=np.float64) for name in names}
    for iteration in range(resamples):
        selected = rng.integers(0, len(unique_patients), size=len(unique_patients))
        patient_weights = np.bincount(selected, minlength=len(unique_patients))
        values, _ = _metrics(
            labels, probabilities, thresholds, patient_weights[patient_index]
        )
        for name in names:
            samples[name][iteration] = values[name]
    return {
        name: {
            "low": float(np.nanpercentile(values, 2.5)),
            "high": float(np.nanpercentile(values, 97.5)),
        }
        for name, values in samples.items()
    }


def _dataset_characteristics(frame):
    positive_labels_per_image = frame[LABELS].sum(axis=1)
    details = {}
    for part in ("train", "val", "test"):
        subset = frame.loc[frame["partition"] == part]
        views = subset["View Position"] if "View Position" in subset else None
        details[part] = {
            "images": int(len(subset)),
            "patients": int(subset["Patient ID"].nunique()),
            "mean_findings_per_image": float(
                positive_labels_per_image.loc[subset.index].mean()
            ),
            "pa_view_fraction": float(views.eq("PA").mean()) if views is not None else None,
            "prevalence": {name: float(subset[name].mean()) for name in LABELS},
        }
    return details


def _autocast(device):
    return torch.autocast(
        device_type=device.type,
        dtype=torch.float16,
        enabled=device.type == "cuda",
    )


def _clip_grad_norm_fp32(parameters, max_norm):
    parameters = [parameter for parameter in parameters if parameter.grad is not None]
    if not parameters:
        return torch.zeros((), dtype=torch.float32)
    norms = [
        torch.linalg.vector_norm(parameter.grad, ord=2, dtype=torch.float32)
        for parameter in parameters
    ]
    total_norm = torch.linalg.vector_norm(torch.stack(norms), ord=2)
    if torch.isfinite(total_norm):
        scale = torch.clamp(max_norm / (total_norm + 1e-6), max=1.0)
        for parameter in parameters:
            parameter.grad.mul_(scale.to(dtype=parameter.grad.dtype))
    return total_norm


def _scaler_step(scaler, optimizer, grad_norm):
    scale_before = scaler.get_scale()
    scaler.step(optimizer)
    scaler.update()
    stepped = scaler.get_scale() >= scale_before
    norm_is_finite = bool(torch.isfinite(grad_norm))
    if stepped and not norm_is_finite:
        raise FloatingPointError("GradScaler did not skip a step with non-finite gradients")
    if not stepped and norm_is_finite:
        raise FloatingPointError("GradScaler skipped a step despite a finite gradient norm")
    return stepped


def measure_latency(model, model_key, device):
    if device.type != "cuda":
        raise RuntimeError("Protocol latency measurement requires a CUDA GPU.")
    cfg = MODEL_CONFIGS[model_key]
    model.eval()
    results = {
        "model_id": cfg["model_id"],
        "device": torch.cuda.get_device_name(device),
        "input_size": cfg["input_size"],
        "precision": "fp16 autocast",
        "tta": "none",
        "warmup_iterations": 50,
        "timed_iterations_per_repeat": 200,
        "repeats": 3,
        "batch_results": {},
    }
    with torch.inference_mode():
        for batch_size in (1, 32):
            inputs = torch.zeros(
                batch_size, 3, cfg["input_size"], cfg["input_size"], device=device
            )
            repeat_medians = []
            for _ in range(3):
                for _ in range(50):
                    with _autocast(device):
                        model(inputs)
                torch.cuda.synchronize(device)
                starts = [torch.cuda.Event(enable_timing=True) for _ in range(200)]
                ends = [torch.cuda.Event(enable_timing=True) for _ in range(200)]
                for start, end in zip(starts, ends):
                    start.record()
                    with _autocast(device):
                        model(inputs)
                    end.record()
                torch.cuda.synchronize(device)
                milliseconds = np.asarray(
                    [start.elapsed_time(end) for start, end in zip(starts, ends)],
                    dtype=np.float64,
                )
                repeat_medians.append(float(np.median(milliseconds)))
            median_batch_ms = float(np.median(repeat_medians))
            results["batch_results"][str(batch_size)] = {
                "repeat_median_batch_ms": repeat_medians,
                "median_batch_ms": median_batch_ms,
                "median_ms_per_image": median_batch_ms / batch_size,
            }
            del inputs, starts, ends
            torch.cuda.empty_cache()
    return results


def train_model(model_key, args):
    cfg = MODEL_CONFIGS[model_key]
    seed_everything()
    if not torch.cuda.is_available():
        raise RuntimeError("Enable a Kaggle GPU accelerator before training.")
    device = torch.device("cuda")
    torch.backends.cuda.matmul.allow_tf32 = False
    frame, split_hash, split_frame = load_protocol_data(args.input_dir, args.data_root)
    output_root = Path(args.output_dir).expanduser().resolve()
    output_root.mkdir(parents=True, exist_ok=True)
    split_frame.to_csv(output_root / "split_NIH_OFFICIAL_PV.csv", index=False, lineterminator="\n")
    run_name = f"NIH_CXR14__NIH_OFFICIAL_PV__{cfg['model_id']}__s{SEED}"
    run_dir = output_root / "runs" / run_name
    run_dir.mkdir(parents=True, exist_ok=True)
    config = {
        "model_id": cfg["model_id"],
        "hub_id": cfg["hub_id"],
        "revision": cfg["revision"],
        "dataset_id": "NIH_CXR14",
        "split_id": "NIH_OFFICIAL_PV",
        "protocol_version": "2.3.0",
        "split_sha256": split_hash,
        "seed": SEED,
        "label_order": LABELS,
        "img_size": cfg["input_size"],
        "batch_size": cfg["batch_size"],
        "warmup_epochs": cfg["warmup_epochs"],
        "finetune_epochs": cfg["finetune_epochs"],
        "patience": 2,
        "lr_head_warmup": cfg["lr_head_warmup"],
        "lr_head": cfg["lr_head"],
        "lr_backbone": cfg["lr_backbone"],
        "weight_decay": 0.05,
        "dropout": 0.2,
        "pooling": cfg["pooling"],
        "preprocessing": cfg["preprocess"],
        "asl": {"gamma_neg": 4, "gamma_pos": 1, "clip": 0.05},
        "augmentation": {
            "random_resized_crop_scale": [0.85, 1.0],
            "random_resized_crop_ratio": [0.9, 1.1],
            "horizontal_flip_p": 0.5,
            "rotation_degrees": 7,
            "rotation_p": 0.5,
            "brightness": 0.1,
            "contrast": 0.1,
        },
        "ema_decay": cfg["ema_decay"],
        "amp": "fp16 on CUDA",
        "tta": "none",
        "device": str(device),
        "pretrained": True,
        "distillation": False,
    }
    (run_dir / "config.json").write_text(json.dumps(config, indent=2), encoding="utf-8")

    train_frame = frame.loc[frame["partition"] == "train"].reset_index(drop=True)
    val_frame = frame.loc[frame["partition"] == "val"].reset_index(drop=True)
    test_frame = frame.loc[frame["partition"] == "test"].reset_index(drop=True)
    model, processors = build_model(model_key)
    model = model.to(device)
    train_loader = _loader(train_frame, model_key, processors, True, args.workers)
    val_loader = _loader(val_frame, model_key, processors, False, args.workers)
    loss_fn = AsymmetricLoss()
    scaler = torch.amp.GradScaler("cuda", enabled=device.type == "cuda")
    ema = ModelEMA(model, cfg["ema_decay"]) if cfg["ema_decay"] is not None else None
    evaluated_weights = "EMA" if ema else "model"
    eval_model = ema.model if ema else model

    best_score = -np.inf
    best_epoch = 0
    best_path = run_dir / "checkpoint_best.pt"
    training_log = []
    bad_finetune_epochs = 0
    epoch_number = 0
    stopped_early = False

    for phase, epochs in (
        ("warmup", cfg["warmup_epochs"]),
        ("finetune", cfg["finetune_epochs"]),
    ):
        optimizer = _phase_optimizer(model, phase)
        scheduler = _schedule(optimizer, len(train_loader) * epochs)
        if phase == "finetune":
            bad_finetune_epochs = 0
        for phase_epoch in range(epochs):
            epoch_number += 1
            epoch_start = time.time()
            model.train()
            if phase == "warmup":
                model.backbone.eval()
            running_loss = 0.0
            samples = 0
            optimizer_steps = 0
            for images, targets in train_loader:
                images = images.to(device, non_blocking=True)
                targets = targets.to(device, non_blocking=True)
                optimizer.zero_grad(set_to_none=True)
                with _autocast(device):
                    logits = model(images)
                    loss = loss_fn(logits, targets)
                if not torch.isfinite(loss):
                    raise FloatingPointError(f"Non-finite training loss at epoch {epoch_number}")
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                grad_norm = _clip_grad_norm_fp32(model.parameters(), 1.0)
                running_loss += float(loss.detach()) * len(images)
                samples += len(images)
                if not _scaler_step(scaler, optimizer, grad_norm):
                    continue
                scheduler.step()
                if ema:
                    ema.update(model)
                optimizer_steps += 1

            if optimizer_steps == 0:
                raise FloatingPointError(
                    f"AMP skipped every optimizer step in epoch {epoch_number}"
                )
            val_logits, val_labels = _predict(eval_model, val_loader, device)
            score = _macro_auroc(val_labels, val_logits)
            if score > best_score:
                best_score = score
                best_epoch = epoch_number
                torch.save(
                    {
                        "model_state": eval_model.state_dict(),
                        "model_id": cfg["model_id"],
                        "split_sha256": split_hash,
                        "best_epoch": best_epoch,
                        "best_val_macro_auroc": best_score,
                        "evaluated_weights": evaluated_weights,
                    },
                    best_path,
                )
                bad_finetune_epochs = 0
            elif phase == "finetune":
                bad_finetune_epochs += 1
            training_log.append(
                {
                    "epoch": epoch_number,
                    "phase": phase,
                    "phase_epoch": phase_epoch + 1,
                    "train_loss": running_loss / max(1, samples),
                    "val_macro_auroc": score,
                    "best_val_macro_auroc": best_score,
                    "epoch_seconds": time.time() - epoch_start,
                }
            )
            print(
                f"{cfg['model_id']} epoch {epoch_number}: "
                f"loss={running_loss / max(1, samples):.5f} "
                f"val_macro_auroc={score:.5f} best={best_score:.5f}",
                flush=True,
            )
            if phase == "finetune" and bad_finetune_epochs >= 2:
                stopped_early = True
                break
        if stopped_early:
            break

    del optimizer, scheduler
    if ema:
        del model, ema
        gc.collect()
        torch.cuda.empty_cache()
    with (run_dir / "train_log.csv").open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=training_log[0].keys())
        writer.writeheader()
        writer.writerows(training_log)
    checkpoint = torch.load(best_path, map_location=device, weights_only=False)
    eval_model.load_state_dict(checkpoint["model_state"])
    eval_model.eval()

    val_logits, val_labels = _predict(eval_model, val_loader, device)
    calibration = _fit_platt(val_labels, val_logits)
    val_probabilities = _calibrate(val_logits, calibration)
    thresholds, fallback_labels = _fit_thresholds(val_labels, val_probabilities)
    (run_dir / "calibration.json").write_text(
        json.dumps(
            {
                "method": "per-class Platt scaling; validation only",
                "label_order": LABELS,
                "parameters": [{"a": float(a), "b": float(b)} for a, b in calibration],
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    (run_dir / "thresholds.json").write_text(
        json.dumps(
            {
                "method": "per-class validation F1 maximization; ties choose smallest threshold",
                "label_order": LABELS,
                "thresholds": {
                    name: float(thresholds[index]) for index, name in enumerate(LABELS)
                },
                "f1_zero_fallback_labels": fallback_labels,
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    timing = measure_latency(eval_model, model_key, device)
    (run_dir / f"timing_{cfg['model_id']}.json").write_text(
        json.dumps(timing, indent=2), encoding="utf-8"
    )
    test_loader = _loader(test_frame, model_key, processors, False, args.workers)
    test_logits, test_labels = _predict(eval_model, test_loader, device)
    _save_logits(run_dir / "logits_val.npz", val_frame, val_logits, val_labels, split_hash)
    _save_logits(run_dir / "logits_test.npz", test_frame, test_logits, test_labels, split_hash)
    metrics, per_class = _metrics(
        test_labels, _calibrate(test_logits, calibration), thresholds
    )
    results = {
        "model_id": cfg["model_id"],
        "split_id": "NIH_OFFICIAL_PV",
        "split_sha256": split_hash,
        "seed": SEED,
        "tta": "none",
        "calibration": "per-class Platt scaling fitted on validation",
        "thresholds": "per-class F1-maximizing validation thresholds",
        "best_epoch": best_epoch,
        "best_val_macro_auroc": best_score,
        "epochs_trained": len(training_log),
        "stopped_early": stopped_early,
        "evaluated_weights": evaluated_weights,
        "test_metrics": metrics,
        "test_metrics_95pct_patient_bootstrap_ci": _patient_bootstrap(
            test_labels,
            _calibrate(test_logits, calibration),
            thresholds,
            test_frame["Patient ID"].to_numpy(),
            resamples=args.bootstrap_resamples,
        ),
        "per_class_test_metrics": per_class,
        "bootstrap": {
            "method": "patient-level percentile bootstrap",
            "resamples": args.bootstrap_resamples,
            "seed": BOOTSTRAP_SEED,
        },
        "dataset_characteristics": _dataset_characteristics(frame),
    }
    (run_dir / "test_metrics.json").write_text(
        json.dumps(results, indent=2, allow_nan=True), encoding="utf-8"
    )
    print(f"Run complete: {run_dir}", flush=True)


def smoke_test(args):
    if not torch.cuda.is_available():
        raise RuntimeError("Enable a Kaggle GPU accelerator before running this smoke test.")
    frame, split_hash, _ = load_protocol_data(args.input_dir, args.data_root)
    train_frame = frame.loc[frame["partition"] == "train"].head(1).reset_index(drop=True)
    device = torch.device("cuda")
    print(
        f"Protocol checks passed: split={split_hash}; GPU={torch.cuda.get_device_name(0)}",
        flush=True,
    )
    for model_key, cfg in MODEL_CONFIGS.items():
        seed_everything()
        model, processors = build_model(model_key)
        model = model.to(device)
        dataset = CXRDataset(train_frame, model_key, processors, training=True)
        images = dataset[0][0].unsqueeze(0).to(device)
        if tuple(images.shape) != (1, 3, cfg["input_size"], cfg["input_size"]):
            raise AssertionError(f"{model_key}: incorrect input shape {tuple(images.shape)}")
        targets = torch.from_numpy(dataset.labels[:1]).to(device)
        model.train()
        ema = ModelEMA(model, cfg["ema_decay"]) if cfg["ema_decay"] is not None else None
        optimizer = _phase_optimizer(model, "finetune")
        scaler = torch.amp.GradScaler("cuda", enabled=True)
        for attempt in range(18):
            optimizer.zero_grad(set_to_none=True)
            with _autocast(device):
                logits = model(images)
                loss = AsymmetricLoss()(logits, targets)
            if logits.shape != (1, len(LABELS)):
                raise AssertionError(f"{model_key}: incorrect logits shape {tuple(logits.shape)}")
            if not torch.isfinite(logits).all() or not torch.isfinite(loss):
                raise FloatingPointError(f"{model_key}: non-finite forward pass or loss")
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            grad_norm = _clip_grad_norm_fp32(model.parameters(), 1.0)
            if _scaler_step(scaler, optimizer, grad_norm):
                break
        else:
            raise FloatingPointError(
                f"{model_key}: AMP could not produce finite gradients after 18 retries"
            )
        if not any(
            parameter.grad is not None and torch.isfinite(parameter.grad).all()
            for parameter in model.head.parameters()
        ):
            raise FloatingPointError(f"{model_key}: classifier head has no finite gradients")
        if not any(
            parameter.grad is not None and torch.isfinite(parameter.grad).all()
            for parameter in model.backbone.parameters()
            if parameter.requires_grad
        ):
            raise FloatingPointError(f"{model_key}: fine-tuned backbone has no finite gradients")
        if ema:
            ema.update(model)
        eval_dataset = CXRDataset(train_frame, model_key, processors, training=False)
        eval_images = eval_dataset[0][0].unsqueeze(0).to(device)
        if tuple(eval_images.shape) != (1, 3, cfg["input_size"], cfg["input_size"]):
            raise AssertionError(
                f"{model_key}: incorrect evaluation input shape {tuple(eval_images.shape)}"
            )
        model.eval()
        with torch.inference_mode(), _autocast(device):
            eval_logits = model(eval_images)
        if eval_logits.shape != (1, len(LABELS)) or not torch.isfinite(eval_logits).all():
            raise FloatingPointError(f"{model_key}: invalid evaluation preprocessing/forward pass")
        print(
            f"{model_key}: pretrained load, train/eval transforms, forward, ASL, "
            f"backward, and optimizer step passed (AMP retries: {attempt})",
            flush=True,
        )
        del model, optimizer, images, eval_images, eval_logits, targets
        del dataset, eval_dataset, processors, scaler, ema
        gc.collect()
        torch.cuda.empty_cache()
    print("All three single-model transformer smoke tests passed.", flush=True)


def train_entry(model_key):
    class Args:
        input_dir = INPUT_DIR
        data_root = DATA_ROOT
        output_dir = OUTPUT_DIR
        workers = NUM_WORKERS
        bootstrap_resamples = BOOTSTRAP_RESAMPLES

    if NUM_WORKERS < 0:
        raise ValueError("NUM_WORKERS must be zero or greater")
    if BOOTSTRAP_RESAMPLES < 1:
        raise ValueError("BOOTSTRAP_RESAMPLES must be greater than zero")
    args = Args()
    args.output_dir = str(Path(OUTPUT_DIR) / model_key)
    train_model(model_key, args)


def smoke_entry():
    class Args:
        input_dir = INPUT_DIR
        data_root = DATA_ROOT
        workers = NUM_WORKERS

    smoke_test(Args())
