"""Config loading: user YAML deep-merged over defaults."""

from __future__ import annotations

import copy
from pathlib import Path

import yaml

DEFAULTS: dict = {
    "names": [],
    "workdir": "runs/selftrain",
    "seed": 0,
    "data": {
        "gt_images": None,          # dir with labeled images; labels live in the sibling .../labels/... tree
        "unlabeled_images": None,   # dir with the unlabeled pool
        "extensions": [".jpg", ".jpeg", ".png", ".bmp", ".webp"],
    },
    "groups": {
        "regex": r"^(?P<group>[^/]+)/",  # first path component relative to the images dir
        "csv": None,                       # optional CSV path,group (overrides regex)
    },
    "split": {
        "val_fraction": 0.15,
        "test_fraction": 0.15,
        "exclude_eval_groups_from_pool": True,
    },
    "teacher": {
        "conf_floor": 0.05,
        "nms_iou": 0.6,
        "tta": True,               # own TTA (flip + scales), works for NMS-free YOLO26 too
        "tta_flip": True,
        "tta_scales": [0.83, 1.17],
        "tta_fuse_iou": 0.55,
        "max_det": 300,
        "chunk": 64,               # images read into memory at once
        "imgsz": 640,
        "batch": 16,
        "device": None,
    },
    "calibration": {
        "iou": 0.5,
        "target_precision": 0.90,
        "low_precision": 0.50,
        "min_support": 20,
        "fallback_high": 0.6,
        "fallback_low": 0.25,
    },
    "filter": {
        "uncertain_policy": "drop_image",  # drop_image | keep_confident
        "min_box_area": 0.0005,
        "temporal": {
            "enabled": False,
            "regex": r"(?P<seq>.+)[/_-](?P<idx>\d+)\.[^.]+$",
            "window": 2,
            "iou": 0.3,
            "min_support": 1,
        },
    },
    "merge": {
        "max_pseudo_ratio": 3.0,
        "background_ratio": 0.1,
        "max_per_group": None,
    },
    "review": {
        "max_items": 500,
    },
    "student": {
        "base_weights": "yolo26s.pt",
        "init": "base",  # base | teacher
        # optimizer is pinned: with "auto" ultralytics picks AdamW or MuSGD by iteration count,
        # so round 0 (GT only) and pseudo-label rounds could silently train with different optimizers
        "train_args": {"epochs": 100, "imgsz": 640, "batch": 16, "patience": 30,
                       "optimizer": "MuSGD", "lr0": 0.01, "momentum": 0.9},
    },
    "eval": {
        "min_group_images": 10,
        "imgsz": 640,
        "batch": 16,
    },
    "accept": {
        "metric": "map50_95",
        "min_delta": 0.003,
        "group_tolerance": 0.02,
    },
    "loop": {
        "max_rounds": 4,
        "patience": 2,
    },
}


def deep_merge(base: dict, override: dict) -> dict:
    out = copy.deepcopy(base)
    for k, v in (override or {}).items():
        if isinstance(v, dict) and isinstance(out.get(k), dict):
            out[k] = deep_merge(out[k], v)
        else:
            out[k] = copy.deepcopy(v)
    return out


def load_config(path: str | Path) -> dict:
    with open(path) as f:
        user = yaml.safe_load(f) or {}
    cfg = deep_merge(DEFAULTS, user)
    validate(cfg)
    return cfg


def validate(cfg: dict) -> None:
    if not cfg["names"]:
        raise ValueError("config: `names` (list of class names) is required")
    if not cfg["data"]["gt_images"]:
        raise ValueError("config: `data.gt_images` is required")
    if cfg["filter"]["uncertain_policy"] not in ("drop_image", "keep_confident"):
        raise ValueError("config: filter.uncertain_policy must be drop_image or keep_confident")
    if cfg["student"]["init"] not in ("base", "teacher"):
        raise ValueError("config: student.init must be base or teacher")
    if cfg["student"]["train_args"].get("optimizer", "auto") == "auto":
        import warnings
        warnings.warn("student.train_args.optimizer=auto: the optimizer may differ between rounds; "
                      "pin it (e.g. MuSGD or AdamW) so rounds are comparable")
    s = cfg["split"]
    if s["val_fraction"] <= 0 or s["val_fraction"] + s["test_fraction"] >= 1:
        raise ValueError("config: need 0 < val_fraction and val_fraction + test_fraction < 1")
