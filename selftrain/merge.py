"""Assemble a round's training set: GT-train + selected pseudo-labeled + background images."""

from __future__ import annotations

from pathlib import Path

import numpy as np

from .dataset import link_image, label_path_for, read_labels, write_data_yaml, write_labels, write_list
from .filter import ImageDecision


def class_counts(label_rows: list[np.ndarray], n_classes: int) -> np.ndarray:
    counts = np.zeros(n_classes, dtype=int)
    for r in label_rows:
        if len(r):
            counts += np.bincount(np.asarray(r)[:, 0].astype(int), minlength=n_classes)[:n_classes]
    return counts


def select_pseudo(cands: list[ImageDecision], gt_counts: np.ndarray, max_images: int,
                  max_per_group: int | None, seed: int) -> list[ImageDecision]:
    """Rare-class-first selection under a total and a per-group cap.

    Each image is scored by its rarest class, rarity = 1 / (GT count + pseudo count available).
    """
    if max_images <= 0 or not cands:
        return []
    n = len(gt_counts)
    avail = class_counts([c.confident[:, [5, 0, 1, 2, 3]] for c in cands], n)
    rarity = 1.0 / (gt_counts + avail + 1.0)
    rng = np.random.default_rng(seed)
    tiebreak = rng.random(len(cands))
    score = np.array([rarity[c.confident[:, 5].astype(int)].max() for c in cands])
    order = np.lexsort((tiebreak, -score))
    picked, per_group = [], {}
    for i in order:
        c = cands[i]
        if max_per_group is not None and per_group.get(c.group, 0) >= max_per_group:
            continue
        picked.append(c)
        per_group[c.group] = per_group.get(c.group, 0) + 1
        if len(picked) >= max_images:
            break
    return picked


def select_background(cands: list[ImageDecision], n: int, max_per_group: int | None, seed: int) -> list[ImageDecision]:
    if n <= 0 or not cands:
        return []
    rng = np.random.default_rng(seed + 1)
    picked, per_group = [], {}
    for i in rng.permutation(len(cands)):
        c = cands[i]
        if max_per_group is not None and per_group.get(c.group, 0) >= max_per_group:
            continue
        picked.append(c)
        per_group[c.group] = per_group.get(c.group, 0) + 1
        if len(picked) >= n:
            break
    return picked


def build_round_dataset(round_dir: str | Path, gt_train: list[str], val_list: str | Path,
                        decisions: list[ImageDecision], names: list[str], mcfg: dict, seed: int) -> dict:
    """Writes round_dir/data/{images,labels}, train.txt and data.yaml. Returns stats."""
    round_dir = Path(round_dir)
    img_dir = round_dir / "data" / "images"
    gt_counts = class_counts([read_labels(label_path_for(p)) for p in gt_train], len(names))

    pseudo_c = [d for d in decisions if d.status == "pseudo"]
    bg_c = [d for d in decisions if d.status == "background"]
    max_pseudo = int(round(mcfg["max_pseudo_ratio"] * len(gt_train)))
    n_bg = int(round(mcfg["background_ratio"] * len(gt_train)))
    pseudo = select_pseudo(pseudo_c, gt_counts, max_pseudo, mcfg["max_per_group"], seed)
    background = select_background(bg_c, n_bg, mcfg["max_per_group"], seed)

    train = list(gt_train)
    for d in pseudo:
        dst = link_image(d.path, img_dir)
        write_labels(label_path_for(dst), d.confident[:, :4], d.confident[:, 5])
        train.append(str(dst))
    for d in background:
        dst = link_image(d.path, img_dir)
        write_labels(label_path_for(dst), np.zeros((0, 4)), np.zeros(0))  # empty file = negative image
        train.append(str(dst))

    train_txt = write_list(round_dir / "train.txt", train)
    data_yaml = write_data_yaml(round_dir / "data.yaml", train_txt, val_list, names)
    pseudo_counts = class_counts([d.confident[:, [5, 0, 1, 2, 3]] for d in pseudo], len(names))
    return {
        "data_yaml": str(data_yaml),
        "gt_images": len(gt_train),
        "pseudo_images": len(pseudo),
        "background_images": len(background),
        "candidates": {s: sum(d.status == s for d in decisions) for s in ("pseudo", "uncertain", "background")},
        "gt_boxes_per_class": dict(zip(names, gt_counts.tolist())),
        "pseudo_boxes_per_class": dict(zip(names, pseudo_counts.tolist())),
    }
