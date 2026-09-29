"""Turn raw teacher predictions into pseudo-labels, background images and a review queue."""

from __future__ import annotations

import re
from dataclasses import dataclass, field

import numpy as np

from .boxes import area, iou_matrix


@dataclass
class ImageDecision:
    path: str
    group: str
    confident: np.ndarray = field(default_factory=lambda: np.zeros((0, 6)))  # x1,y1,x2,y2,conf,cls
    uncertain: np.ndarray = field(default_factory=lambda: np.zeros((0, 6)))
    status: str = "background"  # pseudo | uncertain | background
    uncertainty: float = 0.0


def split_by_confidence(dets: np.ndarray, thresholds: dict, min_area: float):
    """-> (confident, uncertain). Boxes under min_area or below t_low are dropped."""
    dets = np.asarray(dets, dtype=float).reshape(-1, 6)
    dets = dets[area(dets[:, :4]) >= min_area]
    if not len(dets):
        return np.zeros((0, 6)), np.zeros((0, 6))
    cls = dets[:, 5].astype(int)
    hi = np.array([thresholds[c]["t_high"] for c in cls])
    lo = np.array([thresholds[c]["t_low"] for c in cls])
    conf = dets[:, 4]
    return dets[conf >= hi], dets[(conf >= lo) & (conf < hi)]


def temporal_consistency(records: dict[str, np.ndarray], regex: str, window: int,
                         iou_thr: float, min_support: int) -> tuple[dict, dict]:
    """Keep a box only if >= min_support neighbour frames (±window) of the same sequence
    contain a same-class box with IoU >= iou_thr.

    records: path -> (N,6) confident boxes. Paths that do not match the regex are passed through.
    Returns (kept, removed), both path -> (N,6).
    """
    seqs: dict[str, dict[int, str]] = {}
    for p in records:
        m = re.search(regex, p)
        if m:
            seqs.setdefault(m.group("seq"), {})[int(m.group("idx"))] = p
    kept, removed = dict(records), {}
    for frames in seqs.values():
        idxs = sorted(frames)
        max_gap = window * _step(idxs)  # neighbours across a gap in the sequence do not count
        for pos, i in enumerate(idxs):
            p = frames[i]
            boxes = records[p]
            if not len(boxes):
                continue
            around = idxs[max(0, pos - window): pos + window + 1]
            neigh = [frames[j] for j in around if j != i and abs(j - i) <= max_gap]
            support = np.zeros(len(boxes), dtype=int)
            for q in neigh:
                nb = records[q]
                if not len(nb):
                    continue
                ious = iou_matrix(boxes[:, :4], nb[:, :4])
                same = boxes[:, 5][:, None] == nb[:, 5][None, :]
                support += ((ious >= iou_thr) & same).any(axis=1)
            ok = support >= min_support
            kept[p] = boxes[ok]
            if (~ok).any():
                removed[p] = boxes[~ok]
    return kept, removed


def _step(idxs: list[int]) -> int:
    """Typical index step between consecutive sampled frames (frames may be numbered 0, 5, 10...)."""
    if len(idxs) < 2:
        return 1
    return max(1, int(np.median(np.diff(idxs))))


def decide(preds: dict[str, np.ndarray], groups: dict[str, str], thresholds: dict, fcfg: dict) -> list[ImageDecision]:
    conf, unc = {}, {}
    for p, d in preds.items():
        conf[p], unc[p] = split_by_confidence(d, thresholds, fcfg["min_box_area"])
    t = fcfg["temporal"]
    if t["enabled"]:
        conf, removed = temporal_consistency(conf, t["regex"], t["window"], t["iou"], t["min_support"])
        for p, r in removed.items():  # inconsistent "confident" boxes are not background either
            unc[p] = np.concatenate([unc[p], r]) if len(unc[p]) else r

    out = []
    for p in preds:
        c, u = conf[p], unc[p]
        d = ImageDecision(path=p, group=groups.get(p, "?"), confident=c, uncertain=u)
        if len(u):
            d.uncertainty = float(_uncertainty(u, thresholds))
        if len(u) and fcfg["uncertain_policy"] == "drop_image":
            d.status = "uncertain"
        elif len(c):
            d.status = "pseudo"
        elif len(u):
            d.status = "uncertain"
        else:
            d.status = "background"
        out.append(d)
    return out


def _uncertainty(u: np.ndarray, thresholds: dict) -> float:
    """Sum over grey-zone boxes of closeness to the middle of [t_low, t_high] (1 = middle)."""
    s = 0.0
    for b in u:
        t = thresholds[int(b[5])]
        mid, half = (t["t_high"] + t["t_low"]) / 2, max((t["t_high"] - t["t_low"]) / 2, 1e-6)
        s += max(0.0, 1.0 - abs(b[4] - mid) / half)
    return s


def review_queue(decisions: list[ImageDecision], max_items: int) -> list[ImageDecision]:
    """Most uncertain images first, round-robin across groups so one group cannot fill the queue."""
    by_group: dict[str, list] = {}
    for d in decisions:
        if len(d.uncertain):
            by_group.setdefault(d.group, []).append(d)
    for g in by_group:
        by_group[g].sort(key=lambda d: -d.uncertainty)
    queue = []
    while len(queue) < max_items and any(by_group.values()):
        for g in sorted(by_group, key=lambda g: -(by_group[g][0].uncertainty if by_group[g] else -1)):
            if by_group[g] and len(queue) < max_items:
                queue.append(by_group[g].pop(0))
    return queue
