"""Box utilities. All boxes are normalized xyxy in [0, 1] unless stated otherwise."""

from __future__ import annotations

import numpy as np


def iou_matrix(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Pairwise IoU between boxes a (N,4) and b (M,4), xyxy."""
    a = np.asarray(a, dtype=float).reshape(-1, 4)
    b = np.asarray(b, dtype=float).reshape(-1, 4)
    if len(a) == 0 or len(b) == 0:
        return np.zeros((len(a), len(b)))
    x1 = np.maximum(a[:, None, 0], b[None, :, 0])
    y1 = np.maximum(a[:, None, 1], b[None, :, 1])
    x2 = np.minimum(a[:, None, 2], b[None, :, 2])
    y2 = np.minimum(a[:, None, 3], b[None, :, 3])
    inter = np.clip(x2 - x1, 0, None) * np.clip(y2 - y1, 0, None)
    area_a = (a[:, 2] - a[:, 0]) * (a[:, 3] - a[:, 1])
    area_b = (b[:, 2] - b[:, 0]) * (b[:, 3] - b[:, 1])
    union = area_a[:, None] + area_b[None, :] - inter
    return np.where(union > 0, inter / np.maximum(union, 1e-12), 0.0)


def area(boxes: np.ndarray) -> np.ndarray:
    boxes = np.asarray(boxes, dtype=float).reshape(-1, 4)
    return np.clip(boxes[:, 2] - boxes[:, 0], 0, None) * np.clip(boxes[:, 3] - boxes[:, 1], 0, None)


def nms_per_class(dets: np.ndarray, iou_thr: float) -> np.ndarray:
    """Class-wise greedy NMS. dets: (N,6) = x1,y1,x2,y2,conf,cls. Returns kept rows, conf-desc."""
    dets = np.asarray(dets, dtype=float).reshape(-1, 6)
    keep = []
    for c in np.unique(dets[:, 5]):
        d = dets[dets[:, 5] == c]
        d = d[np.argsort(-d[:, 4])]
        while len(d):
            keep.append(d[0])
            if len(d) == 1:
                break
            ious = iou_matrix(d[:1, :4], d[1:, :4])[0]
            d = d[1:][ious < iou_thr]
    if not keep:
        return np.zeros((0, 6))
    out = np.stack(keep)
    return out[np.argsort(-out[:, 4])]


def yolo_to_xyxy(rows: np.ndarray) -> np.ndarray:
    """YOLO label rows (N,5) = cls,cx,cy,w,h -> (N,5) = cls,x1,y1,x2,y2."""
    rows = np.asarray(rows, dtype=float).reshape(-1, 5)
    cls, cx, cy, w, h = rows.T
    return np.stack([cls, cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2], axis=1)


def xyxy_to_yolo(boxes: np.ndarray, cls: np.ndarray) -> np.ndarray:
    """(N,4) xyxy + (N,) cls -> (N,5) YOLO rows, clipped to the image."""
    b = np.clip(np.asarray(boxes, dtype=float).reshape(-1, 4), 0.0, 1.0)
    cx = (b[:, 0] + b[:, 2]) / 2
    cy = (b[:, 1] + b[:, 3]) / 2
    w = b[:, 2] - b[:, 0]
    h = b[:, 3] - b[:, 1]
    return np.stack([np.asarray(cls, dtype=float).reshape(-1), cx, cy, w, h], axis=1)


def unflip_x(dets: np.ndarray) -> np.ndarray:
    """Map boxes predicted on a horizontally flipped image back to the original (normalized coords)."""
    d = np.asarray(dets, dtype=float).reshape(-1, 6).copy()
    x1 = d[:, 0].copy()
    d[:, 0] = 1.0 - d[:, 2]
    d[:, 2] = 1.0 - x1
    return d


def fuse_tta(views: list[np.ndarray], iou_thr: float = 0.55) -> np.ndarray:
    """Fuse detections from several TTA views into one set (weighted box fusion, simplified).

    Boxes of the same class with IoU >= iou_thr across views form a cluster. The fused box is
    the confidence-weighted mean; the fused confidence is the SUM of the best confidence per view
    divided by the number of views, so an object seen in only one view gets its confidence
    diluted. For pseudo-labeling this is the point: agreement between views = reliability.
    """
    n_views = len(views)
    if n_views == 0:
        return np.zeros((0, 6))
    tagged = [np.c_[np.asarray(v, dtype=float).reshape(-1, 6), np.full(len(v), i)] for i, v in enumerate(views)]
    allb = np.concatenate(tagged) if tagged else np.zeros((0, 7))
    out = []
    for c in np.unique(allb[:, 5]):
        d = allb[allb[:, 5] == c]
        d = d[np.argsort(-d[:, 4])]
        used = np.zeros(len(d), dtype=bool)
        for i in range(len(d)):
            if used[i]:
                continue
            ious = iou_matrix(d[i:i + 1, :4], d[:, :4])[0]
            members = np.where(~used & (ious >= iou_thr))[0]
            used[members] = True
            m = d[members]
            w = m[:, 4:5]
            box = (m[:, :4] * w).sum(0) / w.sum()
            best_per_view = [m[m[:, 6] == v, 4].max() for v in np.unique(m[:, 6])]
            out.append([*box, sum(best_per_view) / n_views, c])
    if not out:
        return np.zeros((0, 6))
    out = np.asarray(out)
    return out[np.argsort(-out[:, 4])]
