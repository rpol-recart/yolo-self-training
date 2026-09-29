"""Per-class confidence thresholds chosen on the human-labeled val set."""

from __future__ import annotations

import numpy as np

from .boxes import iou_matrix


def match_predictions(preds: dict[str, np.ndarray], gts: dict[str, np.ndarray], iou_thr: float):
    """Greedy per-image, per-class matching.

    preds: image -> (N,6) x1,y1,x2,y2,conf,cls (normalized)
    gts:   image -> (M,5) cls,x1,y1,x2,y2 (normalized)
    Returns (scored, n_gt): scored[cls] = list of (conf, is_tp); n_gt[cls] = number of GT boxes.
    """
    scored: dict[int, list] = {}
    n_gt: dict[int, int] = {}
    for img in set(preds) | set(gts):
        p = np.asarray(preds.get(img, np.zeros((0, 6)))).reshape(-1, 6)
        g = np.asarray(gts.get(img, np.zeros((0, 5)))).reshape(-1, 5)
        for c in np.unique(g[:, 0]).astype(int):
            n_gt[c] = n_gt.get(c, 0) + int((g[:, 0] == c).sum())
        for c in np.unique(p[:, 5]).astype(int):
            pc = p[p[:, 5] == c]
            pc = pc[np.argsort(-pc[:, 4])]
            gc = g[g[:, 0] == c][:, 1:5]
            used = np.zeros(len(gc), dtype=bool)
            ious = iou_matrix(pc[:, :4], gc)
            for i in range(len(pc)):
                tp = False
                if len(gc):
                    cand = np.where(~used, ious[i], -1.0)
                    j = int(np.argmax(cand))
                    if cand[j] >= iou_thr:
                        used[j] = True
                        tp = True
                scored.setdefault(c, []).append((float(pc[i, 4]), tp))
    return scored, n_gt


def threshold_for_precision(scored: list, target: float) -> tuple[float | None, float, float]:
    """Lowest confidence t such that precision of {conf >= t} >= target.

    Returns (t, precision_at_t, tp_count_at_t); t is None when the target is never reached.
    """
    if not scored:
        return None, 0.0, 0.0
    arr = np.array(sorted(scored, key=lambda x: -x[0]), dtype=float)
    conf, tp = arr[:, 0], arr[:, 1]
    cum_tp = np.cumsum(tp)
    prec = cum_tp / np.arange(1, len(tp) + 1)
    # only evaluate at the last index of each run of equal confidences
    last_of_run = np.r_[conf[1:] != conf[:-1], True]
    ok = np.where((prec >= target) & last_of_run)[0]
    if len(ok) == 0:
        return None, 0.0, 0.0
    k = ok[-1]  # deepest cut that still meets the target = lowest threshold
    return float(conf[k]), float(prec[k]), float(cum_tp[k])


def calibrate(preds, gts, n_classes: int, cal: dict, conf_floor: float) -> dict[int, dict]:
    """Returns {cls: {t_high, t_low, precision, recall, support, fallback}}."""
    scored, n_gt = match_predictions(preds, gts, cal["iou"])
    out = {}
    for c in range(n_classes):
        support = n_gt.get(c, 0)
        s = scored.get(c, [])
        t_high, prec, tp = threshold_for_precision(s, cal["target_precision"])
        t_low, _, _ = threshold_for_precision(s, cal["low_precision"])
        if t_low is not None and s and t_low <= min(conf for conf, _ in s):
            # every val prediction of this class clears low_precision: nothing tells us that
            # lower-confidence boxes are background, so the grey zone extends down to the floor
            t_low = conf_floor
        fallback = support < cal["min_support"] or t_high is None
        if fallback:
            t_high, t_low = cal["fallback_high"], cal["fallback_low"]
            prec, tp = float("nan"), float("nan")
        t_low = conf_floor if t_low is None else t_low
        t_low = min(max(t_low, conf_floor), t_high)
        out[c] = {
            "t_high": float(t_high),
            "t_low": float(t_low),
            "precision": prec,
            "recall": (tp / support) if support and not fallback else float("nan"),
            "support": support,
            "fallback": bool(fallback),
        }
    return out
