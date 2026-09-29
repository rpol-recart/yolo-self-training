"""Thin wrappers over ultralytics: predict, train, evaluate. Imported lazily so the
pure-logic modules and tests work without ultralytics/torch installed."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from .boxes import fuse_tta, nms_per_class, unflip_x
from .dataset import write_data_yaml, write_list


def _yolo(weights: str):
    from ultralytics import YOLO

    return YOLO(weights)


def _views(tcfg: dict) -> list[tuple[int, bool]]:
    """TTA views as (imgsz, hflip). The first view is always the plain one."""
    base = tcfg["imgsz"]
    if not tcfg["tta"]:
        return [(base, False)]
    sizes = [max(32, int(round(base * s / 32)) * 32) for s in tcfg.get("tta_scales", [1.0])]
    views = []
    for sz in dict.fromkeys([base, *sizes]):
        views.append((sz, False))
        if tcfg.get("tta_flip", True):
            views.append((sz, True))
    return views


def _predict_arrays(model, arrays: list, imgsz: int, tcfg: dict) -> list[np.ndarray]:
    out = []
    for r in model.predict(arrays, conf=tcfg["conf_floor"], iou=tcfg["nms_iou"], imgsz=imgsz,
                           augment=False, batch=tcfg["batch"], device=tcfg["device"],
                           max_det=tcfg.get("max_det", 300), stream=True, verbose=False):
        b = r.boxes
        out.append(np.concatenate([b.xyxyn.cpu().numpy(), b.conf.cpu().numpy()[:, None],
                                   b.cls.cpu().numpy()[:, None]], axis=1) if len(b) else np.zeros((0, 6)))
    return out


def predict(weights: str, images: list[str], tcfg: dict, out_jsonl: str | Path) -> dict[str, np.ndarray]:
    """Teacher predictions -> {path: (N,6) normalized x1,y1,x2,y2,conf,cls}. Cached in out_jsonl.

    TTA is done here (flip + scales, fused with fuse_tta) instead of ultralytics' augment=True,
    which end-to-end (NMS-free) models such as YOLO26 silently ignore.
    """
    import cv2

    out_jsonl = Path(out_jsonl)
    if out_jsonl.exists():
        return load_predictions(out_jsonl)
    model = _yolo(weights)
    views = _views(tcfg)
    res = {}
    out_jsonl.parent.mkdir(parents=True, exist_ok=True)
    tmp = out_jsonl.with_suffix(".partial")
    chunk = max(1, int(tcfg.get("chunk", 64)))
    with open(tmp, "w") as f:
        for i in range(0, len(images), chunk):
            paths = images[i:i + chunk]
            arrays = [cv2.imread(p) for p in paths]
            bad = [p for p, a in zip(paths, arrays) if a is None]
            if bad:
                raise IOError(f"cannot read images: {bad[:3]}")
            per_view = []
            for sz, flip in views:
                src = [np.ascontiguousarray(a[:, ::-1]) for a in arrays] if flip else arrays
                dets = _predict_arrays(model, src, sz, tcfg)
                per_view.append([unflip_x(d) if flip else d for d in dets])
            for j, p in enumerate(paths):
                if len(views) == 1:
                    d = nms_per_class(per_view[0][j], tcfg["nms_iou"])
                else:
                    d = fuse_tta([v[j] for v in per_view], tcfg.get("tta_fuse_iou", 0.55))
                    d = d[d[:, 4] >= tcfg["conf_floor"]]
                key = str(Path(p).resolve())  # same key form as the split lists
                res[key] = d
                f.write(json.dumps({"path": key, "dets": np.round(d, 5).tolist()}) + "\n")
    tmp.rename(out_jsonl)
    return res


def load_predictions(path: str | Path) -> dict[str, np.ndarray]:
    out = {}
    with open(path) as f:
        for line in f:
            r = json.loads(line)
            out[r["path"]] = np.asarray(r["dets"], dtype=float).reshape(-1, 6)
    return out


def train(data_yaml: str, init_weights: str, out_dir: str | Path, train_args: dict, seed: int) -> str:
    out_dir = Path(out_dir).resolve()
    model = _yolo(init_weights)
    model.train(data=data_yaml, project=str(out_dir.parent), name=out_dir.name, exist_ok=True,
                seed=seed, deterministic=True, **train_args)
    best = out_dir / "weights" / "best.pt"
    if not best.exists():
        raise FileNotFoundError(f"training finished without {best}")
    return str(best)


def _val(model, data_yaml: str, ecfg: dict, names: list[str], run_dir: Path) -> dict:
    m = model.val(data=data_yaml, imgsz=ecfg["imgsz"], batch=ecfg["batch"], split="val",
                  project=str(run_dir.parent), name=run_dir.name, exist_ok=True, plots=False, verbose=False)
    per_class = {}
    for i, c in enumerate(m.box.ap_class_index.tolist()):
        per_class[names[int(c)]] = float(m.box.maps[int(c)])
    return {"map50_95": float(m.box.map), "map50": float(m.box.map50), "per_class_map50_95": per_class}


def evaluate(weights: str, images: list[str], groups: dict[str, str], names: list[str],
             ecfg: dict, out_dir: str | Path) -> dict:
    """Overall + per-group metrics on the given human-labeled images."""
    out_dir = Path(out_dir)
    model = _yolo(weights)

    def run(tag: str, imgs: list[str]) -> dict:
        lst = write_list(out_dir / f"{tag}.txt", imgs)
        yml = write_data_yaml(out_dir / f"{tag}.yaml", lst, lst, names)
        return _val(model, str(yml), ecfg, names, out_dir / f"val_{tag}")

    res = {"overall": run("all", images), "groups": {}}
    by_group: dict[str, list] = {}
    for p in images:
        by_group.setdefault(groups[p], []).append(p)
    for g, imgs in sorted(by_group.items()):
        if len(imgs) >= ecfg["min_group_images"]:
            safe = "".join(ch if ch.isalnum() or ch in "-_" else "_" for ch in g)
            res["groups"][g] = run(f"group_{safe}", imgs)
    return res
