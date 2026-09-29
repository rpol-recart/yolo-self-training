"""Thin wrappers over ultralytics: predict, train, evaluate. Imported lazily so the
pure-logic modules and tests work without ultralytics/torch installed."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from .boxes import nms_per_class
from .dataset import write_data_yaml, write_list


def _yolo(weights: str):
    from ultralytics import YOLO

    return YOLO(weights)


def predict(weights: str, images: list[str], tcfg: dict, out_jsonl: str | Path) -> dict[str, np.ndarray]:
    """Teacher predictions -> {path: (N,6) normalized x1,y1,x2,y2,conf,cls}. Cached in out_jsonl."""
    out_jsonl = Path(out_jsonl)
    if out_jsonl.exists():
        return load_predictions(out_jsonl)
    model = _yolo(weights)
    res = {}
    out_jsonl.parent.mkdir(parents=True, exist_ok=True)
    tmp = out_jsonl.with_suffix(".partial")
    with open(tmp, "w") as f:
        for i in range(0, len(images), 256):
            chunk = images[i:i + 256]
            for r in model.predict(chunk, conf=tcfg["conf_floor"], iou=tcfg["nms_iou"], imgsz=tcfg["imgsz"],
                                   augment=tcfg["tta"], batch=tcfg["batch"], device=tcfg["device"],
                                   stream=True, verbose=False):
                b = r.boxes
                d = np.concatenate([b.xyxyn.cpu().numpy(), b.conf.cpu().numpy()[:, None],
                                    b.cls.cpu().numpy()[:, None]], axis=1) if len(b) else np.zeros((0, 6))
                d = nms_per_class(d, tcfg["nms_iou"])
                key = str(Path(r.path).resolve())  # same key form as the split lists
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
