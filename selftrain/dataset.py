"""YOLO-format dataset helpers (images/ + labels/ sibling trees, one .txt per image)."""

from __future__ import annotations

import hashlib
import os
from pathlib import Path

import numpy as np
import yaml

from .boxes import xyxy_to_yolo


def list_images(root: str | Path, extensions: list[str]) -> list[Path]:
    root = Path(root)
    exts = {e.lower() for e in extensions}
    return sorted(p.resolve() for p in root.rglob("*") if p.is_file() and p.suffix.lower() in exts)


def label_path_for(image: str | Path) -> Path:
    """Ultralytics convention: replace the last /images/ path component with /labels/, suffix -> .txt."""
    s = str(image)
    marker = f"{os.sep}images{os.sep}"
    i = s.rfind(marker)
    if i < 0:
        raise ValueError(f"image path has no '{marker}' component: {s}")
    s = s[:i] + f"{os.sep}labels{os.sep}" + s[i + len(marker):]
    return Path(s).with_suffix(".txt")


def read_labels(path: str | Path) -> np.ndarray:
    """(N,5) cls,cx,cy,w,h; empty array for a missing/empty file (= background image)."""
    p = Path(path)
    if not p.exists() or p.stat().st_size == 0:
        return np.zeros((0, 5))
    rows = np.loadtxt(p, ndmin=2)
    return rows[:, :5] if rows.size else np.zeros((0, 5))


def write_labels(path: str | Path, boxes_xyxy: np.ndarray, cls: np.ndarray) -> None:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    rows = xyxy_to_yolo(boxes_xyxy, cls) if len(boxes_xyxy) else np.zeros((0, 5))
    with open(p, "w") as f:
        for r in rows:
            f.write(f"{int(r[0])} {r[1]:.6f} {r[2]:.6f} {r[3]:.6f} {r[4]:.6f}\n")


def stable_name(src: str | Path) -> str:
    """Collision-free file name for a linked image: <stem>_<hash8><suffix>."""
    src = Path(src)
    h = hashlib.sha1(str(src).encode()).hexdigest()[:8]
    return f"{src.stem}_{h}{src.suffix}"


def link_image(src: str | Path, dst_dir: str | Path) -> Path:
    dst = Path(dst_dir) / stable_name(src)
    dst.parent.mkdir(parents=True, exist_ok=True)
    if not dst.exists():
        os.symlink(Path(src).resolve(), dst)
    return dst


def write_list(path: str | Path, items) -> Path:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text("".join(f"{i}\n" for i in items))
    return p


def read_list(path: str | Path) -> list[str]:
    return [l.strip() for l in Path(path).read_text().splitlines() if l.strip()]


def write_data_yaml(path: str | Path, train_list: str | Path, val_list: str | Path, names: list[str]) -> Path:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    data = {
        "train": str(Path(train_list).resolve()),
        "val": str(Path(val_list).resolve()),
        "names": {i: n for i, n in enumerate(names)},
    }
    p.write_text(yaml.safe_dump(data, sort_keys=False, allow_unicode=True))
    return p
