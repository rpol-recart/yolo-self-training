"""Group-aware, frozen train/val/test split.

Usage:
    python -m selftrain.split --config cfg.yaml --dry-run   # print groups and sizes
    python -m selftrain.split --config cfg.yaml             # write workdir/splits/ (reused if present)
"""

from __future__ import annotations

import argparse
import csv
import json
import re
from collections import Counter
from pathlib import Path

import numpy as np

from .config import load_config
from .dataset import list_images, read_list, write_list


def group_of(image: Path, root: Path, regex: str, csv_map: dict | None = None) -> str:
    if csv_map is not None:
        key = str(image)
        if key not in csv_map:
            raise KeyError(f"no group in CSV for {image}")
        return csv_map[key]
    rel = image.relative_to(root).as_posix()
    m = re.search(regex, rel)
    if not m or "group" not in m.groupdict():
        raise ValueError(f"groups.regex {regex!r} did not match {rel!r} (needs a (?P<group>...) group)")
    return m.group("group")


def load_group_csv(path: str | Path) -> dict:
    with open(path) as f:
        return {str(Path(r["path"]).resolve()): r["group"] for r in csv.DictReader(f)}


def assign_groups(groups: list[str], val_fraction: float, test_fraction: float, seed: int) -> dict[str, str]:
    """Deterministically assign each unique group to train/val/test."""
    uniq = sorted(set(groups))
    rng = np.random.default_rng(seed)
    order = list(rng.permutation(len(uniq)))
    n = len(uniq)
    n_val = max(1, round(n * val_fraction))
    n_test = max(1, round(n * test_fraction)) if test_fraction > 0 else 0
    if n_val + n_test >= n:
        raise ValueError(f"only {n} groups: not enough to split into train/val/test; add groups or lower fractions")
    out = {}
    for rank, i in enumerate(order):
        g = uniq[i]
        out[g] = "val" if rank < n_val else ("test" if rank < n_val + n_test else "train")
    return out


def check_no_leakage(image_groups: dict[str, str], group_split: dict[str, str]) -> None:
    """Each group must land in exactly one split (true by construction; guards manual edits)."""
    seen: dict[str, set] = {}
    for img, g in image_groups.items():
        seen.setdefault(g, set()).add(group_split[g])
    bad = {g: s for g, s in seen.items() if len(s) > 1}
    if bad:
        raise AssertionError(f"groups present in several splits: {bad}")


def make_or_load_split(cfg: dict, force: bool = False) -> dict:
    """Returns {'train': [...], 'val': [...], 'test': [...], 'pool': [...], 'groups': {img: group}}."""
    split_dir = Path(cfg["workdir"]) / "splits"
    meta = split_dir / "groups.json"
    if meta.exists() and not force:
        groups = json.loads(meta.read_text())
        res = {k: read_list(split_dir / f"{k}.txt") for k in ("train", "val", "test", "pool")}
        res["groups"] = groups
        return res

    d = cfg["data"]
    csv_map = load_group_csv(cfg["groups"]["csv"]) if cfg["groups"]["csv"] else None
    gt_root = Path(d["gt_images"]).resolve()
    gt = list_images(gt_root, d["extensions"])
    if not gt:
        raise FileNotFoundError(f"no images under {gt_root}")
    img_group = {str(p): group_of(p, gt_root, cfg["groups"]["regex"], csv_map) for p in gt}
    gsplit = assign_groups(list(img_group.values()), cfg["split"]["val_fraction"],
                           cfg["split"]["test_fraction"], cfg["seed"])
    check_no_leakage(img_group, gsplit)
    res = {k: sorted(i for i, g in img_group.items() if gsplit[g] == k) for k in ("train", "val", "test")}

    pool = []
    if d["unlabeled_images"]:
        pool_root = Path(d["unlabeled_images"]).resolve()
        eval_groups = {g for g, s in gsplit.items() if s != "train"}
        for p in list_images(pool_root, d["extensions"]):
            g = group_of(p, pool_root, cfg["groups"]["regex"], csv_map)
            img_group[str(p)] = g
            if cfg["split"]["exclude_eval_groups_from_pool"] and g in eval_groups:
                continue  # unlabeled frames from val/test groups would leak into training
            pool.append(str(p))
    res["pool"] = sorted(pool)

    for k in ("train", "val", "test", "pool"):
        write_list(split_dir / f"{k}.txt", res[k])
    meta.write_text(json.dumps(img_group, indent=0))
    res["groups"] = img_group
    return res


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", required=True)
    ap.add_argument("--dry-run", action="store_true", help="print groups, write nothing")
    ap.add_argument("--force", action="store_true", help="re-create the split (invalidates earlier rounds!)")
    args = ap.parse_args()
    cfg = load_config(args.config)
    if args.dry_run:
        d = cfg["data"]
        root = Path(d["gt_images"]).resolve()
        csv_map = load_group_csv(cfg["groups"]["csv"]) if cfg["groups"]["csv"] else None
        counts = Counter(group_of(p, root, cfg["groups"]["regex"], csv_map) for p in list_images(root, d["extensions"]))
        for g, n in sorted(counts.items()):
            print(f"{n:6d}  {g}")
        print(f"{len(counts)} groups, {sum(counts.values())} labeled images")
        return
    res = make_or_load_split(cfg, force=args.force)
    for k in ("train", "val", "test", "pool"):
        print(f"{k:5s} {len(res[k])}")


if __name__ == "__main__":
    main()
