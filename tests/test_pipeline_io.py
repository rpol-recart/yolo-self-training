"""File-level tests: split freezing, pool exclusion, round dataset layout. No ultralytics needed."""

import subprocess
import sys
from pathlib import Path

import numpy as np
import yaml

from selftrain.config import DEFAULTS, deep_merge
from selftrain.dataset import label_path_for, read_labels, read_list
from selftrain.filter import ImageDecision
from selftrain.merge import build_round_dataset
from selftrain.split import make_or_load_split

ROOT = Path(__file__).resolve().parents[1]


def _synthetic(tmp_path):
    subprocess.run([sys.executable, str(ROOT / "scripts" / "make_synthetic.py"), "--out", str(tmp_path / "syn"),
                    "--labeled-seqs", "10", "--unlabeled-seqs", "3", "--frames", "4"], check=True)
    return deep_merge(DEFAULTS, {
        "names": ["a", "b"],
        "workdir": str(tmp_path / "work"),
        "data": {"gt_images": str(tmp_path / "syn/labeled/images"),
                 "unlabeled_images": str(tmp_path / "syn/unlabeled/images")},
        "groups": {"regex": r"^(?P<group>seq_\d+)/"},
        "split": {"val_fraction": 0.2, "test_fraction": 0.2},
    })


def test_split_frozen_and_leak_free(tmp_path):
    cfg = _synthetic(tmp_path)
    s1 = make_or_load_split(cfg)
    g = s1["groups"]
    groups_of = {k: {g[p] for p in s1[k]} for k in ("train", "val", "test")}
    assert not (groups_of["train"] & groups_of["val"]) and not (groups_of["val"] & groups_of["test"])
    assert len(s1["val"]) == 8 and len(s1["test"]) == 8 and len(s1["train"]) == 24
    assert len(s1["pool"]) == 12
    # a different seed must not change a frozen split
    s2 = make_or_load_split(deep_merge(cfg, {"seed": 99}))
    assert s2["val"] == s1["val"]


def test_pool_excludes_eval_groups(tmp_path):
    cfg = _synthetic(tmp_path)
    # put an unlabeled copy of a val group into the pool
    s = make_or_load_split(cfg, force=True)
    val_group = s["groups"][s["val"][0]]
    src = Path(s["val"][0])
    dst = Path(cfg["data"]["unlabeled_images"]) / val_group / "frame_9999.jpg"
    dst.parent.mkdir(parents=True, exist_ok=True)
    dst.write_bytes(src.read_bytes())
    s = make_or_load_split(cfg, force=True)
    assert str(dst.resolve()) not in s["pool"]


def test_build_round_dataset(tmp_path):
    cfg = _synthetic(tmp_path)
    s = make_or_load_split(cfg)
    pool = s["pool"]
    decisions = [
        ImageDecision(path=pool[0], group="p", status="pseudo", confident=np.array([[0.1, 0.1, 0.4, 0.4, 0.9, 1]])),
        ImageDecision(path=pool[1], group="p", status="background"),
        ImageDecision(path=pool[2], group="p", status="uncertain", uncertain=np.ones((1, 6))),
    ]
    val_txt = tmp_path / "val.txt"
    val_txt.write_text("\n".join(s["val"]))
    stats = build_round_dataset(tmp_path / "round1", s["train"], val_txt, decisions, ["a", "b"], cfg["merge"], 0)
    assert stats["pseudo_images"] == 1 and stats["background_images"] == 1
    train = read_list(tmp_path / "round1" / "train.txt")
    assert len(train) == len(s["train"]) + 2
    assert pool[2] not in train and all(Path(p).exists() for p in train)
    added = [p for p in train if "round1" in p]
    labels = sorted(len(read_labels(label_path_for(p))) for p in added)
    assert labels == [0, 1]  # background = empty label file, pseudo = one box
    data = yaml.safe_load((tmp_path / "round1" / "data.yaml").read_text())
    assert data["names"] == {0: "a", 1: "b"}
