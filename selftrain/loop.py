"""Self-training loop: round 0 on GT, then teacher -> pseudo-labels -> student rounds.

Usage:
    python -m selftrain.loop --config cfg.yaml

Re-running resumes: finished rounds (rounds/<k>/result.json) are skipped.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path

import yaml

from . import yolo_ops
from .boxes import yolo_to_xyxy
from .calibrate import calibrate
from .config import load_config
from .dataset import label_path_for, read_labels, write_data_yaml, write_list
from .decide import accept_round, should_stop
from .filter import decide, review_queue
from .merge import build_round_dataset
from .split import make_or_load_split


def _clean(obj):
    """NaN/inf -> None so the JSON stays valid for strict parsers."""
    if isinstance(obj, float) and not math.isfinite(obj):
        return None
    if isinstance(obj, dict):
        return {k: _clean(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_clean(v) for v in obj]
    return obj


def _dump(path: Path, obj) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(_clean(obj), indent=2, ensure_ascii=False, default=float))


def _gt_boxes(images: list[str]) -> dict:
    return {p: yolo_to_xyxy(read_labels(label_path_for(p))) for p in images}


def run_round0(cfg: dict, split: dict, rdir: Path) -> dict:
    val_txt = write_list(rdir / "val.txt", split["val"])
    train_txt = write_list(rdir / "train.txt", split["train"])
    data_yaml = write_data_yaml(rdir / "data.yaml", train_txt, val_txt, cfg["names"])
    s = cfg["student"]
    weights = yolo_ops.train(str(data_yaml), s["base_weights"], rdir / "train", s["train_args"], cfg["seed"])
    metrics = yolo_ops.evaluate(weights, split["val"], split["groups"], cfg["names"], cfg["eval"], rdir / "eval")
    _dump(rdir / "metrics.json", metrics)
    return {"round": 0, "weights": weights, "metrics": metrics, "accepted": True, "reasons": ["baseline on GT only"]}


def run_round(k: int, cfg: dict, split: dict, teacher: str, rdir: Path, prev_rejected: Path | None) -> dict:
    names = cfg["names"]
    select_seed = cfg["seed"] + k  # varies the pseudo-label sample between rounds
    rdir.mkdir(parents=True, exist_ok=True)
    (rdir / "config.yaml").write_text(yaml.safe_dump(cfg, sort_keys=False, allow_unicode=True))

    # 1. thresholds from the teacher's behaviour on human-labeled val
    val_preds = yolo_ops.predict(teacher, split["val"], cfg["teacher"], rdir / "preds_val.jsonl")
    thresholds = calibrate(val_preds, _gt_boxes(split["val"]), len(names), cfg["calibration"],
                           cfg["teacher"]["conf_floor"])
    _dump(rdir / "thresholds.json", {names[c]: t for c, t in thresholds.items()})

    # 2. pseudo-labels on the unlabeled pool
    pool_preds = yolo_ops.predict(teacher, split["pool"], cfg["teacher"], rdir / "preds_pool.jsonl")
    decisions = decide(pool_preds, split["groups"], thresholds, cfg["filter"])

    # 3. active-learning queue
    queue = review_queue(decisions, cfg["review"]["max_items"])
    with open(rdir / "review_queue.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["path", "group", "uncertainty", "n_uncertain", "n_confident"])
        for d in queue:
            w.writerow([d.path, d.group, f"{d.uncertainty:.3f}", len(d.uncertain), len(d.confident)])

    # 4. round dataset
    val_txt = write_list(rdir / "val.txt", split["val"])
    stats = build_round_dataset(rdir, split["train"], val_txt, decisions, names, cfg["merge"], select_seed)
    _dump(rdir / "pseudo_stats.json", stats)
    if prev_rejected is not None and (prev_rejected / "sources.txt").exists() and \
            (prev_rejected / "sources.txt").read_text() == (rdir / "sources.txt").read_text():
        # same teacher + same selected data + same training seed = the rejected round again
        return {"round": k, "weights": None, "metrics": None, "teacher": teacher, "pseudo": stats,
                "duplicate": True}

    # 5. student
    s = cfg["student"]
    init = s["base_weights"] if s["init"] == "base" else teacher
    # same training seed in every round: metric differences should come from the data, not the seed
    weights = yolo_ops.train(stats["data_yaml"], init, rdir / "train", s["train_args"], cfg["seed"])

    # 6. evaluation on the frozen GT-val
    metrics = yolo_ops.evaluate(weights, split["val"], split["groups"], names, cfg["eval"], rdir / "eval")
    _dump(rdir / "metrics.json", metrics)
    return {"round": k, "weights": weights, "metrics": metrics, "teacher": teacher, "pseudo": stats}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", required=True)
    args = ap.parse_args()
    cfg = load_config(args.config)
    work = Path(cfg["workdir"])
    split = make_or_load_split(cfg)
    print({k: len(split[k]) for k in ("train", "val", "test", "pool")})
    if not split["pool"]:
        raise SystemExit("unlabeled pool is empty: set data.unlabeled_images")

    rounds: list[dict] = []
    best = None
    k = 0
    while True:
        rdir = work / "rounds" / str(k)
        done = rdir / "result.json"
        if done.exists():
            result = json.loads(done.read_text())
            print(f"round {k}: cached")
        elif k == 0:
            result = run_round0(cfg, split, rdir)
        else:
            prev = rounds[-1] if rounds and not rounds[-1]["accepted"] else None
            prev_dir = work / "rounds" / str(prev["round"]) if prev else None
            result = run_round(k, cfg, split, best["weights"], rdir, prev_dir)
            if result.get("duplicate"):
                result["accepted"] = False
                result["reasons"] = [f"training set identical to rejected round {prev['round']}: "
                                     "nothing new to try with this teacher"]
            else:
                result["accepted"], result["reasons"] = accept_round(result["metrics"], best["metrics"],
                                                                     cfg["accept"])
        if not done.exists():
            _dump(done, result)
        m = cfg["accept"]["metric"]
        score = f"{result['metrics']['overall'][m]:.4f}" if result["metrics"] else "n/a"
        print(f"round {k}: {m}={score} accepted={result['accepted']} ({'; '.join(result['reasons'])})")
        if result["accepted"]:
            best = result
        rounds.append(result)
        history = rounds[1:]  # the stop rule counts self-training rounds only
        _dump(work / "history.json", {
            "best_round": best["round"],
            "best_weights": best["weights"],
            "rounds": [{"round": r["round"], "accepted": r["accepted"], "reasons": r["reasons"],
                        "overall": r["metrics"]["overall"] if r["metrics"] else None} for r in rounds],
        })
        stop, why = should_stop(history, cfg["loop"])
        if result.get("duplicate"):
            stop, why = True, result["reasons"][0]
        if stop:
            print(f"stop: {why}")
            break
        k += 1

    final = {"best_round": best["round"], "best_weights": best["weights"], "val": best["metrics"]}
    if split["test"]:
        final["test"] = yolo_ops.evaluate(best["weights"], split["test"], split["groups"], cfg["names"],
                                          cfg["eval"], work / "final_test")
    _dump(work / "final.json", final)
    print(f"best round {best['round']}: {best['weights']}")


if __name__ == "__main__":
    main()
