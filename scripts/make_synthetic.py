"""Generate a tiny synthetic detection dataset for the smoke test.

Two classes of coloured shapes moving across noisy backgrounds. Each sequence is a
group ("seq_XX"); frames of a sequence are consecutive, so the temporal filter applies.

    python scripts/make_synthetic.py --out data/synthetic
creates
    data/synthetic/labeled/images/seq_00/frame_0000.jpg  (+ labels/...)
    data/synthetic/unlabeled/images/seq_10/frame_0000.jpg
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

SIZE = 320
COLOURS = [(220, 40, 40), (40, 90, 220)]  # class 0 rectangles, class 1 ellipses


def make_sequence(rng, n_frames: int):
    n_obj = rng.integers(1, 4)
    objs = []
    for _ in range(n_obj):
        cls = int(rng.integers(0, 2))
        w, h = rng.integers(30, 80, size=2)
        x, y = rng.integers(0, SIZE - 80, size=2)
        vx, vy = rng.integers(-4, 5, size=2)
        objs.append([cls, float(x), float(y), int(w), int(h), int(vx), int(vy)])
    bg_tint = rng.integers(60, 180, size=3)
    for _ in range(n_frames):
        noise = rng.normal(0, 25, size=(SIZE, SIZE, 3))
        img = np.clip(bg_tint + noise, 0, 255).astype(np.uint8)
        im = Image.fromarray(img)
        draw = ImageDraw.Draw(im)
        labels = []
        for o in objs:
            cls, x, y, w, h, vx, vy = o
            box = [x, y, x + w, y + h]
            (draw.rectangle if cls == 0 else draw.ellipse)(box, fill=COLOURS[cls])
            labels.append((cls, (x + w / 2) / SIZE, (y + h / 2) / SIZE, w / SIZE, h / SIZE))
            o[1] = float(np.clip(x + vx, 0, SIZE - w))
            o[2] = float(np.clip(y + vy, 0, SIZE - h))
        yield im, labels


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="data/synthetic")
    ap.add_argument("--labeled-seqs", type=int, default=10)
    ap.add_argument("--unlabeled-seqs", type=int, default=12)
    ap.add_argument("--frames", type=int, default=12)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    rng = np.random.default_rng(args.seed)
    out = Path(args.out)
    for s in range(args.labeled_seqs + args.unlabeled_seqs):
        labeled = s < args.labeled_seqs
        split = "labeled" if labeled else "unlabeled"
        for i, (im, labels) in enumerate(make_sequence(rng, args.frames)):
            img_path = out / split / "images" / f"seq_{s:02d}" / f"frame_{i:04d}.jpg"
            img_path.parent.mkdir(parents=True, exist_ok=True)
            im.save(img_path, quality=90)
            if labeled:
                lbl = out / split / "labels" / f"seq_{s:02d}" / f"frame_{i:04d}.txt"
                lbl.parent.mkdir(parents=True, exist_ok=True)
                lbl.write_text("".join(f"{c} {cx:.6f} {cy:.6f} {w:.6f} {h:.6f}\n" for c, cx, cy, w, h in labels))
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
