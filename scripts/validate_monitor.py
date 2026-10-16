"""Select a validation-only threshold, freeze it, then audit the fixed test split.

Only compact final counts are kept. Optional per-image predictions go to a
caller-provided temporary directory for reviewing examples, never the dataset.
"""
from __future__ import annotations

import argparse
from datetime import datetime
import hashlib
import json
import os
from pathlib import Path
from zoneinfo import ZoneInfo

from monitor_helmet import ROOT, choose_device


def match(gt, pred, iou_threshold=0.5):
    """Class-aware matching in descending score order, same as this framework."""
    import numpy as np
    used = set()
    pairs = []
    for j in np.argsort(-pred[:, 4], kind="stable") if len(pred) else []:
        best, best_iou = None, -1.0
        for k, target in enumerate(gt):
            if k in used or int(target[0]) != int(pred[j, 5]):
                continue
            a, b = target[1:], pred[j, :4]
            inter = float(np.maximum(0, np.minimum(a[2:], b[2:]) - np.maximum(a[:2], b[:2])).prod())
            area = float((a[2:] - a[:2]).prod() + (b[2:] - b[:2]).prod() - inter)
            iou = inter / max(area, 1e-9)
            if iou > best_iou:
                best, best_iou = k, iou
        if best is not None and best_iou >= iou_threshold:
            pairs.append((best, int(j)))
            used.add(best)
    return pairs


def row(tp, fp, fn):
    p = tp / (tp + fp) if tp + fp else 0
    r = tp / (tp + fn) if tp + fn else 0
    return dict(tp=int(tp), fp=int(fp), fn=int(fn), precision=p, recall=r,
                f1=2 * p * r / (p + r) if p + r else 0)


def collect(model, split, options):
    import numpy as np
    files = sorted((ROOT / "datasets/hard-hat-workers/yolo/images" / split).glob("*.jpg"))
    if not files:
        raise ValueError("No split images")
    records = []
    def results():
        # The framework treats a Python list as a single batch, so chunk explicitly.
        for start in range(0, len(files), 8):
            yield from model.predict([str(f) for f in files[start:start + 8]], stream=True, **options)
    for i, r in enumerate(results()):
        h, w = r.orig_shape
        label = ROOT / "datasets/hard-hat-workers/yolo/labels" / split / (Path(r.path).stem + ".txt")
        targets = []
        for line in label.read_text().splitlines():
            c, x, y, bw, bh = map(float, line.split())
            targets.append([int(c), (x-bw/2)*w, (y-bh/2)*h, (x+bw/2)*w, (y+bh/2)*h])
        records.append({"image": str(Path(r.path).relative_to(ROOT)), "shape": [h, w],
                        "gt": np.array(targets, dtype=float).reshape(-1, 5),
                        "pred": r.boxes.data.cpu().numpy()[:, :6]})
        if (i + 1) % 200 == 0:
            print(f"{split}: {i+1}/{len(files)}", flush=True)
    return records


def summarize(records, threshold):
    import numpy as np
    totals = np.zeros((2, 3), dtype=int)
    sizes = {"small": np.zeros((2, 2), dtype=int), "other": np.zeros((2, 2), dtype=int)}
    density = {"dense": np.zeros((2, 3), dtype=int), "other": np.zeros((2, 3), dtype=int)}
    image_counts = {"dense": 0, "other": 0}
    for rec in records:
        gt, pred = rec["gt"], rec["pred"]
        pred = pred[pred[:, 4] >= threshold]
        pairs = match(gt, pred)
        mg = {g for g, p in pairs}
        mp = {p for g, p in pairs}
        count = np.zeros((2, 3), dtype=int)
        for c in (0, 1):
            count[c] = [sum(int(pred[j, 5]) == c for j in mp),
                        sum(int(p[5]) == c and j not in mp for j, p in enumerate(pred)),
                        sum(int(g[0]) == c and j not in mg for j, g in enumerate(gt))]
        totals += count
        group = "dense" if len(gt) >= 10 else "other"
        density[group] += count
        image_counts[group] += 1
        scale = 640 / max(rec["shape"])
        for j, g in enumerate(gt):
            area = (g[3]-g[1]) * (g[4]-g[2]) * scale**2
            sizes["small" if area <= 32**2 else "other"][int(g[0])] += [j in mg, 1]
    names = ["helmet", "no_helmet"]
    return {"images": len(records), "conf": threshold,
            "classes": {n: row(*totals[c]) for c, n in enumerate(names)},
            "size_recall": {k: {n: {"tp": int(v[c, 0]), "targets": int(v[c, 1]),
                             "recall": float(v[c, 0]/v[c, 1]) if v[c, 1] else None}
                                for c, n in enumerate(names)} for k, v in sizes.items()},
            "density": {k: {"images": image_counts[k], "classes": {n: row(*v[c]) for c, n in enumerate(names)}}
                        for k, v in density.items()}}


def main():
    from ultralytics import YOLO
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--output", type=Path, default=ROOT / "outputs/acceptance.json")
    p.add_argument("--config-output", type=Path, default=ROOT / "outputs/monitor.json")
    p.add_argument("--review-dir", type=Path)
    p.add_argument("--device", default="auto")
    args = p.parse_args()
    for path in (args.output, args.config_output):
        if path.exists():
            raise FileExistsError(f"Refusing to overwrite {path}")
    weights = ROOT / "weights/best.pt"
    device = choose_device(args.device)
    options = dict(imgsz=640, conf=0.001, iou=0.7, max_det=300, rect=False, augment=False,
                   device=device, verbose=False, save=False)
    model = YOLO(str(weights))
    val = collect(model, "val", options)
    sweep = [summarize(val, i/100) for i in range(10, 81, 5)]
    best = max(sweep, key=lambda s: (s["classes"]["no_helmet"]["f1"],
                                    s["classes"]["no_helmet"]["precision"], s["conf"]))
    conf = best["conf"]
    config = {"weights": str(weights.relative_to(ROOT)),
              "weights_sha256": hashlib.sha256(weights.read_bytes()).hexdigest(),
              **{k: options[k] for k in ("imgsz", "iou", "max_det", "rect", "augment")}, "conf": conf,
              "selected_on": "val", "selection": "max no_helmet F1 on 0.10..0.80 step 0.05; ties: precision then threshold",
              "frozen_at": datetime.now(ZoneInfo("Asia/Shanghai")).isoformat()}
    args.config_output.parent.mkdir(parents=True, exist_ok=True)
    args.config_output.write_text(json.dumps(config, ensure_ascii=False, indent=2) + "\n")
    print("FROZEN", conf, best["classes"], flush=True)
    # Test is read only after the config has been written; its result cannot select the threshold.
    test = collect(model, "test", options)
    result = {"protocol": "Fixed inference threshold; class-aware confidence-greedy one-to-one matching, IoU >= 0.50",
              "size_definition": "GT area after scaling longest image side to 640: small <= 32^2 pixels",
              "density_definition": "dense = >=10 annotated targets per image; not an occlusion label",
              "test_history": "Existing test set was already used for standard AP reporting; no retraining or threshold tuning on test",
              "device": device, "config": config, "validation_sweep": sweep,
              "validation_selected": best, "test_fixed_threshold": summarize(test, conf)}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n")
    if args.review_dir:
        args.review_dir.mkdir(parents=True, exist_ok=True)
        for split, records in (("val", val), ("test", test)):
            serial = [{**r, "gt": r["gt"].tolist(), "pred": r["pred"].tolist()} for r in records]
            (args.review_dir / (split + ".json")).write_text(json.dumps(serial))
    print("RESULT", json.dumps(result["test_fixed_threshold"]), flush=True)


if __name__ == "__main__":
    main()
