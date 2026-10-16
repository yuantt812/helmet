"""Audit COCO data and prepare a reproducible, grouped YOLO dataset.

Run with the project's .venv; raw data is never modified.
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
import math
from pathlib import Path
import random
import shutil

import numpy as np
from PIL import Image, ImageDraw, ImageOps

ROOT = Path(__file__).resolve().parents[1]
BASE = ROOT / "datasets/hard-hat-workers"
RAW = BASE / "raw"
AUDIT = BASE / "audit"
OUT = BASE / "yolo"
SEED = 42
CLASS_MAP = {"helmet": 0, "head": 1}
NAMES = ["helmet", "no_helmet"]


def save(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False) + "\n")


def digest(bits):
    return int.from_bytes(np.packbits(bits).tobytes(), "big")


def read_image(record):
    try:
        path = RAW / record["source"]
        with Image.open(path) as im:
            im.load()
            rgb = im.convert("RGB")
            record["size"] = list(rgb.size)
            record["pixel_sha256"] = hashlib.sha256(
                str(rgb.size).encode() + rgb.tobytes()
            ).hexdigest()
            gray = rgb.convert("L")
            a = np.asarray(gray.resize((32, 32), Image.Resampling.LANCZOS), dtype=float)
            x = np.arange(32)
            c = np.cos(np.pi * np.arange(8)[:, None] * (2 * x + 1) / 64)
            low = (c @ a @ c.T).ravel()
            # Exclude the DC component; all hashes have 63 useful bits.
            record["phash"] = digest(low[1:] > np.median(low[1:]))
            a = np.asarray(gray.resize((9, 8), Image.Resampling.LANCZOS), dtype=float)
            record["dhash"] = digest((a[:, 1:] > a[:, :-1]).ravel())
            record["error"] = None
    except Exception as exc:
        record["error"] = str(exc)
    return record


def sheet(records, path, title, columns=5, cell=(300, 245)):
    rows = math.ceil(len(records) / columns)
    canvas = Image.new("RGB", (columns * cell[0], rows * cell[1] + 35), "white")
    draw = ImageDraw.Draw(canvas)
    draw.text((8, 8), title, fill="black")
    for index, r in enumerate(records):
        with Image.open(RAW / r["source"]) as im:
            im = im.convert("RGB")
        pen = ImageDraw.Draw(im)
        for a in r["annotations"]:
            if a["class"] not in CLASS_MAP:
                continue
            x, y, w, h = a["bbox"]
            color = "lime" if a["class"] == "helmet" else "red"
            pen.rectangle((x, y, x + w, y + h), outline=color, width=2)
        im.thumbnail((cell[0] - 10, cell[1] - 40))
        x, y = (index % columns) * cell[0], 35 + (index // columns) * cell[1]
        canvas.paste(im, (x + 5, y + 5))
        counts = Counter(a["class"] for a in r["annotations"])
        draw.text((x + 5, y + cell[1] - 31), r["key"], fill="black")
        draw.text((x + 5, y + cell[1] - 17), f"helmet={counts['helmet']} head={counts['head']}", fill="black")
    path.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(path)


def audit():
    for split in ("train", "test"):
        if not (RAW / split / "_annotations.coco.json").is_file():
            raise FileNotFoundError(f"Restore the COCO dataset to {RAW} before auditing (see datasets/hard-hat-workers/README.md)")
    records = []
    for f in sorted(RAW.glob("*/_annotations.coco.json")):
        data = json.loads(f.read_text())
        categories = {c["id"]: c["name"] for c in data["categories"]}
        annotations = defaultdict(list)
        ids = {im["id"] for im in data["images"]}
        assert len(ids) == len(data["images"]), "duplicate image ID"
        for a in data["annotations"]:
            assert a["image_id"] in ids, "orphan annotation"
            annotations[a["image_id"]].append({"id": a["id"], "class": categories[a["category_id"]], "bbox": a["bbox"]})
        for im in sorted(data["images"], key=lambda x: x["file_name"]):
            records.append({"key": f"{f.parent.name}_{im['id']:05d}",
                            "source": str((f.parent / im["file_name"]).relative_to(RAW)),
                            "declared_size": [im["width"], im["height"]],
                            "annotations": annotations[im["id"]]})
    with ThreadPoolExecutor(max_workers=8) as pool:
        records = list(pool.map(read_image, records))
    save(AUDIT / "records.json", records)
    summary = {"images": len(records), "decode_errors": [r["key"] for r in records if r["error"]],
               "size_mismatches": [r["key"] for r in records if not r["error"] and r["size"] != r["declared_size"]]}
    issues = []
    for r in records:
        if r["error"]:
            continue
        w, h = r["size"]
        for a in r["annotations"]:
            x, y, bw, bh = a["bbox"]
            if not all(math.isfinite(v) for v in a["bbox"]) or bw <= 0 or bh <= 0:
                issues.append({"key": r["key"], "kind": "invalid", "annotation": a})
            elif x < 0 or y < 0 or x + bw > w or y + bh > h:
                issues.append({"key": r["key"], "kind": "out_of_bounds", "annotation": a})
    save(AUDIT / "box_issues.json", issues)
    summary["box_issue_counts"] = dict(Counter(i["kind"] for i in issues))
    groups = defaultdict(list)
    for r in records:
        if not r["error"]:
            groups[r["pixel_sha256"]].append(r["key"])
    save(AUDIT / "exact_duplicate_groups.json", [v for v in groups.values() if len(v) > 1])
    summary["exact_duplicate_groups"] = sum(len(v) > 1 for v in groups.values())
    save(AUDIT / "initial_summary.json", summary)
    sampled = random.Random(SEED).sample([r for r in records if not r["error"]], 100)
    save(AUDIT / "semantic_sample.json", [r["key"] for r in sampled])
    for i in range(5):
        sheet(sampled[i * 20:(i + 1) * 20], AUDIT / f"semantic_{i+1}.jpg", "Seed 42 semantic review: green=helmet, red=head")
    print(json.dumps(summary, indent=2), flush=True)


def prepare():
    records = json.loads((AUDIT / "records.json").read_text())
    review = json.loads((AUDIT / "manual_review.json").read_text())
    excluded = review.get("exclude", {})
    usable, removed = [], []
    seen = {}
    issues = []
    for r in records:
        reason = r["error"] or excluded.get(r["key"])
        if r.get("size") != r["declared_size"]:
            reason = reason or "image and annotation dimensions disagree"
        if reason:
            removed.append({"key": r["key"], "reason": reason})
            continue
        width, height = r["size"]
        labels = []
        for a in r["annotations"]:
            if a["class"] not in CLASS_MAP:
                continue
            x, y, w, h = a["bbox"]
            if not all(math.isfinite(v) for v in a["bbox"]) or w <= 0 or h <= 0:
                reason = "invalid target annotation"
                break
            x1, y1, x2, y2 = max(0, x), max(0, y), min(width, x + w), min(height, y + h)
            if x2 <= x1 or y2 <= y1:
                reason = "target annotation outside image"
                break
            if [x1, y1, x2-x1, y2-y1] != a["bbox"]:
                issues.append({"key": r["key"], "annotation_id": a["id"], "original": a["bbox"], "clipped": [x1,y1,x2-x1,y2-y1]})
            labels.append([CLASS_MAP[a["class"]], (x1+x2)/2/width, (y1+y2)/2/height, (x2-x1)/width, (y2-y1)/height])
        if reason:
            removed.append({"key": r["key"], "reason": reason})
            continue
        r["labels"] = sorted({tuple(v) for v in labels})
        previous = seen.get(r["pixel_sha256"])
        if previous:
            # Conflicting duplicate labels require review, rather than merging annotations blindly.
            if r["labels"] != previous["labels"]:
                raise ValueError(f"Conflicting duplicate labels: {previous['key']}, {r['key']}")
            removed.append({"key": r["key"], "reason": "exact pixel duplicate", "kept": previous["key"]})
        else:
            seen[r["pixel_sha256"]] = r
            usable.append(r)
    # All pairs at this dataset size are cheap in NumPy; no approximate-search misses.
    parent = list(range(len(usable)))
    def find(i):
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i
    def union(a, b):
        a, b = find(a), find(b)
        if a != b:
            parent[max(a,b)] = min(a,b)
    ph = np.array([r["phash"] for r in usable], dtype=np.uint64)
    dh = np.array([r["dhash"] for r in usable], dtype=np.uint64)
    pairs = []
    for i in range(len(usable)-1):
        pd = np.bitwise_count(ph[i] ^ ph[i+1:])
        dd = np.bitwise_count(dh[i] ^ dh[i+1:])
        for offset in np.flatnonzero((pd <= 12) & (dd <= 16)):
            j = i + 1 + int(offset)
            union(i, j)
            pairs.append({"a": usable[i]["key"], "b": usable[j]["key"], "phash_distance": int(pd[offset]), "dhash_distance": int(dd[offset])})
    key_index = {r["key"]:i for i,r in enumerate(usable)}
    for group in review.get("scene_groups", []):
        indices = [key_index[k] for k in group if k in key_index]
        for i in indices[1:]:
            union(indices[0], i)
    groups = defaultdict(list)
    for i, r in enumerate(usable):
        groups[find(i)].append(r)
    groups = list(groups.values())
    rng = random.Random(SEED)
    rng.shuffle(groups)
    groups.sort(key=len, reverse=True)
    # Greedy allocation balances image counts and both class-box counts by group.
    total = np.array([len(usable)] + [sum(sum(v[0] == c for v in r["labels"]) for r in usable) for c in range(2)], dtype=float)
    ratios = np.array([.7,.2,.1])
    targets = ratios[:, None] * total
    actual = np.zeros((3,3))
    splits = [[],[],[]]
    for group in groups:
        v = np.array([len(group)] + [sum(sum(a[0] == c for a in r["labels"]) for r in group) for c in range(2)])
        costs=[]
        for s in range(3):
            trial=actual.copy();trial[s]+=v
            costs.append(float(np.sum((trial-targets)**2 / np.maximum(targets,1))))
        s=min(range(3),key=costs.__getitem__)
        actual[s]+=v
        for r in group:
            r["group"] = min(x["key"] for x in group)
        splits[s].extend(group)
    if OUT.exists():
        raise FileExistsError(f"Refusing to overwrite {OUT}")
    counts = {}
    manifest = []
    for name, members in zip(["train","val","test"], splits):
        for sub in ["images","labels"]:
            (OUT/sub/name).mkdir(parents=True)
        for r in sorted(members,key=lambda r:r["key"]):
            image=OUT/"images"/name/(r["key"]+".jpg")
            shutil.copy2(RAW/r["source"],image)
            (OUT/"labels"/name/(r["key"]+".txt")).write_text("".join(f"{v[0]} " + " ".join(f"{x:.9f}" for x in v[1:]) + "\n" for v in r["labels"]))
            manifest.append({"key":r["key"],"source":r["source"],"split":name,"group":r["group"],"pixel_sha256":r["pixel_sha256"]})
        counts[name]={"images":len(members),"backgrounds":sum(not r["labels"] for r in members),"helmet":sum(sum(a[0]==0 for a in r["labels"]) for r in members),"no_helmet":sum(sum(a[0]==1 for a in r["labels"]) for r in members)}
        sheet(random.Random(SEED).sample(members,min(20,len(members))),AUDIT/f"final_{name}.jpg",f"{name}: green=helmet, red=no_helmet")
    # Ultralytics resolves paths relative to this YAML when `path` is omitted.
    (OUT/"data.yaml").write_text("train: images/train\nval: images/val\ntest: images/test\nnames:\n  0: helmet\n  1: no_helmet\n")
    save(AUDIT/"split_manifest.json",manifest)
    save(AUDIT/"near_duplicate_pairs.json",pairs)
    save(AUDIT/"group_members.json",[[r["key"] for r in g] for g in groups if len(g)>1])
    save(AUDIT/"excluded.json",removed)
    save(AUDIT/"clipped_boxes.json",issues)
    summary={"seed":SEED,"source_images":len(records),"kept_images":len(usable),"excluded_images":len(removed),"groups":len(groups),"largest_group":max(map(len,groups)),"near_duplicate_pairs":len(pairs),"clipped_boxes":len(issues),"splits":counts}
    save(AUDIT/"summary.json",summary)
    print(json.dumps(summary,indent=2))


def verify():
    """Independently compare every exported box against its source COCO box."""
    records = {r["key"]:r for r in json.loads((AUDIT/"records.json").read_text())}
    manifest = json.loads((AUDIT/"split_manifest.json").read_text())
    by_key = {m["key"]:m for m in manifest}
    assert len(by_key) == len(manifest)
    group_splits, hash_splits = defaultdict(set), defaultdict(set)
    counts = {}
    for m in manifest:
        group_splits[m["group"]].add(m["split"])
        hash_splits[m["pixel_sha256"]].add(m["split"])
    assert all(len(v) == 1 for v in group_splits.values())
    assert all(len(v) == 1 for v in hash_splits.values())
    for pair in json.loads((AUDIT/"near_duplicate_pairs.json").read_text()):
        assert by_key[pair["a"]]["split"] == by_key[pair["b"]]["split"]
    for split in ["train", "val", "test"]:
        members = [m for m in manifest if m["split"] == split]
        assert len(list((OUT/"images"/split).glob("*.jpg"))) == len(members)
        assert len(list((OUT/"labels"/split).glob("*.txt"))) == len(members)
        count = Counter()
        preview = []
        for m in members:
            r = records[m["key"]]
            image_path = OUT/"images"/split/(m["key"]+".jpg")
            assert image_path.read_bytes() == (RAW/m["source"]).read_bytes()
            w, h = r["size"]
            expected = []
            for a in r["annotations"]:
                if a["class"] not in CLASS_MAP:
                    continue
                x, y, bw, bh = a["bbox"]
                x1,y1,x2,y2=max(0,x),max(0,y),min(w,x+bw),min(h,y+bh)
                expected.append((CLASS_MAP[a["class"]],(x1+x2)/2/w,(y1+y2)/2/h,(x2-x1)/w,(y2-y1)/h))
            actual = []
            for line in (OUT/"labels"/split/(m["key"]+".txt")).read_text().splitlines():
                fields=line.split()
                assert len(fields)==5
                c=int(fields[0]); v=list(map(float,fields[1:]))
                assert c in [0,1] and all(math.isfinite(x) for x in v)
                cx,cy,bw,bh=v
                assert bw>0 and bh>0 and cx-bw/2>=-1e-8 and cy-bh/2>=-1e-8 and cx+bw/2<=1+1e-8 and cy+bh/2<=1+1e-8
                actual.append((c,*v));count[c]+=1
            expected=sorted(set(expected));actual=sorted(actual)
            assert len(expected)==len(actual)
            assert all(all(abs(x-y)<1e-8 for x,y in zip(e,v)) for e,v in zip(expected,actual))
            preview.append({"key":m["key"],"source":m["source"],"annotations":[{"class": "helmet" if c==0 else "head", "bbox":[(cx-bw/2)*w,(cy-bh/2)*h,bw*w,bh*h]} for c,cx,cy,bw,bh in actual]})
        counts[split]={"images":len(members),"helmet":count[0],"no_helmet":count[1]}
        sheet(random.Random(SEED).sample(preview,20),AUDIT/f"exported_{split}.jpg",f"Exported YOLO labels: {split}, green=helmet red=no_helmet")
    result={"passed":True,"images_verified":len(manifest),"checks":["image bytes preserved","image-label pairing","finite and bounded coordinates","COCO-to-YOLO class and coordinates match","no exact hashes cross splits","no identified similar pair or group crosses splits"],"statistics":counts}
    save(AUDIT/"verification.json",result)
    print(json.dumps(result,indent=2))


if __name__ == "__main__":
    parser=argparse.ArgumentParser()
    parser.add_argument("stage",choices=["audit","prepare","verify"])
    args=parser.parse_args()
    {"audit":audit,"prepare":prepare,"verify":verify}[args.stage]()
