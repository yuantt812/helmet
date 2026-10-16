"""Detect visible helmet states in one image, a local video, or an optional camera."""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
os.environ.setdefault("YOLO_CONFIG_DIR", str(ROOT / ".config/ultralytics"))
os.environ.setdefault("MPLCONFIGDIR", str(ROOT / ".config/matplotlib"))
os.environ.setdefault("YOLO_AUTOINSTALL", "false")
os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")
for name in ("YOLO_CONFIG_DIR", "MPLCONFIGDIR"):
    Path(os.environ[name]).mkdir(parents=True, exist_ok=True)


def load_config(path):
    config = json.loads(Path(path).read_text())
    weights = ROOT / config["weights"]
    actual = hashlib.sha256(weights.read_bytes()).hexdigest()
    if actual != config["weights_sha256"]:
        raise ValueError("Weight checksum differs from the validated model")
    return config, weights


def choose_device(requested="auto"):
    import torch
    if requested != "auto":
        return requested
    return "mps" if torch.backends.mps.is_available() else ("0" if torch.cuda.is_available() else "cpu")


def find_font(requested=None):
    candidates = [requested, "/System/Library/Fonts/Supplemental/Arial Unicode.ttf",
                  "/System/Library/Fonts/STHeiti Medium.ttc",
                  "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc",
                  "C:/Windows/Fonts/msyh.ttc"]
    for value in candidates:
        if value and Path(value).is_file():
            return value
    raise FileNotFoundError("Chinese font required; specify --font /path/to/font.ttf")


def annotate(frame, result, font_path, confidence):
    import cv2
    import numpy as np
    from PIL import Image, ImageDraw, ImageFont
    h, w = frame.shape[:2]
    font = ImageFont.truetype(font_path, max(14, min(26, w // 45)))
    counts = [0, 0]
    img = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    draw = ImageDraw.Draw(img)
    for box in result.boxes:
        cls = int(box.cls.item())
        if cls not in (0, 1):
            continue
        counts[cls] += 1
        color = (30, 210, 90) if cls == 0 else (240, 55, 55)
        x1, y1, x2, y2 = [int(v) for v in box.xyxy[0].tolist()]
        label = ("戴安全帽" if cls == 0 else "未戴安全帽") + f" {float(box.conf.item()):.2f}"
        draw.rectangle((x1, y1, x2, y2), outline=color, width=max(2, w // 400))
        bounds = draw.textbbox((0, 0), label, font=font)
        tw, th = bounds[2], bounds[3] - bounds[1]
        tx = max(0, min(x1, w - tw - 6))
        ty = max(0, y1 - th - 7)
        draw.rectangle((tx, ty, tx + tw + 6, ty + th + 6), fill=color)
        draw.text((tx + 3, ty + 2 - bounds[1]), label, font=font, fill="white")
    # Add a separate status strip so it never hides a target at the top of the frame.
    strip = max(74, font.size * 3 + 12)
    out = Image.new("RGB", (w, h + strip), (20, 28, 39))
    out.paste(img, (0, strip))
    d = ImageDraw.Draw(out)
    d.text((10, 5), f"当前帧  戴帽 {counts[0]}   未戴帽 {counts[1]}   阈值 {confidence:.2f}", font=font, fill="white")
    message = "检测到未戴安全帽" if counts[1] else "当前帧未检出未戴帽目标"
    d.text((10, font.size + 13), message + "  ·  不累计人数", font=font,
           fill=(255, 100, 100) if counts[1] else (140, 225, 170))
    # Even dimensions are required by common H.264 encoders.
    out = out.crop((0, 0, out.width - out.width % 2, out.height - out.height % 2))
    return cv2.cvtColor(np.asarray(out), cv2.COLOR_RGB2BGR), counts


def main():
    import cv2
    from ultralytics import YOLO
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--source", required=True, help="Image/video path, or camera index with --camera")
    p.add_argument("--output", required=True, type=Path, help="New image path or .mp4 video path")
    p.add_argument("--config", type=Path, default=ROOT / "config/monitor.json")
    p.add_argument("--device", default="auto")
    p.add_argument("--font")
    p.add_argument("--camera", action="store_true")
    p.add_argument("--show", action="store_true", help="Display preview; Q exits")
    p.add_argument("--max-frames", type=int, help="Optional maximum video frames")
    args = p.parse_args()
    if args.output.exists():
        raise FileExistsError(f"Refusing to overwrite {args.output}")
    if args.max_frames is not None and args.max_frames <= 0:
        p.error("--max-frames must be positive")
    if not args.camera and not Path(args.source).is_file():
        raise FileNotFoundError(args.source)
    if not args.camera and Path(args.source).resolve() == args.output.resolve():
        p.error("Input and output must differ")
    config, weights = load_config(args.config)
    font_path = find_font(args.font)
    model = YOLO(str(weights))
    options = {k: config[k] for k in ("imgsz", "conf", "iou", "max_det", "rect", "augment")}
    options.update(device=choose_device(args.device), verbose=False, save=False)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    is_image = not args.camera and Path(args.source).suffix.lower() in {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff"}
    if is_image:
        frame = cv2.imread(args.source)
        if frame is None:
            raise ValueError("Unreadable image")
        result = model.predict(frame, **options)[0]
        annotated, counts = annotate(frame, result, font_path, config["conf"])
        if args.output.suffix.lower() not in {".jpg", ".jpeg", ".png", ".bmp", ".webp"}:
            p.error("Image output must use an image extension")
        if not cv2.imwrite(str(args.output), annotated):
            raise OSError("Image write failed")
        print(json.dumps({"output": str(args.output), "frames": 1, "counts": counts}, ensure_ascii=False))
        return
    if args.output.suffix.lower() != ".mp4":
        p.error("Video output must end in .mp4")
    cap = cv2.VideoCapture(int(args.source) if args.camera else args.source)
    writer = None
    frames = 0
    frames_with_no_helmet = 0
    frames_with_both = 0
    expected = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) if not args.camera else 0
    try:
        if not cap.isOpened():
            raise ValueError("Unable to open video/camera")
        fps = cap.get(cv2.CAP_PROP_FPS)
        fps = fps if math.isfinite(fps) and fps > 0 else 25.0
        while args.max_frames is None or frames < args.max_frames:
            ok, frame = cap.read()
            if not ok:
                break
            annotated, counts = annotate(frame, model.predict(frame, **options)[0], font_path, config["conf"])
            frames_with_no_helmet += counts[1] > 0
            frames_with_both += counts[0] > 0 and counts[1] > 0
            if writer is None:
                writer = cv2.VideoWriter(str(args.output), cv2.VideoWriter_fourcc(*"avc1"), fps,
                                         (annotated.shape[1], annotated.shape[0]))
                if not writer.isOpened():
                    raise RuntimeError("H.264 encoder unavailable in this OpenCV build")
            writer.write(annotated)
            frames += 1
            if args.show:
                cv2.imshow("Helmet monitor", annotated)
                if cv2.waitKey(1) & 0xFF in (ord("q"), ord("Q")):
                    break
        if frames == 0:
            raise ValueError("Video contains no decodable frames")
        if not args.show and args.max_frames is None and expected > 0 and frames != expected:
            raise RuntimeError(f"Video decode stopped early: {frames}/{expected} frames")
    finally:
        cap.release()
        if writer is not None:
            writer.release()
        if args.show:
            cv2.destroyAllWindows()
    # Read the encoded video back to catch failed or truncated writes.
    check = cv2.VideoCapture(str(args.output))
    decoded = 0
    while True:
        ok, _ = check.read()
        if not ok:
            break
        decoded += 1
    check.release()
    if decoded != frames:
        raise RuntimeError(f"Encoded video check failed: {decoded}/{frames} frames")
    print(json.dumps({"output": str(args.output), "frames": frames, "fps": fps, "decoded": decoded,
                      "frames_with_no_helmet": frames_with_no_helmet, "frames_with_both": frames_with_both}, ensure_ascii=False))


if __name__ == "__main__":
    main()
