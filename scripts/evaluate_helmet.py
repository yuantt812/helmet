"""Evaluate the frozen best checkpoint on the held-out test split."""
from pathlib import Path
import argparse
import hashlib
import json
import os

ROOT = Path(__file__).resolve().parents[1]
os.environ["YOLO_CONFIG_DIR"] = str(ROOT / ".config/ultralytics")
os.environ["MPLCONFIGDIR"] = str(ROOT / ".config/matplotlib")
os.environ["YOLO_AUTOINSTALL"] = "false"
os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"


if __name__ == "__main__":
    from ultralytics import YOLO
    from ultralytics.models.yolo.detect import DetectionValidator
    import numpy as np

    from monitor_helmet import choose_device
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-name", required=True, help="New output name; existing directories are refused")
    parser.add_argument("--weights", type=Path, default=ROOT / "weights/best.pt")
    parser.add_argument("--device", default="auto")
    args = parser.parse_args()
    weights = args.weights.resolve()
    if (ROOT / "runs/evaluation" / args.run_name).exists():
        raise FileExistsError("Evaluation output already exists; choose a new --run-name")
    options = dict(data=str(ROOT / "datasets/hard-hat-workers/yolo/data.yaml"), split="test", imgsz=640,
                   batch=8, device=choose_device(args.device), workers=0, conf=0.001, iou=0.7, augment=False,
                   project=str(ROOT / "runs/evaluation"), name=args.run_name, exist_ok=False,
                   plots=True, save_json=False, verbose=True)
    model = YOLO(str(weights))
    captured = []
    class CaptureValidator(DetectionValidator):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            captured.append(self)
    metrics = model.val(validator=CaptureValidator, **options)
    validator = captured[0]
    result = {"weights": str(weights), "weights_sha256": hashlib.file_digest(weights.open("rb"), "sha256").hexdigest(),
              "options": options, "overall": metrics.results_dict, "classes": metrics.summary(decimals=8),
              "confusion_matrix": validator.confusion_matrix.matrix.tolist(),
              "confusion_matrix_note": "Rows=predicted, columns=true; conf=0.001, matching IoU=0.45. Not the P/R operating point.",
              "ap_by_iou": {metrics.names[int(c)]: metrics.box.all_ap[i].tolist() for i,c in enumerate(metrics.box.ap_class_index)}}
    # Record the operating point that the framework used for its reported P/R.
    from ultralytics.utils.metrics import smooth
    idx = int(smooth(metrics.box.f1_curve.mean(0), 0.1).argmax())
    result["framework_pr_confidence"] = float(metrics.box.px[idx])
    result["pr_note"] = "Framework P/R uses the test curve's maximum smoothed mean-F1 point; not a validation-selected deployment threshold."
    out = Path(metrics.save_dir)
    serialized = json.dumps(result, indent=2, default=lambda x: x.item() if isinstance(x, np.generic) else str(x))
    (out / "metrics.json").write_text(serialized + "\n")
    print("EVALUATION_SUMMARY", serialized)
