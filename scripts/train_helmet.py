"""Run the helmet experiment with project-local configuration and status files."""
from __future__ import annotations

import argparse
from datetime import datetime
import json
import os
from pathlib import Path
import platform
import subprocess
import traceback
from zoneinfo import ZoneInfo

from monitor_helmet import choose_device

ROOT = Path(__file__).resolve().parents[1]
os.environ.setdefault("YOLO_CONFIG_DIR", str(ROOT / ".config/ultralytics"))
os.environ.setdefault("MPLCONFIGDIR", str(ROOT / ".config/matplotlib"))
os.environ.setdefault("YOLO_AUTOINSTALL", "false")
os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")
Path(os.environ["YOLO_CONFIG_DIR"]).mkdir(parents=True, exist_ok=True)
Path(os.environ["MPLCONFIGDIR"]).mkdir(parents=True, exist_ok=True)


def now():
    return datetime.now(ZoneInfo("Asia/Shanghai")).isoformat()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-name", required=True)
    parser.add_argument("--device", default="auto", help="auto, cpu, mps, or CUDA device index")
    args = parser.parse_args()
    run_dir = ROOT / "runs/helmet" / args.run_name
    if run_dir.exists():
        raise FileExistsError(f"Refusing to overwrite existing run: {run_dir}")
    job = run_dir / "logs"
    job.mkdir(parents=True)
    status = {"started_at": now(), "pid": os.getpid(), "phase": "initializing", "epochs_requested": 100, "batch": 8}

    def update(**values):
        status.update(values, updated_at=now())
        temp = job / "status.tmp"
        temp.write_text(json.dumps(status, indent=2, ensure_ascii=False) + "\n")
        temp.replace(job / "status.json")

    update()
    try:
        import torch
        import ultralytics
        from ultralytics import YOLO, settings

        settings.update({"sync": False, "weights_dir": str(ROOT / "weights"), "runs_dir": str(ROOT / "runs"), "datasets_dir": str(ROOT / "datasets")})
        device = choose_device(args.device)
        env = {"python": platform.python_version(), "platform": platform.platform(), "torch": torch.__version__,
               "ultralytics": ultralytics.__version__, "device": device,
               "git_commit": subprocess.check_output(["git", "-C", str(ROOT / "ultralytics"), "rev-parse", "HEAD"], text=True).strip()}
        (job / "environment.json").write_text(json.dumps(env, indent=2) + "\n")
        (ROOT / "weights").mkdir(exist_ok=True)
        update(phase="loading_pretrained_weights", device=device)
        model = YOLO(str(ROOT / "weights/yolo26n.pt"))
        options = {"data": str(ROOT / "datasets/hard-hat-workers/yolo/data.yaml"), "epochs": 100, "batch": 8,
                   "imgsz": 640, "patience": 20, "seed": 42, "device": device, "workers": 2,
                   "project": str(ROOT / "runs/helmet"), "name": args.run_name, "exist_ok": True,
                   "save": True, "plots": True, "cache": False}
        (job / "requested-config.json").write_text(json.dumps(options, indent=2) + "\n")

        def started(trainer):
            update(phase="training", output_dir=str(trainer.save_dir), actual_batch=trainer.args.batch)

        def epoch_finished(trainer):
            metrics = {k: float(v) for k, v in trainer.metrics.items()}
            update(phase="training", epochs_completed=trainer.epoch + 1, metrics=metrics,
                   best_fitness=float(trainer.best_fitness))

        model.add_callback("on_train_start", started)
        model.add_callback("on_fit_epoch_end", epoch_finished)
        model.train(**options)
        update(phase="completed", finished_at=now(), output_dir=str(model.trainer.save_dir),
               epochs_completed=model.trainer.epoch + 1,
               best_weights=str(model.trainer.best), last_weights=str(model.trainer.last))
    except BaseException as exc:
        update(phase="failed", finished_at=now(), error=repr(exc))
        traceback.print_exc()
        raise


if __name__ == "__main__":
    main()
