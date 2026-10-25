"""Run inference with the trained sewer-obstruction YOLO checkpoint."""
from __future__ import annotations

import argparse
from pathlib import Path
from ultralytics import YOLO


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_MODEL = REPO_ROOT / "models" / "best.pt"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Detect the trained tree_root and obstacle classes."
    )
    parser.add_argument(
        "--source",
        required=True,
        help="Image path, directory, video, or other Ultralytics-supported source.",
    )
    parser.add_argument(
        "--model",
        default=str(DEFAULT_MODEL),
        help=f"Path to YOLO checkpoint (default: {DEFAULT_MODEL})",
    )
    parser.add_argument(
        "--conf",
        type=float,
        default=0.25,
        help="Confidence threshold between 0 and 1 (default: 0.25).",
    )
    parser.add_argument(
        "--imgsz",
        type=int,
        default=640,
        help="Inference image size (default: 640).",
    )
    parser.add_argument(
        "--project",
        default="runs/detect",
        help="Directory under which prediction output is saved.",
    )
    parser.add_argument(
        "--name",
        default="predict",
        help="Name of the prediction output folder.",
    )
    parser.add_argument(
        "--show",
        action="store_true",
        help="Display prediction windows where supported.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    if not 0.0 <= args.conf <= 1.0:
        raise SystemExit("--conf must be between 0 and 1.")
    if args.imgsz <= 0:
        raise SystemExit("--imgsz must be a positive integer.")

    model_path = Path(args.model).expanduser().resolve()
    if not model_path.is_file():
        raise SystemExit(
            f"Checkpoint not found: {model_path}\n"
            "Place the checkpoint at models/best.pt or pass --model PATH."
        )

    source_path = Path(args.source).expanduser()
    if not source_path.exists():
        raise SystemExit(f"Prediction source not found: {source_path}")

    model = YOLO(str(model_path))
    print(f"Loaded model: {model_path}")
    print(f"Task: {model.task}")
    print(f"Classes: {model.names}")

    results = model.predict(
        source=str(source_path),
        conf=args.conf,
        imgsz=args.imgsz,
        save=True,
        show=args.show,
        project=args.project,
        name=args.name,
        exist_ok=False,
    )

    total_detections = 0
    for result in results:
        count = 0 if result.boxes is None else len(result.boxes)
        total_detections += count
        print(f"{result.path}: {count} detection(s)")
        if result.boxes is not None:
            for box in result.boxes:
                class_id = int(box.cls.item())
                confidence = float(box.conf.item())
                class_name = model.names.get(class_id, str(class_id))
                print(f"  - {class_name}: confidence={confidence:.3f}")

    print(f"Total detections: {total_detections}")
    print("Prediction run completed. See the Ultralytics output folder above.")


if __name__ == "__main__":
    main()
