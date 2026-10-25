"""Run the trained checkpoint over all supported images in a folder."""
from __future__ import annotations

import argparse
from pathlib import Path
from ultralytics import YOLO


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_MODEL = REPO_ROOT / "models" / "best.pt"


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Batch-predict a folder of sewer inspection images."
    )
    parser.add_argument("--source", required=True, help="Folder containing images.")
    parser.add_argument("--model", default=str(DEFAULT_MODEL), help="YOLO checkpoint path.")
    parser.add_argument("--conf", type=float, default=0.25, help="Confidence threshold.")
    parser.add_argument("--imgsz", type=int, default=640, help="Inference image size.")
    parser.add_argument("--name", default="batch_predict", help="Output folder name.")
    args = parser.parse_args()

    source = Path(args.source).expanduser().resolve()
    model_path = Path(args.model).expanduser().resolve()

    if not source.is_dir():
        raise SystemExit(f"Source must be an existing directory: {source}")
    if not model_path.is_file():
        raise SystemExit(f"Checkpoint not found: {model_path}")
    if not 0.0 <= args.conf <= 1.0:
        raise SystemExit("--conf must be between 0 and 1.")

    model = YOLO(str(model_path))
    print(f"Classes: {model.names}")
    results = model.predict(
        source=str(source),
        conf=args.conf,
        imgsz=args.imgsz,
        save=True,
        project="runs/detect",
        name=args.name,
        exist_ok=False,
    )

    image_count = 0
    detection_count = 0
    for result in results:
        image_count += 1
        count = 0 if result.boxes is None else len(result.boxes)
        detection_count += count
        print(f"{Path(result.path).name}: {count} detection(s)")

    print(f"Processed images: {image_count}")
    print(f"Total detections: {detection_count}")
    print("Batch prediction complete. Check the output path printed by Ultralytics.")


if __name__ == "__main__":
    main()
