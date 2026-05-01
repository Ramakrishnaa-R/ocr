"""
train_yolo.py
─────────────
Trains a YOLO26n model on the medical-pills 2-class dataset.

Classes:
    0 = occupied  (pill inside pocket)
    1 = vacant    (empty pocket)

Usage:
    python train_yolo.py
    python train_yolo.py --epochs 30 --device cuda
    python train_yolo.py --epochs 30 --device cpu
"""

import argparse
from pathlib import Path
from ultralytics import YOLO


DATASET_YAML = str(Path(__file__).parent / "archive" / "medical-pills-2class.yaml")
MODEL_WEIGHTS = str(Path(__file__).parent / "yolo26n.pt")
OUTPUT_DIR    = "runs/detect"
RUN_NAME      = "train_2class_yolo26"


def train(epochs: int = 30, device: str = "cpu", imgsz: int = 640, batch: int = 8):
    print(f"\n{'='*60}")
    print(f"  YOLO26n Training  |  device={device}  epochs={epochs}")
    print(f"{'='*60}\n")

    # Load YOLO26n pretrained weights
    model = YOLO(MODEL_WEIGHTS)

    results = model.train(
        data=DATASET_YAML,
        epochs=epochs,
        imgsz=imgsz,
        batch=batch,
        device=device,
        plots=True,
        project=OUTPUT_DIR,
        name=RUN_NAME,
        exist_ok=True,
        # YOLO26-specific: already NMS-free, no extra flags needed
    )

    best_weights = Path(OUTPUT_DIR) / RUN_NAME / "weights" / "best.pt"
    print(f"\n{'='*60}")
    print(f"  Training Complete!")
    print(f"  Best weights saved to: {best_weights}")
    print(f"  Run analyzer with:")
    print(f"    python analyzer_yolo.py <image> --model {best_weights}")
    print(f"{'='*60}\n")

    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train YOLO26n on Medical Pills (2-class)")
    parser.add_argument("--epochs", type=int,   default=30,    help="Number of training epochs")
    parser.add_argument("--device", type=str,   default="cpu", help="'cpu' or 'cuda'")
    parser.add_argument("--imgsz",  type=int,   default=640,   help="Image size")
    parser.add_argument("--batch",  type=int,   default=8,     help="Batch size")
    args = parser.parse_args()

    train(epochs=args.epochs, device=args.device, imgsz=args.imgsz, batch=args.batch)
