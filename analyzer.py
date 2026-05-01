"""
analyzer.py
──────────────
Simplified blister pack analyzer:

  Locates pills in the image using YOLO26n and outputs the total quantity.

Usage:
    python analyzer.py path/to/image.jpg
    python analyzer.py path/to/image.jpg --conf 0.25
"""

import cv2
import numpy as np
import pytesseract
import argparse
import sys
from pathlib import Path
from ultralytics import YOLO

if sys.platform.startswith("win"):
    pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"

# ── Config ────────────────────────────────────────────────────────────────────
DEFAULT_MODEL = r"runs\detect\train_2class_yolo26\weights\best.pt"
CONF_THRESHOLD = 0.25   # Default confidence for pill detection

# Minimum box area to consider (filters noise)
MIN_BOX_AREA = 100


# ── OCR ───────────────────────────────────────────────────────────────────────
def extract_med_name(image: np.ndarray) -> str:
    h, w = image.shape[:2]
    crop = image[0 : int(h * 0.25), 0:w]
    gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
    thresh = cv2.adaptiveThreshold(
        gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2
    )
    try:
        text = pytesseract.image_to_string(thresh, config="--psm 6").strip()
        if len(text) < 2:
            text = pytesseract.image_to_string(gray).strip()
    except Exception:
        text = ""
    return text if len(text) >= 2 else "Could not extract text via OCR"


# ── Main Pipeline ─────────────────────────────────────────────────────────────
def analyze(image_path: str, model_path: str, conf: float, debug: bool, save: bool) -> dict:

    # Load image
    print(f"\nLoading image: {image_path}")
    image = cv2.imread(image_path)
    if image is None:
        raise FileNotFoundError(f"Cannot read: {image_path}")

    # Auto-rotate portrait
    h, w = image.shape[:2]
    if h > w:
        print("Portrait image — rotating to landscape...")
        image = cv2.rotate(image, cv2.ROTATE_90_CLOCKWISE)

    # ── Stage 1: OCR ──────────────────────────────────────────────────────────
    print("Reading medication name via OCR...")
    med_name = extract_med_name(image)
    print(f"  → {med_name!r}\n")

    # ── Stage 2: YOLO to LOCATE pills ──────────────────────────────────────
    print(f"Locating pills with YOLO26n (conf={conf})...")
    model = YOLO(model_path)
    results = model(image, verbose=False, conf=conf)
    raw_boxes = results[0].boxes

    # Collect all detected boxes
    pills = []
    for box in raw_boxes:
        x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
        area = (x2 - x1) * (y2 - y1)
        if area < MIN_BOX_AREA:
            continue
        pills.append((x1, y1, x2, y2))

    quantity = len(pills)
    print(f"  → {quantity} pills located by YOLO\n")

    if quantity == 0:
        print("No pills found. Try lowering --conf (e.g. --conf 0.1)")
        return {}

    output = image.copy()
    color = (0, 220, 60)  # Green

    for (x1, y1, x2, y2) in pills:
        cv2.rectangle(output, (x1, y1), (x2, y2), color, 3)
        # Add label "Pill"
        text = "Pill"
        (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2)
        cv2.rectangle(output, (x1, y1 - th - 6), (x1 + tw + 4, y1), color, -1)
        cv2.putText(output, text, (x1 + 2, y1 - 4),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)

    # ── Report ────────────────────────────────────────────────────────────────
    print("\n" + "=" * 50)
    print("  BLISTER PACK ANALYSIS REPORT  (v2)")
    print("=" * 50)
    print(f"  Medication            : {med_name}")
    print(f"  Total Pills Found     : {quantity}")
    print("=" * 50)

    if save:
        stem = Path(image_path).stem
        out_path = f"output_v2_{stem}.jpg"
        cv2.imwrite(out_path, output)
        print(f"\n  Annotated image saved → {out_path}")

    return {
        "med_name": med_name,
        "quantity": quantity,
    }


# ── CLI ───────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Blister Pack Analyzer v2 — YOLO Pill Counter")
    parser.add_argument("image_path", help="Path to the blister pack image")
    parser.add_argument("--model",  default=DEFAULT_MODEL, help="YOLO .pt weights path")
    parser.add_argument("--conf",   type=float, default=CONF_THRESHOLD,
                        help=f"YOLO confidence threshold for pill location (default {CONF_THRESHOLD})")
    parser.add_argument("--debug",  action="store_true", help="Debug flag (unused)")
    parser.add_argument("--no-save", dest="save", action="store_false", help="Don't save output image")
    parser.set_defaults(save=True)
    args = parser.parse_args()

    analyze(args.image_path, args.model, args.conf, args.debug, args.save)
