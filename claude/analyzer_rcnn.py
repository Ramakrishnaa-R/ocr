"""
analyzer_rcnn.py  — Inference script for medicine blister pack analysis
───────────────────────────────────────────────────────────────────────
Model: Faster R-CNN (ResNet-50-FPN)
    Class 1 = Occupied  (pill inside pocket)
    Class 2 = Vacant    (empty pocket)

Key fixes over original:
  • build_model() identical to train_rcnn.py (no weight-init mismatch)
  • CLAHE preprocessing matches training pipeline
  • Portrait-to-landscape rotation applied before OCR AND detection
  • Correct device fallback with user-visible warning
  • draw_detections() shows per-class count summary on image

Usage:
    python analyzer_rcnn.py path/to/blister.jpg --model runs/rcnn/best.pt
    python analyzer_rcnn.py path/to/blister.jpg --model runs/rcnn/best.pt --conf 0.4
    python analyzer_rcnn.py path/to/blister.jpg --model runs/rcnn/best.pt --no-save
"""

import sys, argparse
from pathlib import Path

import cv2
import numpy as np
import pytesseract
import torch
import torchvision.transforms.functional as TF
from PIL import Image

from torchvision.models.detection import fasterrcnn_resnet50_fpn
from torchvision.models.detection.rpn import AnchorGenerator
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor

# ── Windows Tesseract path ────────────────────────────────────────────────────
if sys.platform.startswith("win"):
    pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"

# ── Constants ─────────────────────────────────────────────────────────────────
NUM_CLASSES    = 3
CLASS_NAMES    = {1: "Occupied", 2: "Vacant"}
CLASS_COLORS   = {1: (0, 220, 50), 2: (0, 50, 220)}   # BGR: green / red
DEFAULT_MODEL  = "runs/rcnn/best.pt"
CONF_THRESHOLD = 0.5

# ── CLAHE (must match training) ───────────────────────────────────────────────

def apply_clahe_bgr(image_bgr: np.ndarray) -> np.ndarray:
    """CLAHE in LAB colour space — reduces blue-foil reflection artefacts."""
    lab = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2LAB)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    lab[:, :, 0] = clahe.apply(lab[:, :, 0])
    return cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)

# ── OCR ───────────────────────────────────────────────────────────────────────

def extract_medication_name(image_bgr: np.ndarray) -> str:
    """OCR on top 20% of the (already-rotated) image."""
    h, w = image_bgr.shape[:2]
    crop = image_bgr[0:max(1, int(h * 0.20)), 0:w]
    gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
    thresh = cv2.adaptiveThreshold(
        gray, 255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY,
        blockSize=11, C=2,
    )
    try:
        text = pytesseract.image_to_string(thresh, config="--psm 6").strip()
        if len(text) < 2:
            text = pytesseract.image_to_string(gray).strip()
    except pytesseract.pytesseract.TesseractNotFoundError:
        print("Warning: Tesseract not found — OCR skipped.")
        return "Could not extract text via OCR"
    except Exception as e:
        print(f"Warning: OCR failed ({e}).")
        return "Could not extract text via OCR"
    return text if len(text) >= 2 else "Could not extract text via OCR"

# ── Model (MUST be identical to train_rcnn.py build_model) ───────────────────

def build_model(num_classes: int) -> torch.nn.Module:
    """
    Identical architecture to train_rcnn.py.
    weights=None here because we always load our own checkpoint.
    """
    anchor_generator = AnchorGenerator(
        sizes=((32,), (64,), (128,), (256,), (512,)),
        aspect_ratios=((0.5, 1.0, 2.0),) * 5,
    )
    model = fasterrcnn_resnet50_fpn(
        weights=None,
        rpn_anchor_generator=anchor_generator,
    )
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)
    return model


def load_model(model_path: str, device: torch.device) -> torch.nn.Module:
    print(f"Loading Faster R-CNN weights: {model_path}")
    model = build_model(NUM_CLASSES)
    state = torch.load(model_path, map_location=device)
    model.load_state_dict(state)
    model.to(device).eval()
    print("Model loaded.\n")
    return model

# ── Inference ─────────────────────────────────────────────────────────────────

@torch.no_grad()
def detect_pockets(image_bgr: np.ndarray, model, device, conf_threshold: float):
    """
    Run Faster R-CNN on the CLAHE-preprocessed image.
    Returns list of {"box": (x1,y1,x2,y2), "label": int, "score": float}.
    """
    # Apply CLAHE before inference (matches training preprocessing)
    processed = apply_clahe_bgr(image_bgr)
    image_rgb  = cv2.cvtColor(processed, cv2.COLOR_BGR2RGB)
    img_tensor = TF.to_tensor(Image.fromarray(image_rgb)).to(device)

    preds = model([img_tensor])[0]

    detections = []
    for box, label, score in zip(
        preds["boxes"].cpu(),
        preds["labels"].cpu(),
        preds["scores"].cpu(),
    ):
        if score.item() < conf_threshold: continue
        x1, y1, x2, y2 = box.int().tolist()
        detections.append({
            "box":   (x1, y1, x2, y2),
            "label": label.item(),
            "score": round(score.item(), 3),
        })
    return detections

# ── Annotated image ───────────────────────────────────────────────────────────

def draw_detections(image_bgr: np.ndarray, detections: list,
                    occupied: int, vacant: int) -> np.ndarray:
    """Draws boxes + summary banner onto a copy of the image."""
    out = image_bgr.copy()
    for det in detections:
        x1, y1, x2, y2 = det["box"]
        lbl   = det["label"]
        name  = CLASS_NAMES.get(lbl, f"cls{lbl}")
        color = CLASS_COLORS.get(lbl, (200, 200, 200))
        cv2.rectangle(out, (x1, y1), (x2, y2), color, 3)
        text = f"{name} {det['score']:.2f}"
        (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.65, 2)
        cv2.rectangle(out, (x1, y1 - th - 8), (x1 + tw + 4, y1), color, -1)
        cv2.putText(out, text, (x1 + 2, y1 - 5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 255, 255), 2)

    # Summary banner at bottom
    h, w = out.shape[:2]
    banner = f"Occupied={occupied}  Vacant={vacant}  Remaining={occupied}"
    cv2.rectangle(out, (0, h - 36), (w, h), (30, 30, 30), -1)
    cv2.putText(out, banner, (8, h - 10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
    return out

# ── Full pipeline ─────────────────────────────────────────────────────────────

def analyze_blister_pack(
    image_path: str,
    model_path: str = DEFAULT_MODEL,
    conf_threshold: float = CONF_THRESHOLD,
    device_str: str = "cpu",
    save_output: bool = True,
) -> dict:
    # ── Load image ────────────────────────────────────────────────────────
    print(f"Loading image: {image_path}")
    image = cv2.imread(image_path)
    if image is None:
        raise ValueError(f"Could not read image: {image_path}")

    # Rotate portrait → landscape (same as training)
    h, w = image.shape[:2]
    rotated = False
    if h > w:
        print("Portrait image detected — rotating to landscape...")
        image   = cv2.rotate(image, cv2.ROTATE_90_CLOCKWISE)
        rotated = True

    # ── Step 1: OCR ───────────────────────────────────────────────────────
    print("Step 1 — OCR (top 20%)...")
    med_name = extract_medication_name(image)
    print(f"  → {med_name!r}\n")

    # ── Step 2: Load model ────────────────────────────────────────────────
    if device_str == "cuda" and not torch.cuda.is_available():
        print("Warning: CUDA not available — falling back to CPU.")
        device_str = "cpu"
    device = torch.device(device_str)

    model_file = Path(model_path)
    if not model_file.exists():
        raise FileNotFoundError(
            f"Weights not found: {model_path}\n"
            "Train first:  python train_rcnn.py --data medical-pills-2class.yaml"
        )
    rcnn = load_model(str(model_file), device)

    # ── Step 3: Detect ────────────────────────────────────────────────────
    print(f"Step 2 — Faster R-CNN inference (conf ≥ {conf_threshold})...")
    detections = detect_pockets(image, rcnn, device, conf_threshold)

    # ── Step 4: Count ─────────────────────────────────────────────────────
    occupied  = sum(1 for d in detections if d["label"] == 1)
    vacant    = sum(1 for d in detections if d["label"] == 2)
    total     = occupied + vacant
    remaining = occupied   # spec: Remaining = Occupied count

    # ── Step 5: Save output ───────────────────────────────────────────────
    output_path = None
    if save_output:
        annotated   = draw_detections(image, detections, occupied, vacant)
        stem        = Path(image_path).stem
        output_path = f"rcnn_output_{stem}.jpg"
        cv2.imwrite(output_path, annotated)
        print(f"  Annotated image → {output_path}\n")

    # ── Report ────────────────────────────────────────────────────────────
    sep = "=" * 52
    print(f"\n{sep}")
    print("  FASTER R-CNN  ANALYSIS REPORT")
    print(sep)
    print(f"  Medication Name        : {med_name}")
    print(f"  Rotated (portrait fix) : {rotated}")
    print(f"  Confidence Threshold   : {conf_threshold}")
    print(f"  Total Pockets Detected : {total}")
    print(f"    ✓ Occupied (pill in) : {occupied}")
    print(f"    ✗ Vacant   (empty)   : {vacant}")
    print(f"  Remaining Pill Count   : {remaining}")
    if output_path:
        print(f"  Annotated Output       : {output_path}")
    print(f"{sep}\n")

    return {
        "med_name":   med_name,
        "total":      total,
        "occupied":   occupied,
        "vacant":     vacant,
        "remaining":  remaining,
        "detections": detections,
    }

# ── CLI ───────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Analyze blister pack with Faster R-CNN + Tesseract OCR"
    )
    parser.add_argument("image_path")
    parser.add_argument("--model",  default=DEFAULT_MODEL)
    parser.add_argument("--conf",   type=float, default=CONF_THRESHOLD)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--no-save", dest="save", action="store_false")
    parser.set_defaults(save=True)
    args = parser.parse_args()

    analyze_blister_pack(
        image_path     = args.image_path,
        model_path     = args.model,
        conf_threshold = args.conf,
        device_str     = args.device,
        save_output    = args.save,
    )
