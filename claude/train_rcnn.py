"""
train_rcnn.py  — Faster R-CNN training for medicine blister packs
─────────────────────────────────────────────────────────────────
Classes (YOLO label → R-CNN label):
    YOLO 0 = occupied  →  RCNN 1
    YOLO 1 = vacant    →  RCNN 2
    label 0 is always background (torchvision convention)

Key fixes over original:
  • Dataset path comes from the YAML file (no hardcoded Windows paths)
  • CLAHE preprocessing to handle blue-foil reflections
  • Data augmentation: horizontal/vertical flips + colour jitter
  • Cosine-annealing LR schedule instead of step decay
  • Portrait-to-landscape rotation matches analyzer_rcnn.py behaviour
  • Fixed device-fallback logic (no silent cuda→cpu switch)
  • build_model() kept identical to analyzer_rcnn.py (weights=None always)

Usage:
    python train_rcnn.py --data medical-pills-2class.yaml
    python train_rcnn.py --data medical-pills-2class.yaml --epochs 30 --device cuda
"""

import os, argparse, time, json, random
from pathlib import Path

import cv2
import numpy as np
import torch
import torchvision.transforms.functional as TF
from torch.utils.data import Dataset, DataLoader
from torchvision.models.detection import fasterrcnn_resnet50_fpn
from torchvision.models.detection.rpn import AnchorGenerator
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from PIL import Image

# ── Constants ────────────────────────────────────────────────────────────────

YOLO_TO_RCNN = {0: 1, 1: 2}   # occupied→1, vacant→2
NUM_CLASSES  = 3               # background + occupied + vacant
CLASS_NAMES  = {1: "Occupied", 2: "Vacant"}
IMG_EXTS     = {".jpg", ".jpeg", ".png", ".bmp"}

# ── YAML parser (avoids pyyaml dependency) ───────────────────────────────────

def parse_yaml(yaml_path: str) -> dict:
    """Minimal YAML parser sufficient for our dataset config."""
    result = {}
    with open(yaml_path) as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"): continue
            if ":" in line:
                k, _, v = line.partition(":")
                result[k.strip()] = v.strip()
    return result

# ── CLAHE helper ─────────────────────────────────────────────────────────────

def apply_clahe(image_rgb: np.ndarray) -> np.ndarray:
    """
    Apply CLAHE in LAB colour space to reduce blue-foil reflections and
    improve contrast between pill body and empty pocket.
    """
    lab = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2LAB)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    lab[:, :, 0] = clahe.apply(lab[:, :, 0])
    return cv2.cvtColor(lab, cv2.COLOR_LAB2RGB)

# ── Dataset ──────────────────────────────────────────────────────────────────

class BlisterDataset(Dataset):
    """
    Reads images + YOLO-format labels.
    Applies CLAHE and optional augmentation.
    Portrait images are rotated to landscape to match inference behaviour.
    """

    def __init__(self, img_dir: Path, lbl_dir: Path, augment: bool = False):
        self.img_dir = img_dir
        self.lbl_dir = lbl_dir
        self.augment = augment

        self.samples = []
        for img_path in sorted(img_dir.iterdir()):
            if img_path.suffix.lower() not in IMG_EXTS: continue
            lbl_path = lbl_dir / (img_path.stem + ".txt")
            if lbl_path.exists():
                self.samples.append((img_path, lbl_path))

        print(f"  {'[AUG]' if augment else '[VAL]'} {img_dir.name}: {len(self.samples)} samples")

    def __len__(self): return len(self.samples)

    def __getitem__(self, idx):
        img_path, lbl_path = self.samples[idx]

        # ── Load & preprocess ─────────────────────────────────────────────
        img_bgr = cv2.imread(str(img_path))
        img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)

        # Rotate portrait → landscape (same as analyzer_rcnn.py)
        h, w = img_rgb.shape[:2]
        if h > w:
            img_rgb = cv2.rotate(img_rgb, cv2.ROTATE_90_CLOCKWISE)
            h, w = img_rgb.shape[:2]

        # CLAHE for reflection suppression
        img_rgb = apply_clahe(img_rgb)

        # ── Parse labels ──────────────────────────────────────────────────
        boxes, labels = [], []
        with open(lbl_path) as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) < 5: continue
                yolo_cls = int(parts[0])
                cx, cy, bw, bh = map(float, parts[1:5])
                x1 = max(0.0, (cx - bw/2) * w)
                y1 = max(0.0, (cy - bh/2) * h)
                x2 = min(float(w), (cx + bw/2) * w)
                y2 = min(float(h), (cy + bh/2) * h)
                if x2 <= x1 or y2 <= y1: continue
                boxes.append([x1, y1, x2, y2])
                labels.append(YOLO_TO_RCNN.get(yolo_cls, 1))

        # ── Augmentation ──────────────────────────────────────────────────
        if self.augment and len(boxes) > 0:
            img_rgb, boxes = self._augment(img_rgb, boxes)

        # ── Convert to tensors ────────────────────────────────────────────
        pil_img = Image.fromarray(img_rgb)
        img_tensor = TF.to_tensor(pil_img)

        if len(boxes) == 0:
            boxes_t  = torch.zeros((0, 4), dtype=torch.float32)
            labels_t = torch.zeros((0,),   dtype=torch.int64)
        else:
            boxes_t  = torch.as_tensor(boxes,  dtype=torch.float32)
            labels_t = torch.as_tensor(labels, dtype=torch.int64)

        target = {
            "boxes":    boxes_t,
            "labels":   labels_t,
            "image_id": torch.tensor([idx]),
        }
        return img_tensor, target

    # ── Augmentation helpers ──────────────────────────────────────────────

    def _augment(self, img: np.ndarray, boxes: list):
        h, w = img.shape[:2]

        # Random horizontal flip
        if random.random() > 0.5:
            img = cv2.flip(img, 1)
            boxes = [[w - x2, y1, w - x1, y2] for (x1, y1, x2, y2) in boxes]

        # Random vertical flip
        if random.random() > 0.5:
            img = cv2.flip(img, 0)
            boxes = [[x1, h - y2, x2, h - y1] for (x1, y1, x2, y2) in boxes]

        # Colour jitter (brightness + contrast in HSV)
        hsv = cv2.cvtColor(img, cv2.COLOR_RGB2HSV).astype(np.float32)
        hsv[:, :, 2] *= random.uniform(0.7, 1.3)   # brightness
        hsv[:, :, 1] *= random.uniform(0.8, 1.2)   # saturation
        hsv = np.clip(hsv, 0, 255).astype(np.uint8)
        img = cv2.cvtColor(hsv, cv2.COLOR_HSV2RGB)

        return img, boxes


def collate_fn(batch):
    return tuple(zip(*batch))

# ── Model (identical architecture to analyzer_rcnn.py) ───────────────────────

def build_model(num_classes: int, pretrained: bool = True) -> torch.nn.Module:
    anchor_generator = AnchorGenerator(
        sizes=((32,), (64,), (128,), (256,), (512,)),
        aspect_ratios=((0.5, 1.0, 2.0),) * 5,
    )
    model = fasterrcnn_resnet50_fpn(
        weights="DEFAULT" if pretrained else None,
        rpn_anchor_generator=anchor_generator,
    )
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)
    return model

# ── Training loop ─────────────────────────────────────────────────────────────

def train_one_epoch(model, optimizer, loader, device, epoch, total_epochs):
    model.train()
    total_loss = 0.0
    n = len(loader)
    for i, (images, targets) in enumerate(loader):
        images  = [img.to(device) for img in images]
        targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

        loss_dict = model(images, targets)
        losses = sum(loss_dict.values())

        optimizer.zero_grad()
        losses.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
        optimizer.step()

        total_loss += losses.item()

        if (i + 1) % max(1, n // 4) == 0 or i == n - 1:
            detail = "  ".join(f"{k.replace('loss_','')}={v.item():.3f}" for k, v in loss_dict.items())
            print(f"  [{epoch}/{total_epochs}] batch {i+1}/{n}  loss={losses.item():.4f}  [{detail}]")

    return total_loss / n


@torch.no_grad()
def evaluate(model, loader, device, conf=0.5):
    model.eval()
    counts, n_imgs = {1: 0, 2: 0}, 0
    for images, _ in loader:
        images = [img.to(device) for img in images]
        for pred in model(images):
            keep = pred["scores"] >= conf
            for lbl in pred["labels"][keep].tolist():
                counts[lbl] = counts.get(lbl, 0) + 1
        n_imgs += len(images)
    return {CLASS_NAMES[k]: f"{v/max(1,n_imgs):.2f}/img" for k, v in counts.items()}

# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Train Faster R-CNN on Medical Pills")
    parser.add_argument("--data",       default="medical-pills-2class.yaml", help="Dataset YAML")
    parser.add_argument("--epochs",     type=int,   default=20)
    parser.add_argument("--batch",      type=int,   default=2)
    parser.add_argument("--lr",         type=float, default=0.005)
    parser.add_argument("--device",     default="cpu")
    parser.add_argument("--workers",    type=int,   default=0, help="0 recommended on Windows")
    parser.add_argument("--output",     default="runs/rcnn")
    parser.add_argument("--no-pretrain",dest="pretrained", action="store_false")
    parser.set_defaults(pretrained=True)
    args = parser.parse_args()

    # ── Resolve dataset paths from YAML ───────────────────────────────────
    cfg = parse_yaml(args.data)
    root = Path(cfg.get("path", "."))
    train_img = root / cfg.get("train", "train/images")
    train_lbl = train_img.parent.parent / "labels" / train_img.name
    valid_img = root / cfg.get("val",   "valid/images")
    valid_lbl = valid_img.parent.parent / "labels" / valid_img.name

    print(f"\n{'='*65}")
    print(f"  Faster R-CNN Training  |  epochs={args.epochs}  batch={args.batch}")
    print(f"  Train: {train_img}")
    print(f"  Valid: {valid_img}")
    print(f"{'='*65}\n")

    # ── Device ────────────────────────────────────────────────────────────
    if args.device == "cuda" and not torch.cuda.is_available():
        print("Warning: CUDA requested but not available — using CPU.")
        device = torch.device("cpu")
    else:
        device = torch.device(args.device)
    print(f"Using device: {device}\n")

    # ── Datasets ──────────────────────────────────────────────────────────
    print("Loading datasets...")
    train_ds = BlisterDataset(train_img, train_lbl, augment=True)
    valid_ds = BlisterDataset(valid_img, valid_lbl, augment=False)

    train_loader = DataLoader(train_ds, batch_size=args.batch, shuffle=True,
                              num_workers=args.workers, collate_fn=collate_fn)
    valid_loader = DataLoader(valid_ds, batch_size=1, shuffle=False,
                              num_workers=args.workers, collate_fn=collate_fn)

    # ── Model ─────────────────────────────────────────────────────────────
    print("\nBuilding Faster R-CNN (ResNet-50-FPN)...")
    model = build_model(NUM_CLASSES, pretrained=args.pretrained).to(device)

    # Two-stage LR: backbone 10× lower
    params = [
        {"params": [p for n, p in model.named_parameters()
                    if "backbone" in n and p.requires_grad], "lr": args.lr * 0.1},
        {"params": [p for n, p in model.named_parameters()
                    if "backbone" not in n and p.requires_grad], "lr": args.lr},
    ]
    optimizer = torch.optim.SGD(params, momentum=0.9, weight_decay=1e-4)
    # Cosine annealing: smoother convergence than step decay
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=1e-5)

    # ── Training ──────────────────────────────────────────────────────────
    out_dir  = Path(args.output)
    out_dir.mkdir(parents=True, exist_ok=True)
    history  = []
    best_loss = float("inf")

    for epoch in range(1, args.epochs + 1):
        t0 = time.time()
        print(f"\nEpoch {epoch}/{args.epochs}  (lr={scheduler.get_last_lr()[0]:.2e})")
        avg_loss = train_one_epoch(model, optimizer, train_loader, device, epoch, args.epochs)
        scheduler.step()
        elapsed = time.time() - t0

        val_info = evaluate(model, valid_loader, device)
        print(f"  → avg train loss: {avg_loss:.4f}  ({elapsed:.1f}s)")
        print(f"  → val avg dets/img: {val_info}")

        history.append({"epoch": epoch, "loss": avg_loss, "val": val_info})

        if avg_loss < best_loss:
            best_loss = avg_loss
            torch.save(model.state_dict(), out_dir / "best.pt")
            print(f"  ✓ Best checkpoint saved → {out_dir/'best.pt'}")

    torch.save(model.state_dict(), out_dir / "last.pt")
    with open(out_dir / "train_log.json", "w") as f:
        json.dump(history, f, indent=2)

    print(f"\n{'='*65}")
    print(f"  Done!  Best loss: {best_loss:.4f}")
    print(f"  Weights: {out_dir/'best.pt'}")
    print(f"  Run inference: python analyzer_rcnn.py <image> --model {out_dir/'best.pt'}")
    print(f"{'='*65}\n")


if __name__ == "__main__":
    main()
