import cv2
import pytesseract
import re
import numpy as np
import sys
from pathlib import Path
from ultralytics import YOLO

# --- CONFIGURATION ---
BASE_MODEL_PATH = r"E:\FOS\ocr\yolo26n.pt"
TRAINED_MODEL_PATH = r"E:\FOS\ocr\runs\detect\train_2class_yolo26\weights\best.pt"
if sys.platform.startswith("win"):
    pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"

# --- PILL COUNTING (YOLO) ---
def get_pill_count(img, model):
    if model is None:
        return 0, img.copy()

    results = model(img, conf=0.15, verbose=False)
    boxes = results[0].boxes
    return len(boxes), results[0].plot()


def choose_inference_model_path():
    trained = Path(TRAINED_MODEL_PATH)
    base = Path(BASE_MODEL_PATH)

    if trained.exists():
        return trained, "trained_best"
    if base.exists():
        print("Warning: trained best.pt not found, falling back to base YOLO checkpoint.")
        print("         Detection quality may be worse until you train the model.")
        return base, "base_checkpoint"

    return None, "missing"


def load_model():
    model_file, model_kind = choose_inference_model_path()
    if model_file is None:
        print("Warning: no YOLO model found.")
        print(f"         Expected trained model: {TRAINED_MODEL_PATH}")
        print(f"         Optional base model   : {BASE_MODEL_PATH}")
        return None, "missing"
    return YOLO(str(model_file)), model_kind


def get_blister_roi(img):
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    lower = np.array([85, 50, 40], dtype=np.uint8)
    upper = np.array([140, 255, 255], dtype=np.uint8)
    mask = cv2.inRange(hsv, lower, upper)
    kernel = np.ones((7, 7), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=2)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=1)

    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return img.copy(), (0, 0, img.shape[1], img.shape[0]), None

    largest = max(contours, key=cv2.contourArea)
    x, y, w, h = cv2.boundingRect(largest)
    if w <= 0 or h <= 0:
        return img.copy(), (0, 0, img.shape[1], img.shape[0]), None

    return img[y:y + h, x:x + w].copy(), (x, y, w, h), largest


# --- PILL / SLOT COUNTING ---
def cluster_positions(values, tolerance):
    groups = []
    for value in sorted(values):
        if not groups or abs(value - groups[-1][-1]) > tolerance:
            groups.append([value])
        else:
            groups[-1].append(value)
    return [int(sum(group) / len(group)) for group in groups]


def draw_fallback_grid(annotated_img, rect, centers):
    x0, y0, _, _ = rect
    for x, y, r in centers:
        cv2.circle(annotated_img, (x0 + x, y0 + y), r, (0, 255, 255), 3)
        cv2.circle(annotated_img, (x0 + x, y0 + y), 2, (0, 0, 255), -1)


def draw_component_boxes(annotated_img, rect, boxes):
    x0, y0, _, _ = rect
    for x, y, w, h in boxes:
        cv2.rectangle(
            annotated_img,
            (x0 + x, y0 + y),
            (x0 + x + w, y0 + y + h),
            (0, 255, 255),
            2,
        )


def estimate_blister_slots(img):
    roi, rect, _ = get_blister_roi(img)
    roi_h, roi_w = roi.shape[:2]
    roi_gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    roi_blur = cv2.GaussianBlur(roi_gray, (7, 7), 0)
    _, pill_mask = cv2.threshold(roi_blur, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    kernel = np.ones((5, 5), np.uint8)
    pill_mask = cv2.morphologyEx(pill_mask, cv2.MORPH_OPEN, kernel, iterations=1)
    pill_mask = cv2.morphologyEx(pill_mask, cv2.MORPH_CLOSE, kernel, iterations=2)

    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(pill_mask, connectivity=8)
    if num_labels <= 1:
        return 0, img.copy()

    min_area = max(300, (roi_h * roi_w) // 120)
    max_area = max(2000, (roi_h * roi_w) // 8)
    component_boxes = []
    centers = []

    for label_index in range(1, num_labels):
        x, y, w, h, area = stats[label_index]
        if area < min_area or area > max_area:
            continue
        aspect_ratio = w / max(h, 1)
        fill_ratio = area / max(w * h, 1)
        if not (0.55 <= aspect_ratio <= 1.8):
            continue
        if fill_ratio < 0.45:
            continue
        if x < 3 or y < 3 or x + w > roi_w - 3 or y + h > roi_h - 3:
            continue

        component_boxes.append((x, y, w, h))
        centers.append((x + w // 2, y + h // 2, max(w, h) // 2))

    if not centers:
        return 0, img.copy()

    median_radius = int(np.median([r for _, _, r in centers]))
    x_groups = cluster_positions([x for x, _, _ in centers], tolerance=max(24, median_radius))
    y_groups = cluster_positions([y for _, y, _ in centers], tolerance=max(24, median_radius))
    grid_count = len(x_groups) * len(y_groups)
    component_count = len(component_boxes)
    estimated_slots = max(component_count, grid_count)

    annotated = img.copy()
    draw_component_boxes(annotated, rect, component_boxes)
    cv2.rectangle(
        annotated,
        (rect[0], rect[1]),
        (rect[0] + rect[2], rect[1] + rect[3]),
        (0, 255, 0),
        2,
    )

    return estimated_slots, annotated


def get_pill_count_with_fallback(img, model):
    yolo_count, yolo_annotated = get_pill_count(img, model)
    fallback_count, fallback_annotated = estimate_blister_slots(img)

    if fallback_count >= max(6, yolo_count + 2):
        return fallback_count, fallback_annotated, "fallback_slot_counter"
    return yolo_count, yolo_annotated, "yolo"


# --- MEDICINE INFO (OCR) ---
def collect_ocr_tokens(image, config="--oem 3 --psm 11"):
    data = pytesseract.image_to_data(
        image,
        config=config,
        output_type=pytesseract.Output.DICT,
    )

    tokens = []
    for text, conf in zip(data["text"], data["conf"]):
        text = text.strip()
        if not text:
            continue

        try:
            conf_value = float(conf)
        except ValueError:
            conf_value = -1

        tokens.append((text, conf_value))
    return tokens


def normalize_token(token):
    return re.sub(r"[^A-Za-z0-9.+-]", "", token).strip()


def normalize_quantity_text(text):
    replacements = {
        "O": "0",
        "o": "0",
        "I": "1",
        "l": "1",
    }
    for src, dst in replacements.items():
        text = text.replace(src, dst)
    return text


def extract_quantity_from_text(text):
    normalized = normalize_quantity_text(text)
    patterns = [
        r"\b\d+\.?\d*\s*(mg|g|ml|l|mcg)\b",
        r"\b\d+\.?\d*(mg|g|ml|l|mcg)\b",
    ]
    for pattern in patterns:
        match = re.search(pattern, normalized, re.IGNORECASE)
        if match:
            return match.group(0)
    return "Not found"


def get_quantity_text_candidates(img):
    roi, _, _ = get_blister_roi(img)
    h, _ = roi.shape[:2]
    top_band = roi[:max(40, int(h * 0.22)), :]
    gray = cv2.cvtColor(top_band, cv2.COLOR_BGR2GRAY)
    enlarged = cv2.resize(gray, None, fx=3, fy=3, interpolation=cv2.INTER_CUBIC)
    variants = [
        enlarged,
        cv2.equalizeHist(enlarged),
        cv2.adaptiveThreshold(
            enlarged, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 31, 5
        ),
    ]

    texts = []
    for variant in variants:
        for rot in [None, cv2.ROTATE_90_CLOCKWISE, cv2.ROTATE_90_COUNTERCLOCKWISE]:
            rotated = variant if rot is None else cv2.rotate(variant, rot)
            texts.append(
                pytesseract.image_to_string(
                    rotated,
                    config="--oem 3 --psm 6 -c tessedit_char_whitelist=ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789.- ",
                )
            )
            tokens = collect_ocr_tokens(rotated, config="--oem 3 --psm 6")
            texts.append(" ".join(token for token, conf in tokens if conf >= 35))
    return texts


def get_ocr_info(img):
    quantity = "Not found"
    for candidate_text in get_quantity_text_candidates(img):
        quantity = extract_quantity_from_text(candidate_text)
        if quantity != "Not found":
            break

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Try multiple variants and rotations. We only accept text that has
    # meaningful OCR confidence so foil texture is less likely to be mistaken
    # for a medicine name.
    variants = [
        gray,
        cv2.filter2D(gray, -1, np.array([[-1, -1, -1], [-1, 9, -1], [-1, -1, -1]])),
        cv2.adaptiveThreshold(
            gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2
        ),
    ]

    all_tokens = []
    for variant in variants:
        for rot in [None, cv2.ROTATE_90_CLOCKWISE, cv2.ROTATE_90_COUNTERCLOCKWISE]:
            rotated = variant if rot is None else cv2.rotate(variant, rot)
            all_tokens.extend(collect_ocr_tokens(rotated))

    cleaned = []
    seen = set()
    for raw_text, conf in all_tokens:
        token = normalize_token(raw_text)
        if not token:
            continue
        key = token.lower()
        if key in seen:
            continue
        seen.add(key)
        cleaned.append((token, conf))

    good_tokens = [
        token
        for token, conf in cleaned
        if conf >= 45 and re.search(r"[A-Za-z]", token)
    ]

    if not good_tokens:
        return "Not readable from this image", quantity

    combined_text = " ".join(good_tokens)
    quantity_pattern = r"(\d+\.?\d*\s*(mg|g|ml|l|mcg|tablets?|caps?|capsules?))\b"
    if quantity == "Not found":
        quantity_match = re.search(quantity_pattern, combined_text, re.IGNORECASE)
        quantity = quantity_match.group(0) if quantity_match else "Not found"

    stopwords = {
        "tablet",
        "tablets",
        "capsule",
        "capsules",
        "strip",
        "blister",
        "dose",
        "mg",
        "ml",
        "mcg",
    }
    name_candidates = [
        token
        for token in good_tokens
        if len(token) >= 4
        and not re.search(quantity_pattern, token, re.IGNORECASE)
        and token.lower() not in stopwords
    ]
    name = name_candidates[0] if name_candidates else "Not readable from this image"

    return name, quantity

# --- MASTER PIPELINE ---
def analyze_blister_pack(image_path):
    img = cv2.imread(image_path)
    if img is None:
        print("Error: Could not read image.")
        return

    # 1. Count Pills
    model, model_kind = load_model()
    count, annotated_img, count_method = get_pill_count_with_fallback(img, model)

    # 2. Extract Info
    name, quantity = get_ocr_info(img)

    # --- OUTPUT ---
    print("\n" + "="*40)
    print("      BLISTER PACK ANALYSIS REPORT")
    print("="*40)
    print(f"PILL COUNT      : {count}")
    print(f"QUANTITY (DOSE) : {quantity}")
    print(f"MEDICINE NAME   : {name}")
    print("="*40)
    print(f"MODEL USED      : {model_kind}")
    print(f"COUNT METHOD    : {count_method}")

    if name == "Not readable from this image" and quantity == "Not readable from this image":
        print("NOTE            : Front-side blister photos usually do not contain enough")
        print("                  readable text for medicine name or dose extraction.")
        print("                  Use the printed foil side or the outer box for OCR.")

    # Save and Show Result
    output_path = "final_analysis_result.jpg"
    cv2.imwrite(output_path, annotated_img)
    print(f"\nAnnotated result saved to: {output_path}")

if __name__ == "__main__":
    path = input("Enter image path for full analysis: ").strip()
    if path:
        analyze_blister_pack(path)
