import re
import sys
from pathlib import Path

import cv2
import pytesseract
from ultralytics import YOLO

if sys.platform.startswith("win"):
    pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"

MODEL_PATH = r"E:\FOS\ocr\pill_count_yolo26_best (1).pt"

# Choose input source
print("Select input source:")
print("1. Image file")
print("2. Camera (webcam)")
choice = input("Enter choice (1 or 2): ").strip()

if choice == "2":
    USE_CAMERA = True
    IMAGE_PATH = None
else:
    USE_CAMERA = False
    IMAGE_PATH = input("enter path").strip()

OCR_STOP_WORDS = {
    "all",
    "age",
    "bottle",
    "capsule",
    "capsules",
    "enjoy",
    "for",
    "group",
    "herbs",
    "mg",
    "ml",
    "oral",
    "pill",
    "suspension",
    "syrup",
    "tablets",
    "tonic",
    "with",
}
OCR_CONFIDENCE_THRESHOLD = 35

def rotate_image(image, angle):
    if angle == 90:
        return cv2.rotate(image, cv2.ROTATE_90_CLOCKWISE)
    if angle == 180:
        return cv2.rotate(image, cv2.ROTATE_180)
    if angle == 270:
        return cv2.rotate(image, cv2.ROTATE_90_COUNTERCLOCKWISE)
    return image

def preprocess_for_ocr(image_bgr):
    gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
    gray = cv2.resize(gray, None, fx=2, fy=2, interpolation=cv2.INTER_CUBIC)
    gray = cv2.GaussianBlur(gray, (3, 3), 0)
    return cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]


def collect_ocr_tokens(image_bgr, config="--oem 3 --psm 11"):
    data = pytesseract.image_to_data(
        image_bgr,
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


def ocr_variants(image_bgr, fast_mode=False):
    """OCR with optional fast mode for camera (2x-5x faster)."""
    gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
    height = gray.shape[0]
    crops = [gray[: max(int(height * 0.20), 1), :]]
    
    if fast_mode:
        # Fast mode: single crop, minimal variants
        variants = [crops[0]]
        variants.append(cv2.resize(crops[0], None, fx=2, fy=2, interpolation=cv2.INTER_CUBIC))
        _, otsu = cv2.threshold(crops[0], 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        variants.append(otsu)
    else:
        # Full mode: multiple crops, many variants
        crops.append(gray)
        variants = []
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
        
        for crop in crops:
            variants.append(crop)
            variants.append(cv2.resize(crop, None, fx=3, fy=3, interpolation=cv2.INTER_CUBIC))
            variants.append(cv2.GaussianBlur(crop, (3, 3), 0))
            variants.append(cv2.morphologyEx(crop, cv2.MORPH_CLOSE, kernel))
            _, otsu = cv2.threshold(crop, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
            variants.append(otsu)
            adaptive = cv2.adaptiveThreshold(crop, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 31, 11)
            variants.append(adaptive)

    texts = []
    seen = set()
    
    for variant in variants:
        # Fast mode: only try 0 and 90 degree rotations
        angles = [0, 90] if fast_mode else [0, 90, 180, 270]
        for angle in angles:
            rotated = rotate_image(variant, angle)
            # Fast mode: only try PSM 6 (single text block)
            psms = [6] if fast_mode else [6, 11, 7]
            
            for psm in psms:
                if fast_mode:
                    # Fast: just raw OCR
                    raw_text = pytesseract.image_to_string(rotated, config=f"--oem 3 --psm {psm}").strip()
                    if raw_text and raw_text not in seen:
                        seen.add(raw_text)
                        texts.append(raw_text)
                else:
                    # Full: confidence-filtered + raw
                    confident_tokens = [token for token, conf in collect_ocr_tokens(rotated, config=f"--oem 3 --psm {psm}") if conf >= OCR_CONFIDENCE_THRESHOLD]
                    confident_text = " ".join(confident_tokens).strip()
                    if confident_text and confident_text not in seen:
                        seen.add(confident_text)
                        texts.append(confident_text)
                    
                    raw_text = pytesseract.image_to_string(rotated, config=f"--oem 3 --psm {psm}").strip()
                    if raw_text and raw_text not in seen:
                        seen.add(raw_text)
                        texts.append(raw_text)
    
    return texts


def best_ocr_text(image_bgr, fast_mode=False):
    texts = ocr_variants(image_bgr, fast_mode=fast_mode)
    return "\n".join(texts)


def score_candidate(line, frequency):
    words = re.findall(r"[A-Za-z][A-Za-z'&+-]*", line)
    if not words:
        return -1

    lowered_words = [word.lower() for word in words]
    if all(word in OCR_STOP_WORDS for word in lowered_words):
        return -1

    alpha_count = sum(1 for char in line if char.isalpha())
    digit_count = sum(1 for char in line if char.isdigit())
    stopword_count = sum(1 for word in lowered_words if word in OCR_STOP_WORDS)
    score = alpha_count * 2 + frequency * 6 + len(words) * 3 - digit_count * 2 - stopword_count * 4

    if 1 <= len(words) <= 4:
        score += 5
    if any(word not in OCR_STOP_WORDS and len(word) > 2 for word in lowered_words):
        score += 4
    if any(char.isupper() for char in line):
        score += 2

    return score

def parse_name_and_quantity(text):
    lines = [line.strip() for line in text.splitlines() if line.strip()]
    quantity = None
    quantity_match = re.search(
        r'\b\d+(?:\.\d+)?\s*(?:ml|mg|mcg|g|l|liters?|tablets?|capsules?|tabs?|syrup|tonic)\b',
        text,
        flags=re.I,
    )
    if quantity_match:
        quantity = quantity_match.group(0)

    line_frequency = {}
    normalized_lines = []
    for line in lines:
        normalized = re.sub(r'\s+', ' ', line).strip()
        if not normalized:
            continue
        normalized_lines.append(normalized)
        key = normalized.lower()
        line_frequency[key] = line_frequency.get(key, 0) + 1

    candidates = []
    for line in normalized_lines:
        if quantity and quantity.lower() in line.lower():
            continue
        if re.search(r'\b\d+\s*(?:ml|mg|mcg|g|l|liters?|tablets?|capsules?|tabs?|syrup|tonic)\b', line, flags=re.I):
            continue

        cleaned = re.sub(r'[^A-Za-z0-9+&\- ]', ' ', line).strip()
        cleaned = re.sub(r'\s+', ' ', cleaned)
        if len(cleaned) < 3:
            continue

        if not any(char.isalpha() for char in cleaned):
            continue

        score = score_candidate(cleaned, line_frequency.get(cleaned.lower(), 1))
        if score >= 0:
            candidates.append((score, cleaned))

    medicine_name = max(candidates, key=lambda item: item[0], default=(None, None))[1]
    return medicine_name, quantity

image_path = Path(IMAGE_PATH) if IMAGE_PATH else None
model_path = Path(MODEL_PATH)

if not USE_CAMERA and not image_path.exists():
    raise FileNotFoundError(f'Image file does not exist: {image_path}')
if not model_path.exists():
    raise FileNotFoundError(f'Model file does not exist: {model_path}')

model = YOLO(str(model_path))

if USE_CAMERA:
    # Camera mode (headless - no GUI display)
    import queue
    import threading
    
    # Try to open camera with multiple fallback options
    cap = None
    camera_source = None
    
    # Option 1: Try local device (index 0)
    cap = cv2.VideoCapture(0)
    if cap.isOpened():
        camera_source = "Local device (index 0)"
    else:
        cap.release()
        # Option 2: Try other local device indices (1, 2)
        for i in range(1, 3):
            cap = cv2.VideoCapture(i)
            if cap.isOpened():
                camera_source = f"Local device (index {i})"
                break
            cap.release()
        
        # Option 3: Try IP camera (DroidCam/IP Webcam default)
        if not camera_source:
            ip_urls = [
                "http://192.168.1.100:4747/video",  # DroidCam common
                "http://127.0.0.1:8080/video",      # IP Webcam localhost
                "http://192.168.0.100:8080/video",  # IP Webcam common
            ]
            for url in ip_urls:
                cap = cv2.VideoCapture(url)
                if cap.isOpened():
                    camera_source = f"IP camera ({url})"
                    break
                cap.release()
    
    if not cap or not cap.isOpened():
        print("\n❌ CAMERA NOT FOUND")
        print("\nTroubleshooting:")
        print("1. Local Webcam: Make sure your webcam is connected and not in use")
        print("2. Phone Webcam (DroidCam/IP Webcam):")
        print("   - Install app: 'DroidCam' or 'IP Webcam' on your phone")
        print("   - Start the app on your phone")
        print("   - Update the IP address in test.py (search for 'http://192.168' in code)")
        print("   - For DroidCam: http://YOUR_PHONE_IP:4747/video")
        print("   - For IP Webcam: http://YOUR_PHONE_IP:8080/video")
        print("\n3. Check your firewall/network settings\n")
        raise RuntimeError("Could not open camera. See troubleshooting above.")
    
    print(f"✓ Camera found: {camera_source}")
    
    print("\n" + "="*60)
    print("CAMERA MODE (Headless)")
    print("="*60)
    print("Type commands to control:")
    print("  'c' + Enter  → Capture and analyze current frame")
    print("  'q' + Enter  → Quit")
    print("="*60 + "\n")
    
    frame_count = 0
    latest_frame = None
    command_queue = queue.Queue()
    
    # Warm up the camera
    for _ in range(5):
        cap.read()
    
    def input_listener():
        """Listen for user input in a separate thread."""
        while True:
            try:
                user_cmd = input().strip().lower()
                if user_cmd in ['c', 'q']:
                    command_queue.put(user_cmd)
            except EOFError:
                break
    
    # Start input thread as daemon
    input_thread = threading.Thread(target=input_listener, daemon=True)
    input_thread.start()
    
    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                print("Failed to read frame from camera.")
                break
            
            latest_frame = frame.copy()
            
            # Check for commands from user input thread
            try:
                cmd = command_queue.get_nowait()  # Non-blocking get
                
                if cmd == 'c' and latest_frame is not None:
                    frame_count += 1
                    print(f"\n--- Analyzing captured frame {frame_count} (fast mode) ---")
                    
                    # Run OCR (fast mode for camera)
                    ocr_text = best_ocr_text(latest_frame, fast_mode=True)
                    medicine_name, quantity = parse_name_and_quantity(ocr_text)
                    
                    # Run detection
                    results = model.predict(source=latest_frame, conf=0.25, save=False, verbose=False)
                    
                    for r in results:
                        pill_count = len(r.boxes)
                        print(f'Pill count: {pill_count}')
                        print(f'Medicine name: {medicine_name or "Not found"}')
                        print(f'Quantity: {quantity or "Not found"}')
                        print('OCR text (top 300 chars):')
                        print(ocr_text[:300] if ocr_text.strip() else 'No text detected')
                        
                        # Save annotated frame
                        annotated = r.plot()
                        output_filename = f'camera_capture_{frame_count}_annotated.jpg'
                        if cv2.imwrite(output_filename, annotated):
                            print(f'✓ Annotated frame saved to: {output_filename}')
                        else:
                            print(f'✗ Failed to save annotated frame.')
                    print("Ready for next capture. Type 'c' to capture or 'q' to quit.\n")
                
                elif cmd == 'q':
                    print("Quitting camera mode...")
                    break
            
            except queue.Empty:
                # No command in queue, continue camera loop
                pass
    
    finally:
        cap.release()

else:
    # File mode (original behavior)
    image_bgr = cv2.imread(str(image_path))
    if image_bgr is None:
        raise FileNotFoundError(f'Could not read image: {IMAGE_PATH}')

    results = model.predict(source=str(image_path), conf=0.25, save=False)

    ocr_text = best_ocr_text(image_bgr)
    medicine_name, quantity = parse_name_and_quantity(ocr_text)

    for r in results:
        pill_count = len(r.boxes)
        print(f'Pill count: {pill_count}')
        print(f'Medicine name: {medicine_name or "Not found"}')
        print(f'Quantity: {quantity or "Not found"}')
        print('OCR text:')
        print(ocr_text if ocr_text.strip() else 'No text detected')
        annotated = r.plot()
        output_path = image_path.with_name(f'{image_path.stem}_annotated.jpg')
        if not cv2.imwrite(str(output_path), annotated):
            raise RuntimeError(f'Failed to save annotated image: {output_path}')
        print(f'Annotated image saved to: {output_path}')