import cv2
from ultralytics import YOLO
import sys
from pathlib import Path

# Path to the pre-trained YOLO model (using the one from your main project)
MODEL_PATH = r"E:\FOS\ocr\runs\detect\train_2class_yolo26\weights\best.pt"

def count_pills(image_path: str):
    # Check if model exists
    if not Path(MODEL_PATH).exists():
        print(f"Error: YOLO model not found at {MODEL_PATH}")
        return None

    # Load image
    img = cv2.imread(image_path)
    if img is None:
        print(f"Error: Could not read image at {image_path}")
        return None

    # Load YOLO model
    print(f"Loading YOLO model: {MODEL_PATH}...")
    model = YOLO(MODEL_PATH)

    # Run inference
    # conf=0.25 is usually a good balance
    results = model(img, conf=0.25, verbose=False)
    
    # Process results
    boxes = results[0].boxes
    pill_count = len(boxes)
    
    # Draw detections on image
    annotated_img = results[0].plot()

    # Show results
    print("-" * 30)
    print(f"Pills Found: {pill_count}")
    print("-" * 30)

    # Save output
    output_path = Path("pill_count_output.jpg")
    cv2.imwrite(str(output_path), annotated_img)
    print(f"Annotated image saved as: {output_path}")

    return pill_count

if __name__ == "__main__":
    path = input("Enter image path for pill counting: ").strip()
    if path:
        count_pills(path)
