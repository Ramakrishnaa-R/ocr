import cv2
import pytesseract
import re
import numpy as np
import sys
from pathlib import Path

# Configure Tesseract path for Windows
if sys.platform.startswith("win"):
    pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"

def preprocess_variants(image_path):
    img = cv2.imread(str(image_path))
    if img is None: return []
    
    # 1. Use original resolution
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    # We will try both the top crop and the full image
    h, w = gray.shape
    crops = [gray[0:int(h * 0.3), :], gray]
    
    variants = []
    for crop in crops:
        # Variant A: Standard
        variants.append(crop)
        
        # Variant B: Sharp
        kernel = np.array([[-1,-1,-1], [-1,9,-1], [-1,-1,-1]])
        variants.append(cv2.filter2D(crop, -1, kernel))
        
        # Variant C: Laplacian (highlights edges of engraved text)
        laplacian = cv2.Laplacian(crop, cv2.CV_64F)
        variants.append(np.uint8(np.absolute(laplacian)))
        
        # Variant D: Thresholded
        _, thresh = cv2.threshold(crop, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        variants.append(thresh)

    return variants

def extract_medicine_info(image_path: str) -> dict:
    variants = preprocess_variants(image_path)
    if not variants:
        return {"name": "Error", "quantity": "File not found", "all_text": ""}

    combined_text = ""
    for variant in variants:
        # Try all 4 rotations for each variant
        for rotation in [None, cv2.ROTATE_90_CLOCKWISE, cv2.ROTATE_180, cv2.ROTATE_90_COUNTERCLOCKWISE]:
            rotated = variant if rotation is None else cv2.rotate(variant, rotation)
            
            # Use PSM 11 for sparse/randomly oriented text
            text = pytesseract.image_to_string(rotated, config='--oem 3 --psm 11')
            combined_text += "\n" + text
            
            # If we find a quantity match early, we can prioritize it
            if re.search(r'\d+\.?\d*\s*(mg|g|ml|l|mcg|tablets?|caps?)', text, re.IGNORECASE):
                # Small optimization: if we found a good match, we can stop or keep going
                pass 

    print("--- Detected Text (Combined) ---\n", combined_text.strip()[:500] + "...")
    print("--------------------")

    # Priority 1: Extract Quantity
    # Regex for numbers followed by common units
    quantity_pattern = r'(\d+\.?\d*\s*(mg|g|ml|l|mcg|tablets?|caps?|capsules?|mcg|unit|iu))\b'
    quantity_matches = re.findall(quantity_pattern, combined_text, re.IGNORECASE)
    
    quantity = "Not found"
    if quantity_matches:
        # Filter out obviously wrong matches (like just "1 tablets" if it's suspicious)
        # But for now, we take the most specific one (longest)
        matches = [m[0] for m in quantity_matches]
        quantity = max(matches, key=len)

    # Priority 2: Extract Name
    # Filter lines to find likely medicine names (Upper case or CamelCase, 4+ chars)
    name = "Unknown"
    potential_names = []
    
    for line in combined_text.split('\n'):
        clean = line.strip()
        if len(clean) < 4: continue
        # Must have at least some letters
        if not any(c.isalpha() for c in clean): continue
        # Ignore lines that are just quantities
        if re.search(quantity_pattern, clean, re.IGNORECASE): continue
        
        potential_names.append(clean)

    if potential_names:
        # Heuristic: the most frequently occurring word/line across variants is likely the name
        from collections import Counter
        word_counts = Counter(potential_names)
        name = word_counts.most_common(1)[0][0]

    return {
        "name": name,
        "quantity": quantity,
        "all_text": combined_text
    }

if __name__ == "__main__":
    path = input("Enter image path: ").strip()
    if not path:
        print("No path entered.")
    else:
        info = extract_medicine_info(path)
        print(f"\nFINAL RESULTS:")
        print(f"Quantity      : {info['quantity']}")
        print(f"Medicine Name : {info['name']}")