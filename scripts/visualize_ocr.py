"""
ANPR OCR Visual Review Generator
================================
Generates visual review images categorized by quality for inspections:
- highest_quality/ (top highest scoring GOOD results)
- lowest_confidence/ (lowest confidence results among detected plates)
- review/ (results flagged for manual review)
- no_text/ (crops where no valid plate text could be detected)

Overlays bounding box (if available) and an informational banner with:
Detected Text, Final Score, Raw OCR Confidence, Status, and Preprocessing Variant.
"""

from pathlib import Path
import json
import cv2
import pandas as pd

OCR_RESULTS_FILE = Path("outputs/ocr_results.csv")
OCR_CANDIDATES_FILE = Path("outputs/ocr_candidates.csv")
CROP_DIR = Path("outputs/plate_crops")
REVIEW_BASE_DIR = Path("outputs/ocr_review")


def create_annotated_card(image, detected_text, status, score, conf, variant, bbox=None):
    """
    Creates an annotated image with bounding box (if available)
    and a clean top banner with metadata.
    """
    h, w = image.shape[:2]
    annotated = image.copy()

    # Draw bounding box if provided [x, y, bw, bh]
    if bbox and len(bbox) == 4:
        bx, by, bw, bh = int(bbox[0]), int(bbox[1]), int(bbox[2]), int(bbox[3])
        # Box color depends on status
        if status == "GOOD":
            color = (0, 200, 0)      # Green
        elif status == "REVIEW":
            color = (0, 165, 255)    # Orange
        else:
            color = (0, 0, 255)      # Red
        cv2.rectangle(annotated, (bx, by), (bx + bw, by + bh), color, 2)

    # Scale image to a convenient display size if too small or large
    target_w = max(450, min(900, w))
    scale = target_w / w
    new_h = int(h * scale)
    resized = cv2.resize(annotated, (target_w, new_h), interpolation=cv2.INTER_LINEAR)

    # Add header banner of height 70px
    banner_h = 75
    card = 255 * (cv2.imread(str(CROP_DIR / "dummy.jpg")) if False else __import__("numpy").ones((new_h + banner_h, target_w, 3), dtype="uint8"))

    # Dark banner background
    card[:banner_h, :] = (35, 35, 35)
    # Place image below banner
    card[banner_h:, :] = resized

    # Status color
    if status == "GOOD":
        tag_color = (80, 220, 80)
    elif status == "REVIEW":
        tag_color = (0, 200, 255)
    else:
        tag_color = (80, 80, 255)

    # Line 1: Detected Text and Status
    text_display = f"TEXT: {detected_text}" if detected_text else "NO TEXT DETECTED"
    cv2.putText(card, text_display, (12, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (255, 255, 255), 2, cv2.LINE_AA)
    cv2.putText(card, f"[{status}]", (target_w - 130, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.70, tag_color, 2, cv2.LINE_AA)

    # Line 2: Metrics
    metrics_str = f"Score: {score:.3f} | Conf: {conf:.3f} | Variant: {variant}"
    cv2.putText(card, metrics_str, (12, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.52, (200, 200, 200), 1, cv2.LINE_AA)

    return card


def generate_review_images():
    if not OCR_RESULTS_FILE.exists():
        print(f"ERROR: {OCR_RESULTS_FILE} not found. Run scripts/ocr_plates.py first.")
        return

    df = pd.read_csv(OCR_RESULTS_FILE)
    df["confidence"] = pd.to_numeric(df["confidence"], errors="coerce").fillna(0.0)
    df["score"] = pd.to_numeric(df["score"], errors="coerce").fillna(0.0)

    # Load selected candidates bbox mapping if candidates file exists
    bbox_map = {}
    if OCR_CANDIDATES_FILE.exists():
        cand_df = pd.read_csv(OCR_CANDIDATES_FILE)
        selected_cands = cand_df[cand_df["selected"] == True]
        for _, row in selected_cands.iterrows():
            try:
                b = json.loads(row["bbox"])
                bbox_map[row["plate_image"]] = b
            except Exception:
                pass

    # Create category directories
    categories = {
        "highest_quality": REVIEW_BASE_DIR / "highest_quality",
        "lowest_confidence": REVIEW_BASE_DIR / "lowest_confidence",
        "review": REVIEW_BASE_DIR / "review",
        "no_text": REVIEW_BASE_DIR / "no_text"
    }

    for dir_path in categories.values():
        dir_path.mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print("ANPR OCR VISUAL REVIEW GENERATION")
    print("=" * 60)

    # 1. Highest quality (top 15 GOOD)
    good_df = df[df["status"] == "GOOD"].sort_values("score", ascending=False).head(15)
    print(f"Generating Highest Quality reviews ({len(good_df)} images)...")
    for idx, row in enumerate(good_df.itertuples(), 1):
        img_path = CROP_DIR / row.plate_image
        if not img_path.exists():
            continue
        img = cv2.imread(str(img_path))
        if img is None:
            continue
        card = create_annotated_card(
            img, str(row.detected_text), str(row.status),
            float(row.score), float(row.confidence), str(row.variant),
            bbox_map.get(row.plate_image)
        )
        out_name = f"{idx:02d}_score_{row.score:.3f}_{row.plate_image}"
        cv2.imwrite(str(categories["highest_quality"] / out_name), card)

    # 2. Lowest confidence (15 lowest confidence among detected plates)
    detected_df = df[df["status"] != "NO_TEXT"].sort_values("confidence", ascending=True).head(15)
    print(f"Generating Lowest Confidence reviews ({len(detected_df)} images)...")
    for idx, row in enumerate(detected_df.itertuples(), 1):
        img_path = CROP_DIR / row.plate_image
        if not img_path.exists():
            continue
        img = cv2.imread(str(img_path))
        if img is None:
            continue
        card = create_annotated_card(
            img, str(row.detected_text), str(row.status),
            float(row.score), float(row.confidence), str(row.variant),
            bbox_map.get(row.plate_image)
        )
        out_name = f"{idx:02d}_conf_{row.confidence:.3f}_{row.plate_image}"
        cv2.imwrite(str(categories["lowest_confidence"] / out_name), card)

    # 3. REVIEW category (up to 20 REVIEW samples)
    review_df = df[df["status"] == "REVIEW"].sort_values("score", ascending=False).head(20)
    print(f"Generating Manual Review samples ({len(review_df)} images)...")
    for idx, row in enumerate(review_df.itertuples(), 1):
        img_path = CROP_DIR / row.plate_image
        if not img_path.exists():
            continue
        img = cv2.imread(str(img_path))
        if img is None:
            continue
        card = create_annotated_card(
            img, str(row.detected_text), str(row.status),
            float(row.score), float(row.confidence), str(row.variant),
            bbox_map.get(row.plate_image)
        )
        out_name = f"{idx:02d}_review_{row.plate_image}"
        cv2.imwrite(str(categories["review"] / out_name), card)

    # 4. NO_TEXT category
    notext_df = df[df["status"] == "NO_TEXT"].head(15)
    print(f"Generating NO_TEXT samples ({len(notext_df)} images)...")
    for idx, row in enumerate(notext_df.itertuples(), 1):
        img_path = CROP_DIR / row.plate_image
        if not img_path.exists():
            continue
        img = cv2.imread(str(img_path))
        if img is None:
            continue
        card = create_annotated_card(
            img, "", "NO_TEXT",
            0.0, 0.0, str(row.variant),
            None
        )
        out_name = f"{idx:02d}_notext_{row.plate_image}"
        cv2.imwrite(str(categories["no_text"] / out_name), card)

    print()
    print("Visual review generation complete.")
    print(f"Saved review categories to: {REVIEW_BASE_DIR}")
    print("=" * 60)


if __name__ == "__main__":
    generate_review_images()