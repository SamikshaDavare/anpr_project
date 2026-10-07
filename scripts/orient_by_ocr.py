from pathlib import Path
import cv2
import easyocr
import pandas as pd
import re


# ============================================================
# PATHS
# ============================================================

INPUT_DIR = Path("outputs/rectified/polygon")

OUTPUT_DIR = Path("outputs/rectified/oriented/polygon")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

RESULTS_FILE = Path(
    "outputs/rectified/orientation_ocr_results.csv"
)

CANDIDATES_FILE = Path(
    "outputs/rectified/orientation_candidates.csv"
)


# ============================================================
# OCR MODEL
# ============================================================

print("=" * 60)
print("LOADING ENGLISH OCR MODEL")
print("=" * 60)

reader = easyocr.Reader(
    ["en"],
    gpu=False
)

print("OCR model loaded.")
print()


# ============================================================
# ROTATION FUNCTION
# ============================================================

def rotate_image(image, angle):

    if angle == 0:
        return image

    elif angle == 90:
        return cv2.rotate(
            image,
            cv2.ROTATE_90_CLOCKWISE
        )

    elif angle == 180:
        return cv2.rotate(
            image,
            cv2.ROTATE_180
        )

    elif angle == 270:
        return cv2.rotate(
            image,
            cv2.ROTATE_90_COUNTERCLOCKWISE
        )

    return image


# ============================================================
# CLEAN OCR TEXT
# ============================================================

def clean_text(text):

    # Keep ONLY English letters and digits.
    text = re.sub(
        r"[^A-Za-z0-9]",
        "",
        str(text)
    )

    return text


# ============================================================
# OCR
# ============================================================

def run_ocr(image):

    results = reader.readtext(
        image,
        detail=1,
        paragraph=False,
        width_ths=0.7,
        mag_ratio=1.5
    )

    if not results:
        return {
            "text": "",
            "confidence": 0.0,
            "detections": 0,
            "characters": 0
        }

    texts = []
    confidences = []

    for detection in results:

        if len(detection) < 3:
            continue

        raw_text = str(detection[1]).strip()

        confidence = float(
            detection[2]
        )

        # Convert to English letters/digits only.
        cleaned = clean_text(
            raw_text
        )

        if cleaned:

            texts.append(cleaned)

            confidences.append(
                confidence
            )

    if not confidences:

        return {
            "text": "",
            "confidence": 0.0,
            "detections": 0,
            "characters": 0
        }

    final_text = "".join(
        texts
    )

    average_confidence = (
        sum(confidences)
        / len(confidences)
    )

    return {
        "text": final_text,
        "confidence": average_confidence,
        "detections": len(confidences),
        "characters": len(final_text)
    }


# ============================================================
# ORIENTATION SCORING
# ============================================================

def orientation_score(
    confidence,
    text,
    detections
):

    length = len(text)

    if length == 0:
        return 0.0

    # --------------------------------------------------------
    # Start with OCR confidence
    # --------------------------------------------------------

    score = confidence

    # --------------------------------------------------------
    # Penalize extremely short OCR results.
    #
    # "2" with 0.99 confidence should NOT automatically
    # beat "IND42557" with 0.70 confidence.
    # --------------------------------------------------------

    if length == 1:
        score *= 0.20

    elif length == 2:
        score *= 0.40

    elif length == 3:
        score *= 0.65

    elif length == 4:
        score *= 0.80

    elif length >= 5:
        score *= 1.00

    # --------------------------------------------------------
    # Slight bonus for reasonable plate-like lengths.
    # --------------------------------------------------------

    if 5 <= length <= 12:
        score *= 1.15

    # --------------------------------------------------------
    # Bonus if both letters AND digits are present.
    #
    # Indian plates commonly contain both.
    # --------------------------------------------------------

    has_letters = bool(
        re.search(
            r"[A-Za-z]",
            text
        )
    )

    has_digits = bool(
        re.search(
            r"[0-9]",
            text
        )
    )

    if has_letters and has_digits:
        score *= 1.20

    # --------------------------------------------------------
    # If only letters OR only numbers are present,
    # don't give the mixed-character bonus.
    # --------------------------------------------------------

    elif has_letters or has_digits:
        score *= 0.90

    # --------------------------------------------------------
    # Multiple detections can indicate that OCR found
    # multiple useful character regions.
    # --------------------------------------------------------

    if detections >= 2:
        score *= 1.05

    return score


# ============================================================
# MAIN
# ============================================================

files = sorted(
    INPUT_DIR.glob("*.jpg")
)

print("=" * 60)
print("OCR-BASED ORIENTATION CORRECTION")
print("=" * 60)

print(
    f"Input plates: {len(files)}"
)

print()


all_results = []

candidate_results = []


# ============================================================
# PROCESS EACH PLATE
# ============================================================

for index, image_path in enumerate(
    files,
    start=1
):

    image = cv2.imread(
        str(image_path)
    )

    if image is None:

        print(
            f"[SKIP] Could not read: "
            f"{image_path.name}"
        )

        continue

    candidates = []


    # --------------------------------------------------------
    # TEST FOUR ORIENTATIONS
    # --------------------------------------------------------

    for angle in [
        0,
        90,
        180,
        270
    ]:

        rotated = rotate_image(
            image,
            angle
        )

        ocr = run_ocr(
            rotated
        )

        score = orientation_score(
            confidence=ocr["confidence"],
            text=ocr["text"],
            detections=ocr["detections"]
        )

        candidate = {
            "angle": angle,
            "text": ocr["text"],
            "confidence": ocr["confidence"],
            "detections": ocr["detections"],
            "characters": ocr["characters"],
            "score": score
        }

        candidates.append(
            candidate
        )


        # ----------------------------------------------------
        # SAVE ALL CANDIDATES
        # ----------------------------------------------------

        candidate_results.append({

            "plate_image":
                image_path.name,

            "orientation":
                angle,

            "detected_text":
                ocr["text"],

            "confidence":
                ocr["confidence"],

            "detections":
                ocr["detections"],

            "characters":
                ocr["characters"],

            "orientation_score":
                score
        })


    # --------------------------------------------------------
    # SELECT BEST ORIENTATION
    # --------------------------------------------------------

    best = max(
        candidates,
        key=lambda x: x["score"]
    )


    best_angle = best["angle"]


    # --------------------------------------------------------
    # CREATE WINNING ORIENTATION
    # --------------------------------------------------------

    best_image = rotate_image(
        image,
        best_angle
    )


    # --------------------------------------------------------
    # SAVE ORIENTED IMAGE
    # --------------------------------------------------------

    output_path = (
        OUTPUT_DIR
        / image_path.name
    )

    cv2.imwrite(
        str(output_path),
        best_image,
        [
            cv2.IMWRITE_JPEG_QUALITY,
            95
        ]
    )


    # --------------------------------------------------------
    # SAVE FINAL RESULT
    # --------------------------------------------------------

    all_results.append({

        "plate_image":
            image_path.name,

        "orientation":
            best_angle,

        "detected_text":
            best["text"],

        "confidence":
            best["confidence"],

        "detections":
            best["detections"],

        "characters":
            best["characters"],

        "orientation_score":
            best["score"]
    })


    # --------------------------------------------------------
    # TERMINAL OUTPUT
    # --------------------------------------------------------

    print(
        f"{index:03d}/{len(files):03d} | "
        f"{best_angle:3d}° | "
        f"OCR: {best['confidence']:.3f} | "
        f"Score: {best['score']:.3f} | "
        f"Text: {best['text']}"
    )


# ============================================================
# SAVE RESULTS
# ============================================================

results_df = pd.DataFrame(
    all_results
)

candidates_df = pd.DataFrame(
    candidate_results
)


results_df.to_csv(
    RESULTS_FILE,
    index=False,
    encoding="utf-8-sig"
)


candidates_df.to_csv(
    CANDIDATES_FILE,
    index=False,
    encoding="utf-8-sig"
)


# ============================================================
# SUMMARY
# ============================================================

print()

print("=" * 60)
print("ORIENTATION CORRECTION COMPLETE")
print("=" * 60)

print(
    f"Processed plates : "
    f"{len(results_df)}"
)


if len(results_df) > 0:

    print()

    print(
        "Selected orientations:"
    )

    print(
        results_df[
            "orientation"
        ]
        .value_counts()
        .sort_index()
        .to_string()
    )


    print()

    print(
        f"Average OCR confidence : "
        f"{results_df['confidence'].mean():.4f}"
    )

    print(
        f"Minimum OCR confidence : "
        f"{results_df['confidence'].min():.4f}"
    )

    print(
        f"Maximum OCR confidence : "
        f"{results_df['confidence'].max():.4f}"
    )

    print()

    print(
        f"Average orientation score : "
        f"{results_df['orientation_score'].mean():.4f}"
    )


print()

print(
    "Oriented images:"
)

print(
    OUTPUT_DIR
)

print()

print(
    "Final results:"
)

print(
    RESULTS_FILE
)

print()

print(
    "All orientation candidates:"
)

print(
    CANDIDATES_FILE
)

print("=" * 60)