from pathlib import Path
import cv2

# ============================================================
# PATHS
# ============================================================

INPUT_DIR = Path("outputs/plate_crops")
OUTPUT_DIR = Path("outputs/preprocessed")

ORIGINAL_DIR = OUTPUT_DIR / "original"
GRAYSCALE_DIR = OUTPUT_DIR / "grayscale"
ENHANCED_DIR = OUTPUT_DIR / "enhanced"
THRESHOLD_DIR = OUTPUT_DIR / "threshold"

# ============================================================
# SETTINGS
# ============================================================

MIN_HEIGHT = 100
TARGET_HEIGHT = 150


# ============================================================
# PREPROCESSING
# ============================================================

def preprocess_image(image):

    # --------------------------------------------------------
    # 1. ORIGINAL
    # --------------------------------------------------------

    original = image.copy()

    # --------------------------------------------------------
    # 2. GRAYSCALE
    # --------------------------------------------------------

    gray = cv2.cvtColor(
        image,
        cv2.COLOR_BGR2GRAY
    )

    # --------------------------------------------------------
    # 3. UPSCALE SMALL PLATES
    # --------------------------------------------------------

    height, width = gray.shape

    if height < MIN_HEIGHT:

        scale = TARGET_HEIGHT / height

        new_width = int(width * scale)

        gray = cv2.resize(
            gray,
            (new_width, TARGET_HEIGHT),
            interpolation=cv2.INTER_CUBIC
        )

    # --------------------------------------------------------
    # 4. CLAHE CONTRAST ENHANCEMENT
    # --------------------------------------------------------

    clahe = cv2.createCLAHE(
        clipLimit=2.0,
        tileGridSize=(8, 8)
    )

    enhanced = clahe.apply(gray)

    # --------------------------------------------------------
    # 5. OTSU THRESHOLD
    # --------------------------------------------------------

    _, threshold = cv2.threshold(
        enhanced,
        0,
        255,
        cv2.THRESH_BINARY + cv2.THRESH_OTSU
    )

    return (
        original,
        gray,
        enhanced,
        threshold
    )


# ============================================================
# MAIN
# ============================================================

def main():

    for directory in [
        ORIGINAL_DIR,
        GRAYSCALE_DIR,
        ENHANCED_DIR,
        THRESHOLD_DIR
    ]:
        directory.mkdir(
            parents=True,
            exist_ok=True
        )

    images = sorted(
        INPUT_DIR.glob("*.jpg")
    )

    print("=" * 60)
    print("ANPR PLATE PREPROCESSING")
    print("=" * 60)

    print(f"Input plates: {len(images)}")
    print()

    processed = 0

    for i, image_path in enumerate(images, 1):

        print(
            f"[{i}/{len(images)}] "
            f"{image_path.name}"
        )

        image = cv2.imread(
            str(image_path)
        )

        if image is None:

            print(
                "   WARNING: Could not read image"
            )

            continue

        try:

            (
                original,
                gray,
                enhanced,
                threshold
            ) = preprocess_image(image)

            filename = image_path.name

            cv2.imwrite(
                str(ORIGINAL_DIR / filename),
                original
            )

            cv2.imwrite(
                str(GRAYSCALE_DIR / filename),
                gray
            )

            cv2.imwrite(
                str(ENHANCED_DIR / filename),
                enhanced
            )

            cv2.imwrite(
                str(THRESHOLD_DIR / filename),
                threshold
            )

            processed += 1

        except Exception as e:

            print(
                f"   ERROR: {e}"
            )

    print()
    print("=" * 60)
    print("PREPROCESSING COMPLETE")
    print("=" * 60)

    print(f"Processed: {processed}")

    print()
    print("Generated:")
    print(f"  Original:  {ORIGINAL_DIR}")
    print(f"  Grayscale: {GRAYSCALE_DIR}")
    print(f"  Enhanced:  {ENHANCED_DIR}")
    print(f"  Threshold: {THRESHOLD_DIR}")

    print("=" * 60)


if __name__ == "__main__":
    main()