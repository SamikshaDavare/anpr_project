from pathlib import Path

import cv2


INPUT_DIR = Path("outputs/plate_crops")
OUTPUT_DIR = Path("outputs/preprocessed")

MIN_HEIGHT = 100
TARGET_HEIGHT = 100


def preprocess_image(image):
    """
    Create several preprocessing versions of a plate crop.
    """

    # Convert to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Improve local contrast
    clahe = cv2.createCLAHE(
        clipLimit=2.0,
        tileGridSize=(8, 8)
    )
    enhanced = clahe.apply(gray)

    # Upscale small plates
    height, width = enhanced.shape

    if height < MIN_HEIGHT:
        scale = TARGET_HEIGHT / height
        new_width = int(width * scale)

        enhanced = cv2.resize(
            enhanced,
            (new_width, TARGET_HEIGHT),
            interpolation=cv2.INTER_CUBIC
        )

    # Otsu threshold
    _, threshold = cv2.threshold(
        enhanced,
        0,
        255,
        cv2.THRESH_BINARY + cv2.THRESH_OTSU
    )

    return gray, enhanced, threshold


def main():

    OUTPUT_DIR.joinpath("grayscale").mkdir(
        parents=True,
        exist_ok=True
    )

    OUTPUT_DIR.joinpath("enhanced").mkdir(
        parents=True,
        exist_ok=True
    )

    OUTPUT_DIR.joinpath("threshold").mkdir(
        parents=True,
        exist_ok=True
    )

    images = list(INPUT_DIR.glob("*.jpg"))

    print("=" * 60)
    print("PLATE PREPROCESSING")
    print("=" * 60)
    print(f"Input plates: {len(images)}")

    processed = 0

    for image_path in images:

        image = cv2.imread(str(image_path))

        if image is None:
            print(f"WARNING: Could not read {image_path.name}")
            continue

        gray, enhanced, threshold = preprocess_image(image)

        filename = image_path.name

        cv2.imwrite(
            str(OUTPUT_DIR / "grayscale" / filename),
            gray
        )

        cv2.imwrite(
            str(OUTPUT_DIR / "enhanced" / filename),
            enhanced
        )

        cv2.imwrite(
            str(OUTPUT_DIR / "threshold" / filename),
            threshold
        )

        processed += 1

    print(f"Processed: {processed}")
    print("=" * 60)


if __name__ == "__main__":
    main()