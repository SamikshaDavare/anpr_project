from pathlib import Path
import cv2
import csv


# =========================
# PATHS
# =========================

BASE_DIR = Path(__file__).resolve().parent.parent

IMAGE_DIR = BASE_DIR / "student_1" / "images"
LABEL_DIR = BASE_DIR / "student_1" / "labels"

OUTPUT_DIR = BASE_DIR / "outputs" / "plate_crops"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

METADATA_FILE = OUTPUT_DIR / "metadata.csv"


# =========================
# SETTINGS
# =========================

PLATE_CLASS = 1


# =========================
# FUNCTIONS
# =========================

def polygon_to_bbox(points, image_width, image_height):
    """
    Convert normalized polygon coordinates to pixel bounding box.
    """

    xs = points[0::2]
    ys = points[1::2]

    x_min = max(0, int(min(xs) * image_width))
    y_min = max(0, int(min(ys) * image_height))

    x_max = min(image_width, int(max(xs) * image_width))
    y_max = min(image_height, int(max(ys) * image_height))

    return x_min, y_min, x_max, y_max


def yolo_bbox_to_pixels(values, image_width, image_height):
    """
    Convert normalized YOLO bounding box to pixel coordinates.
    """

    x_center, y_center, width, height = values

    x_center *= image_width
    y_center *= image_height
    width *= image_width
    height *= image_height

    x_min = max(0, int(x_center - width / 2))
    y_min = max(0, int(y_center - height / 2))

    x_max = min(image_width, int(x_center + width / 2))
    y_max = min(image_height, int(y_center + height / 2))

    return x_min, y_min, x_max, y_max


# =========================
# MAIN
# =========================

metadata = []

images_processed = 0
plates_extracted = 0
bbox_count = 0
polygon_count = 0


for image_path in sorted(IMAGE_DIR.glob("*.jpg")):

    label_path = LABEL_DIR / f"{image_path.stem}.txt"

    if not label_path.exists():
        continue

    image = cv2.imread(str(image_path))

    if image is None:
        print(f"WARNING: Could not read {image_path.name}")
        continue

    image_height, image_width = image.shape[:2]

    images_processed += 1

    lines = label_path.read_text().splitlines()

    plate_index = 0

    for line in lines:

        parts = line.strip().split()

        if not parts:
            continue

        try:
            class_id = int(parts[0])
        except ValueError:
            continue

        # Only process number plates
        if class_id != PLATE_CLASS:
            continue

        coordinates = list(map(float, parts[1:]))

        # --------------------------------
        # YOLO bounding box
        # class x_center y_center width height
        # --------------------------------

        if len(coordinates) == 4:

            x_min, y_min, x_max, y_max = yolo_bbox_to_pixels(
                coordinates,
                image_width,
                image_height
            )

            annotation_type = "bbox"
            bbox_count += 1

        # --------------------------------
        # YOLO segmentation polygon
        # class x1 y1 x2 y2 ...
        # --------------------------------

        elif len(coordinates) >= 6 and len(coordinates) % 2 == 0:

            x_min, y_min, x_max, y_max = polygon_to_bbox(
                coordinates,
                image_width,
                image_height
            )

            annotation_type = "polygon"
            polygon_count += 1

        else:
            print(
                f"WARNING: Unsupported annotation in {label_path.name}: "
                f"{len(coordinates)} coordinates"
            )
            continue

        # Make sure crop is valid

        if x_max <= x_min or y_max <= y_min:
            print(
                f"WARNING: Invalid crop in {image_path.name}"
            )
            continue

        plate_crop = image[y_min:y_max, x_min:x_max]

        if plate_crop.size == 0:
            continue

        # --------------------------------
        # Save crop
        # --------------------------------

        crop_name = (
            f"{image_path.stem}_plate_{plate_index + 1}.jpg"
        )

        crop_path = OUTPUT_DIR / crop_name

        cv2.imwrite(str(crop_path), plate_crop)

        # --------------------------------
        # Metadata
        # --------------------------------

        metadata.append({
            "plate_id": f"plate_{plates_extracted + 1:04d}",
            "original_image": image_path.name,
            "crop_filename": crop_name,
            "annotation_type": annotation_type,
            "class_id": class_id,
            "x_min": x_min,
            "y_min": y_min,
            "x_max": x_max,
            "y_max": y_max,
            "image_width": image_width,
            "image_height": image_height
        })

        plates_extracted += 1
        plate_index += 1


# =========================
# WRITE METADATA
# =========================

with open(
    METADATA_FILE,
    "w",
    newline="",
    encoding="utf-8"
) as f:

    fieldnames = [
        "plate_id",
        "original_image",
        "crop_filename",
        "annotation_type",
        "class_id",
        "x_min",
        "y_min",
        "x_max",
        "y_max",
        "image_width",
        "image_height"
    ]

    writer = csv.DictWriter(
        f,
        fieldnames=fieldnames
    )

    writer.writeheader()
    writer.writerows(metadata)


# =========================
# SUMMARY
# =========================

print()
print("=" * 50)
print("PLATE EXTRACTION COMPLETE")
print("=" * 50)

print(f"Images processed     : {images_processed}")
print(f"Plates extracted     : {plates_extracted}")
print(f"Bounding boxes       : {bbox_count}")
print(f"Polygons             : {polygon_count}")
print(f"Output directory     : {OUTPUT_DIR}")
print(f"Metadata file        : {METADATA_FILE}")
print("=" * 50)