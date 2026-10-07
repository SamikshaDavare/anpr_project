from pathlib import Path
import cv2
import numpy as np
import pandas as pd


# ============================================================
# PATHS
# ============================================================

IMAGE_DIR = Path("student_1/images")
LABEL_DIR = Path("student_1/labels")
OUTPUT_DIR = Path("outputs/rectified/polygon")

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


# ============================================================
# FUNCTIONS
# ============================================================

def order_points(points):
    """
    Order four points as:
    top-left, top-right, bottom-right, bottom-left
    """

    points = np.array(points, dtype=np.float32)

    s = points.sum(axis=1)
    diff = np.diff(points, axis=1).flatten()

    top_left = points[np.argmin(s)]
    bottom_right = points[np.argmax(s)]
    top_right = points[np.argmin(diff)]
    bottom_left = points[np.argmax(diff)]

    return np.array(
        [top_left, top_right, bottom_right, bottom_left],
        dtype=np.float32
    )


def get_four_corners(points):
    """
    Convert an arbitrary polygon into four corner points.

    Uses convex hull + approxPolyDP.
    """

    points = np.array(points, dtype=np.float32)

    hull = cv2.convexHull(points)

    perimeter = cv2.arcLength(hull, True)

    # Try several approximation strengths
    for epsilon_ratio in [0.01, 0.02, 0.03, 0.04, 0.05, 0.08]:

        approx = cv2.approxPolyDP(
            hull,
            epsilon_ratio * perimeter,
            True
        )

        if len(approx) == 4:
            return approx.reshape(4, 2)

    # If approximation does not give exactly four points,
    # use minimum-area rectangle.
    rect = cv2.minAreaRect(points)
    box = cv2.boxPoints(rect)

    return box


def read_polygon(label_file, image_width, image_height):
    """
    Read YOLO polygon annotation.

    Format:
    class x1 y1 x2 y2 x3 y3 ...
    """

    polygons = []

    with open(label_file, "r", encoding="utf-8") as f:

        for line in f:

            values = line.strip().split()

            if len(values) < 7:
                continue

            class_id = values[0]

            # We only want class 1 plates
            if class_id != "1":
                continue

            coords = np.array(
                list(map(float, values[1:])),
                dtype=np.float32
            )

            # x,y pairs
            points = coords.reshape(-1, 2)

            # YOLO coordinates → pixels
            points[:, 0] *= image_width
            points[:, 1] *= image_height

            polygons.append(points)

    return polygons


def rectify_plate(image, polygon):
    """
    Perspective-correct a polygon plate.
    """

    corners = get_four_corners(polygon)
    corners = order_points(corners)

    tl, tr, br, bl = corners

    # Calculate width
    width_top = np.linalg.norm(tr - tl)
    width_bottom = np.linalg.norm(br - bl)

    max_width = int(max(width_top, width_bottom))

    # Calculate height
    height_left = np.linalg.norm(bl - tl)
    height_right = np.linalg.norm(br - tr)

    max_height = int(max(height_left, height_right))

    if max_width < 10 or max_height < 10:
        return None

    # --------------------------------------------------------
    # Important:
    # Keep the natural orientation.
    # If the plate is vertical, don't force it horizontal.
    # --------------------------------------------------------

    destination = np.array(
        [
            [0, 0],
            [max_width - 1, 0],
            [max_width - 1, max_height - 1],
            [0, max_height - 1]
        ],
        dtype=np.float32
    )

    matrix = cv2.getPerspectiveTransform(
        corners,
        destination
    )

    rectified = cv2.warpPerspective(
        image,
        matrix,
        (max_width, max_height)
    )

    # --------------------------------------------------------
    # Normalize orientation
    # License plates should normally be horizontal for OCR.
    # If the rectified plate is significantly taller than wide,
    # rotate it 90 degrees.
    # --------------------------------------------------------

    h, w = rectified.shape[:2]

    if h > w * 1.2:
        rectified = cv2.rotate(
            rectified,
            cv2.ROTATE_90_CLOCKWISE
        )

    return rectified


# ============================================================
# MAIN
# ============================================================

metadata_file = Path("outputs/plate_crops/metadata.csv")

metadata = pd.read_csv(metadata_file)

polygon_rows = metadata[
    metadata["annotation_type"] == "polygon"
]

print("=" * 60)
print("POLYGON PERSPECTIVE RECTIFICATION")
print("=" * 60)

processed = 0
failed = 0

for _, row in polygon_rows.iterrows():

    image_name = row["original_image"]

    image_path = IMAGE_DIR / image_name

    label_name = Path(image_name).with_suffix(".txt").name
    label_path = LABEL_DIR / label_name

    if not image_path.exists():
        print(f"[SKIP] Image not found: {image_name}")
        failed += 1
        continue

    if not label_path.exists():
        print(f"[SKIP] Label not found: {label_name}")
        failed += 1
        continue

    image = cv2.imread(str(image_path))

    if image is None:
        print(f"[SKIP] Could not read: {image_name}")
        failed += 1
        continue

    height, width = image.shape[:2]

    polygons = read_polygon(
        label_path,
        width,
        height
    )

    if not polygons:
        print(f"[SKIP] No polygon: {image_name}")
        failed += 1
        continue

    # Find polygon corresponding to this crop.
    # Usually there is one, but some images can contain multiple.
    # We use the metadata bounding box to select the closest polygon.

    target_center = np.array([
        (float(row["x_min"]) + float(row["x_max"])) / 2,
        (float(row["y_min"]) + float(row["y_max"])) / 2
    ])

    best_polygon = None
    best_distance = float("inf")

    for polygon in polygons:

        center = polygon.mean(axis=0)

        distance = np.linalg.norm(
            center - target_center
        )

        if distance < best_distance:
            best_distance = distance
            best_polygon = polygon

    if best_polygon is None:
        failed += 1
        continue

    rectified = rectify_plate(
        image,
        best_polygon
    )

    if rectified is None:
        print(f"[FAILED] {image_name}")
        failed += 1
        continue

    output_name = (
        Path(row["crop_filename"]).stem
        + "_rectified.jpg"
    )

    output_path = OUTPUT_DIR / output_name

    cv2.imwrite(
        str(output_path),
        rectified,
        [cv2.IMWRITE_JPEG_QUALITY, 95]
    )

    processed += 1

    if processed % 10 == 0:
        print(f"Processed: {processed}")


print()
print("=" * 60)
print("RECTIFICATION COMPLETE")
print("=" * 60)
print(f"Polygon annotations : {len(polygon_rows)}")
print(f"Processed           : {processed}")
print(f"Failed              : {failed}")
print(f"Output directory    : {OUTPUT_DIR}")
print("=" * 60)