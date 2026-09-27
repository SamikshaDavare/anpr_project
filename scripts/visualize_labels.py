from pathlib import Path
import cv2

DATASET = Path("student_1")
IMAGE_DIR = DATASET / "images"
LABEL_DIR = DATASET / "labels"
OUTPUT_DIR = Path("outputs") / "label_preview"

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# Colors are intentionally different for the two classes.
CLASS_COLORS = {
    0: (0, 255, 0),
    1: (0, 0, 255),
}

for image_path in list(IMAGE_DIR.glob("*.jpg"))[:10]:

    label_path = LABEL_DIR / f"{image_path.stem}.txt"

    image = cv2.imread(str(image_path))

    if image is None:
        print(f"Could not read: {image_path.name}")
        continue

    height, width = image.shape[:2]

    if not label_path.exists():
        print(f"No label found: {image_path.name}")
        continue

    with open(label_path, "r") as f:
        lines = f.readlines()

    for line in lines:
        parts = line.strip().split()

        if len(parts) != 5:
            continue

        class_id = int(parts[0])
        x_center = float(parts[1])
        y_center = float(parts[2])
        box_width = float(parts[3])
        box_height = float(parts[4])

        x_center *= width
        y_center *= height
        box_width *= width
        box_height *= height

        x1 = int(x_center - box_width / 2)
        y1 = int(y_center - box_height / 2)
        x2 = int(x_center + box_width / 2)
        y2 = int(y_center + box_height / 2)

        color = CLASS_COLORS.get(class_id, (255, 255, 255))

        cv2.rectangle(
            image,
            (x1, y1),
            (x2, y2),
            color,
            4
        )

        cv2.putText(
            image,
            f"class {class_id}",
            (x1, max(y1 - 10, 30)),
            cv2.FONT_HERSHEY_SIMPLEX,
            1.2,
            color,
            3
        )

    output_path = OUTPUT_DIR / image_path.name
    cv2.imwrite(str(output_path), image)

    print(f"Saved: {output_path}")

print("\nDone.")
print(f"Preview images are in: {OUTPUT_DIR}")