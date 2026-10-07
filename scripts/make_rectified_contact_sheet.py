from pathlib import Path
import cv2
import math

INPUT_DIR = Path("outputs/rectified/polygon")
OUTPUT = Path("outputs/rectified/polygon_contact_sheet.jpg")

files = sorted(INPUT_DIR.glob("*.jpg"))

thumb_w = 300
thumb_h = 180
cols = 4
rows = math.ceil(len(files) / cols)

sheet = 255 * __import__("numpy").ones(
    (rows * thumb_h, cols * thumb_w, 3),
    dtype="uint8"
)

for i, f in enumerate(files):

    img = cv2.imread(str(f))

    if img is None:
        continue

    h, w = img.shape[:2]

    scale = min(
        (thumb_w - 10) / w,
        (thumb_h - 10) / h
    )

    nw = max(1, int(w * scale))
    nh = max(1, int(h * scale))

    img = cv2.resize(img, (nw, nh))

    x = (i % cols) * thumb_w
    y = (i // cols) * thumb_h

    px = x + (thumb_w - nw) // 2
    py = y + (thumb_h - nh) // 2

    sheet[py:py + nh, px:px + nw] = img

    cv2.putText(
        sheet,
        str(i + 1),
        (x + 5, y + 18),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.55,
        (0, 0, 0),
        1,
        cv2.LINE_AA
    )

cv2.imwrite(
    str(OUTPUT),
    sheet,
    [cv2.IMWRITE_JPEG_QUALITY, 95]
)

print(f"Created: {OUTPUT}")
print(f"Images: {len(files)}")