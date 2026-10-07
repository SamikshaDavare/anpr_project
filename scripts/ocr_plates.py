"""
ANPR OCR Pipeline - Candidate-Selection Architecture
====================================================
Performs Optical Character Recognition on license-plate crops using EasyOCR
(English + Arabic). Implements geometric bounding-box grouping, peripheral-marking
filtering, Arabic/Persian digit normalization, structural plausibility evaluation,
repetition penalties, multi-signal candidate scoring, and staged preprocessing retries.

Never concatenates all detections across a crop into one string.
Distinguishes raw OCR confidence from composite quality score, and marks ambiguous
or problematic results as REVIEW rather than returning confidently false plates.
"""

import csv
import json
from pathlib import Path
import re
import sys
import cv2
import easyocr
import torch

# Ensure UTF-8 output on Windows consoles
if sys.stdout.encoding and sys.stdout.encoding.lower() != "utf-8":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass


# ============================================================
# CONFIGURATION & CONSTANTS
# ============================================================

BASE_DIR = Path("outputs/preprocessed")
ENHANCED_DIR = BASE_DIR / "enhanced"
ORIGINAL_DIR = BASE_DIR / "original"
THRESHOLD_DIR = BASE_DIR / "threshold"

RESULTS_FILE = Path("outputs/ocr_results.csv")
CANDIDATES_FILE = Path("outputs/ocr_candidates.csv")

# EasyOCR detection parameters
OCR_WIDTH_THS = 0.7
OCR_TEXT_THRESHOLD = 0.4
OCR_LOW_TEXT = 0.3
OCR_LINK_THRESHOLD = 0.3
OCR_MAG_RATIO = 1.0

# Arabic-Indic and Eastern Arabic-Indic / Persian digits to Western digits
DIGIT_MAP = str.maketrans({
    # Arabic-Indic digits
    "٠": "0", "١": "1", "٢": "2", "٣": "3", "٤": "4",
    "٥": "5", "٦": "6", "٧": "7", "٨": "8", "٩": "9",
    # Persian / Eastern Arabic-Indic digits
    "۰": "0", "۱": "1", "۲": "2", "۳": "3", "۴": "4",
    "۵": "5", "۶": "6", "۷": "7", "۸": "8", "۹": "9",
})

# Peripheral markings, province codes, and category identifiers
PERIPHERAL_KEYWORDS = {
    "IND", "IN", "INDIA", "PRV", "PRK", "HRT", "KBL",
    "HERAT", "KABUL", "GOVT", "POLICE", "COMMERCIAL",
    "PRIVATE", "TRANSPORT", "شخصی", "هرات", "کابل", "تک"
}

# Subtitle / banner noise fragments found on government plate borders
BANNER_KEYWORDS = {
    "REPUBLIC", "ISLAMIC", "AFGHANISTAN", "SLG", "PUBLG",
    "ESLEU", "IBLAIIC", "MMSUIS", "DDZIGH", "DAAFGHANISTAN",
    "STATE", "GOVERNMENT"
}

# Quality thresholds
MIN_GOOD_FINAL_SCORE = 0.42
MIN_GOOD_OCR_CONF = 0.28
MIN_GOOD_LENGTH = 4
MAX_GOOD_LENGTH = 12


# ============================================================
# INITIALIZE OCR READER
# ============================================================

def init_reader():
    """Initializes EasyOCR reader with GPU if available, else CPU."""
    use_gpu = torch.cuda.is_available()
    print("=" * 60)
    print("ANPR OCR - ADVANCED CANDIDATE SELECTION")
    print("=" * 60)
    print(f"Device: {'CUDA GPU' if use_gpu else 'CPU'}")
    print("Languages: English + Arabic")
    print("Loading EasyOCR models...")
    reader = easyocr.Reader(["en", "ar"], gpu=use_gpu, verbose=False)
    print("Models loaded successfully.\n")
    return reader


# ============================================================
# TEXT NORMALIZATION
# ============================================================

def normalize_text(text):
    """
    Normalizes OCR text:
    - Uppercase conversion
    - Arabic/Persian digits normalized to 0-9
    - Removal of punctuation and non-alphanumeric noise
    - Preservation of A-Z and 0-9
    Does NOT blindly substitute letters for numbers (e.g. O->0, I->1, B->8).
    """
    if not text:
        return ""
    text = str(text).upper().strip()
    text = text.translate(DIGIT_MAP)
    # Remove non-alphanumeric characters
    text = re.sub(r"[^A-Z0-9]", "", text)
    return text


# ============================================================
# DETECTION GEOMETRY EXTRACTION
# ============================================================

def extract_detection_geometry(detection, img_w, img_h):
    """
    Extracts spatial geometry and metadata for an individual EasyOCR detection:
    - Bounding box coordinates (xmin, xmax, ymin, ymax, w, h)
    - Center coordinates (cx, cy)
    - Relative height, width, and position normalized by crop dimensions
    - Peripheral marking and banner noise detection
    """
    bbox, raw_text, conf = detection
    xs = [p[0] for p in bbox]
    ys = [p[1] for p in bbox]
    xmin = max(0.0, float(min(xs)))
    xmax = min(float(img_w), float(max(xs)))
    ymin = max(0.0, float(min(ys)))
    ymax = min(float(img_h), float(max(ys)))
    w = max(1.0, xmax - xmin)
    h = max(1.0, ymax - ymin)
    cx = (xmin + xmax) / 2.0
    cy = (ymin + ymax) / 2.0

    clean_text = normalize_text(raw_text)
    raw_upper = str(raw_text).upper().strip()

    # Identify thin banner strip noise (e.g. tiny government slogans along edges)
    rel_h = h / max(1.0, float(img_h))
    rel_cy = cy / max(1.0, float(img_h))
    is_banner = False
    if (rel_h < 0.18 and rel_cy > 0.70 and float(conf) < 0.20) or any(k in raw_upper for k in BANNER_KEYWORDS):
        is_banner = True

    # Identify peripheral markings (country/province/category identifiers)
    is_peripheral = (
        clean_text in PERIPHERAL_KEYWORDS
        or raw_upper in PERIPHERAL_KEYWORDS
        or any(k == clean_text for k in PERIPHERAL_KEYWORDS)
    )

    return {
        "raw_text": raw_text,
        "text": clean_text,
        "conf": float(conf),
        "bbox": bbox,
        "xmin": xmin, "xmax": xmax,
        "ymin": ymin, "ymax": ymax,
        "w": w, "h": h,
        "cx": cx, "cy": cy,
        "rel_w": w / max(1.0, float(img_w)),
        "rel_h": rel_h,
        "rel_cx": cx / max(1.0, float(img_w)),
        "rel_cy": rel_cy,
        "area": w * h,
        "is_banner": is_banner,
        "is_peripheral": is_peripheral,
        "detections_count": 1
    }


# ============================================================
# GEOMETRIC GROUPING & LINE FORMATION
# ============================================================

def group_detections_into_lines(detections):
    """
    Groups spatial detections that likely belong to the same physical text line based on:
    - Vertical center proximity
    - Bounding-box height similarity
    - Vertical overlap
    Partitions detections into sorted horizontal lines (top to bottom, left to right).
    """
    if not detections:
        return []

    # Sort primarily by vertical center
    sorted_dets = sorted(detections, key=lambda d: d["cy"])
    lines = []

    for det in sorted_dets:
        placed = False
        for line in lines:
            avg_cy = sum(d["cy"] for d in line) / len(line)
            avg_h = sum(d["h"] for d in line) / len(line)

            vert_diff = abs(det["cy"] - avg_cy)
            height_ratio = min(det["h"], avg_h) / max(det["h"], avg_h)

            # Compatible if vertical centers are close and heights are similar
            if vert_diff < 0.50 * max(det["h"], avg_h) and height_ratio > 0.35:
                line.append(det)
                placed = True
                break
        if not placed:
            lines.append([det])

    # Sort detections within each line from left to right by xmin
    for line in lines:
        line.sort(key=lambda d: d["xmin"])

    # Sort lines from top to bottom by average cy
    lines.sort(key=lambda line: sum(d["cy"] for d in line) / len(line))
    return lines


def build_candidate_from_group(det_list, img_w, img_h):
    """
    Combines a group of detections into a single coherent candidate representation:
    - Concatenates texts in spatial order
    - Computes character-length-weighted OCR confidence
    - Calculates enclosing bounding box
    """
    if not det_list:
        return None

    valid = [d for d in det_list if d.get("text")]
    if not valid:
        return None

    combined_text = "".join(d["text"] for d in valid)
    total_len = sum(len(d["text"]) for d in valid)
    if total_len > 0:
        weighted_conf = sum(d["conf"] * len(d["text"]) for d in valid) / total_len
    else:
        weighted_conf = sum(d["conf"] for d in valid) / len(valid)

    xmin = min(d["xmin"] for d in valid)
    xmax = max(d["xmax"] for d in valid)
    ymin = min(d["ymin"] for d in valid)
    ymax = max(d["ymax"] for d in valid)
    w = max(1.0, xmax - xmin)
    h = max(1.0, ymax - ymin)

    is_all_peripheral = all(d.get("is_peripheral", False) for d in valid)

    return {
        "text": combined_text,
        "conf": weighted_conf,
        "xmin": xmin, "xmax": xmax,
        "ymin": ymin, "ymax": ymax,
        "w": w, "h": h,
        "cx": (xmin + xmax) / 2.0,
        "cy": (ymin + ymax) / 2.0,
        "area": w * h,
        "detections_count": len(valid),
        "is_all_peripheral": is_all_peripheral
    }


# ============================================================
# SCORING FUNCTIONS
# ============================================================

def repetition_penalty(text):
    """
    Detects suspicious repeated substrings / duplication artifacts caused by
    concatenating duplicate detections (e.g. Arabic vs English lines, or overlapping windows).
    Protects legitimate repeated single digits (e.g. 0001, 9999).
    """
    if not text or len(text) < 6:
        return 0.0

    n = len(text)
    # Exact half repeat: ABCDABCD -> penalty
    if n % 2 == 0 and text[:n // 2] == text[n // 2:]:
        return 0.60

    # Near repeat in halves: e.g. 29278 vs 29378 or J27648 vs R27648
    half = n // 2
    part1, part2 = text[:half], text[n - half:]
    matches = sum(1 for a, b in zip(part1, part2) if a == b)
    if matches >= 4 and (matches / half) >= 0.75:
        return 0.50

    # Repeated block of length >= 4 appearing elsewhere in text
    for k in range(4, half + 1):
        for i in range(n - 2 * k + 1):
            sub = text[i:i + k]
            if sub in text[i + k:]:
                return 0.40

    return 0.0


def structure_score(text):
    """
    Evaluates soft plate structure plausibility:
    - Works across Indian, Afghan, Arabic, international, and partial formats.
    - Favors plausible registration lengths (5 to 10 chars).
    - Recognizes standard Indian alphanumeric patterns.
    - Recognizes standard international/Afghan 4 to 6 digit registrations.
    - Penalizes extreme lengths (< 4 or >= 13) and pure-letter non-registration codes.
    """
    if not text:
        return 0.0

    n = len(text)
    score = 0.0

    # Length plausibility
    if 5 <= n <= 10:
        score += 0.40
    elif n == 4 or 11 <= n <= 12:
        score += 0.20
    elif n < 4:
        score -= 0.30
    else:  # >= 13
        score -= 0.40

    has_letters = bool(re.search(r"[A-Z]", text))
    has_digits = bool(re.search(r"[0-9]", text))

    # Standard Indian plate format: e.g. MH12AB1234 or CH01AN0001
    indian_pattern = bool(re.match(r"^[A-Z]{2}[0-9]{1,2}[A-Z]{0,3}[0-9]{3,4}$", text))
    if indian_pattern:
        score += 0.40
    elif has_letters and has_digits:
        score += 0.25
    elif has_digits and not has_letters:
        # Pure numeric: 4 to 6 digits is standard for Afghan/international plates
        if 4 <= n <= 6:
            score += 0.35
        else:
            score += 0.10
    elif has_letters and not has_digits:
        # Pure letters: usually markings or words, not registration
        if n <= 3:
            score -= 0.35
        else:
            score -= 0.20

    return max(-1.0, min(1.0, score))


def geometry_score(candidate, img_w, img_h):
    """
    Calculates spatial coherence score based on character prominence and location:
    - Height ratio (main plate text typically occupies 30% to 85% of crop height)
    - Width ratio (main text occupies significant horizontal width)
    - Centrality (centered text preferred over edge markings)
    """
    rel_h = candidate["h"] / max(1.0, float(img_h))
    rel_w = candidate["w"] / max(1.0, float(img_w))
    cx, cy = candidate["cx"], candidate["cy"]

    if 0.30 <= rel_h <= 0.85:
        h_score = 1.0
    elif 0.20 <= rel_h < 0.30:
        h_score = 0.60
    elif 0.85 < rel_h <= 1.0:
        h_score = 0.80
    else:
        h_score = 0.20

    if rel_w >= 0.35:
        w_score = 1.0
    elif rel_w >= 0.20:
        w_score = 0.60
    else:
        w_score = 0.20

    dx = abs(cx - img_w / 2.0) / (img_w / 2.0)
    dy = abs(cy - img_h / 2.0) / (img_h / 2.0)
    c_score = max(0.0, 1.0 - 0.5 * (dx + dy))

    return round(0.40 * h_score + 0.35 * w_score + 0.25 * c_score, 4)


def peripheral_penalty(candidate):
    """
    Penalizes candidates that consist of or are contaminated by peripheral markings:
    - Detections flagged as all-peripheral
    - Pure keyword matches (e.g. IND, PRV, KBL)
    - Concatenated peripheral keywords (e.g. KBLPRV)
    - Strings ending or starting with peripheral indicators
    """
    text = candidate["text"]

    if candidate.get("is_all_peripheral", False):
        return 0.50

    if text in PERIPHERAL_KEYWORDS:
        return 0.50

    if any(k in text for k in BANNER_KEYWORDS):
        return 0.60

    # Check if text is composed solely of peripheral keywords
    rem = text
    for k in sorted(PERIPHERAL_KEYWORDS, key=len, reverse=True):
        rem = rem.replace(k, "")
    if len(rem) == 0:
        return 0.50

    # Mild penalty if peripheral code is appended/prepended to registration
    for k in PERIPHERAL_KEYWORDS:
        if (text.endswith(k) or text.startswith(k)) and len(text) > len(k):
            return 0.15

    return 0.0


def candidate_score(candidate, img_w, img_h):
    """
    Computes explainable composite quality score from:
    OCR confidence + Geometry + Structure - Repetition penalty - Peripheral penalty.
    """
    text = candidate["text"]
    conf = float(candidate["conf"])
    geom = geometry_score(candidate, img_w, img_h)
    struct = structure_score(text)
    rep_pen = repetition_penalty(text)
    per_pen = peripheral_penalty(candidate)

    final = (
        0.35 * conf
        + 0.25 * geom
        + 0.25 * struct
        - rep_pen
        - per_pen
    )

    bbox_list = [
        round(candidate["xmin"], 1),
        round(candidate["ymin"], 1),
        round(candidate["w"], 1),
        round(candidate["h"], 1)
    ]

    return {
        "text": text,
        "ocr_confidence": round(conf, 4),
        "geometry_score": round(geom, 4),
        "structure_score": round(struct, 4),
        "repetition_penalty": round(rep_pen, 4),
        "peripheral_penalty": round(per_pen, 4),
        "final_score": round(final, 4),
        "bbox": bbox_list
    }


# ============================================================
# CANDIDATE GENERATION & SELECTION
# ============================================================

def generate_candidates(detections, img_w, img_h):
    """
    Generates competing candidate plate interpretations:
    1. Filter out obvious bottom banner noise
    2. Treat each valid detection as an independent atomic candidate
    3. Form line-grouped combinations
    4. Form peripheral-filtered line combinations
    5. Form multi-line combinations (only when not duplicate parallel scripts)
    """
    valid_dets = [d for d in detections if not d["is_banner"] and d["text"]]
    if not valid_dets:
        valid_dets = [d for d in detections if d["text"]]
        if not valid_dets:
            return []

    candidates_dict = {}

    def add_candidate(cand_obj):
        if not cand_obj or not cand_obj.get("text"):
            return
        scored = candidate_score(cand_obj, img_w, img_h)
        text = scored["text"]
        # Keep candidate with highest final score if duplicate text
        if text not in candidates_dict or scored["final_score"] > candidates_dict[text]["final_score"]:
            candidates_dict[text] = scored

    # 1. Atomic candidates
    for d in valid_dets:
        add_candidate(d)

    # 2. Line grouping
    lines = group_detections_into_lines(valid_dets)

    for line in lines:
        if len(line) > 1:
            # Full line combination
            full_line_cand = build_candidate_from_group(line, img_w, img_h)
            add_candidate(full_line_cand)

            # Filtered combination (excluding boundary peripheral markings)
            non_periph = [d for d in line if not d.get("is_peripheral", False)]
            if non_periph and len(non_periph) < len(line):
                filtered_cand = build_candidate_from_group(non_periph, img_w, img_h)
                add_candidate(filtered_cand)

    # 3. Multi-line stacking (support two-line plates)
    if len(lines) == 2:
        cand1 = build_candidate_from_group(lines[0], img_w, img_h)
        cand2 = build_candidate_from_group(lines[1], img_w, img_h)

        t1 = cand1["text"] if cand1 else ""
        t2 = cand2["text"] if cand2 else ""

        # Check if lines are duplicate representations (e.g. Arabic digits vs English digits)
        d1 = re.sub(r"[^0-9]", "", t1)
        d2 = re.sub(r"[^0-9]", "", t2)
        is_duplicate = False
        if d1 and d2 and (d1 == d2 or (len(d1) >= 4 and len(d2) >= 4 and (d1 in d2 or d2 in d1))):
            is_duplicate = True

        if not is_duplicate:
            stacked_cand = build_candidate_from_group(lines[0] + lines[1], img_w, img_h)
            add_candidate(stacked_cand)

    candidate_list = list(candidates_dict.values())
    candidate_list.sort(key=lambda c: c["final_score"], reverse=True)
    return candidate_list


def determine_status(candidate):
    """
    Determines status: GOOD, REVIEW, or NO_TEXT.
    Does not assume non-empty equals good.
    """
    if not candidate or not candidate.get("text"):
        return "NO_TEXT"

    text = candidate["text"]
    conf = candidate["ocr_confidence"]
    score = candidate["final_score"]
    struct = candidate["structure_score"]
    rep_pen = candidate["repetition_penalty"]
    per_pen = candidate["peripheral_penalty"]

    is_good = (
        score >= MIN_GOOD_FINAL_SCORE
        and conf >= MIN_GOOD_OCR_CONF
        and MIN_GOOD_LENGTH <= len(text) <= MAX_GOOD_LENGTH
        and struct >= 0.15
        and rep_pen == 0.0
        and per_pen == 0.0
    )

    return "GOOD" if is_good else "REVIEW"


def should_retry(candidate):
    """
    Checks if a preprocessing retry should be triggered:
    - No text detected
    - Very low confidence (< 0.40)
    - Very short text (< 4 chars)
    - Excessively long text (> 12 chars)
    - Poor structure score (< 0.20)
    - Non-zero repetition or peripheral penalty
    - Status is REVIEW
    """
    if candidate is None or not candidate.get("text"):
        return True

    text = candidate["text"]
    conf = candidate["ocr_confidence"]
    score = candidate["final_score"]
    struct = candidate["structure_score"]

    if conf < 0.40 or len(text) < 4 or len(text) > 12:
        return True
    if struct < 0.20 or score < 0.45:
        return True
    if candidate["repetition_penalty"] > 0 or candidate["peripheral_penalty"] > 0:
        return True

    return False


# ============================================================
# OCR INFERENCE ON A SINGLE IMAGE
# ============================================================

def run_ocr_on_file(reader, image_path):
    """
    Runs EasyOCR inference on a single image file, extracts geometries,
    generates all candidates, and returns candidate diagnostics and raw text.
    """
    image = cv2.imread(str(image_path))
    if image is None:
        return {
            "candidates": [],
            "detections": 0,
            "raw_text": ""
        }

    h, w = image.shape[:2]
    raw_results = reader.readtext(
        image,
        detail=1,
        paragraph=False,
        mag_ratio=OCR_MAG_RATIO,
        width_ths=OCR_WIDTH_THS,
        text_threshold=OCR_TEXT_THRESHOLD,
        low_text=OCR_LOW_TEXT,
        link_threshold=OCR_LINK_THRESHOLD
    )

    geometries = [extract_detection_geometry(det, w, h) for det in raw_results]
    candidates = generate_candidates(geometries, w, h)

    raw_tokens = [d["text"] for d in geometries if d.get("text")]
    raw_text = " ".join(raw_tokens)

    return {
        "candidates": candidates,
        "detections": len(raw_results),
        "raw_text": raw_text
    }


# ============================================================
# MAIN PIPELINE EXECUTION
# ============================================================

def process_all_plates():
    """
    Processes all plate crops through the staged OCR retry architecture:
    Pass 1: enhanced
    Pass 2: original (if retry needed)
    Pass 3: threshold (if still needed)
    Saves outputs/ocr_results.csv and outputs/ocr_candidates.csv.
    """
    reader = init_reader()

    plate_files = sorted(ENHANCED_DIR.glob("*.jpg"))
    total_count = len(plate_files)

    print(f"Total plate crops found: {total_count}")
    print("Beginning staged OCR processing...\n")

    final_results = []
    all_evaluated_candidates = []

    good_count = 0
    review_count = 0
    no_text_count = 0

    for i, enhanced_path in enumerate(plate_files, 1):
        filename = enhanced_path.name
        print(f"[{i:03d}/{total_count:03d}] {filename}")

        crop_candidates_by_variant = {}

        # ----------------------------------------------------
        # PASS 1: ENHANCED (Primary)
        # ----------------------------------------------------
        pass1 = run_ocr_on_file(reader, enhanced_path)
        for c in pass1["candidates"]:
            c_copy = dict(c)
            c_copy["variant"] = "enhanced"
            crop_candidates_by_variant.setdefault("enhanced", []).append(c_copy)

        best_cand = pass1["candidates"][0] if pass1["candidates"] else None
        winning_variant = "enhanced"
        winning_detections = pass1["detections"]
        winning_raw_text = pass1["raw_text"]

        # ----------------------------------------------------
        # RETRY EVALUATION
        # ----------------------------------------------------
        if should_retry(best_cand):
            # Pass 2: original
            original_path = ORIGINAL_DIR / filename
            if original_path.exists():
                pass2 = run_ocr_on_file(reader, original_path)
                for c in pass2["candidates"]:
                    c_copy = dict(c)
                    c_copy["variant"] = "original"
                    crop_candidates_by_variant.setdefault("original", []).append(c_copy)

                cand2 = pass2["candidates"][0] if pass2["candidates"] else None
                if cand2 and (best_cand is None or cand2["final_score"] > best_cand["final_score"]):
                    best_cand = cand2
                    winning_variant = "original"
                    winning_detections = pass2["detections"]
                    winning_raw_text = pass2["raw_text"]

            # Pass 3: threshold (if still needed)
            if should_retry(best_cand):
                threshold_path = THRESHOLD_DIR / filename
                if threshold_path.exists():
                    pass3 = run_ocr_on_file(reader, threshold_path)
                    for c in pass3["candidates"]:
                        c_copy = dict(c)
                        c_copy["variant"] = "threshold"
                        crop_candidates_by_variant.setdefault("threshold", []).append(c_copy)

                    cand3 = pass3["candidates"][0] if pass3["candidates"] else None
                    if cand3 and (best_cand is None or cand3["final_score"] > best_cand["final_score"]):
                        best_cand = cand3
                        winning_variant = "threshold"
                        winning_detections = pass3["detections"]
                        winning_raw_text = pass3["raw_text"]

        # ----------------------------------------------------
        # RECORD CANDIDATES FOR DIAGNOSTICS
        # ----------------------------------------------------
        total_cand_count = sum(len(c_list) for c_list in crop_candidates_by_variant.values())
        winning_text = best_cand["text"] if best_cand else ""

        for var_name, cand_list in crop_candidates_by_variant.items():
            for c in cand_list:
                is_selected = (
                    var_name == winning_variant
                    and c["text"] == winning_text
                    and winning_text != ""
                )
                all_evaluated_candidates.append({
                    "plate_image": filename,
                    "variant": var_name,
                    "candidate_text": c["text"],
                    "ocr_confidence": c["ocr_confidence"],
                    "geometry_score": c["geometry_score"],
                    "structure_score": c["structure_score"],
                    "repetition_penalty": c["repetition_penalty"],
                    "peripheral_penalty": c["peripheral_penalty"],
                    "final_score": c["final_score"],
                    "bbox": json.dumps(c["bbox"]),
                    "selected": is_selected
                })

        # ----------------------------------------------------
        # RECORD FINAL RESULT
        # ----------------------------------------------------
        status = determine_status(best_cand)
        if status == "GOOD":
            good_count += 1
        elif status == "REVIEW":
            review_count += 1
        else:
            no_text_count += 1

        detected_text = best_cand["text"] if best_cand else ""
        ocr_conf = best_cand["ocr_confidence"] if best_cand else 0.0
        final_sc = best_cand["final_score"] if best_cand else 0.0
        geom_sc = best_cand["geometry_score"] if best_cand else 0.0
        struct_sc = best_cand["structure_score"] if best_cand else 0.0
        rep_pen = best_cand["repetition_penalty"] if best_cand else 0.0
        per_pen = best_cand["peripheral_penalty"] if best_cand else 0.0

        final_results.append({
            "plate_image": filename,
            "detected_text": detected_text,
            "confidence": ocr_conf,
            "score": final_sc,
            "variant": winning_variant,
            "detections": winning_detections,
            "candidate_count": total_cand_count,
            "raw_text": winning_raw_text,
            "status": status,
            "ocr_confidence": ocr_conf,
            "geometry_score": geom_sc,
            "structure_score": struct_sc,
            "repetition_penalty": rep_pen,
            "peripheral_penalty": per_pen
        })

        print(
            f"   -> Text: {detected_text:14s} | "
            f"Score: {final_sc:.3f} | "
            f"Conf: {ocr_conf:.3f} | "
            f"Status: {status:7s} | "
            f"Variant: {winning_variant}"
        )

    # --------------------------------------------------------
    # WRITE CSV FILES
    # --------------------------------------------------------
    RESULTS_FILE.parent.mkdir(parents=True, exist_ok=True)

    result_fieldnames = [
        "plate_image",
        "detected_text",
        "confidence",
        "score",
        "variant",
        "detections",
        "candidate_count",
        "raw_text",
        "status",
        "ocr_confidence",
        "geometry_score",
        "structure_score",
        "repetition_penalty",
        "peripheral_penalty"
    ]

    with open(RESULTS_FILE, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=result_fieldnames)
        writer.writeheader()
        writer.writerows(final_results)

    candidate_fieldnames = [
        "plate_image",
        "variant",
        "candidate_text",
        "ocr_confidence",
        "geometry_score",
        "structure_score",
        "repetition_penalty",
        "peripheral_penalty",
        "final_score",
        "bbox",
        "selected"
    ]

    with open(CANDIDATES_FILE, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=candidate_fieldnames)
        writer.writeheader()
        writer.writerows(all_evaluated_candidates)

    # --------------------------------------------------------
    # SUMMARY
    # --------------------------------------------------------
    print("\n" + "=" * 60)
    print("OCR PROCESSING COMPLETE")
    print("=" * 60)
    print(f"Total processed : {total_count}")
    print(f"GOOD            : {good_count} ({good_count / total_count * 100:.1f}%)")
    print(f"REVIEW          : {review_count} ({review_count / total_count * 100:.1f}%)")
    print(f"NO_TEXT         : {no_text_count} ({no_text_count / total_count * 100:.1f}%)")
    print(f"Total candidates logged: {len(all_evaluated_candidates)}")
    print(f"Final results file     : {RESULTS_FILE}")
    print(f"Candidates file        : {CANDIDATES_FILE}")
    print("=" * 60 + "\n")


if __name__ == "__main__":
    process_all_plates()