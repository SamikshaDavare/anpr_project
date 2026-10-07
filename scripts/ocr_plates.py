"""
ANPR OCR Pipeline - Complete Registration Reconstruction
=========================================================
Performs Optical Character Recognition on license-plate crops using EasyOCR
(English + Arabic). Implements:
- Spatial line grouping and contiguous adjacent span combination to reconstruct
  complete registration numbers (e.g. MH + 01 + AB + 1234 -> MH01AB1234).
- Position-aware OCR correction for Indian license plates (e.g. CH01AN0001, MH01AB1234).
- Soft structural scoring supporting both Indian plates and international/Afghan plates.
- Peripheral-marking and banner noise filtering.
- Digit normalization (Arabic/Persian digits -> Western 0-9).
- Staged preprocessing retries (enhanced -> original -> threshold).
- Generation of outputs/ocr_results.csv, outputs/ocr_candidates.csv, and
  comprehensive multi-sheet Excel workbook outputs/ANPR_results.xlsx.
"""

import csv
import json
from pathlib import Path
import re
import sys
import cv2
import easyocr
import openpyxl
from openpyxl.styles import Alignment, Border, Font, PatternFill, Side
from openpyxl.utils import get_column_letter
import pandas as pd
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
METADATA_FILE = Path("outputs/plate_crops/metadata.csv")

RESULTS_FILE = Path("outputs/ocr_results.csv")
CANDIDATES_FILE = Path("outputs/ocr_candidates.csv")
EXCEL_FILE = Path("outputs/ANPR_results.xlsx")

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

# Valid Indian State and Union Territory registration prefixes
INDIAN_STATE_CODES = {
    "AN", "AP", "AR", "AS", "BR", "CG", "CH", "DD", "DL", "DN",
    "GA", "GJ", "HP", "HR", "JH", "JK", "KA", "KL", "LA", "LD",
    "MH", "ML", "MN", "MP", "MZ", "NL", "OD", "OR", "PB", "PY",
    "RJ", "SK", "TN", "TR", "TS", "UK", "UA", "UP", "WB", "BH"
}

# Ambiguous OCR character substitution mappings for position-aware correction
CHAR_TO_DIGIT = {
    "O": "0", "D": "0", "Q": "0",
    "I": "1", "L": "1",
    "Z": "2",
    "S": "5",
    "B": "8",
    "G": "6"
}

CHAR_TO_LETTER = {
    "0": "O",
    "1": "I",
    "2": "Z",
    "5": "S",
    "8": "B",
    "6": "G"
}

# State-code OCR visual confusions (1-char edits to valid state codes)
STATE_CODE_CORRECTIONS = {
    "GH": "CH", "DH": "DL", "KH": "KA", "NH": "MH",
    "RH": "RJ", "TH": "TN", "UH": "UP", "WH": "WB"
}

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
    print("ANPR OCR - COMPLETE REGISTRATION RECONSTRUCTION")
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
    Does NOT perform blind character substitution.
    """
    if not text:
        return ""
    text = str(text).upper().strip()
    text = text.translate(DIGIT_MAP)
    text = re.sub(r"[^A-Z0-9]", "", text)
    return text


def is_indian_pattern(text):
    """Checks if text matches standard Indian license plate structure with valid state code."""
    if not text or not (7 <= len(text) <= 10):
        return False
    state = text[:2]
    if state not in INDIAN_STATE_CODES:
        return False
    return bool(re.match(r"^[A-Z]{2}[0-9]{1,2}[A-Z]{0,3}[0-9]{1,4}$", text))


def correct_indian_positions(text):
    """
    Applies conservative position-aware corrections when text structurally supports
    an Indian registration pattern:
    Structure: [State 2L] + [District 1-2D] + [Series 0-3L] + [Number 1-4D]
    Does NOT hallucinate missing characters.
    """
    if not text or len(text) < 7 or len(text) > 10:
        return text

    # Guard against non-Indian plates with peripheral markings (e.g. HRT, KBL, IND, PRV)
    for p in PERIPHERAL_KEYWORDS:
        if text.startswith(p) or text.endswith(p):
            return text

    chars = list(text)
    n = len(chars)

    # State prefix check / correction: GH -> CH
    prefix = "".join(chars[:2])
    if prefix in STATE_CODE_CORRECTIONS:
        chars[0], chars[1] = STATE_CODE_CORRECTIONS[prefix][0], STATE_CODE_CORRECTIONS[prefix][1]
        prefix = "".join(chars[:2])
    else:
        for i in [0, 1]:
            if chars[i] in CHAR_TO_LETTER:
                chars[i] = CHAR_TO_LETTER[chars[i]]
        prefix = "".join(chars[:2])

    if prefix not in INDIAN_STATE_CODES:
        return text

    # Try standard Indian plate splits: [State 2L] + [District 1-2D] + [Series 0-3L] + [Number 1-4D]
    best_reconstruction = None
    for dist_len in [2, 1]:
        for series_len in [2, 1, 0, 3]:
            num_len = n - 2 - dist_len - series_len
            if 1 <= num_len <= 4:
                p_state = chars[0:2]
                p_dist = chars[2 : 2 + dist_len]
                p_series = chars[2 + dist_len : 2 + dist_len + series_len]
                p_num = chars[2 + dist_len + series_len :]

                c_dist = [CHAR_TO_DIGIT.get(c, c) for c in p_dist]
                c_series = [CHAR_TO_LETTER.get(c, c) for c in p_series]
                c_num = [CHAR_TO_DIGIT.get(c, c) for c in p_num]

                if all(c.isdigit() for c in c_dist) and (not c_series or all(c.isalpha() for c in c_series)) and all(c.isdigit() for c in c_num):
                    cand = "".join(p_state + c_dist + c_series + c_num)
                    if is_indian_pattern(cand):
                        best_reconstruction = cand
                        break
        if best_reconstruction:
            break

    return best_reconstruction if best_reconstruction else text


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

    rel_h = h / max(1.0, float(img_h))
    rel_cy = cy / max(1.0, float(img_h))
    is_banner = False
    if (rel_h < 0.18 and rel_cy > 0.70 and float(conf) < 0.20) or any(k in raw_upper for k in BANNER_KEYWORDS):
        is_banner = True

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
        "detections_count": 1,
        "line_index": 0
    }


# ============================================================
# GEOMETRIC GROUPING & LINE FORMATION
# ============================================================

def group_detections_into_lines(detections):
    """
    Groups spatial detections into physical horizontal text lines based on:
    - Vertical center proximity
    - Bounding-box height similarity
    - Vertical overlap
    Returns sorted horizontal lines (top to bottom, left to right).
    """
    if not detections:
        return []

    sorted_dets = sorted(detections, key=lambda d: d["cy"])
    lines = []

    for det in sorted_dets:
        placed = False
        for line in lines:
            avg_cy = sum(d["cy"] for d in line) / len(line)
            avg_h = sum(d["h"] for d in line) / len(line)

            vert_diff = abs(det["cy"] - avg_cy)
            height_ratio = min(det["h"], avg_h) / max(det["h"], avg_h)

            if vert_diff < 0.50 * max(det["h"], avg_h) and height_ratio > 0.35:
                line.append(det)
                placed = True
                break
        if not placed:
            lines.append([det])

    # Sort detections within each line from left to right by xmin
    for line_idx, line in enumerate(lines, 1):
        line.sort(key=lambda d: d["xmin"])
        for d in line:
            d["line_index"] = line_idx

    # Sort lines from top to bottom
    lines.sort(key=lambda line: sum(d["cy"] for d in line) / len(line))
    return lines


def build_candidate_from_group(det_list, img_w, img_h):
    """
    Combines a group of detections into a candidate dictionary:
    - Concatenates texts in spatial reading order
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
    line_nums = sorted(list(set(d.get("line_index", 1) for d in valid)))
    line_desc = ",".join(str(l) for l in line_nums)

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
        "is_all_peripheral": is_all_peripheral,
        "line": line_desc
    }


# ============================================================
# SCORING & EVALUATION
# ============================================================

def repetition_penalty(text):
    """
    Detects duplicate substring patterns resulting from concatenated dual-line
    or overlapping detections (e.g. 42375...42375 or 27648...27648).
    Protects legitimate repeated single digits (e.g. 0001, 9999).
    """
    if not text or len(text) < 6:
        return 0.0

    n = len(text)
    # Exact half repeat: ABCDABCD
    if n % 2 == 0 and text[:n // 2] == text[n // 2:]:
        return 0.60

    # Near repeat in halves
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
    Evaluates license plate structure:
    - Strong positive signal for Indian format with valid state prefix (e.g. MH01AB1234, CH01AN0001)
    - Strong positive signal for valid international 4-6 digit registrations (e.g. 42375, 29378)
    - Positive signal for general alphanumeric plates (e.g. RJ4545, KL203841)
    - Penalties for short fragments (< 4 chars) or pure non-registration letters
    """
    if not text:
        return 0.0

    n = len(text)
    score = 0.0

    # Length baseline
    if 5 <= n <= 10:
        score += 0.35
    elif n == 4 or 11 <= n <= 12:
        score += 0.20
    elif n < 4:
        score -= 0.40
    else:  # >= 13
        score -= 0.40

    has_letters = bool(re.search(r"[A-Z]", text))
    has_digits = bool(re.search(r"[0-9]", text))

    # Indian plate pattern check
    if is_indian_pattern(text):
        score += 0.35
        state_code = text[:2]
        if state_code in INDIAN_STATE_CODES:
            score += 0.20  # Strong preference for recognized state/UT code
        if 9 <= n <= 10:
            score += 0.10  # Full complete registration length
    elif has_letters and has_digits:
        # General mixed alphanumeric plate (e.g. RJ4545, 148439, etc.)
        score += 0.25
    elif has_digits and not has_letters:
        # Pure numeric: standard 4-6 digits for Afghan/international plates
        if 4 <= n <= 6:
            score += 0.35
        else:
            score += 0.10
    elif has_letters and not has_digits:
        # Pure letters: usually markings or words, not registration
        if n <= 3:
            score -= 0.40
        else:
            score -= 0.20

    return max(-1.0, min(1.0, score))


def geometry_score(candidate, img_w, img_h):
    """
    Calculates spatial prominence:
    - Character height ratio (prominent text 30% to 85% of crop height)
    - Width coverage
    - Centrality in crop
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

    if rel_w >= 0.40:
        w_score = 1.0
    elif rel_w >= 0.25:
        w_score = 0.70
    else:
        w_score = 0.30

    dx = abs(cx - img_w / 2.0) / (img_w / 2.0)
    dy = abs(cy - img_h / 2.0) / (img_h / 2.0)
    c_score = max(0.0, 1.0 - 0.5 * (dx + dy))

    return round(0.40 * h_score + 0.35 * w_score + 0.25 * c_score, 4)


def peripheral_penalty(candidate):
    """
    Penalizes candidates consisting of or contaminated by peripheral markings:
    - Pure keyword matches (e.g. IND, PRV, HRT, KBL)
    - Concatenated peripheral keywords (e.g. KBLPRV)
    - Banner substrings (e.g. REPUBLIC, ISLAMIC)
    """
    text = candidate["text"]

    if candidate.get("is_all_peripheral", False):
        return 0.50

    if text in PERIPHERAL_KEYWORDS:
        return 0.50

    if any(k in text for k in BANNER_KEYWORDS):
        return 0.60

    rem = text
    for k in sorted(PERIPHERAL_KEYWORDS, key=len, reverse=True):
        rem = rem.replace(k, "")
    if len(rem) == 0:
        return 0.50

    for k in PERIPHERAL_KEYWORDS:
        if (text.endswith(k) or text.startswith(k)) and len(text) > len(k):
            return 0.35

    return 0.0


def candidate_score(candidate, img_w, img_h):
    """
    Computes explainable composite quality score from:
    OCR confidence + Geometry + Structure - Repetition penalty - Peripheral penalty
    + Multi-token reconstruction bonus.
    """
    text = candidate["text"]
    conf = float(candidate["conf"])
    geom = geometry_score(candidate, img_w, img_h)
    struct = structure_score(text)
    rep_pen = repetition_penalty(text)
    per_pen = peripheral_penalty(candidate)

    # Multi-detection reconstruction bonus:
    # A complete string reconstructed from multiple adjacent detections covering
    # the plate region is rewarded over an isolated single-token fragment.
    det_count = candidate.get("detections_count", 1)
    reconstruction_bonus = 0.0
    if det_count >= 2 and geom >= 0.70 and len(text) >= 6 and per_pen == 0.0:
        reconstruction_bonus = 0.08

    final = (
        0.35 * conf
        + 0.25 * geom
        + 0.25 * struct
        + reconstruction_bonus
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
        "bbox": bbox_list,
        "detections_count": det_count,
        "line": candidate.get("line", "1")
    }


# ============================================================
# CANDIDATE GENERATION & RECONSTRUCTION
# ============================================================

def generate_candidates(detections, img_w, img_h):
    """
    Reconstructs complete candidate plate numbers from individual detections:
    1. Filters out banner noise
    2. Groups detections into physical lines
    3. Evaluates all contiguous spans (sub-sequences) of adjacent detections on each line
       e.g. MH + 01 + AB + 1234 -> MH01AB1234
    4. Evaluates spans with boundary peripheral tokens excluded
    5. Evaluates multi-line combinations (stacking) when lines are not duplicate scripts
    6. Applies position-aware Indian plate corrections when structurally supported
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
        if text not in candidates_dict or scored["final_score"] > candidates_dict[text]["final_score"]:
            candidates_dict[text] = scored

        # Test position-aware Indian plate correction if applicable
        corrected = correct_indian_positions(text)
        if corrected and corrected != text:
            corr_obj = dict(cand_obj)
            corr_obj["text"] = corrected
            scored_corr = candidate_score(corr_obj, img_w, img_h)
            if corrected not in candidates_dict or scored_corr["final_score"] > candidates_dict[corrected]["final_score"]:
                candidates_dict[corrected] = scored_corr

    # 1. Atomic candidates
    for d in valid_dets:
        add_candidate(d)

    # 2. Line grouping and contiguous span reconstruction
    lines = group_detections_into_lines(valid_dets)

    for line in lines:
        k = len(line)
        if k > 1:
            # Generate all contiguous spans of adjacent detections on the line
            for start_i in range(k):
                for end_j in range(start_i + 1, k + 1):
                    span = line[start_i:end_j]
                    comb_cand = build_candidate_from_group(span, img_w, img_h)
                    add_candidate(comb_cand)

                    # Filtered span (without peripheral tokens on the boundary)
                    non_periph_span = [d for d in span if not d.get("is_peripheral", False)]
                    if non_periph_span and len(non_periph_span) < len(span):
                        filt_cand = build_candidate_from_group(non_periph_span, img_w, img_h)
                        add_candidate(filt_cand)

    # 3. Multi-line stacking (support two-line plates)
    if len(lines) == 2:
        cand1 = build_candidate_from_group(lines[0], img_w, img_h)
        cand2 = build_candidate_from_group(lines[1], img_w, img_h)

        t1 = cand1["text"] if cand1 else ""
        t2 = cand2["text"] if cand2 else ""

        # Check if lines are duplicate parallel representations (e.g. Arabic vs English digits)
        d1 = re.sub(r"[^0-9]", "", t1)
        d2 = re.sub(r"[^0-9]", "", t2)
        is_duplicate = False
        if d1 and d2 and (d1 == d2 or (len(d1) >= 4 and len(d2) >= 4 and (d1 in d2 or d2 in d1))):
            is_duplicate = True

        if not is_duplicate:
            stacked_cand = build_candidate_from_group(lines[0] + lines[1], img_w, img_h)
            add_candidate(stacked_cand)

            # Also try stacking non-peripheral portions
            np_l0 = [d for d in lines[0] if not d.get("is_peripheral", False)]
            np_l1 = [d for d in lines[1] if not d.get("is_peripheral", False)]
            if np_l0 and np_l1 and (len(np_l0) < len(lines[0]) or len(np_l1) < len(lines[1])):
                stacked_filt = build_candidate_from_group(np_l0 + np_l1, img_w, img_h)
                add_candidate(stacked_filt)

    candidate_list = list(candidates_dict.values())
    candidate_list.sort(key=lambda c: c["final_score"], reverse=True)
    return candidate_list


def determine_status(candidate):
    """Determines status: GOOD, REVIEW, or NO_TEXT."""
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
    """Checks if a preprocessing retry should be triggered."""
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
    generates all reconstructed candidates, and returns candidate diagnostics.
    """
    image = cv2.imread(str(image_path))
    if image is None:
        return {
            "candidates": [],
            "detections": 0,
            "raw_text": "",
            "line_count": 0
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
    lines = group_detections_into_lines([d for d in geometries if not d["is_banner"] and d["text"]])
    candidates = generate_candidates(geometries, w, h)

    raw_tokens = [d["text"] for d in geometries if d.get("text")]
    raw_text = " ".join(raw_tokens)

    return {
        "candidates": candidates,
        "detections": len(raw_results),
        "raw_text": raw_text,
        "line_count": len(lines)
    }


# ============================================================
# EXCEL GENERATION (3 SHEETS)
# ============================================================

def generate_excel_workbook(final_results, all_evaluated_candidates, summary_stats):
    """
    Generates outputs/ANPR_results.xlsx with:
    - Sheet 1: Final Results
    - Sheet 2: OCR Candidates
    - Sheet 3: Summary
    Applies clean headers, borders, column widths, and formatting.
    """
    wb = openpyxl.Workbook()
    # Remove default sheet
    wb.remove(wb.active)

    # Styling definitions
    header_font = Font(name="Calibri", size=11, bold=True, color="FFFFFF")
    header_fill = PatternFill(start_color="1F497D", end_color="1F497D", fill_type="solid")
    accent_fill = PatternFill(start_color="DCE6F1", end_color="DCE6F1", fill_type="solid")
    thin_border = Border(
        left=Side(style="thin", color="D9D9D9"),
        right=Side(style="thin", color="D9D9D9"),
        top=Side(style="thin", color="D9D9D9"),
        bottom=Side(style="thin", color="D9D9D9")
    )

    # --------------------------------------------------------
    # SHEET 1: Final Results
    # --------------------------------------------------------
    ws1 = wb.create_sheet(title="Final Results")
    sheet1_headers = [
        "Image Name",
        "Plate Crop",
        "Detected Number",
        "Normalized Number",
        "OCR Confidence",
        "Final Score",
        "Status",
        "Preprocessing Variant",
        "Detection Count",
        "Line Count",
        "Indian Format Match",
        "State Code",
        "Review Required"
    ]
    ws1.append(sheet1_headers)

    for row_data in final_results:
        ws1.append([
            row_data.get("image_name", ""),
            row_data.get("plate_image", ""),
            row_data.get("detected_text", ""),
            row_data.get("detected_text", ""),
            row_data.get("confidence", 0.0),
            row_data.get("score", 0.0),
            row_data.get("status", ""),
            row_data.get("variant", ""),
            row_data.get("detections", 0),
            row_data.get("line_count", 1),
            row_data.get("indian_format_match", "No"),
            row_data.get("state_code", ""),
            row_data.get("review_required", "No")
        ])

    # --------------------------------------------------------
    # SHEET 2: OCR Candidates
    # --------------------------------------------------------
    ws2 = wb.create_sheet(title="OCR Candidates")
    sheet2_headers = [
        "Image Name",
        "Plate Crop",
        "Candidate Text",
        "Normalized Text",
        "OCR Confidence",
        "Candidate Score",
        "Final Score",
        "Preprocessing Variant",
        "Detection Count",
        "Line",
        "Geometry/Position Information",
        "Peripheral Penalty",
        "Structure Score",
        "Repetition Penalty",
        "Selected"
    ]
    ws2.append(sheet2_headers)

    for cand_row in all_evaluated_candidates:
        ws2.append([
            cand_row.get("image_name", ""),
            cand_row.get("plate_image", ""),
            cand_row.get("candidate_text", ""),
            cand_row.get("candidate_text", ""),
            cand_row.get("ocr_confidence", 0.0),
            cand_row.get("final_score", 0.0),
            cand_row.get("final_score", 0.0),
            cand_row.get("variant", ""),
            cand_row.get("detections_count", 1),
            cand_row.get("line", "1"),
            cand_row.get("bbox", ""),
            cand_row.get("peripheral_penalty", 0.0),
            cand_row.get("structure_score", 0.0),
            cand_row.get("repetition_penalty", 0.0),
            "Yes" if cand_row.get("selected", False) else "No"
        ])

    # --------------------------------------------------------
    # SHEET 3: Summary
    # --------------------------------------------------------
    ws3 = wb.create_sheet(title="Summary")
    ws3.append(["Metric", "Count / Value", "Percentage / Note"])

    ws3.append(["Total Plates Processed", summary_stats["total"], "100.0%"])
    ws3.append(["GOOD Status", summary_stats["good"], f"{summary_stats['good_pct']:.1f}%"])
    ws3.append(["REVIEW Status", summary_stats["review"], f"{summary_stats['review_pct']:.1f}%"])
    ws3.append(["NO_TEXT Status", summary_stats["no_text"], f"{summary_stats['no_text_pct']:.1f}%"])
    ws3.append(["Indian-Format Matches", summary_stats["indian_matches"], f"{summary_stats['indian_matches'] / max(1, summary_stats['total']) * 100:.1f}%"])
    ws3.append(["Non-Indian / General Format Matches", summary_stats["non_indian_matches"], f"{summary_stats['non_indian_matches'] / max(1, summary_stats['total']) * 100:.1f}%"])
    ws3.append(["Number Requiring Manual Review", summary_stats["review_required_count"], f"{summary_stats['review_required_count'] / max(1, summary_stats['total']) * 100:.1f}%"])
    ws3.append(["Total Evaluated Candidates Logged", summary_stats["candidate_count"], "Diagnostic records across variants"])
    ws3.append([])
    ws3.append(["Note", "Final Score is the system's composite confidence/quality metric combining raw OCR model confidence, spatial bounding-box geometry, structural plate formatting, and penalty factors. It is NOT the same as ground-truth accuracy.", ""])

    # --------------------------------------------------------
    # APPLY STYLING ACROSS SHEETS
    # --------------------------------------------------------
    for ws in [ws1, ws2, ws3]:
        # Style header row
        for cell in ws[1]:
            cell.font = header_font
            cell.fill = header_fill
            cell.alignment = Alignment(horizontal="center", vertical="center")

        # Auto-adjust column widths
        for col in ws.columns:
            max_len = 0
            col_letter = get_column_letter(col[0].column)
            for cell in col:
                cell.border = thin_border
                val_str = str(cell.value or "")
                if len(val_str) > max_len:
                    max_len = len(val_str)
            ws.column_dimensions[col_letter].width = max(12, min(45, max_len + 3))

    EXCEL_FILE.parent.mkdir(parents=True, exist_ok=True)
    wb.save(str(EXCEL_FILE))
    print(f"Excel workbook successfully generated: {EXCEL_FILE}")


# ============================================================
# MAIN PIPELINE EXECUTION
# ============================================================

def process_all_plates():
    """
    Processes all 207 plate crops:
    1. Loads metadata to map crop to original image.
    2. Runs staged OCR passes (enhanced -> original -> threshold).
    3. Reconstructs complete candidate plate numbers from adjacent detections.
    4. Applies position-aware Indian plate correction.
    5. Saves outputs/ocr_results.csv, outputs/ocr_candidates.csv, and outputs/ANPR_results.xlsx.
    """
    reader = init_reader()

    # Load image metadata mapping
    crop_to_image_map = {}
    if METADATA_FILE.exists():
        try:
            meta_df = pd.read_csv(METADATA_FILE)
            for _, row in meta_df.iterrows():
                crop_to_image_map[str(row["crop_filename"])] = str(row["original_image"])
        except Exception as e:
            print(f"Warning: Could not read metadata: {e}")

    plate_files = sorted(ENHANCED_DIR.glob("*.jpg"))
    total_count = len(plate_files)

    print(f"Total plate crops found: {total_count}")
    print("Beginning complete registration reconstruction...\n")

    final_results = []
    all_evaluated_candidates = []

    good_count = 0
    review_count = 0
    no_text_count = 0
    indian_matches = 0

    for i, enhanced_path in enumerate(plate_files, 1):
        filename = enhanced_path.name
        orig_image_name = crop_to_image_map.get(filename, filename)
        print(f"[{i:03d}/{total_count:03d}] {filename}")

        crop_candidates_by_variant = {}

        # ----------------------------------------------------
        # PASS 1: ENHANCED (Primary)
        # ----------------------------------------------------
        pass1 = run_ocr_on_file(reader, enhanced_path)
        for c in pass1["candidates"]:
            c_copy = dict(c)
            c_copy["variant"] = "enhanced"
            c_copy["image_name"] = orig_image_name
            crop_candidates_by_variant.setdefault("enhanced", []).append(c_copy)

        best_cand = pass1["candidates"][0] if pass1["candidates"] else None
        winning_variant = "enhanced"
        winning_detections = pass1["detections"]
        winning_raw_text = pass1["raw_text"]
        winning_line_count = pass1["line_count"]

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
                    c_copy["image_name"] = orig_image_name
                    crop_candidates_by_variant.setdefault("original", []).append(c_copy)

                cand2 = pass2["candidates"][0] if pass2["candidates"] else None
                if cand2 and (best_cand is None or cand2["final_score"] > best_cand["final_score"]):
                    best_cand = cand2
                    winning_variant = "original"
                    winning_detections = pass2["detections"]
                    winning_raw_text = pass2["raw_text"]
                    winning_line_count = pass2["line_count"]

            # Pass 3: threshold (if still needed)
            if should_retry(best_cand):
                threshold_path = THRESHOLD_DIR / filename
                if threshold_path.exists():
                    pass3 = run_ocr_on_file(reader, threshold_path)
                    for c in pass3["candidates"]:
                        c_copy = dict(c)
                        c_copy["variant"] = "threshold"
                        c_copy["image_name"] = orig_image_name
                        crop_candidates_by_variant.setdefault("threshold", []).append(c_copy)

                    cand3 = pass3["candidates"][0] if pass3["candidates"] else None
                    if cand3 and (best_cand is None or cand3["final_score"] > best_cand["final_score"]):
                        best_cand = cand3
                        winning_variant = "threshold"
                        winning_detections = pass3["detections"]
                        winning_raw_text = pass3["raw_text"]
                        winning_line_count = pass3["line_count"]

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
                    "image_name": orig_image_name,
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
                    "detections_count": c.get("detections_count", 1),
                    "line": c.get("line", "1"),
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

        is_ind = is_indian_pattern(detected_text)
        state_cd = detected_text[:2] if (is_ind and detected_text[:2] in INDIAN_STATE_CODES) else ""
        if is_ind:
            indian_matches += 1

        final_results.append({
            "image_name": orig_image_name,
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
            "peripheral_penalty": per_pen,
            "line_count": winning_line_count,
            "indian_format_match": "Yes" if is_ind else "No",
            "state_code": state_cd,
            "review_required": "Yes" if status != "GOOD" else "No"
        })

        print(
            f"   -> Text: {detected_text:14s} | "
            f"Score: {final_sc:.3f} | "
            f"Conf: {ocr_conf:.3f} | "
            f"Status: {status:7s} | "
            f"Indian: {'Yes (' + state_cd + ')' if is_ind else 'No'} | "
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
        writer = csv.DictWriter(f, fieldnames=result_fieldnames, extrasaction="ignore")
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
        writer = csv.DictWriter(f, fieldnames=candidate_fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(all_evaluated_candidates)

    # --------------------------------------------------------
    # WRITE EXCEL WORKBOOK
    # --------------------------------------------------------
    summary_stats = {
        "total": total_count,
        "good": good_count,
        "good_pct": (good_count / max(1, total_count)) * 100,
        "review": review_count,
        "review_pct": (review_count / max(1, total_count)) * 100,
        "no_text": no_text_count,
        "no_text_pct": (no_text_count / max(1, total_count)) * 100,
        "indian_matches": indian_matches,
        "non_indian_matches": total_count - indian_matches - no_text_count,
        "review_required_count": review_count + no_text_count,
        "candidate_count": len(all_evaluated_candidates)
    }

    generate_excel_workbook(final_results, all_evaluated_candidates, summary_stats)

    # --------------------------------------------------------
    # SUMMARY
    # --------------------------------------------------------
    print("\n" + "=" * 60)
    print("OCR PROCESSING COMPLETE")
    print("=" * 60)
    print(f"Total processed     : {total_count}")
    print(f"GOOD                : {good_count} ({summary_stats['good_pct']:.1f}%)")
    print(f"REVIEW              : {review_count} ({summary_stats['review_pct']:.1f}%)")
    print(f"NO_TEXT             : {no_text_count} ({summary_stats['no_text_pct']:.1f}%)")
    print(f"Indian format match : {indian_matches}")
    print(f"Non-Indian / General: {summary_stats['non_indian_matches']}")
    print(f"Candidates logged   : {len(all_evaluated_candidates)}")
    print(f"Final results CSV   : {RESULTS_FILE}")
    print(f"Candidates CSV      : {CANDIDATES_FILE}")
    print(f"Excel workbook      : {EXCEL_FILE}")
    print("=" * 60 + "\n")


if __name__ == "__main__":
    process_all_plates()