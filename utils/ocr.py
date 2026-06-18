"""License plate OCR helpers for TrafficVision AI.

This module does four things:
1. Finds detections whose class name means "license plate".
2. Crops the license plate bounding box from the uploaded image.
3. Enhances the crop using grayscale, resizing, denoising, and thresholding.
4. Runs EasyOCR or PaddleOCR and returns plate text + OCR confidence.

The OCR engines are optional. If EasyOCR/PaddleOCR are not installed, the app
uses a deterministic mock fallback so the Streamlit demo still runs.
"""

from __future__ import annotations

import hashlib
import random
import re
import string
from typing import Any, Dict, Iterable, List, Optional, Tuple

import numpy as np
from PIL import Image


VEHICLE_WORDS = ("car", "truck", "bus", "motorcycle", "motorbike", "vehicle", "auto")
PLATE_WORDS = (
    "license_plate",
    "licence_plate",
    "license plate",
    "licence plate",
    "number_plate",
    "number plate",
    "plate",
)

_EASYOCR_READER = None
_PADDLEOCR_READER = None


def normalize_class_name(class_name: str) -> str:
    """Normalize class names so different model labels still work."""

    return str(class_name or "").strip().lower().replace("-", "_")


def is_vehicle(class_name: str) -> bool:
    """Return True when a class name looks like a vehicle class."""

    lower_name = normalize_class_name(class_name).replace("_", " ")
    return any(word in lower_name for word in VEHICLE_WORDS)


def is_license_plate(class_name: str) -> bool:
    """Return True when a class name looks like a license plate class."""

    normalized = normalize_class_name(class_name)
    readable = normalized.replace("_", " ")
    return normalized in PLATE_WORDS or readable in PLATE_WORDS or "plate" in normalized


def crop_bbox(image: Image.Image, bbox: Iterable[float], padding: int = 4) -> Image.Image:
    """Crop a PIL image using a YOLO-style [x1, y1, x2, y2] bbox.

    A small padding is useful because plate detectors can be tight around text.
    """

    width, height = image.size
    x1, y1, x2, y2 = [int(round(float(v))) for v in bbox]

    x1 = max(0, min(x1 - padding, width - 1))
    y1 = max(0, min(y1 - padding, height - 1))
    x2 = max(x1 + 1, min(x2 + padding, width))
    y2 = max(y1 + 1, min(y2 + padding, height))

    return image.crop((x1, y1, x2, y2))


def enhance_plate_crop(plate_crop: Image.Image, resize_scale: int = 3) -> Image.Image:
    """Enhance a plate crop before OCR.

    Steps:
    - convert to grayscale
    - resize to make characters larger
    - denoise lightly
    - apply Otsu thresholding to create a high-contrast black/white image
    """

    try:
        import cv2

        rgb_array = np.array(plate_crop.convert("RGB"))
        gray = cv2.cvtColor(rgb_array, cv2.COLOR_RGB2GRAY)

        resized = cv2.resize(
            gray,
            None,
            fx=resize_scale,
            fy=resize_scale,
            interpolation=cv2.INTER_CUBIC,
        )

        denoised = cv2.bilateralFilter(resized, d=7, sigmaColor=60, sigmaSpace=60)
        _, thresholded = cv2.threshold(
            denoised,
            0,
            255,
            cv2.THRESH_BINARY + cv2.THRESH_OTSU,
        )

        return Image.fromarray(thresholded)

    except Exception:
        # Fallback if OpenCV is not available for any reason.
        gray = plate_crop.convert("L")
        new_size = (gray.width * resize_scale, gray.height * resize_scale)
        resized = gray.resize(new_size)
        thresholded = resized.point(lambda pixel: 255 if pixel > 140 else 0)
        return thresholded


def clean_plate_text(text: str) -> str:
    """Clean OCR output into a plate-like string."""

    text = str(text or "").upper()
    text = re.sub(r"[^A-Z0-9]", "", text)
    return text.strip()


def _mock_plate_from_bbox(bbox: Iterable[float]) -> Tuple[str, float]:
    """Generate a stable demo plate when OCR libraries are unavailable."""

    raw_text = ",".join(str(round(float(v), 2)) for v in bbox)
    digest = hashlib.md5(raw_text.encode("utf-8")).hexdigest()
    seed = int(digest[:12], 16)
    rng = random.Random(seed)

    letters_1 = "".join(rng.choices(string.ascii_uppercase, k=2))
    digits_1 = "".join(rng.choices(string.digits, k=2))
    letters_2 = "".join(rng.choices(string.ascii_uppercase, k=2))
    digits_2 = "".join(rng.choices(string.digits, k=4))

    # Indian-style demo format, for example: MH12AB1234
    return f"{letters_1}{digits_1}{letters_2}{digits_2}", 0.0


def _get_easyocr_reader(gpu: bool = False) -> Any:
    """Load EasyOCR once and reuse the reader."""

    global _EASYOCR_READER
    if _EASYOCR_READER is None:
        import easyocr

        _EASYOCR_READER = easyocr.Reader(["en"], gpu=gpu)
    return _EASYOCR_READER


def _get_paddleocr_reader() -> Any:
    """Load PaddleOCR once and reuse the reader.

    Different PaddleOCR versions expose slightly different constructor options,
    so we try a current/simple setup first and then fall back to older options.
    """

    global _PADDLEOCR_READER
    if _PADDLEOCR_READER is None:
        from paddleocr import PaddleOCR

        try:
            _PADDLEOCR_READER = PaddleOCR(lang="en")
        except TypeError:
            _PADDLEOCR_READER = PaddleOCR(use_angle_cls=True, lang="en")
    return _PADDLEOCR_READER


def _easyocr_read(enhanced_crop: Image.Image) -> Tuple[str, float]:
    """Run EasyOCR and return the best text/confidence pair."""

    reader = _get_easyocr_reader(gpu=False)
    crop_array = np.array(enhanced_crop)
    results = reader.readtext(crop_array)

    best_text = ""
    best_confidence = 0.0

    for item in results:
        if len(item) < 3:
            continue

        text = clean_plate_text(item[1])
        confidence = float(item[2])

        if len(text) >= 4 and confidence > best_confidence:
            best_text = text
            best_confidence = confidence

    return best_text, best_confidence


def _extract_text_score_pairs_from_paddle(value: Any) -> List[Tuple[str, float]]:
    """Extract text/confidence pairs from several PaddleOCR output shapes."""

    pairs: List[Tuple[str, float]] = []

    if value is None:
        return pairs

    if isinstance(value, dict):
        # Newer PaddleOCR result dictionaries may contain parallel lists.
        texts = value.get("rec_texts") or value.get("texts") or []
        scores = value.get("rec_scores") or value.get("scores") or []
        for text, score in zip(texts, scores):
            pairs.append((str(text), float(score)))

        # Also recursively inspect nested values.
        for nested_value in value.values():
            pairs.extend(_extract_text_score_pairs_from_paddle(nested_value))
        return pairs

    if isinstance(value, (list, tuple)):
        # Classic PaddleOCR shape often contains: [box, (text, confidence)]
        if (
            len(value) >= 2
            and isinstance(value[1], (list, tuple))
            and len(value[1]) >= 2
            and isinstance(value[1][0], str)
        ):
            pairs.append((str(value[1][0]), float(value[1][1])))
            return pairs

        # Some outputs can be directly shaped as: (text, confidence)
        if len(value) >= 2 and isinstance(value[0], str) and isinstance(value[1], (int, float)):
            pairs.append((str(value[0]), float(value[1])))
            return pairs

        for item in value:
            pairs.extend(_extract_text_score_pairs_from_paddle(item))

    return pairs


def _paddleocr_read(enhanced_crop: Image.Image) -> Tuple[str, float]:
    """Run PaddleOCR and return the best text/confidence pair."""

    reader = _get_paddleocr_reader()
    crop_array = np.array(enhanced_crop.convert("RGB"))

    try:
        raw_result = reader.ocr(crop_array, cls=True)
    except Exception:
        raw_result = reader.predict(crop_array)

    pairs = _extract_text_score_pairs_from_paddle(raw_result)

    best_text = ""
    best_confidence = 0.0
    for text, confidence in pairs:
        cleaned = clean_plate_text(text)
        if len(cleaned) >= 4 and confidence > best_confidence:
            best_text = cleaned
            best_confidence = float(confidence)

    return best_text, best_confidence


def recognize_license_plate(
    image: Image.Image,
    bbox: Iterable[float],
    ocr_engine: str = "auto",
) -> Dict[str, Any]:
    """Crop, enhance, and OCR one license plate detection.

    Returns a dictionary containing the OCR text, OCR confidence, selected engine,
    original crop, and enhanced crop. The Streamlit UI uses these values so a
    human can review/edit the plate before saving.
    """

    bbox_list = [float(v) for v in bbox]
    plate_crop = crop_bbox(image, bbox_list, padding=4)
    enhanced_crop = enhance_plate_crop(plate_crop)

    selected_engine = str(ocr_engine or "auto").strip().lower()
    if selected_engine in {"mock", "fallback", "demo"}:
        plate_text, confidence = _mock_plate_from_bbox(bbox_list)
        return {
            "plate_text": plate_text,
            "ocr_confidence": confidence,
            "ocr_engine": "mock_fallback",
            "bbox": bbox_list,
            "plate_crop": plate_crop,
            "enhanced_crop": enhanced_crop,
            "error": "Mock OCR selected.",
        }

    engines_to_try: List[str]
    if selected_engine == "easyocr":
        engines_to_try = ["easyocr"]
    elif selected_engine == "paddleocr":
        engines_to_try = ["paddleocr"]
    else:
        engines_to_try = ["easyocr", "paddleocr"]

    last_error = ""
    for engine in engines_to_try:
        try:
            if engine == "easyocr":
                plate_text, confidence = _easyocr_read(enhanced_crop)
            else:
                plate_text, confidence = _paddleocr_read(enhanced_crop)

            if plate_text:
                return {
                    "plate_text": plate_text,
                    "ocr_confidence": round(float(confidence), 4),
                    "ocr_engine": engine,
                    "bbox": bbox_list,
                    "plate_crop": plate_crop,
                    "enhanced_crop": enhanced_crop,
                    "error": "",
                }

            last_error = f"{engine} ran but did not return readable plate text."
        except Exception as exc:  # OCR dependencies may not be installed.
            last_error = f"{engine} failed: {exc}"

    plate_text, confidence = _mock_plate_from_bbox(bbox_list)
    return {
        "plate_text": plate_text,
        "ocr_confidence": confidence,
        "ocr_engine": "mock_fallback",
        "bbox": bbox_list,
        "plate_crop": plate_crop,
        "enhanced_crop": enhanced_crop,
        "error": last_error,
    }


def extract_license_plate_ocr(
    image: Image.Image,
    detections: List[Dict[str, Any]],
    ocr_engine: str = "auto",
) -> List[Dict[str, Any]]:
    """Run OCR for every license_plate detection."""

    results: List[Dict[str, Any]] = []

    for index, detection in enumerate(detections):
        class_name = detection.get("class_name", "")
        if not is_license_plate(class_name):
            continue

        bbox = detection.get("bbox", [0, 0, 0, 0])
        ocr_result = recognize_license_plate(
            image=image,
            bbox=bbox,
            ocr_engine=ocr_engine,
        )

        ocr_result.update(
            {
                "detection_index": index,
                "detection_confidence": float(detection.get("confidence", 0.0)),
                "class_name": class_name,
            }
        )
        results.append(ocr_result)

    return results


def read_license_plate(image: Image.Image, bbox: List[float], class_name: str = "vehicle") -> str:
    """Backward-compatible helper used by older app code.

    New code should prefer recognize_license_plate() or extract_license_plate_ocr()
    because they return confidence and crop images too.
    """

    result = recognize_license_plate(image=image, bbox=bbox, ocr_engine="auto")
    return str(result.get("plate_text", "")) or "N/A"
