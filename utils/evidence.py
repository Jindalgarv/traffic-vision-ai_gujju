"""Evidence image utilities for TrafficVision AI.

This module saves per-violation annotated evidence images and optional crops.
"""

from __future__ import annotations

from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Iterable, List, Set

from PIL import Image, ImageDraw, ImageFont


CLASS_COLORS = {
    "motobike": "deepskyblue",
    "motorbike": "deepskyblue",
    "motorcycle": "deepskyblue",
    "bike": "deepskyblue",
    "car": "lime",
    "truck": "lime",
    "bus": "lime",
    "vehicle": "lime",
    "rider": "orange",
    "person": "orange",
    "no_helmet": "red",
    "no helmet": "red",
    "license_plate": "yellow",
    "licence_plate": "yellow",
    "number_plate": "yellow",
    "plate": "yellow",
}


def _safe_font(size: int = 14) -> ImageFont.ImageFont:
    """Load a default font without requiring external font files."""

    for font_name in ("arial.ttf", "DejaVuSans.ttf"):
        try:
            return ImageFont.truetype(font_name, size)
        except Exception:
            pass
    return ImageFont.load_default()


def normalize_name(class_name: str) -> str:
    """Normalize class names for color matching."""

    return str(class_name or "").strip().lower().replace("-", "_")


def get_box_color(class_name: str, is_violation: bool) -> str:
    """Choose a color for a detection box."""

    if is_violation:
        return "red"

    name = normalize_name(class_name)
    readable_name = name.replace("_", " ")
    return CLASS_COLORS.get(name, CLASS_COLORS.get(readable_name, "lime"))


def crop_region(image: Image.Image, bbox: Iterable[float], padding: int = 10) -> Image.Image:
    """Crop a bbox region from an image with optional padding."""

    width, height = image.size
    x1, y1, x2, y2 = [int(round(float(v))) for v in bbox]

    x1 = max(0, x1 - padding)
    y1 = max(0, y1 - padding)
    x2 = min(width, x2 + padding)
    y2 = min(height, y2 + padding)

    if x2 <= x1:
        x2 = min(width, x1 + 1)
    if y2 <= y1:
        y2 = min(height, y1 + 1)

    return image.crop((x1, y1, x2, y2))


def save_evidence_crop(
    image: Image.Image,
    bbox: Iterable[float],
    violation_id: int | str,
    evidence_dir: str | Path = "data/crops",
) -> str:
    """Save cropped evidence as data/crops/{violation_id}.jpg and return path."""

    evidence_dir = Path(evidence_dir)
    evidence_dir.mkdir(parents=True, exist_ok=True)

    crop = crop_region(image, bbox)
    evidence_path = evidence_dir / f"{violation_id}.jpg"
    crop.convert("RGB").save(evidence_path, quality=92)
    return str(evidence_path)


def collect_violating_indexes(violations: List[Dict[str, Any]]) -> Set[int]:
    """Collect all detection indexes that should be highlighted as violating."""

    indexes: Set[int] = set()

    for violation in violations:
        if "detection_index" in violation and violation.get("detection_index") is not None:
            indexes.add(int(violation["detection_index"]))

        if "related_detection_index" in violation and violation.get("related_detection_index") is not None:
            indexes.add(int(violation["related_detection_index"]))

        for idx in violation.get("related_detection_indexes", []):
            indexes.add(int(idx))

        if "plate_detection_index" in violation and violation.get("plate_detection_index") is not None:
            indexes.add(int(violation["plate_detection_index"]))

    return indexes


def draw_text_with_background(
    draw: ImageDraw.ImageDraw,
    xy: tuple[int, int],
    text: str,
    font: ImageFont.ImageFont,
    fill: str = "black",
    background: str = "yellow",
    padding: int = 4,
) -> None:
    """Draw readable text with a solid background."""

    x, y = xy
    bbox = draw.textbbox((x, y), text, font=font)
    width = bbox[2] - bbox[0]
    height = bbox[3] - bbox[1]
    draw.rectangle([(x, y), (x + width + padding * 2, y + height + padding * 2)], fill=background)
    draw.text((x + padding, y + padding), text, fill=fill, font=font)


def draw_annotated_image(
    image: Image.Image,
    detections: List[Dict[str, Any]],
    violations: List[Dict[str, Any]],
    stop_line_ratio: float = 0.60,
    traffic_light_status: str = "RED",
    show_stop_line: bool = False,
) -> Image.Image:
    """Draw bounding boxes, labels, and violation flags."""

    annotated = image.convert("RGB").copy()
    draw = ImageDraw.Draw(annotated)
    font = _safe_font(14)
    width, height = annotated.size

    violating_indexes = collect_violating_indexes(violations)

    # Optional only. Default is False because helmet-only detection should not
    # visually imply red-light/stop-line enforcement.
    if show_stop_line:
        stop_line_y = int(height * stop_line_ratio)
        line_color = "red" if traffic_light_status.upper() == "RED" else "orange"
        draw.line([(0, stop_line_y), (width, stop_line_y)], fill=line_color, width=4)
        draw.text((10, max(0, stop_line_y - 22)), "Virtual stop line", fill=line_color, font=font)

    for idx, detection in enumerate(detections):
        bbox = detection.get("bbox", [0, 0, 0, 0])
        x1, y1, x2, y2 = [int(round(float(v))) for v in bbox]
        class_name = detection.get("class_name", "object")
        confidence = float(detection.get("confidence", 0.0))

        is_violation = idx in violating_indexes
        box_color = get_box_color(class_name, is_violation)
        box_width = 4 if is_violation else 3

        label = f"{idx}: {class_name} {confidence:.2f}"
        if is_violation:
            label += " | VIOLATION"

        draw.rectangle([(x1, y1), (x2, y2)], outline=box_color, width=box_width)

        text_bbox = draw.textbbox((x1, y1), label, font=font)
        text_width = text_bbox[2] - text_bbox[0]
        text_height = text_bbox[3] - text_bbox[1]
        text_y1 = max(0, y1 - text_height - 8)

        draw.rectangle(
            [(x1, text_y1), (min(width, x1 + text_width + 8), text_y1 + text_height + 8)],
            fill=box_color,
        )
        draw.text((x1 + 4, text_y1 + 4), label, fill="black", font=font)

    return annotated


def save_annotated_evidence_image(
    image: Image.Image,
    detections: List[Dict[str, Any]],
    violation: Dict[str, Any],
    violation_id: str,
    annotated_dir: str | Path = "data/annotated",
) -> str:
    """Save one full annotated evidence image for a single violation.

    The saved image highlights the detection(s) involved in that violation and
    writes the violation ID, type, plate, confidence, and timestamp on the image.
    """

    annotated_dir = Path(annotated_dir)
    annotated_dir.mkdir(parents=True, exist_ok=True)

    annotated = draw_annotated_image(
        image=image,
        detections=detections,
        violations=[violation],
    )

    draw = ImageDraw.Draw(annotated)
    font = _safe_font(16)
    timestamp = datetime.now().isoformat(timespec="seconds")
    plate = violation.get("license_plate") or violation.get("ocr_plate_number") or "N/A"
    confidence = float(violation.get("confidence", 0.0))
    banner = (
        f"Violation ID: {violation_id} | "
        f"Type: {violation.get('violation_type', 'unknown')} | "
        f"Plate: {plate} | Confidence: {confidence:.2f} | {timestamp}"
    )
    draw_text_with_background(draw, (10, 10), banner, font=font, background="yellow")

    output_path = annotated_dir / f"{violation_id}.jpg"
    annotated.save(output_path, quality=92)
    return str(output_path)
