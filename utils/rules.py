"""Helmet-only traffic violation rule engine for TrafficVision AI.

This version intentionally creates a violation ONLY when both of these are true:
1. a motorcycle/motobike is detected
2. a no_helmet detection is near that motorcycle/motobike

It does NOT flag red-light, stop-line, speeding, or pedestrian rules. This avoids
false violations when an image only contains a motorcycle crossing a drawn virtual
line but no no_helmet object is detected.
"""

from __future__ import annotations

import math
from typing import Any, Dict, Iterable, List, Tuple


# Roboflow model class names may vary. These helpers normalize the common names.
MOTORCYCLE_NAMES = {
    "motorcycle",
    "motorbike",
    "motobike",
    "motor_bike",
    "motor_cycle",
    "bike",
}

NO_HELMET_NAMES = {
    "no_helmet",
    "no helmet",
    "no-helmet",
    "without_helmet",
    "without helmet",
    "helmet_missing",
    "helmet missing",
}

RIDER_NAMES = {"rider", "person", "pedestrian"}


def normalize_class_name(class_name: str) -> str:
    """Make class names easier to compare."""

    name = str(class_name or "").strip().lower()
    name = name.replace("-", "_")
    return name


def is_motorcycle(class_name: str) -> bool:
    """Return True for motorcycle/motobike classes."""

    name = normalize_class_name(class_name)
    readable = name.replace("_", " ")
    return name in MOTORCYCLE_NAMES or readable in MOTORCYCLE_NAMES


def is_no_helmet(class_name: str) -> bool:
    """Return True for no-helmet classes."""

    name = normalize_class_name(class_name)
    readable = name.replace("_", " ")
    return name in NO_HELMET_NAMES or readable in NO_HELMET_NAMES


def is_rider(class_name: str) -> bool:
    """Return True for rider/person classes. Kept for future rules."""

    name = normalize_class_name(class_name)
    return name in RIDER_NAMES


def bbox_center(bbox: Iterable[float]) -> Tuple[float, float]:
    """Return center point of bbox [x1, y1, x2, y2]."""

    x1, y1, x2, y2 = [float(v) for v in bbox]
    return (x1 + x2) / 2.0, (y1 + y2) / 2.0


def bbox_size(bbox: Iterable[float]) -> Tuple[float, float]:
    """Return width and height of bbox."""

    x1, y1, x2, y2 = [float(v) for v in bbox]
    return max(1.0, x2 - x1), max(1.0, y2 - y1)


def point_inside_box(point: Tuple[float, float], box: Iterable[float]) -> bool:
    """Check whether a point is inside a bbox."""

    px, py = point
    x1, y1, x2, y2 = [float(v) for v in box]
    return x1 <= px <= x2 and y1 <= py <= y2


def distance_between_boxes(box_a: Iterable[float], box_b: Iterable[float]) -> float:
    """Distance between two bbox centers."""

    ax, ay = bbox_center(box_a)
    bx, by = bbox_center(box_b)
    return math.sqrt((ax - bx) ** 2 + (ay - by) ** 2)


def expand_motorcycle_box_for_head_area(motorcycle_bbox: Iterable[float]) -> List[float]:
    """Expand a motorcycle bbox upward and sideways to cover the rider head area.

    In many helmet datasets, the motobike box covers the bike/body while the
    no_helmet box is near the rider's head above the bike. Direct overlap is
    therefore too strict. This expanded box makes the association practical.
    """

    x1, y1, x2, y2 = [float(v) for v in motorcycle_bbox]
    width, height = bbox_size(motorcycle_bbox)

    return [
        x1 - 0.35 * width,   # expand left
        y1 - 0.85 * height,  # expand upward for rider head
        x2 + 0.35 * width,   # expand right
        y2 + 0.10 * height,  # small lower tolerance
    ]


def no_helmet_belongs_to_motorcycle(no_helmet_bbox: Iterable[float], motorcycle_bbox: Iterable[float]) -> bool:
    """Return True when no_helmet appears related to the motorcycle.

    The check is intentionally simple:
    - if the center of no_helmet is inside the expanded motorcycle/head area, yes
    - otherwise, if the boxes are close relative to motorcycle size, yes
    """

    expanded_box = expand_motorcycle_box_for_head_area(motorcycle_bbox)
    no_helmet_center = bbox_center(no_helmet_bbox)

    if point_inside_box(no_helmet_center, expanded_box):
        return True

    motorcycle_width, motorcycle_height = bbox_size(motorcycle_bbox)
    allowed_distance = max(motorcycle_width, motorcycle_height) * 1.10
    actual_distance = distance_between_boxes(no_helmet_bbox, motorcycle_bbox)

    return actual_distance <= allowed_distance


def average_confidence(*detections: Dict[str, Any]) -> float:
    """Average confidence for related detections."""

    if not detections:
        return 0.0

    values = [float(det.get("confidence", 0.0)) for det in detections]
    return round(sum(values) / len(values), 4)


def evaluate_violations(
    image_size: Tuple[int, int],
    detections: List[Dict[str, Any]],
    speed_limit_mph: int = 35,
    traffic_light_status: str = "RED",
    stop_line_ratio: float = 0.60,
    pedestrian_distance_ratio: float = 0.12,
    manual_speed_mph: int | None = None,
) -> List[Dict[str, Any]]:
    """Return helmet non-compliance violations only.

    The extra parameters are kept so app.py does not need major changes, but
    they are intentionally ignored in this helmet-only version.
    """

    violations: List[Dict[str, Any]] = []

    motorcycles = [
        (idx, det)
        for idx, det in enumerate(detections)
        if is_motorcycle(det.get("class_name", ""))
    ]

    no_helmets = [
        (idx, det)
        for idx, det in enumerate(detections)
        if is_no_helmet(det.get("class_name", ""))
    ]

    # No no_helmet detection means no violation. A motorcycle alone is legal here.
    if not motorcycles or not no_helmets:
        return violations

    used_no_helmet_indexes: set[int] = set()

    for motorcycle_idx, motorcycle in motorcycles:
        motorcycle_bbox = motorcycle.get("bbox", [0, 0, 0, 0])

        related_no_helmets = []
        for no_helmet_idx, no_helmet in no_helmets:
            if no_helmet_idx in used_no_helmet_indexes:
                continue

            no_helmet_bbox = no_helmet.get("bbox", [0, 0, 0, 0])
            if no_helmet_belongs_to_motorcycle(no_helmet_bbox, motorcycle_bbox):
                related_no_helmets.append((no_helmet_idx, no_helmet))

        for no_helmet_idx, no_helmet in related_no_helmets:
            used_no_helmet_indexes.add(no_helmet_idx)

            violations.append(
                {
                    "detection_index": motorcycle_idx,
                    "related_detection_index": no_helmet_idx,
                    "related_detection_indexes": [motorcycle_idx, no_helmet_idx],
                    "vehicle_type": motorcycle.get("class_name", "motobike"),
                    "violation_type": "Helmet Non-Compliance",
                    "confidence": average_confidence(motorcycle, no_helmet),
                    "bbox": motorcycle_bbox,
                    "speed_mph": None,
                    "details": (
                        "Helmet violation: a no_helmet detection is near the detected "
                        "motorcycle/motobike. Motorbike alone is not treated as a violation."
                    ),
                }
            )

    return violations


# Kept only because older app.py imports this function. It is no longer used for violations.
def simulated_speed_mph(detection: Dict[str, Any]) -> int:
    """Compatibility helper for older app.py versions."""

    return 0
