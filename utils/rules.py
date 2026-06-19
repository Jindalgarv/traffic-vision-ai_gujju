"""Multi-violation rule engine for IntelliTraffic AI.

Detects the following violations by analysing object-detection outputs:

    1. Helmet Non-Compliance
       Motorcycle detected + no_helmet box associated with it.

    2. Triple Riding
       Motorcycle detected + 3 or more person/rider boxes associated with it.

    3. Stop-Line Violation
       Vehicle crosses a virtual stop line when traffic signal is RED.

    4. Red-Light Violation
       Same logic as stop-line but explicitly named for signal state.

    5. Speeding
       Detected vehicle speed (manual override or simulated) exceeds the
       configured speed limit.

Violations are grouped per vehicle and returned as a flat list that is
compatible with the existing evidence/database pipeline in app.py.
"""

from __future__ import annotations

import math
from typing import Any, Dict, Iterable, List, Optional, Set, Tuple


# ---------------------------------------------------------------------------
# Class-name normalisation helpers
# ---------------------------------------------------------------------------

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

HELMET_NAMES = {
    "helmet",
    "with_helmet",
    "with helmet",
}

RIDER_NAMES = {
    "rider",
    "person",
    "pedestrian",
}

VEHICLE_NAMES = {
    "car",
    "truck",
    "bus",
    "auto",
    "auto_rickshaw",
    "auto rickshaw",
    "vehicle",
    *MOTORCYCLE_NAMES,
}


def normalize_class_name(class_name: str) -> str:
    name = str(class_name or "").strip().lower()
    return name.replace("-", "_")


def is_motorcycle(class_name: str) -> bool:
    name = normalize_class_name(class_name)
    return name in MOTORCYCLE_NAMES or name.replace("_", " ") in MOTORCYCLE_NAMES


def is_no_helmet(class_name: str) -> bool:
    name = normalize_class_name(class_name)
    return name in NO_HELMET_NAMES or name.replace("_", " ") in NO_HELMET_NAMES


def is_rider_or_person(class_name: str) -> bool:
    name = normalize_class_name(class_name)
    return name in RIDER_NAMES


def is_any_vehicle(class_name: str) -> bool:
    name = normalize_class_name(class_name)
    return name in VEHICLE_NAMES or name.replace("_", " ") in VEHICLE_NAMES


# ---------------------------------------------------------------------------
# Geometry helpers
# ---------------------------------------------------------------------------

def bbox_center(bbox: Iterable[float]) -> Tuple[float, float]:
    x1, y1, x2, y2 = [float(v) for v in bbox]
    return (x1 + x2) / 2.0, (y1 + y2) / 2.0


def bbox_size(bbox: Iterable[float]) -> Tuple[float, float]:
    x1, y1, x2, y2 = [float(v) for v in bbox]
    return max(1.0, x2 - x1), max(1.0, y2 - y1)


def point_inside_box(point: Tuple[float, float], box: Iterable[float]) -> bool:
    px, py = point
    x1, y1, x2, y2 = [float(v) for v in box]
    return x1 <= px <= x2 and y1 <= py <= y2


def distance_between_boxes(box_a: Iterable[float], box_b: Iterable[float]) -> float:
    ax, ay = bbox_center(box_a)
    bx, by = bbox_center(box_b)
    return math.sqrt((ax - bx) ** 2 + (ay - by) ** 2)


def expand_motorcycle_box_for_head_area(motorcycle_bbox: Iterable[float]) -> List[float]:
    """Expand motorcycle bbox upward to cover rider head area."""
    x1, y1, x2, y2 = [float(v) for v in motorcycle_bbox]
    width, height = bbox_size(motorcycle_bbox)
    return [
        x1 - 0.35 * width,
        y1 - 0.85 * height,
        x2 + 0.35 * width,
        y2 + 0.10 * height,
    ]


def no_helmet_belongs_to_motorcycle(
    no_helmet_bbox: Iterable[float],
    motorcycle_bbox: Iterable[float],
) -> bool:
    expanded = expand_motorcycle_box_for_head_area(motorcycle_bbox)
    if point_inside_box(bbox_center(no_helmet_bbox), expanded):
        return True
    w, h = bbox_size(motorcycle_bbox)
    allowed = max(w, h) * 1.10
    return distance_between_boxes(no_helmet_bbox, motorcycle_bbox) <= allowed


def person_belongs_to_motorcycle(
    person_bbox: Iterable[float],
    motorcycle_bbox: Iterable[float],
) -> bool:
    """Return True if a person/rider detection is associated with a motorcycle.

    Uses an expanded bounding box that covers the typical seat+rider envelope.
    """
    x1, y1, x2, y2 = [float(v) for v in motorcycle_bbox]
    width, height = bbox_size(motorcycle_bbox)
    expanded = [
        x1 - 0.50 * width,
        y1 - 1.10 * height,   # riders sit above the motorcycle
        x2 + 0.50 * width,
        y2 + 0.20 * height,
    ]
    center = bbox_center(person_bbox)
    if point_inside_box(center, expanded):
        return True
    w, h = bbox_size(motorcycle_bbox)
    allowed = max(w, h) * 1.30
    return distance_between_boxes(person_bbox, motorcycle_bbox) <= allowed


def vehicle_crosses_stop_line(
    vehicle_bbox: Iterable[float],
    image_height: int,
    stop_line_ratio: float,
) -> bool:
    """Return True when the vehicle's bottom edge is below the stop line."""
    _, _, _, y2 = [float(v) for v in vehicle_bbox]
    stop_line_y = image_height * stop_line_ratio
    return y2 >= stop_line_y


# ---------------------------------------------------------------------------
# Per-violation scoring
# ---------------------------------------------------------------------------

def average_confidence(*detections: Dict[str, Any]) -> float:
    if not detections:
        return 0.0
    values = [float(det.get("confidence", 0.0)) for det in detections]
    return round(sum(values) / len(values), 4)


# ---------------------------------------------------------------------------
# Violation severity weights (used for risk score later)
# ---------------------------------------------------------------------------

VIOLATION_SEVERITY: Dict[str, int] = {
    "Helmet Non-Compliance": 5,
    "Triple Riding": 7,
    "Stop-Line Violation": 6,
    "Red-Light Violation": 8,
    "Speeding": 6,
}


# ---------------------------------------------------------------------------
# Simulated speed helper (kept for backwards compatibility)
# ---------------------------------------------------------------------------

def simulated_speed_mph(detection: Dict[str, Any]) -> int:
    """Backwards-compatible stub. Always returns 0 for static images."""
    return 0


# ---------------------------------------------------------------------------
# Main evaluation function
# ---------------------------------------------------------------------------

def evaluate_violations(
    image_size: Tuple[int, int],
    detections: List[Dict[str, Any]],
    speed_limit_mph: int = 35,
    traffic_light_status: str = "RED",
    stop_line_ratio: float = 0.60,
    pedestrian_distance_ratio: float = 0.12,
    manual_speed_mph: Optional[int] = None,
) -> List[Dict[str, Any]]:
    """Detect and group all traffic violations from a list of detections.

    Violations are grouped by vehicle: one motorcycle can contribute multiple
    violations (e.g. helmet + triple riding + red-light) that will appear as
    separate violation records but share the same vehicle detection index.

    Args:
        image_size: (width, height) of the source image.
        detections: List of detection dicts with keys bbox, confidence, class_name.
        speed_limit_mph: Configured speed limit for the scene.
        traffic_light_status: "RED", "YELLOW", or "GREEN".
        stop_line_ratio: Fraction of image height where the stop line is drawn.
        pedestrian_distance_ratio: Unused; kept for API compatibility.
        manual_speed_mph: If set, overrides simulated speed for all vehicles.

    Returns:
        List of violation dicts, each containing at minimum:
            detection_index, vehicle_type, violation_type, confidence, bbox,
            details, related_detection_indexes, speed_mph.
    """
    image_width, image_height = image_size
    violations: List[Dict[str, Any]] = []
    signal_is_red = traffic_light_status.upper() in {"RED", "YELLOW"}

    # Partition detections by type
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
    persons = [
        (idx, det)
        for idx, det in enumerate(detections)
        if is_rider_or_person(det.get("class_name", ""))
    ]
    all_vehicles = [
        (idx, det)
        for idx, det in enumerate(detections)
        if is_any_vehicle(det.get("class_name", ""))
    ]

    used_no_helmet_indexes: Set[int] = set()

    # ------------------------------------------------------------------
    # 1. Helmet Non-Compliance + Triple Riding (motorcycle-centric)
    # ------------------------------------------------------------------
    for moto_idx, moto in motorcycles:
        moto_bbox = moto.get("bbox", [0, 0, 0, 0])
        moto_class = moto.get("class_name", "motorcycle")

        # --- Helmet Non-Compliance ---
        related_no_helmets = [
            (nh_idx, nh)
            for nh_idx, nh in no_helmets
            if nh_idx not in used_no_helmet_indexes
            and no_helmet_belongs_to_motorcycle(nh.get("bbox", [0, 0, 0, 0]), moto_bbox)
        ]
        for nh_idx, nh in related_no_helmets:
            used_no_helmet_indexes.add(nh_idx)
            violations.append(
                {
                    "detection_index": moto_idx,
                    "related_detection_index": nh_idx,
                    "related_detection_indexes": [moto_idx, nh_idx],
                    "vehicle_type": moto_class,
                    "violation_type": "Helmet Non-Compliance",
                    "confidence": average_confidence(moto, nh),
                    "severity": VIOLATION_SEVERITY["Helmet Non-Compliance"],
                    "bbox": moto_bbox,
                    "speed_mph": None,
                    "details": (
                        "Helmet violation: a no_helmet detection is near the detected "
                        "motorcycle/motobike. Motorbike alone is not treated as a violation."
                    ),
                }
            )

        # --- Triple Riding ---
        associated_persons = [
            (p_idx, p)
            for p_idx, p in persons
            if person_belongs_to_motorcycle(p.get("bbox", [0, 0, 0, 0]), moto_bbox)
        ]
        if len(associated_persons) >= 3:
            person_indexes = [p_idx for p_idx, _ in associated_persons]
            all_detections_for_violation = [moto] + [p for _, p in associated_persons]
            violations.append(
                {
                    "detection_index": moto_idx,
                    "related_detection_indexes": [moto_idx] + person_indexes,
                    "vehicle_type": moto_class,
                    "violation_type": "Triple Riding",
                    "confidence": average_confidence(*all_detections_for_violation),
                    "severity": VIOLATION_SEVERITY["Triple Riding"],
                    "bbox": moto_bbox,
                    "speed_mph": None,
                    "details": (
                        f"Triple riding: {len(associated_persons)} persons detected "
                        f"on a single motorcycle (legal limit is 2)."
                    ),
                }
            )

    # ------------------------------------------------------------------
    # 2. Stop-Line / Red-Light Violation (all vehicles)
    # ------------------------------------------------------------------
    if signal_is_red:
        for veh_idx, veh in all_vehicles:
            veh_bbox = veh.get("bbox", [0, 0, 0, 0])
            veh_class = veh.get("class_name", "vehicle")

            if vehicle_crosses_stop_line(veh_bbox, image_height, stop_line_ratio):
                vtype = (
                    "Red-Light Violation"
                    if traffic_light_status.upper() == "RED"
                    else "Stop-Line Violation"
                )
                violations.append(
                    {
                        "detection_index": veh_idx,
                        "related_detection_indexes": [veh_idx],
                        "vehicle_type": veh_class,
                        "violation_type": vtype,
                        "confidence": round(float(veh.get("confidence", 0.0)), 4),
                        "severity": VIOLATION_SEVERITY.get(vtype, 6),
                        "bbox": veh_bbox,
                        "speed_mph": None,
                        "details": (
                            f"Vehicle detected past the virtual stop line "
                            f"(position {stop_line_ratio:.0%} of image height) "
                            f"while signal is {traffic_light_status.upper()}."
                        ),
                    }
                )

    # ------------------------------------------------------------------
    # 3. Speeding (all vehicles)
    # ------------------------------------------------------------------
    for veh_idx, veh in all_vehicles:
        speed = manual_speed_mph if manual_speed_mph and manual_speed_mph > 0 else None
        if speed is not None and speed > speed_limit_mph:
            veh_bbox = veh.get("bbox", [0, 0, 0, 0])
            violations.append(
                {
                    "detection_index": veh_idx,
                    "related_detection_indexes": [veh_idx],
                    "vehicle_type": veh.get("class_name", "vehicle"),
                    "violation_type": "Speeding",
                    "confidence": round(float(veh.get("confidence", 0.0)), 4),
                    "severity": VIOLATION_SEVERITY["Speeding"],
                    "bbox": veh_bbox,
                    "speed_mph": speed,
                    "details": (
                        f"Vehicle speed ({speed} mph) exceeds speed limit "
                        f"({speed_limit_mph} mph)."
                    ),
                }
            )

    return violations
