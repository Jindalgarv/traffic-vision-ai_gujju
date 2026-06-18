"""Basic pipeline checks for TrafficVision AI.

Run from the project root:
    python scratch/test_pipeline.py
"""

from __future__ import annotations

import sys
from pathlib import Path

from PIL import Image, ImageDraw

# Add project root to Python path when running from scratch/.
PROJECT_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_DIR))

from utils.database import (  # noqa: E402
    delete_all_violations,
    get_all_violations,
    init_db,
    insert_violation,
    update_evidence_path,
)
from utils.evidence import draw_annotated_image, save_evidence_crop  # noqa: E402
from utils.ocr import read_license_plate  # noqa: E402
from utils.rules import evaluate_violations  # noqa: E402


def make_dummy_image() -> Image.Image:
    """Create a simple fake traffic scene for local tests."""

    image = Image.new("RGB", (640, 420), "white")
    draw = ImageDraw.Draw(image)
    draw.rectangle((220, 250, 410, 360), outline="black", width=4)  # car
    draw.rectangle((470, 210, 515, 365), outline="black", width=4)  # person
    return image


def main() -> None:
    image = make_dummy_image()
    test_db = PROJECT_DIR / "scratch" / "test_traffic_violations.db"
    evidence_dir = PROJECT_DIR / "scratch" / "test_evidence"
    evidence_dir.mkdir(exist_ok=True)

    detections = [
        {
            "bbox": [220, 250, 410, 360],
            "confidence": 0.92,
            "class_id": 2,
            "class_name": "car",
        },
        {
            "bbox": [470, 210, 515, 365],
            "confidence": 0.88,
            "class_id": 0,
            "class_name": "person",
        },
    ]

    # OCR fallback test.
    plate = read_license_plate(image, detections[0]["bbox"], "car")
    assert len(plate) >= 4, "OCR fallback did not produce a plate"

    # Rules test.
    violations = evaluate_violations(
        image_size=image.size,
        detections=detections,
        speed_limit_mph=35,
        traffic_light_status="RED",
        stop_line_ratio=0.60,
        manual_speed_mph=55,
    )
    assert violations, "Expected at least one violation"

    # Evidence crop test.
    evidence_path = save_evidence_crop(image, detections[0]["bbox"], "test_car", evidence_dir)
    assert Path(evidence_path).exists(), "Evidence crop was not saved"

    # Annotation test.
    annotated = draw_annotated_image(image, detections, violations)
    annotated_path = evidence_dir / "annotated_test.jpg"
    annotated.save(annotated_path)
    assert annotated_path.exists(), "Annotated image was not saved"

    # Database test.
    init_db(test_db)
    delete_all_violations(test_db)
    row_id = insert_violation(
        vehicle_type="car",
        violation_type="Speeding",
        license_plate=plate,
        confidence=0.92,
        speed_mph=55,
        details="Unit test insert",
        original_image_path="dummy.jpg",
        evidence_path="",
        db_path=test_db,
    )
    update_evidence_path(row_id, evidence_path, test_db)
    df = get_all_violations(test_db)
    assert len(df) == 1, "Database insert/query failed"

    print("All TrafficVision AI pipeline checks passed.")
    print(f"Mock/fallback license plate: {plate}")
    print(f"Test evidence saved at: {evidence_path}")

    # Detector model loading is intentionally not required in this test because
    # it may download yolov8n.pt and needs internet the first time.
    model_path = PROJECT_DIR / "models" / "best.pt"
    if model_path.exists():
        print(f"Detector model found at: {model_path}")
    else:
        print("Detector model not found yet. Add models/best.pt or run app with internet once.")


if __name__ == "__main__":
    main()
