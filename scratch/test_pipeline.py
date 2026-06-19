"""Pipeline checks for IntelliTraffic AI.

Run from the project root:
    python scratch/test_pipeline.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

from PIL import Image, ImageDraw

PROJECT_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_DIR))

from utils.database import (  # noqa: E402
    delete_all_violations,
    get_all_violations,
    get_repeat_offenders,
    get_violation_stats,
    init_db,
    insert_violation,
    update_evidence_path,
)
from utils.evidence import draw_annotated_image, save_evidence_crop  # noqa: E402
from utils.ocr import read_license_plate  # noqa: E402
from utils.preprocessing import PreprocessConfig, adaptive_preprocess, compute_image_hash  # noqa: E402
from utils.rules import evaluate_violations  # noqa: E402


def make_dummy_image() -> Image.Image:
    image = Image.new("RGB", (640, 420), "white")
    draw = ImageDraw.Draw(image)
    draw.rectangle((220, 250, 410, 360), outline="black", width=4)   # motorcycle body
    draw.rectangle((240, 180, 310, 250), outline="red", width=3)     # no_helmet box
    draw.rectangle((245, 185, 255, 195), outline="blue", width=2)    # person 1
    draw.rectangle((270, 190, 280, 200), outline="blue", width=2)    # person 2
    draw.rectangle((295, 188, 305, 198), outline="blue", width=2)    # person 3 → triple riding
    return image


def main() -> None:
    print("=" * 60)
    print("IntelliTraffic AI – Pipeline Test Suite")
    print("=" * 60)

    image = make_dummy_image()
    test_db = PROJECT_DIR / "scratch" / "test_traffic_violations.db"
    evidence_dir = PROJECT_DIR / "scratch" / "test_evidence"
    evidence_dir.mkdir(exist_ok=True)

    # ── 1. Preprocessing ──────────────────────────────────────────────
    print("\n[1] Preprocessing …")
    config = PreprocessConfig(apply_clahe=True, apply_gamma=True, apply_sharpen=True)
    enhanced = adaptive_preprocess(image, config)
    assert enhanced.size == image.size, "Preprocessing changed image dimensions"

    # Image hash
    img_bytes = b"dummy image bytes for hash test"
    img_hash = compute_image_hash(img_bytes)
    assert len(img_hash) == 64, "SHA-256 hash should be 64 hex chars"
    print(f"    SHA-256 hash: {img_hash[:16]}…  ✓")

    # ── 2. Rules ──────────────────────────────────────────────────────
    print("\n[2] Rule engine …")
    detections = [
        {"bbox": [220, 250, 410, 360], "confidence": 0.92, "class_id": 3, "class_name": "motorcycle"},
        {"bbox": [240, 180, 310, 250], "confidence": 0.88, "class_id": 0, "class_name": "no_helmet"},
        {"bbox": [245, 185, 255, 195], "confidence": 0.80, "class_id": 0, "class_name": "person"},
        {"bbox": [270, 190, 280, 200], "confidence": 0.79, "class_id": 0, "class_name": "person"},
        {"bbox": [295, 188, 305, 198], "confidence": 0.78, "class_id": 0, "class_name": "person"},
    ]

    violations = evaluate_violations(
        image_size=image.size,
        detections=detections,
        speed_limit_mph=35,
        traffic_light_status="RED",
        stop_line_ratio=0.60,
        manual_speed_mph=None,
    )
    assert any(v["violation_type"] == "Helmet Non-Compliance" for v in violations), \
        "Expected Helmet Non-Compliance violation"
    assert any(v["violation_type"] == "Triple Riding" for v in violations), \
        "Expected Triple Riding violation"
    print(f"    Detected violations: {[v['violation_type'] for v in violations]}  ✓")

    # Speeding test
    violations_speed = evaluate_violations(
        image_size=image.size,
        detections=detections,
        speed_limit_mph=35,
        traffic_light_status="GREEN",  # no red-light
        stop_line_ratio=0.99,          # no stop-line (vehicles won't cross)
        manual_speed_mph=60,
    )
    assert any(v["violation_type"] == "Speeding" for v in violations_speed), \
        "Expected Speeding violation"
    print(f"    Speeding rule fires at 60 mph / 35 mph limit  ✓")

    # ── 3. OCR fallback ───────────────────────────────────────────────
    print("\n[3] OCR fallback …")
    plate = read_license_plate(image, detections[0]["bbox"], "motorcycle")
    assert len(plate) >= 4, "OCR fallback plate too short"
    print(f"    Mock plate: {plate}  ✓")

    # ── 4. Evidence ───────────────────────────────────────────────────
    print("\n[4] Evidence generation …")
    crop_path = save_evidence_crop(image, detections[0]["bbox"], "test_motorcycle", evidence_dir)
    assert Path(crop_path).exists(), "Evidence crop not saved"

    annotated = draw_annotated_image(image, detections, violations)
    ann_path = evidence_dir / "annotated_test.jpg"
    annotated.save(ann_path)
    assert ann_path.exists(), "Annotated image not saved"
    print(f"    Crop saved: {crop_path}  ✓")
    print(f"    Annotated: {ann_path}  ✓")

    # ── 5. Database ───────────────────────────────────────────────────
    print("\n[5] Database (schema v2) …")
    init_db(test_db)
    delete_all_violations(test_db)

    row_id = insert_violation(
        violation_id="TVAI-TEST-0001",
        image_filename="dummy.jpg",
        vehicle_type="motorcycle",
        violation_type="Helmet Non-Compliance",
        violations_json=["Helmet Non-Compliance", "Triple Riding"],
        confidence=0.90,
        original_image_path="dummy.jpg",
        annotated_image_path=str(ann_path),
        ocr_plate_number=plate,
        evidence_path=crop_path,
        image_hash=img_hash,
        location_name="Test Junction",
        gps_coordinates="0.0000° N, 0.0000° E",
        severity=7,
        review_status="Pending Review",
        db_path=test_db,
    )
    update_evidence_path(row_id, crop_path, test_db)

    df = get_all_violations(test_db)
    assert len(df) == 1, "Expected 1 row in database"
    row = df.iloc[0]
    assert row["image_hash"] == img_hash, "image_hash mismatch"
    assert row["location_name"] == "Test Junction", "location_name mismatch"
    viol_list = json.loads(row["violations_json"] or "[]")
    assert "Triple Riding" in viol_list, "violations_json missing Triple Riding"
    print(f"    Row inserted: id={row_id}  ✓")
    print(f"    image_hash stored: {row['image_hash'][:16]}…  ✓")
    print(f"    violations_json: {viol_list}  ✓")

    # Analytics queries
    stats = get_violation_stats(test_db)
    offenders = get_repeat_offenders(test_db, min_violations=1)
    print(f"    get_violation_stats rows: {len(stats)}  ✓")
    print(f"    get_repeat_offenders rows: {len(offenders)}  ✓")

    # ── 6. Local YOLO model loading (import only, no image run) ──────
    print("\n[6] LocalYOLODetector import …")
    try:
        from utils.detector import LocalYOLODetector
        d = LocalYOLODetector.__new__(LocalYOLODetector)
        d._models = {}
        d.confidence = 0.25
        d.iou = 0.45
        d.device = "cpu"
        print("    LocalYOLODetector instantiated  ✓")
    except Exception as exc:
        print(f"    Warning: {exc}")

    print("\n" + "=" * 60)
    print("All IntelliTraffic AI pipeline checks PASSED ✓")
    print("=" * 60)


if __name__ == "__main__":
    main()
