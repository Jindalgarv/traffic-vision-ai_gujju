"""TrafficVision AI - Streamlit traffic violation detection app.

Run with:
    streamlit run app.py
"""

from __future__ import annotations

import os
from datetime import datetime
from io import BytesIO
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple
from uuid import uuid4

import pandas as pd
import streamlit as st
from dotenv import load_dotenv
from PIL import Image, UnidentifiedImageError

from utils.database import (
    delete_all_violations,
    get_all_violations,
    init_db,
    insert_violation,
    update_review_status,
)
from utils.detector import TrafficDetector
from utils.evidence import draw_annotated_image, save_annotated_evidence_image, save_evidence_crop
from utils.ocr import clean_plate_text, extract_license_plate_ocr, is_vehicle
from utils.rules import evaluate_violations, simulated_speed_mph


# -----------------------------------------------------------------------------
# Project root and environment configuration
# -----------------------------------------------------------------------------
PROJECT_DIR = Path(__file__).resolve().parent

# Load local .env values once. Never print or display the API key.
# The project-root path keeps `streamlit run app.py` working even if launched from another folder.
load_dotenv(PROJECT_DIR / ".env")
load_dotenv()

ROBOFLOW_API_KEY = os.getenv("ROBOFLOW_API_KEY", "").strip()
ROBOFLOW_HELMET_MODEL_ID = os.getenv("ROBOFLOW_HELMET_MODEL_ID", "").strip()
ROBOFLOW_PLATE_MODEL_ID = os.getenv("ROBOFLOW_PLATE_MODEL_ID", "").strip()
ROBOFLOW_API_URL = os.getenv("ROBOFLOW_API_URL", "https://serverless.roboflow.com").strip()

REQUIRED_ENV_VARS = {
    "ROBOFLOW_API_KEY": ROBOFLOW_API_KEY,
    "ROBOFLOW_HELMET_MODEL_ID": ROBOFLOW_HELMET_MODEL_ID,
    "ROBOFLOW_PLATE_MODEL_ID": ROBOFLOW_PLATE_MODEL_ID,
    "ROBOFLOW_API_URL": ROBOFLOW_API_URL,
}


# -----------------------------------------------------------------------------
# Project folders
# -----------------------------------------------------------------------------
MODEL_PATH = PROJECT_DIR / "models" / "best.pt"  # Not used when using Roboflow API.
DATA_DIR = PROJECT_DIR / "data"
UPLOADS_DIR = DATA_DIR / "uploads"
ANNOTATED_DIR = DATA_DIR / "annotated"
CROPS_DIR = DATA_DIR / "crops"
DB_PATH = DATA_DIR / "traffic_violations.db"

DATA_DIR.mkdir(exist_ok=True)
UPLOADS_DIR.mkdir(parents=True, exist_ok=True)
ANNOTATED_DIR.mkdir(parents=True, exist_ok=True)
CROPS_DIR.mkdir(parents=True, exist_ok=True)


# -----------------------------------------------------------------------------
# Streamlit setup
# -----------------------------------------------------------------------------
st.set_page_config(
    page_title="TrafficVision AI",
    page_icon="🚦",
    layout="wide",
)

init_db(DB_PATH)


# -----------------------------------------------------------------------------
# Styling helpers
# -----------------------------------------------------------------------------
st.markdown(
    """
    <style>
        .block-container {padding-top: 1.4rem; padding-bottom: 2rem;}
        .tvai-title {font-size: 2.2rem; font-weight: 800; margin-bottom: 0.15rem;}
        .tvai-subtitle {font-size: 1.02rem; color: #667085; margin-bottom: 1.1rem;}
        .status-card {
            border: 1px solid #e6e8ec;
            border-radius: 14px;
            padding: 14px 16px;
            background: #ffffff;
            box-shadow: 0 1px 2px rgba(16, 24, 40, 0.04);
            min-height: 92px;
        }
        .status-title {font-size: 0.84rem; color: #667085; margin-bottom: 0.45rem;}
        .status-value {font-size: 1.04rem; font-weight: 700; color: #101828;}
        .status-badge {
            display: inline-block;
            border-radius: 999px;
            padding: 0.22rem 0.58rem;
            font-size: 0.78rem;
            font-weight: 700;
            border: 1px solid transparent;
            white-space: nowrap;
        }
        .status-loaded, .status-ready, .status-approved {
            color: #067647; background: #ecfdf3; border-color: #abefc6;
        }
        .status-missing, .status-rejected {
            color: #b42318; background: #fef3f2; border-color: #fecdca;
        }
        .status-pending-review {
            color: #b54708; background: #fffaeb; border-color: #fedf89;
        }
        .status-needs-manual-check {
            color: #3538cd; background: #eef4ff; border-color: #c7d7fe;
        }
        .status-neutral {
            color: #344054; background: #f2f4f7; border-color: #e4e7ec;
        }
        .violation-card {
            border: 1px solid #e6e8ec;
            border-radius: 16px;
            padding: 1rem;
            background: #ffffff;
            box-shadow: 0 1px 3px rgba(16, 24, 40, 0.06);
            margin-bottom: 0.85rem;
        }
        .violation-title {font-weight: 800; font-size: 1.05rem; margin-bottom: 0.3rem;}
        .muted {color: #667085; font-size: 0.88rem;}
    </style>
    """,
    unsafe_allow_html=True,
)


# -----------------------------------------------------------------------------
# Core helpers
# -----------------------------------------------------------------------------
def get_missing_env_vars() -> List[str]:
    """Return required Roboflow environment variables that are missing."""

    return [name for name, value in REQUIRED_ENV_VARS.items() if not str(value or "").strip()]


def status_text(value: str) -> str:
    """Return a safe Loaded/Missing status without exposing secret values."""

    return "Loaded" if str(value or "").strip() else "Missing"


def status_slug(status: str) -> str:
    """Convert a review/config status into a CSS class suffix."""

    return str(status or "neutral").strip().lower().replace(" ", "-").replace("/", "-")


def status_badge(status: str) -> str:
    """Return a color-coded badge for statuses used in the app."""

    known_statuses = {
        "loaded",
        "ready",
        "approved",
        "missing",
        "rejected",
        "pending-review",
        "needs-manual-check",
    }
    slug = status_slug(status)
    css_class = f"status-{slug}" if slug in known_statuses else "status-neutral"
    return f'<span class="status-badge {css_class}">{status}</span>'


def render_status_card(title: str, value: str, status: str) -> None:
    """Render one polished top-level status card."""

    st.markdown(
        f"""
        <div class="status-card">
            <div class="status-title">{title}</div>
            <div class="status-value">{value}</div>
            <div style="margin-top: 0.55rem;">{status_badge(status)}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_config_line(label: str, value: str) -> None:
    """Render a safe sidebar configuration status line."""

    st.markdown(
        f"**{label}:** {status_badge(status_text(value))}",
        unsafe_allow_html=True,
    )


@st.cache_resource(show_spinner=False)
def load_detector(
    confidence_threshold: float,
    roboflow_api_key: str,
    roboflow_model_id: str,
    roboflow_api_url: str,
) -> TrafficDetector:
    """Create one Roboflow API detector and reuse it across Streamlit reruns.

    The same helper is used twice: once for the helmet/no-helmet model and once
    for the license-plate model. Streamlit caches separate detectors because the
    model_id and confidence threshold are part of the function arguments.
    """

    return TrafficDetector(
        api_key=roboflow_api_key,
        model_id=roboflow_model_id,
        api_url=roboflow_api_url,
        confidence=confidence_threshold,
    )


def add_model_metadata(
    detections: List[Dict[str, Any]],
    model_role: str,
    model_id: str,
) -> List[Dict[str, Any]]:
    """Tag detections with the model that produced them."""

    tagged: List[Dict[str, Any]] = []
    for detection in detections:
        item = dict(detection)
        item["model_role"] = model_role
        item["model_id"] = model_id
        tagged.append(item)
    return tagged


def save_uploaded_image(uploaded_file, image_bytes: bytes) -> Path:
    """Save the uploaded image so the database can reference the original file."""

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    safe_name = uploaded_file.name.replace(" ", "_")
    output_path = UPLOADS_DIR / f"{timestamp}_{safe_name}"

    with open(output_path, "wb") as file:
        file.write(image_bytes)

    return output_path


def detections_to_dataframe(detections: List[Dict[str, Any]]) -> pd.DataFrame:
    """Convert detections to a user-friendly table."""

    rows = []
    for idx, detection in enumerate(detections):
        class_name = detection.get("class_name", "")
        rows.append(
            {
                "index": idx,
                "model_role": detection.get("model_role", ""),
                "class_name": class_name,
                "confidence": detection.get("confidence"),
                "bbox": detection.get("bbox"),
                "simulated_speed_mph": simulated_speed_mph(detection)
                if is_vehicle(class_name)
                else None,
            }
        )
    return pd.DataFrame(rows)


def bbox_center(bbox: Iterable[float]) -> Tuple[float, float]:
    """Return the center point of a bounding box."""

    x1, y1, x2, y2 = [float(v) for v in bbox]
    return (x1 + x2) / 2.0, (y1 + y2) / 2.0


def point_inside_bbox(point: Tuple[float, float], bbox: Iterable[float]) -> bool:
    """Return True if a point is inside a bounding box."""

    x, y = point
    x1, y1, x2, y2 = [float(v) for v in bbox]
    return x1 <= x <= x2 and y1 <= y <= y2


def center_distance(box_a: Iterable[float], box_b: Iterable[float]) -> float:
    """Distance between the centers of two bounding boxes."""

    ax, ay = bbox_center(box_a)
    bx, by = bbox_center(box_b)
    return ((ax - bx) ** 2 + (ay - by) ** 2) ** 0.5


def find_best_plate_for_vehicle(
    vehicle_bbox: Iterable[float],
    plate_results: List[Dict[str, Any]],
) -> Optional[Dict[str, Any]]:
    """Find the most likely license plate OCR result for one vehicle.

    Preference order:
    1. A license-plate box whose center is inside the vehicle box.
    2. Otherwise, the nearest license-plate box by center distance.
    """

    if not plate_results:
        return None

    inside_matches = []
    for plate in plate_results:
        plate_bbox = plate.get("bbox", [0, 0, 0, 0])
        if point_inside_bbox(bbox_center(plate_bbox), vehicle_bbox):
            inside_matches.append(plate)

    candidates = inside_matches or plate_results
    return min(
        candidates,
        key=lambda plate: center_distance(vehicle_bbox, plate.get("bbox", [0, 0, 0, 0])),
    )


def enrich_violations_with_plates(
    detections: List[Dict[str, Any]],
    violations: List[Dict[str, Any]],
    plate_results: List[Dict[str, Any]],
) -> List[Dict[str, Any]]:
    """Attach OCR plate text/confidence to each vehicle violation."""

    enriched = []
    for violation in violations:
        enriched_violation = dict(violation)
        detection_index = violation.get("detection_index")

        if detection_index is None or detection_index >= len(detections):
            enriched.append(enriched_violation)
            continue

        detection = detections[detection_index]
        class_name = detection.get("class_name", "vehicle")
        bbox = detection.get("bbox", violation.get("bbox", [0, 0, 0, 0]))

        if is_vehicle(class_name):
            plate = find_best_plate_for_vehicle(bbox, plate_results)
            if plate:
                enriched_violation["license_plate"] = plate.get("plate_text", "N/A") or "N/A"
                enriched_violation["ocr_confidence"] = plate.get("ocr_confidence")
                enriched_violation["ocr_engine"] = plate.get("ocr_engine", "")
                enriched_violation["plate_detection_index"] = plate.get(
                    "combined_detection_index",
                    plate.get("detection_index"),
                )
            else:
                enriched_violation["license_plate"] = "N/A"
                enriched_violation["ocr_confidence"] = None
                enriched_violation["ocr_engine"] = ""
        else:
            enriched_violation["license_plate"] = "N/A"
            enriched_violation["ocr_confidence"] = None
            enriched_violation["ocr_engine"] = ""

        enriched.append(enriched_violation)

    return enriched


def violations_to_dataframe(violations: List[Dict[str, Any]]) -> pd.DataFrame:
    """Convert violations to a display table."""

    rows = []
    for violation in violations:
        ocr_confidence = violation.get("ocr_confidence")
        rows.append(
            {
                "vehicle_type": violation.get("vehicle_type"),
                "violation_type": violation.get("violation_type"),
                "license_plate": violation.get("license_plate", "N/A"),
                "ocr_confidence": round(float(ocr_confidence), 3)
                if ocr_confidence is not None
                else None,
                "ocr_engine": violation.get("ocr_engine", ""),
                "confidence": round(float(violation.get("confidence", 0.0)), 3),
                "speed_mph": violation.get("speed_mph"),
                "details": violation.get("details"),
            }
        )
    return pd.DataFrame(rows)


def show_plate_review_ui(plate_results: List[Dict[str, Any]], run_id: int) -> List[Dict[str, Any]]:
    """Show cropped/enhanced plates and let the user edit plate text."""

    st.subheader("🔎 Plate OCR Review")

    if not plate_results:
        st.info(
            "No license plate boxes were detected. Confirm that ROBOFLOW_PLATE_MODEL_ID points "
            "to a plate object-detection model and that its class name contains `plate`."
        )
        return []

    edited_results: List[Dict[str, Any]] = []

    for plate_number, plate in enumerate(plate_results, start=1):
        with st.container(border=True):
            st.markdown(f"**Plate detection #{plate_number}**")
            image_col, enhanced_col, edit_col = st.columns([1, 1, 2])

            with image_col:
                st.image(
                    plate.get("plate_crop"),
                    caption="Original plate crop",
                    use_container_width=True,
                )

            with enhanced_col:
                st.image(
                    plate.get("enhanced_crop"),
                    caption="Enhanced OCR crop",
                    use_container_width=True,
                )

            with edit_col:
                default_text = str(plate.get("plate_text", ""))
                key = f"plate_text_{run_id}_{plate.get('detection_index', plate_number)}"
                edited_text = st.text_input(
                    "Review / edit OCR plate text before saving",
                    value=default_text,
                    key=key,
                )

                confidence = plate.get("ocr_confidence")
                st.caption(
                    f"Engine: {plate.get('ocr_engine', 'unknown')} | "
                    f"OCR confidence: {confidence if confidence is not None else 'N/A'}"
                )

                if plate.get("error"):
                    st.caption(f"OCR note: {plate.get('error')}")

            edited_plate = dict(plate)
            edited_plate["plate_text"] = clean_plate_text(edited_text) or edited_text.strip().upper()
            edited_results.append(edited_plate)

    return edited_results


def generate_violation_id() -> str:
    """Create a human-readable unique violation ID."""

    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    suffix = uuid4().hex[:8].upper()
    return f"TVAI-{timestamp}-{suffix}"


def log_violations(
    image: Image.Image,
    original_image_path: Path,
    detections: List[Dict[str, Any]],
    violations: List[Dict[str, Any]],
) -> List[Dict[str, Any]]:
    """Save annotated evidence images and insert violation rows into SQLite."""

    saved_records: List[Dict[str, Any]] = []

    for violation in violations:
        violation_id = generate_violation_id()
        saved_timestamp = datetime.now().isoformat(timespec="seconds")

        annotated_path = save_annotated_evidence_image(
            image=image,
            detections=detections,
            violation=violation,
            violation_id=violation_id,
            annotated_dir=ANNOTATED_DIR,
        )

        crop_path = save_evidence_crop(
            image=image,
            bbox=violation.get("bbox", [0, 0, 0, 0]),
            violation_id=violation_id,
            evidence_dir=CROPS_DIR,
        )

        row_id = insert_violation(
            violation_id=violation_id,
            image_filename=Path(original_image_path).name,
            vehicle_type=violation.get("vehicle_type", "unknown"),
            violation_type=violation.get("violation_type", "unknown"),
            confidence=float(violation.get("confidence", 0.0)),
            ocr_plate_number=violation.get("license_plate", "N/A"),
            ocr_confidence=violation.get("ocr_confidence"),
            ocr_engine=violation.get("ocr_engine", ""),
            speed_mph=violation.get("speed_mph"),
            details=violation.get("details", ""),
            original_image_path=str(original_image_path),
            annotated_image_path=annotated_path,
            evidence_path=crop_path,
            review_status="Pending Review",
            db_path=DB_PATH,
        )

        saved_record = dict(violation)
        saved_record.update(
            {
                "id": row_id,
                "violation_id": violation_id,
                "timestamp": saved_timestamp,
                "image_filename": Path(original_image_path).name,
                "annotated_image_path": annotated_path,
                "evidence_path": crop_path,
                "review_status": "Pending Review",
            }
        )
        saved_records.append(saved_record)

    return saved_records


def run_detection_pipeline(
    image: Image.Image,
    uploaded_file,
    image_bytes: bytes,
    helmet_confidence_threshold: float,
    plate_confidence_threshold: float,
    ocr_engine: str,
    traffic_light_status: str,
    speed_limit_mph: int,
    manual_speed: int,
    stop_line_ratio: float,
    pedestrian_distance_ratio: float,
) -> None:
    """Run both Roboflow models, OCR, rules, annotation, and store latest run in session."""

    original_image_path = save_uploaded_image(uploaded_file, image_bytes)

    helmet_detector = load_detector(
        confidence_threshold=helmet_confidence_threshold,
        roboflow_api_key=ROBOFLOW_API_KEY,
        roboflow_model_id=ROBOFLOW_HELMET_MODEL_ID,
        roboflow_api_url=ROBOFLOW_API_URL,
    )
    helmet_detections = add_model_metadata(
        helmet_detector.detect(image),
        model_role="helmet",
        model_id=ROBOFLOW_HELMET_MODEL_ID,
    )

    plate_detector = load_detector(
        confidence_threshold=plate_confidence_threshold,
        roboflow_api_key=ROBOFLOW_API_KEY,
        roboflow_model_id=ROBOFLOW_PLATE_MODEL_ID,
        roboflow_api_url=ROBOFLOW_API_URL,
    )
    plate_detections = add_model_metadata(
        plate_detector.detect(image),
        model_role="plate",
        model_id=ROBOFLOW_PLATE_MODEL_ID,
    )

    # Helmet violations are evaluated only from the helmet model.
    # Plate detections are merged later only for annotation and OCR.
    detections = helmet_detections + plate_detections

    plate_results = extract_license_plate_ocr(
        image=image,
        detections=plate_detections,
        ocr_engine=ocr_engine,
    )
    for plate_result in plate_results:
        # The OCR helper receives only plate_detections, so its detection_index is
        # local to the plate model. Store the combined index too so annotated
        # evidence highlights the correct plate box after both models are merged.
        plate_result["combined_detection_index"] = (
            len(helmet_detections) + int(plate_result.get("detection_index", 0))
        )

    violations = evaluate_violations(
        image_size=image.size,
        detections=helmet_detections,
        speed_limit_mph=speed_limit_mph,
        traffic_light_status=traffic_light_status,
        stop_line_ratio=stop_line_ratio,
        pedestrian_distance_ratio=pedestrian_distance_ratio,
        manual_speed_mph=manual_speed if manual_speed > 0 else None,
    )

    violations = enrich_violations_with_plates(
        detections=detections,
        violations=violations,
        plate_results=plate_results,
    )

    annotated = draw_annotated_image(
        image=image,
        detections=detections,
        violations=violations,
        stop_line_ratio=stop_line_ratio,
        traffic_light_status=traffic_light_status,
    )

    st.session_state["last_run_id"] = st.session_state.get("last_run_id", 0) + 1
    st.session_state["last_run_timestamp"] = datetime.now().isoformat(timespec="seconds")
    st.session_state["last_image"] = image
    st.session_state["last_original_image_path"] = str(original_image_path)
    st.session_state["last_detections"] = detections
    st.session_state["last_helmet_detections"] = helmet_detections
    st.session_state["last_plate_detections"] = plate_detections
    st.session_state["last_raw_violations"] = violations
    st.session_state["last_plate_results"] = plate_results
    st.session_state["last_annotated"] = annotated
    st.session_state["last_saved_records"] = []


def find_plate_for_violation(
    violation: Dict[str, Any],
    plate_results: List[Dict[str, Any]],
) -> Optional[Dict[str, Any]]:
    """Find the OCR result attached to a violation, if any."""

    target_index = violation.get("plate_detection_index")
    if target_index is None:
        return None

    for plate in plate_results:
        if plate.get("combined_detection_index", plate.get("detection_index")) == target_index:
            return plate
    return None


def get_saved_record_for_violation(index: int) -> Dict[str, Any]:
    """Return the saved record matching a latest-run violation index, if saved."""

    saved_records = st.session_state.get("last_saved_records", [])
    if 0 <= index < len(saved_records):
        return saved_records[index]
    return {}


def render_violation_card(
    violation: Dict[str, Any],
    index: int,
    saved_record: Optional[Dict[str, Any]] = None,
) -> None:
    """Render one readable violation card."""

    saved_record = saved_record or {}
    violation_id = saved_record.get("violation_id", "Not saved yet")
    review_status = saved_record.get("review_status", "Pending Review")
    timestamp = saved_record.get("timestamp", st.session_state.get("last_run_timestamp", "N/A"))
    confidence = float(violation.get("confidence", 0.0))
    plate_number = violation.get("license_plate", "N/A")

    with st.container(border=True):
        top_col, badge_col = st.columns([3, 1])
        with top_col:
            st.markdown(f"### Violation #{index + 1}: {violation.get('violation_type', 'Unknown')}")
            st.caption(f"Violation ID: {violation_id}")
        with badge_col:
            st.markdown(status_badge(review_status), unsafe_allow_html=True)

        metric_col1, metric_col2, metric_col3, metric_col4 = st.columns(4)
        metric_col1.metric("Confidence", f"{confidence:.3f}")
        metric_col2.metric("OCR Plate", str(plate_number or "N/A"))
        metric_col3.metric("Vehicle", str(violation.get("vehicle_type", "unknown")))
        metric_col4.metric("Timestamp", str(timestamp))

        st.caption(violation.get("details", ""))

        if saved_record.get("annotated_image_path"):
            st.markdown(f"**Saved annotated evidence:** `{saved_record.get('annotated_image_path')}`")
        else:
            st.info("This violation has not been saved to SQLite/evidence folders yet.")


def render_review_record(selected_row: pd.Series) -> None:
    """Render the selected SQLite review record with action buttons."""

    evidence_col, metadata_col = st.columns([2, 1])
    with evidence_col:
        annotated_path = selected_row.get("annotated_image_path", "")
        if annotated_path and Path(str(annotated_path)).exists():
            st.image(
                str(annotated_path),
                caption=f"Annotated evidence: {selected_row.get('violation_id', selected_row.get('id'))}",
                use_container_width=True,
            )
        else:
            st.warning("Annotated evidence image not found for this record.")

    with metadata_col:
        st.markdown(status_badge(str(selected_row.get("review_status", "Pending Review"))), unsafe_allow_html=True)
        st.markdown(f"**Violation ID:** {selected_row.get('violation_id', '')}")
        st.markdown(f"**Timestamp:** {selected_row.get('timestamp', '')}")
        st.markdown(f"**Image file:** {selected_row.get('image_filename', '')}")
        st.markdown(f"**Violation type:** {selected_row.get('violation_type', '')}")
        st.markdown(f"**Confidence:** {float(selected_row.get('confidence', 0.0)):.3f}")
        st.markdown(f"**OCR plate:** {selected_row.get('ocr_plate_number', 'N/A')}")

        approve_col, reject_col, manual_col = st.columns(3)
        with approve_col:
            if st.button("✅ Approve", type="primary", key=f"approve_{selected_row.get('id')}"):
                update_review_status(int(selected_row.get("id")), "Approved", DB_PATH)
                st.success("Marked as Approved.")
                st.rerun()
        with reject_col:
            if st.button("❌ Reject", key=f"reject_{selected_row.get('id')}"):
                update_review_status(int(selected_row.get("id")), "Rejected", DB_PATH)
                st.success("Marked as Rejected.")
                st.rerun()
        with manual_col:
            if st.button("🟦 Manual Check", key=f"manual_{selected_row.get('id')}"):
                update_review_status(int(selected_row.get("id")), "Needs Manual Check", DB_PATH)
                st.success("Marked as Needs Manual Check.")
                st.rerun()

    crop_path = selected_row.get("evidence_path", "")
    if crop_path and Path(str(crop_path)).exists():
        with st.expander("Show violation crop"):
            st.image(str(crop_path), caption="Saved violation crop", use_container_width=True)


# -----------------------------------------------------------------------------
# Sidebar
# -----------------------------------------------------------------------------
with st.sidebar:
    st.header("🚦 TrafficVision AI")
    st.caption("Two-model traffic violation detection and human review.")

    st.subheader("Project Configuration")
    render_config_line("Roboflow API key", ROBOFLOW_API_KEY)
    render_config_line("Helmet model ID", ROBOFLOW_HELMET_MODEL_ID)
    render_config_line("Plate model ID", ROBOFLOW_PLATE_MODEL_ID)
    render_config_line("Roboflow API URL", ROBOFLOW_API_URL)

    missing_env_vars = get_missing_env_vars()
    if missing_env_vars:
        st.error("Missing .env variable(s): " + ", ".join(missing_env_vars))
    else:
        st.success("Roboflow configuration loaded from .env.")

    st.divider()
    st.subheader("Detection Settings")
    helmet_confidence_threshold = st.slider(
        "Helmet model confidence threshold",
        min_value=0.10,
        max_value=0.90,
        value=0.25,
        step=0.05,
    )
    plate_confidence_threshold = st.slider(
        "Plate model confidence threshold",
        min_value=0.10,
        max_value=0.90,
        value=0.25,
        step=0.05,
    )

    st.subheader("License Plate OCR")
    ocr_engine = st.selectbox(
        "OCR engine",
        options=["auto", "easyocr", "paddleocr", "mock"],
        index=0,
        help=(
            "Auto tries EasyOCR first, then PaddleOCR, then mock fallback. "
            "Mock is useful when OCR libraries are not installed."
        ),
    )

    st.subheader("Violation Rules")
    traffic_light_status = st.selectbox(
        "Traffic light status",
        options=["RED", "YELLOW", "GREEN"],
        index=0,
    )
    speed_limit_mph = st.slider("Speed limit (mph)", 10, 80, 35, 5)
    manual_speed = st.slider(
        "Manual vehicle speed override (mph, 0 = auto simulated)",
        min_value=0,
        max_value=120,
        value=0,
        step=5,
    )
    stop_line_ratio = st.slider(
        "Virtual stop line position",
        min_value=0.30,
        max_value=0.90,
        value=0.60,
        step=0.05,
        help="0.60 means the stop line is 60% down from the top of the image.",
    )
    pedestrian_distance_ratio = st.slider(
        "Pedestrian hazard sensitivity",
        min_value=0.05,
        max_value=0.30,
        value=0.12,
        step=0.01,
    )

    st.info(
        "For image-only demos, speed is simulated. Real speed detection requires video, "
        "timestamps, camera calibration, radar, or another sensor."
    )

    with st.expander("How to configure .env"):
        st.code(
            "ROBOFLOW_API_KEY=your_roboflow_api_key_here\n"
            "ROBOFLOW_HELMET_MODEL_ID=your-helmet-model/1\n"
            "ROBOFLOW_PLATE_MODEL_ID=your-license-plate-model/1\n"
            "ROBOFLOW_API_URL=https://serverless.roboflow.com",
            language="bash",
        )


# -----------------------------------------------------------------------------
# Main UI
# -----------------------------------------------------------------------------
st.markdown('<div class="tvai-title">🚦 TrafficVision AI Evidence Review System</div>', unsafe_allow_html=True)
st.markdown(
    '<div class="tvai-subtitle">Analyze traffic images, detect helmet violations, read license plates, save annotated evidence, and review records from SQLite.</div>',
    unsafe_allow_html=True,
)

status_col1, status_col2, status_col3, status_col4 = st.columns(4)
with status_col1:
    render_status_card("Helmet model status", "Helmet / no-helmet", status_text(ROBOFLOW_HELMET_MODEL_ID))
with status_col2:
    render_status_card("Plate model status", "License plate", status_text(ROBOFLOW_PLATE_MODEL_ID))
with status_col3:
    render_status_card("OCR engine status", ocr_engine.upper(), "Ready")
with status_col4:
    render_status_card("SQLite database status", "traffic_violations.db", "Ready" if DB_PATH.exists() else "Missing")

analyze_tab, evidence_tab, review_tab, system_tab = st.tabs(
    ["📷 Analyze Image", "🧾 Violation Evidence", "✅ Review Dashboard", "⚙️ System Info"]
)


with analyze_tab:
    st.subheader("📷 Analyze Traffic Image")
    missing_env_vars = get_missing_env_vars()
    if missing_env_vars:
        st.error(
            "The app cannot run detection because these required `.env` variables are missing: "
            + ", ".join(f"`{name}`" for name in missing_env_vars)
            + ". Copy `.env.example` to `.env`, fill the values, then restart Streamlit."
        )

    upload_col, instruction_col = st.columns([2, 1])
    with upload_col:
        uploaded_file = st.file_uploader(
            "Upload a traffic image",
            type=["jpg", "jpeg", "png", "webp"],
            help="Upload one image frame containing traffic riders/vehicles and visible plates.",
        )
    with instruction_col:
        st.info(
            "Workflow: upload image → run both Roboflow models → crop/enhance plates → OCR → "
            "review detected violations → save evidence."
        )

    if uploaded_file is None:
        st.warning("Upload a traffic image to begin.")
    else:
        try:
            image_bytes = uploaded_file.getvalue()
            image = Image.open(BytesIO(image_bytes)).convert("RGB")
        except UnidentifiedImageError:
            st.error("The uploaded file could not be opened as an image.")
            st.stop()

        original_col, annotated_col = st.columns(2)
        with original_col:
            st.markdown("#### Original Image")
            st.image(image, use_container_width=True)
            st.caption(f"File: {uploaded_file.name} | Size: {image.width} × {image.height}")

        with annotated_col:
            st.markdown("#### Annotated Evidence Preview")
            if "last_annotated" in st.session_state:
                st.image(st.session_state["last_annotated"], use_container_width=True)
                st.caption("Latest annotated run preview")
            else:
                st.info("Run detection to generate an annotated preview.")

        run_button = st.button(
            "🚀 Run Helmet Detection, Plate OCR, and Rules",
            type="primary",
            disabled=bool(missing_env_vars),
            use_container_width=True,
        )

        if run_button:
            with st.spinner("Running helmet model, plate model, OCR, and violation rules..."):
                try:
                    run_detection_pipeline(
                        image=image,
                        uploaded_file=uploaded_file,
                        image_bytes=image_bytes,
                        helmet_confidence_threshold=helmet_confidence_threshold,
                        plate_confidence_threshold=plate_confidence_threshold,
                        ocr_engine=ocr_engine,
                        traffic_light_status=traffic_light_status,
                        speed_limit_mph=speed_limit_mph,
                        manual_speed=manual_speed,
                        stop_line_ratio=stop_line_ratio,
                        pedestrian_distance_ratio=pedestrian_distance_ratio,
                    )
                    st.success("Detection completed. Open the Violation Evidence tab to review OCR and save records.")
                    st.rerun()
                except Exception as error:
                    st.error(str(error))

        if "last_detections" in st.session_state:
            st.divider()
            st.subheader("Detection Summary")
            current_detections = st.session_state.get("last_detections", [])
            helmet_detections = st.session_state.get("last_helmet_detections", [])
            plate_detections = st.session_state.get("last_plate_detections", [])
            current_violations = st.session_state.get("last_raw_violations", [])
            plate_results = st.session_state.get("last_plate_results", [])

            m1, m2, m3, m4 = st.columns(4)
            m1.metric("Total detections", len(current_detections))
            m2.metric("Helmet-model detections", len(helmet_detections))
            m3.metric("Plate detections", len(plate_detections))
            m4.metric("Violations", len(current_violations))

            detection_df = detections_to_dataframe(current_detections)
            if detection_df.empty:
                st.info("No traffic-related objects were detected.")
            else:
                st.dataframe(detection_df, use_container_width=True, hide_index=True)

            if plate_results:
                st.caption(f"OCR processed {len(plate_results)} detected plate crop(s).")
            else:
                st.caption("No plate OCR results are available for the latest run.")


with evidence_tab:
    st.subheader("🧾 Latest Violation Evidence")

    if "last_detections" not in st.session_state:
        st.info("Run analysis first to generate violation evidence for the latest uploaded image.")
    else:
        current_detections = st.session_state.get("last_detections", [])
        edited_plate_results = show_plate_review_ui(
            st.session_state.get("last_plate_results", []),
            st.session_state.get("last_run_id", 0),
        )

        current_violations = enrich_violations_with_plates(
            detections=current_detections,
            violations=st.session_state.get("last_raw_violations", []),
            plate_results=edited_plate_results,
        )
        st.session_state["last_reviewed_violations"] = current_violations

        st.divider()
        st.subheader("Detected Violations")
        violation_df = violations_to_dataframe(current_violations)
        if violation_df.empty:
            st.success("No violations were detected using the current rule settings.")
        else:
            summary_col1, summary_col2, summary_col3 = st.columns(3)
            summary_col1.metric("Violations", len(current_violations))
            summary_col2.metric("OCR plates", sum(1 for v in current_violations if v.get("license_plate") not in (None, "", "N/A")))
            summary_col3.metric("Saved records", len(st.session_state.get("last_saved_records", [])))

            for index, violation in enumerate(current_violations):
                render_violation_card(
                    violation=violation,
                    index=index,
                    saved_record=get_saved_record_for_violation(index),
                )

                plate = find_plate_for_violation(violation, edited_plate_results)
                if plate:
                    crop_col, enhanced_col, ocr_col = st.columns([1, 1, 2])
                    with crop_col:
                        st.image(plate.get("plate_crop"), caption="Plate crop", use_container_width=True)
                    with enhanced_col:
                        st.image(plate.get("enhanced_crop"), caption="Enhanced OCR crop", use_container_width=True)
                    with ocr_col:
                        st.markdown(f"**OCR text:** `{plate.get('plate_text', 'N/A') or 'N/A'}`")
                        st.markdown(f"**OCR engine:** {plate.get('ocr_engine', 'unknown')}")
                        confidence = plate.get("ocr_confidence")
                        st.markdown(f"**OCR confidence:** {confidence if confidence is not None else 'N/A'}")
                else:
                    st.caption("No plate crop linked to this violation.")

            st.warning(
                "Review and edit plate text above before saving. The reviewed plate text is stored in SQLite."
            )

            if st.button("💾 Process and Log Violation Evidence", type="primary", use_container_width=True):
                saved_records = log_violations(
                    image=st.session_state["last_image"],
                    original_image_path=Path(st.session_state["last_original_image_path"]),
                    detections=st.session_state["last_detections"],
                    violations=st.session_state["last_reviewed_violations"],
                )
                st.session_state["last_saved_records"] = saved_records
                st.success(
                    f"Saved {len(saved_records)} violation evidence record(s) to SQLite, data/annotated, and data/crops."
                )
                st.rerun()


with review_tab:
    st.subheader("✅ Human Review Dashboard")
    db_df = get_all_violations(DB_PATH)

    if db_df.empty:
        st.info("No violation records saved yet. Run detection, then click Process and Log Violation Evidence.")
    else:
        metric_col1, metric_col2, metric_col3, metric_col4, metric_col5 = st.columns(5)
        metric_col1.metric("Total", len(db_df))
        metric_col2.metric("Pending", int((db_df["review_status"] == "Pending Review").sum()))
        metric_col3.metric("Approved", int((db_df["review_status"] == "Approved").sum()))
        metric_col4.metric("Rejected", int((db_df["review_status"] == "Rejected").sum()))
        metric_col5.metric("Manual Check", int((db_df["review_status"] == "Needs Manual Check").sum()))

        filter_col1, filter_col2, filter_col3 = st.columns(3)
        with filter_col1:
            status_options = ["All", "Pending Review", "Approved", "Rejected", "Needs Manual Check"]
            selected_status = st.selectbox("Filter by review status", status_options)
        with filter_col2:
            vehicle_options = ["All"] + sorted(db_df["vehicle_type"].dropna().unique().tolist())
            selected_vehicle = st.selectbox("Filter by vehicle type", vehicle_options)
        with filter_col3:
            violation_options = ["All"] + sorted(db_df["violation_type"].dropna().unique().tolist())
            selected_violation = st.selectbox("Filter by violation type", violation_options)

        filtered_df = db_df.copy()
        if selected_status != "All":
            filtered_df = filtered_df[filtered_df["review_status"] == selected_status]
        if selected_vehicle != "All":
            filtered_df = filtered_df[filtered_df["vehicle_type"] == selected_vehicle]
        if selected_violation != "All":
            filtered_df = filtered_df[filtered_df["violation_type"] == selected_violation]

        display_columns = [
            "id",
            "violation_id",
            "timestamp",
            "image_filename",
            "violation_type",
            "confidence",
            "ocr_plate_number",
            "review_status",
        ]
        st.dataframe(filtered_df[display_columns], use_container_width=True, hide_index=True)

        if filtered_df.empty:
            st.info("No records match the selected filters.")
        else:
            st.subheader("Review Selected Evidence")
            selected_id = st.selectbox(
                "Select violation record",
                filtered_df["id"].tolist(),
                format_func=lambda row_id: (
                    filtered_df.loc[filtered_df["id"] == row_id, "violation_id"].iloc[0]
                ),
            )
            selected_row = filtered_df[filtered_df["id"] == selected_id].iloc[0]
            render_review_record(selected_row)

        st.divider()
        with st.expander("Danger zone"):
            st.warning("This deletes only SQLite records. Saved images in data/annotated and data/crops are not removed.")
            if st.button("Clear All Database Records", type="secondary"):
                delete_all_violations(DB_PATH)
                st.success("All violation records were deleted.")
                st.rerun()


with system_tab:
    st.subheader("⚙️ System Info")
    st.caption("Safe operational information only. Secrets are never displayed.")

    paths = [
        ("Uploads folder", UPLOADS_DIR),
        ("Annotated evidence folder", ANNOTATED_DIR),
        ("Violation crops folder", CROPS_DIR),
        ("SQLite database", DB_PATH),
    ]
    path_rows = []
    for label, path in paths:
        path_rows.append(
            {
                "Item": label,
                "Path": str(path.relative_to(PROJECT_DIR)),
                "Exists": "Yes" if path.exists() else "No",
            }
        )
    st.dataframe(pd.DataFrame(path_rows), use_container_width=True, hide_index=True)

    config_rows = [
        {"Config": "ROBOFLOW_API_KEY", "Status": status_text(ROBOFLOW_API_KEY), "Secret displayed": "No"},
        {"Config": "ROBOFLOW_HELMET_MODEL_ID", "Status": status_text(ROBOFLOW_HELMET_MODEL_ID), "Secret displayed": "No"},
        {"Config": "ROBOFLOW_PLATE_MODEL_ID", "Status": status_text(ROBOFLOW_PLATE_MODEL_ID), "Secret displayed": "No"},
        {"Config": "ROBOFLOW_API_URL", "Status": status_text(ROBOFLOW_API_URL), "Secret displayed": "No"},
    ]
    st.dataframe(pd.DataFrame(config_rows), use_container_width=True, hide_index=True)

    db_df = get_all_violations(DB_PATH)
    c1, c2, c3 = st.columns(3)
    c1.metric("SQLite records", len(db_df))
    c2.metric("Annotated files", len(list(ANNOTATED_DIR.glob("*.jpg"))))
    c3.metric("Crop files", len(list(CROPS_DIR.glob("*.jpg"))))
