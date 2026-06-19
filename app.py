"""IntelliTraffic AI – Multi-Violation Detection & Evidence Platform.

Run with:
    streamlit run app.py

Two detection backends are available (sidebar toggle):
    Local Models  – Uses free pre-trained YOLOv8 models (no API key needed).
                    Models download automatically on first run.
    Roboflow API  – Uses your custom Roboflow models (requires .env config).
"""

from __future__ import annotations

import json
import logging
import os
import subprocess
import sys
from datetime import datetime
from io import BytesIO
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
from uuid import uuid4

# --- Streamlit Cloud OpenCV Auto-Fix ---
# ultralytics forces the installation of opencv-python which requires system-level
# GLib and GL libraries. On Streamlit Cloud, installing these via packages.txt
# currently fails due to mixed apt repositories (bullseye vs trixie).
# This block catches the missing library error and forcefully replaces
# opencv-python with opencv-python-headless at runtime.
try:
    import cv2
except ImportError as e:
    if "libgthread" in str(e) or "libGL" in str(e):
        import streamlit as st
        
        print("Detected Streamlit Cloud OpenCV dependency error. Applying headless hotfix...")
        subprocess.check_call([sys.executable, "-m", "pip", "uninstall", "-y", "opencv-python", "opencv-python-headless"])
        subprocess.check_call([sys.executable, "-m", "pip", "install", "opencv-python-headless>=4.8.0"])
        # Now cv2 should import cleanly
        import cv2
    else:
        raise

import pandas as pd
import streamlit as st
from dotenv import load_dotenv
from PIL import Image, UnidentifiedImageError

from utils.database import (
    delete_all_violations,
    get_all_violations,
    get_location_stats,
    get_repeat_offenders,
    get_violation_stats,
    init_db,
    insert_violation,
    update_review_status,
)
from utils.detector import LocalYOLODetector, TrafficDetector
from utils.evidence import draw_annotated_image, save_annotated_evidence_image, save_evidence_crop
from utils.ocr import clean_plate_text, extract_license_plate_ocr, is_vehicle
from utils.preprocessing import PreprocessConfig, adaptive_preprocess, compute_image_hash
from utils.rules import evaluate_violations, simulated_speed_mph, VIOLATION_SEVERITY
from utils.gemini_vision import analyze_traffic_scene

# ---------------------------------------------------------------------------
# Project paths & env config
# ---------------------------------------------------------------------------
PROJECT_DIR = Path(__file__).resolve().parent
load_dotenv(PROJECT_DIR / ".env")
load_dotenv()

ROBOFLOW_API_KEY = os.getenv("ROBOFLOW_API_KEY", "").strip()
ROBOFLOW_HELMET_MODEL_ID = os.getenv("ROBOFLOW_HELMET_MODEL_ID", "").strip()
ROBOFLOW_PLATE_MODEL_ID = os.getenv("ROBOFLOW_PLATE_MODEL_ID", "").strip()
ROBOFLOW_API_URL = os.getenv("ROBOFLOW_API_URL", "https://serverless.roboflow.com").strip()
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "").strip()

DATA_DIR = PROJECT_DIR / "data"
UPLOADS_DIR = DATA_DIR / "uploads"
ANNOTATED_DIR = DATA_DIR / "annotated"
CROPS_DIR = DATA_DIR / "crops"
DB_PATH = DATA_DIR / "traffic_violations.db"

for _d in (DATA_DIR, UPLOADS_DIR, ANNOTATED_DIR, CROPS_DIR):
    _d.mkdir(parents=True, exist_ok=True)

# Known locations with simulated GPS (for demo / hackathon)
KNOWN_LOCATIONS: Dict[str, str] = {
    "Hazratganj Junction, Lucknow":   "26.8467° N, 80.9462° E",
    "Vastrapur Crossroad, Ahmedabad": "23.0389° N, 72.5298° E",
    "Lal Darwaza, Ahmedabad":         "23.0225° N, 72.5714° E",
    "MG Road, Bangalore":             "12.9716° N, 77.6094° E",
    "Connaught Place, Delhi":         "28.6315° N, 77.2167° E",
    "Dadar TT Circle, Mumbai":        "19.0176° N, 72.8562° E",
    "Anna Nagar, Chennai":            "13.0850° N, 80.2101° E",
    "Custom Location":                "",
}

# ---------------------------------------------------------------------------
# Page setup
# ---------------------------------------------------------------------------
st.set_page_config(
    page_title="IntelliTraffic AI",
    page_icon="IT",
    layout="wide",
)
init_db(DB_PATH)

# ---------------------------------------------------------------------------
# CSS
# ---------------------------------------------------------------------------
st.markdown(
    """
    <style>
        @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;600;700;800&display=swap');

        html, body, [class*="css"] { font-family: 'Inter', sans-serif; }
        .block-container { padding-top: 1.2rem; padding-bottom: 2rem; }

        /* Hero */
        .hero-title {
            font-size: 2.4rem; font-weight: 800; letter-spacing: -0.5px;
            background: linear-gradient(135deg, #e63946 0%, #f4a261 55%, #e63946 100%);
            -webkit-background-clip: text; -webkit-text-fill-color: transparent;
            margin-bottom: 0.1rem;
        }
        .hero-sub { font-size: 1rem; color: #667085; margin-bottom: 1.2rem; }

        /* Metric cards */
        .metric-card {
            background: linear-gradient(145deg, #1a1a2e 0%, #16213e 100%);
            border: 1px solid #0f3460;
            border-radius: 16px; padding: 1.1rem 1.2rem;
            text-align: center; min-height: 100px;
        }
        .metric-card .label { font-size: 0.8rem; color: #a0aec0; margin-bottom: 0.3rem; }
        .metric-card .value { font-size: 2rem; font-weight: 800; color: #e2e8f0; }
        .metric-card .sub   { font-size: 0.75rem; color: #718096; margin-top: 0.2rem; }

        /* Status badges */
        .badge {
            display: inline-block; border-radius: 999px;
            padding: 0.2rem 0.6rem; font-size: 0.75rem; font-weight: 700;
            border: 1px solid transparent; white-space: nowrap;
        }
        .badge-green  { color: #067647; background: #ecfdf3; border-color: #abefc6; }
        .badge-red    { color: #b42318; background: #fef3f2; border-color: #fecdca; }
        .badge-yellow { color: #b54708; background: #fffaeb; border-color: #fedf89; }
        .badge-blue   { color: #3538cd; background: #eef4ff; border-color: #c7d7fe; }
        .badge-gray   { color: #344054; background: #f2f4f7; border-color: #e4e7ec; }

        /* Violation card */
        .vcard {
            border: 1px solid #e6e8ec; border-radius: 16px;
            padding: 1rem 1.1rem; background: #ffffff;
            box-shadow: 0 1px 4px rgba(16,24,40,0.06); margin-bottom: 0.8rem;
        }

        /* Risk score colours */
        .risk-high   { color: #dc2626; font-weight: 800; }
        .risk-medium { color: #d97706; font-weight: 700; }
        .risk-low    { color: #16a34a; font-weight: 600; }

        /* Hash display */
        .hash-box {
            font-family: monospace; font-size: 0.72rem; color: #4a5568;
            background: #f7fafc; border: 1px solid #e2e8f0;
            border-radius: 6px; padding: 4px 8px; word-break: break-all;
        }
    </style>
    """,
    unsafe_allow_html=True,
)

# ---------------------------------------------------------------------------
# Helper: status badge HTML
# ---------------------------------------------------------------------------
def badge(text: str, cls: str = "gray") -> str:
    css_map = {
        "green": "badge-green", "red": "badge-red",
        "yellow": "badge-yellow", "blue": "badge-blue", "gray": "badge-gray",
    }
    css_class = css_map.get(cls, "badge-gray")
    return f'<span class="badge {css_class}">{text}</span>'


def review_badge(status: str) -> str:
    mapping = {
        "Approved": "green", "Rejected": "red",
        "Pending Review": "yellow", "Needs Manual Check": "blue",
    }
    return badge(status, mapping.get(status, "gray"))


def risk_color_class(score: float) -> str:
    if score >= 70:
        return "risk-high"
    if score >= 40:
        return "risk-medium"
    return "risk-low"


# ---------------------------------------------------------------------------
# Helper: save uploaded image
# ---------------------------------------------------------------------------
def save_uploaded_image(uploaded_file, image_bytes: bytes) -> Path:
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    safe = uploaded_file.name.replace(" ", "_")
    path = UPLOADS_DIR / f"{ts}_{safe}"
    path.write_bytes(image_bytes)
    return path


# ---------------------------------------------------------------------------
# Detector caches
# ---------------------------------------------------------------------------

@st.cache_resource(show_spinner=False)
def get_local_detector(confidence: float) -> LocalYOLODetector:
    return LocalYOLODetector(confidence=confidence)


@st.cache_resource(show_spinner=False)
def get_roboflow_detector(api_key: str, model_id: str, api_url: str, confidence: float) -> TrafficDetector:
    return TrafficDetector(
        api_key=api_key, model_id=model_id, api_url=api_url, confidence=confidence,
    )


# ---------------------------------------------------------------------------
# Metadata helpers
# ---------------------------------------------------------------------------
def add_model_metadata(detections: List[Dict], model_role: str, model_id: str) -> List[Dict]:
    return [{**d, "model_role": model_role, "model_id": model_id} for d in detections]


def generate_violation_id() -> str:
    ts = datetime.now().strftime("%Y%m%d-%H%M%S")
    suffix = uuid4().hex[:8].upper()
    return f"TVAI-{ts}-{suffix}"


# ---------------------------------------------------------------------------
# Plate helpers (same as before)
# ---------------------------------------------------------------------------
def bbox_center(bbox) -> Tuple[float, float]:
    x1, y1, x2, y2 = [float(v) for v in bbox]
    return (x1 + x2) / 2.0, (y1 + y2) / 2.0


def point_inside_bbox(point, bbox) -> bool:
    x, y = point
    x1, y1, x2, y2 = [float(v) for v in bbox]
    return x1 <= x <= x2 and y1 <= y <= y2


def center_distance(a, b) -> float:
    ax, ay = bbox_center(a)
    bx, by = bbox_center(b)
    return ((ax - bx) ** 2 + (ay - by) ** 2) ** 0.5


def find_best_plate(vehicle_bbox, plate_results: List[Dict]) -> Optional[Dict]:
    if not plate_results:
        return None
    inside = [
        p for p in plate_results
        if point_inside_bbox(bbox_center(p.get("bbox", [0, 0, 0, 0])), vehicle_bbox)
    ]
    candidates = inside or plate_results
    return min(candidates, key=lambda p: center_distance(vehicle_bbox, p.get("bbox", [0, 0, 0, 0])))


def enrich_violations_with_plates(detections, violations, plate_results) -> List[Dict]:
    enriched = []
    for v in violations:
        ev = dict(v)
        idx = v.get("detection_index")
        if idx is not None and idx < len(detections):
            det = detections[idx]
            bbox = det.get("bbox", v.get("bbox", [0, 0, 0, 0]))
            if is_vehicle(det.get("class_name", "")):
                plate = find_best_plate(bbox, plate_results)
                if plate:
                    ev["license_plate"] = plate.get("plate_text") or "N/A"
                    ev["ocr_confidence"] = plate.get("ocr_confidence")
                    ev["ocr_engine"] = plate.get("ocr_engine", "")
                    ev["plate_detection_index"] = plate.get(
                        "combined_detection_index", plate.get("detection_index")
                    )
                else:
                    ev.setdefault("license_plate", "N/A")
            else:
                ev.setdefault("license_plate", "N/A")
        else:
            ev.setdefault("license_plate", "N/A")
        enriched.append(ev)
    return enriched


# ---------------------------------------------------------------------------
# Detection pipeline
# ---------------------------------------------------------------------------
def run_detection_pipeline(
    image: Image.Image,
    uploaded_file,
    image_bytes: bytes,
    backend: str,
    confidence_threshold: float,
    ocr_engine: str,
    traffic_light_status: str,
    speed_limit_mph: int,
    manual_speed: int,
    stop_line_ratio: float,
    pedestrian_distance_ratio: float,
    preprocess_config: PreprocessConfig,
    location_name: str,
    gps_coordinates: str,
    gemini_api_key: str = "",
) -> None:
    """Run preprocessing, detection, OCR, rules, annotation and store results."""

    original_image_path = save_uploaded_image(uploaded_file, image_bytes)
    image_hash = compute_image_hash(image_bytes)

    # Preprocessing
    enhanced_image = adaptive_preprocess(image, preprocess_config)

    # --- Detection ---
    gemini_result = None  # set when Hybrid mode is used

    if backend == "Hybrid AI (YOLO + Gemini Vision)":
        if not gemini_api_key:
            raise RuntimeError("Gemini API key is required for Hybrid AI mode. Enter it in the sidebar.")

        # 1. YOLO for bounding boxes (used for visual annotation + plate cropping)
        detector = get_local_detector(confidence_threshold)
        helmet_dets_raw, plate_dets_raw, vehicle_dets_raw = detector.detect_all(enhanced_image)
        helmet_detections = add_model_metadata(
            vehicle_dets_raw,
            model_role="vehicle",
            model_id="local:yolov8n-coco",
        )
        plate_detections = add_model_metadata(
            plate_dets_raw,
            model_role="plate",
            model_id="local:yolov8n-license-plate",
        )

        # 2. Gemini Vision for intelligent analysis
        gemini_result = analyze_traffic_scene(
            image=enhanced_image,
            api_key=gemini_api_key,
            traffic_light_status=traffic_light_status,
            speed_limit_mph=speed_limit_mph,
            location_name=location_name,
        )
        if gemini_result.get("error"):
            raise RuntimeError(gemini_result["error"])

    elif backend == "Local Models (Free · No API Key)":
        detector = get_local_detector(confidence_threshold)
        helmet_detections_raw, plate_detections_raw, vehicle_detections_raw = detector.detect_all(enhanced_image)
        helmet_detections = add_model_metadata(
            helmet_detections_raw + vehicle_detections_raw,
            model_role="helmet+vehicle",
            model_id="local:yolov8n-coco + yolov8s-protective-equipment",
        )
        plate_detections = add_model_metadata(
            plate_detections_raw,
            model_role="plate",
            model_id="local:yolov8n-license-plate",
        )
    else:
        # Roboflow API path
        try:
            h_detector = get_roboflow_detector(
                ROBOFLOW_API_KEY, ROBOFLOW_HELMET_MODEL_ID, ROBOFLOW_API_URL, confidence_threshold
            )
            p_detector = get_roboflow_detector(
                ROBOFLOW_API_KEY, ROBOFLOW_PLATE_MODEL_ID, ROBOFLOW_API_URL, confidence_threshold
            )
            helmet_detections = add_model_metadata(
                h_detector.detect(enhanced_image), "helmet", ROBOFLOW_HELMET_MODEL_ID
            )
            plate_detections = add_model_metadata(
                p_detector.detect(enhanced_image), "plate", ROBOFLOW_PLATE_MODEL_ID
            )
        except Exception as exc:
            raise RuntimeError(f"Roboflow API error: {exc}") from exc

    detections = helmet_detections + plate_detections

    # ── Hybrid mode: use Gemini results; other modes: use rule engine ──
    if gemini_result is not None:
        # Violations come from Gemini's intelligent analysis
        violations = gemini_result["violations"]

        # Plate crops come from YOLO bounding boxes (proper crops),
        # but the plate TEXT is overlaid from Gemini's superior OCR.
        plate_results = extract_license_plate_ocr(
            image=enhanced_image, detections=plate_detections, ocr_engine="mock"
        )
        for pr in plate_results:
            pr["combined_detection_index"] = len(helmet_detections) + int(pr.get("detection_index", 0))

        # Overlay Gemini's plate readings onto YOLO crops
        gemini_plates = list(gemini_result.get("plate_results", []))
        for pr in plate_results:
            if gemini_plates:
                gp = gemini_plates.pop(0)
                pr["plate_text"] = gp.get("plate_text", pr.get("plate_text", ""))
                pr["ocr_confidence"] = gp.get("ocr_confidence", pr.get("ocr_confidence", 0))
                pr["ocr_engine"] = "gemini_vision"
    else:
        # Traditional: crop-based OCR + geometric rule engine
        plate_results = extract_license_plate_ocr(
            image=enhanced_image, detections=plate_detections, ocr_engine=ocr_engine
        )
        for pr in plate_results:
            pr["combined_detection_index"] = len(helmet_detections) + int(pr.get("detection_index", 0))

        violations = evaluate_violations(
            image_size=enhanced_image.size,
            detections=helmet_detections,
            speed_limit_mph=speed_limit_mph,
            traffic_light_status=traffic_light_status,
            stop_line_ratio=stop_line_ratio,
            pedestrian_distance_ratio=pedestrian_distance_ratio,
            manual_speed_mph=manual_speed if manual_speed > 0 else None,
        )
        violations = enrich_violations_with_plates(
            detections=detections, violations=violations, plate_results=plate_results
        )

    annotated = draw_annotated_image(
        image=enhanced_image,
        detections=detections,
        violations=violations,
        stop_line_ratio=stop_line_ratio,
        traffic_light_status=traffic_light_status,
    )

    st.session_state.update(
        {
            "last_run_id": st.session_state.get("last_run_id", 0) + 1,
            "last_run_timestamp": datetime.now().isoformat(timespec="seconds"),
            "last_image": image,
            "last_enhanced_image": enhanced_image,
            "last_original_image_path": str(original_image_path),
            "last_image_hash": image_hash,
            "last_location_name": location_name,
            "last_gps_coordinates": gps_coordinates,
            "last_detections": detections,
            "last_helmet_detections": helmet_detections,
            "last_plate_detections": plate_detections,
            "last_raw_violations": violations,
            "last_plate_results": plate_results,
            "last_annotated": annotated,
            "last_saved_records": [],
            "last_backend": backend,
            # Gemini-specific
            "last_gemini_evidence": gemini_result.get("evidence_summary", "") if gemini_result else "",
            "last_gemini_scene": gemini_result.get("scene_description", "") if gemini_result else "",
            "last_gemini_raw": gemini_result.get("raw", {}) if gemini_result else {},
        }
    )


# ---------------------------------------------------------------------------
# Log violations to DB
# ---------------------------------------------------------------------------
def log_violations(
    image: Image.Image,
    original_image_path: Path,
    detections: List[Dict],
    violations: List[Dict],
    image_hash: str,
    location_name: str,
    gps_coordinates: str,
) -> List[Dict]:
    saved = []
    for v in violations:
        vid = generate_violation_id()
        ts = datetime.now().isoformat(timespec="seconds")
        ann_path = save_annotated_evidence_image(
            image=image, detections=detections,
            violation=v, violation_id=vid, annotated_dir=ANNOTATED_DIR,
        )
        crop_path = save_evidence_crop(
            image=image, bbox=v.get("bbox", [0, 0, 0, 0]),
            violation_id=vid, evidence_dir=CROPS_DIR,
        )
        # Collect all violation types for this vehicle (grouped)
        viol_types = [vv["violation_type"] for vv in violations
                      if vv.get("detection_index") == v.get("detection_index")]
        row_id = insert_violation(
            violation_id=vid,
            image_filename=Path(original_image_path).name,
            vehicle_type=v.get("vehicle_type", "unknown"),
            violation_type=v.get("violation_type", "unknown"),
            violations_json=viol_types,
            confidence=float(v.get("confidence", 0.0)),
            ocr_plate_number=v.get("license_plate", "N/A"),
            ocr_confidence=v.get("ocr_confidence"),
            ocr_engine=v.get("ocr_engine", ""),
            speed_mph=v.get("speed_mph"),
            severity=v.get("severity", VIOLATION_SEVERITY.get(v.get("violation_type", ""), 5)),
            details=v.get("details", ""),
            original_image_path=str(original_image_path),
            annotated_image_path=ann_path,
            evidence_path=crop_path,
            image_hash=image_hash,
            location_name=location_name,
            gps_coordinates=gps_coordinates,
            review_status="Pending Review",
            db_path=DB_PATH,
        )
        saved.append({
            **v,
            "id": row_id, "violation_id": vid, "timestamp": ts,
            "image_filename": Path(original_image_path).name,
            "annotated_image_path": ann_path,
            "evidence_path": crop_path,
            "review_status": "Pending Review",
        })
    return saved


# ---------------------------------------------------------------------------
# Plate review UI
# ---------------------------------------------------------------------------
def show_plate_review_ui(plate_results: List[Dict], run_id: int) -> List[Dict]:
    st.subheader("Plate OCR Review")
    if not plate_results:
        st.info("No license plates detected by the plate model.")
        return []
    edited = []
    for i, plate in enumerate(plate_results, 1):
        with st.container(border=True):
            st.markdown(f"**Plate #{i}**")
            c1, c2, c3 = st.columns([1, 1, 2])
            with c1:
                st.image(plate.get("plate_crop"), caption="Crop", width="stretch")
            with c2:
                st.image(plate.get("enhanced_crop"), caption="Enhanced", width="stretch")
            with c3:
                key = f"plate_{run_id}_{plate.get('detection_index', i)}"
                edited_text = st.text_input(
                    "Review / edit plate text", value=str(plate.get("plate_text", "")), key=key
                )
                conf = plate.get("ocr_confidence")
                st.caption(
                    f"Engine: {plate.get('ocr_engine', '?')} | "
                    f"Confidence: {conf if conf is not None else 'N/A'}"
                )
                if plate.get("error"):
                    st.caption(f"Note: {plate.get('error')}")
            ep = dict(plate)
            ep["plate_text"] = clean_plate_text(edited_text) or edited_text.strip().upper()
            edited.append(ep)
    return edited


# ---------------------------------------------------------------------------
# Violation card
# ---------------------------------------------------------------------------
def render_violation_card(v: Dict, index: int, saved: Optional[Dict] = None) -> None:
    saved = saved or {}
    with st.container(border=True):
        c1, c2 = st.columns([3, 1])
        with c1:
            st.markdown(f"### #{index + 1}: {v.get('violation_type', 'Unknown')}")
            st.caption(f"ID: {saved.get('violation_id', 'Not saved')}")
        with c2:
            st.markdown(review_badge(saved.get("review_status", "Pending Review")), unsafe_allow_html=True)

        m1, m2, m3, m4 = st.columns(4)
        m1.metric("Confidence", f"{float(v.get('confidence', 0)):.3f}")
        m2.metric("OCR Plate", str(v.get("license_plate", "N/A")))
        m3.metric("Vehicle", str(v.get("vehicle_type", "?")))
        m4.metric("Severity", f"{v.get('severity', '?')}/10")
        st.caption(v.get("details", ""))
        if saved.get("annotated_image_path"):
            st.markdown(f"**Evidence:** `{saved.get('annotated_image_path')}`")
        else:
            st.info("Not yet saved to database.")


# ============================================================================
# SIDEBAR
# ============================================================================
with st.sidebar:
    st.markdown("## IntelliTraffic AI")
    st.caption("Multi-Violation Detection · Evidence Platform")
    st.divider()

    # --- Backend ---
    st.subheader("Detection Backend")
    backend = st.radio(
        "Choose inference engine",
        options=[
            "Hybrid AI (YOLO + Gemini Vision)",
            "Local Models (Free · No API Key)",
            "Roboflow API (Custom Models)",
        ],
        index=0,
        help=(
            "Hybrid AI: YOLO detects vehicles, Gemini Vision reasons about violations, "
            "reads plates, and generates evidence descriptions. Best accuracy.\n\n"
            "Local Models: Uses pre-trained YOLOv8 weights only. Free, no API key.\n\n"
            "Roboflow API: Uses your custom trained models (requires .env config)."
        ),
    )

    if backend == "Hybrid AI (YOLO + Gemini Vision)":
        gemini_key = st.text_input(
            "Gemini API Key",
            value=GEMINI_API_KEY,
            type="password",
            help="Get a free key at https://aistudio.google.com/apikey",
        )
        st.markdown(
            f"**Vehicle detection:** {badge('YOLOv8n (local)', 'blue')}  \n"
            f"**Violation reasoning:** {badge('Gemini Vision AI', 'green')}  \n"
            f"**License plate OCR:** {badge('Gemini Vision AI', 'green')}  \n"
            f"**Evidence generation:** {badge('Gemini Vision AI', 'green')}",
            unsafe_allow_html=True,
        )
        if not gemini_key:
            st.warning("Enter your Gemini API key above. Free at [aistudio.google.com](https://aistudio.google.com/apikey)")
    elif backend == "Roboflow API (Custom Models)":
        gemini_key = ""
        st.markdown(
            f"**API Key:** {badge('Loaded', 'green') if ROBOFLOW_API_KEY else badge('Missing', 'red')}",
            unsafe_allow_html=True,
        )
        st.markdown(
            f"**Helmet model:** {badge('Loaded', 'green') if ROBOFLOW_HELMET_MODEL_ID else badge('Missing', 'red')}",
            unsafe_allow_html=True,
        )
        st.markdown(
            f"**Plate model:** {badge('Loaded', 'green') if ROBOFLOW_PLATE_MODEL_ID else badge('Missing', 'red')}",
            unsafe_allow_html=True,
        )
    else:
        gemini_key = ""
        st.markdown(
            f"**COCO (vehicles/persons):** {badge('yolov8n.pt', 'blue')}  \n"
            f"**Helmet model:** {badge('keremberke/yolov8s-ppe', 'blue')}  \n"
            f"**Plate model:** {badge('Koushim/yolov8-lp', 'blue')}",
            unsafe_allow_html=True,
        )

    st.divider()
    st.subheader("Detection Settings")
    confidence_threshold = st.slider("Confidence threshold", 0.10, 0.90, 0.25, 0.05)

    st.subheader("OCR Engine")
    ocr_engine = st.selectbox(
        "OCR engine",
        ["auto", "easyocr", "paddleocr", "mock"],
        index=0,
        help="Auto tries EasyOCR → PaddleOCR → mock fallback.",
    )

    st.subheader("Violation Rules")
    traffic_light_status = st.selectbox("Traffic light status", ["RED", "YELLOW", "GREEN"], index=0)
    speed_limit_mph = st.slider("Speed limit (mph)", 10, 80, 35, 5)
    manual_speed = st.slider("Manual speed override (mph, 0=auto)", 0, 120, 0, 5)
    stop_line_ratio = st.slider(
        "Stop line position (% of image height)", 0.30, 0.90, 0.60, 0.05,
        help="0.60 = stop line drawn 60% down from top."
    )

    st.subheader("Location")
    location_name = st.selectbox("Camera location", list(KNOWN_LOCATIONS.keys()), index=0)
    gps_coords = KNOWN_LOCATIONS[location_name]
    if location_name == "Custom Location":
        gps_coords = st.text_input("Custom GPS coordinates", "")
    else:
        st.caption(f"GPS: {gps_coords}")

    st.divider()
    st.subheader("Image Preprocessing")
    apply_clahe   = st.checkbox("CLAHE (low-light enhancement)", value=True)
    apply_gamma   = st.checkbox("Gamma correction (exposure)", value=True)
    gamma_val     = st.slider("Gamma value", 0.5, 2.0, 0.85, 0.05,
                              help="< 1.0 brightens; > 1.0 darkens")
    apply_sharpen = st.checkbox("Sharpening / deblur", value=True)
    apply_dehaze  = st.checkbox("Dehazing (fog/rain removal)", value=False)

    preprocess_config = PreprocessConfig(
        apply_clahe=apply_clahe,
        apply_gamma=apply_gamma,
        gamma=gamma_val,
        apply_sharpen=apply_sharpen,
        apply_dehaze=apply_dehaze,
    )

    with st.expander(".env configuration"):
        st.code(
            "ROBOFLOW_API_KEY=your_key\n"
            "ROBOFLOW_HELMET_MODEL_ID=your-helmet-model/1\n"
            "ROBOFLOW_PLATE_MODEL_ID=your-plate-model/1\n"
            "ROBOFLOW_API_URL=https://serverless.roboflow.com",
            language="bash",
        )


# ============================================================================
# HERO HEADER
# ============================================================================
st.markdown('<div class="hero-title">IntelliTraffic AI</div>', unsafe_allow_html=True)
st.markdown(
    '<div class="hero-sub">Multi-Violation Detection · Automated Evidence Generation · Enforcement Analytics</div>',
    unsafe_allow_html=True,
)

# Top KPI bar
db_df_top = get_all_violations(DB_PATH)
total = len(db_df_top)
pending = int((db_df_top["review_status"] == "Pending Review").sum()) if total else 0
approved = int((db_df_top["review_status"] == "Approved").sum()) if total else 0
unique_plates = int(db_df_top["ocr_plate_number"].nunique()) if total else 0

kpi1, kpi2, kpi3, kpi4 = st.columns(4)
with kpi1:
    st.markdown(
        '<div class="metric-card">'
        '<div class="label">Total Violations</div>'
        f'<div class="value">{total}</div>'
        '<div class="sub">in database</div>'
        '</div>',
        unsafe_allow_html=True,
    )
with kpi2:
    st.markdown(
        '<div class="metric-card">'
        '<div class="label">Pending Review</div>'
        f'<div class="value">{pending}</div>'
        '<div class="sub">awaiting officer</div>'
        '</div>',
        unsafe_allow_html=True,
    )
with kpi3:
    st.markdown(
        '<div class="metric-card">'
        '<div class="label">Approved Violations</div>'
        f'<div class="value">{approved}</div>'
        '<div class="sub">legally admitted</div>'
        '</div>',
        unsafe_allow_html=True,
    )
with kpi4:
    st.markdown(
        '<div class="metric-card">'
        '<div class="label">Unique Vehicles</div>'
        f'<div class="value">{unique_plates}</div>'
        '<div class="sub">license plates tracked</div>'
        '</div>',
        unsafe_allow_html=True,
    )

st.markdown("<br>", unsafe_allow_html=True)

# ============================================================================
# TABS
# ============================================================================
analyze_tab, evidence_tab, review_tab, analytics_tab, system_tab = st.tabs(
    [
        "Analyze Image",
        "Violation Evidence",
        "Review Dashboard",
        "Analytics & Risk",
        "System Info",
    ]
)

# ─────────────────────────────────────────────────────────────────────────────
# TAB 1: Analyze Image
# ─────────────────────────────────────────────────────────────────────────────
with analyze_tab:
    st.subheader("Analyze Traffic Image")

    upload_col, info_col = st.columns([2, 1])
    with upload_col:
        uploaded_file = st.file_uploader(
            "Upload a traffic image",
            type=["jpg", "jpeg", "png", "webp"],
            help="Upload one image frame containing traffic riders/vehicles.",
        )
    with info_col:
        st.info(
            f"**Backend:** {backend}  \n"
            f"**Location:** {location_name}  \n"
            f"**Signal:** {traffic_light_status}  \n"
            f"**Speed limit:** {speed_limit_mph} mph"
        )

    if uploaded_file is None:
        st.warning("Upload a traffic image to begin detection.")
    else:
        try:
            image_bytes = uploaded_file.getvalue()
            image = Image.open(BytesIO(image_bytes)).convert("RGB")
        except UnidentifiedImageError:
            st.error("Could not open the uploaded file as an image.")
            st.stop()

        # Preview: original + enhanced
        orig_col, enh_col, ann_col = st.columns(3)
        with orig_col:
            st.markdown("#### Original")
            st.image(image, width="stretch")
            st.caption(f"{uploaded_file.name} · {image.width}×{image.height}")
        with enh_col:
            st.markdown("#### Preprocessed")
            enhanced_preview = adaptive_preprocess(image, preprocess_config)
            st.image(enhanced_preview, width="stretch")
            st.caption("After adaptive enhancement")
        with ann_col:
            st.markdown("#### Annotated")
            if "last_annotated" in st.session_state:
                st.image(st.session_state["last_annotated"], width="stretch")
                st.caption("Latest detection run")
            else:
                st.info("Run detection to see annotated output.")

        run_btn = st.button(
            "Run IntelliTraffic Detection",
            type="primary",
            width="stretch",
        )

        if run_btn:
            with st.spinner("Running all detection models…"):
                try:
                    run_detection_pipeline(
                        image=image,
                        uploaded_file=uploaded_file,
                        image_bytes=image_bytes,
                        backend=backend,
                        confidence_threshold=confidence_threshold,
                        ocr_engine=ocr_engine,
                        traffic_light_status=traffic_light_status,
                        speed_limit_mph=speed_limit_mph,
                        manual_speed=manual_speed,
                        stop_line_ratio=stop_line_ratio,
                        pedestrian_distance_ratio=0.12,
                        preprocess_config=preprocess_config,
                        location_name=location_name,
                        gps_coordinates=gps_coords,
                        gemini_api_key=gemini_key if backend == "Hybrid AI (YOLO + Gemini Vision)" else "",
                    )
                    st.success(
                        "Detection complete. Open the **Violation Evidence** tab to review & save records."
                    )
                    st.rerun()
                except Exception as err:
                    st.error(str(err))

        # Post-run summary
        if "last_detections" in st.session_state:
            st.divider()
            st.subheader("Detection Summary")

            all_dets = st.session_state.get("last_detections", [])
            helmet_dets = st.session_state.get("last_helmet_detections", [])
            plate_dets = st.session_state.get("last_plate_detections", [])
            all_viols = st.session_state.get("last_raw_violations", [])
            img_hash = st.session_state.get("last_image_hash", "")

            s1, s2, s3, s4 = st.columns(4)
            s1.metric("Total detections", len(all_dets))
            s2.metric("Helmet-model dets", len(helmet_dets))
            s3.metric("Plate detections", len(plate_dets))
            s4.metric("Violations", len(all_viols))

            if img_hash:
                st.markdown("**Evidence Integrity Hash (SHA-256)**")
                st.markdown(f'<div class="hash-box">{img_hash}</div>', unsafe_allow_html=True)
                st.caption(f"Location: {st.session_state.get('last_location_name', 'N/A')} · "
                           f"{st.session_state.get('last_gps_coordinates', '')}")

            det_rows = []
            for i, d in enumerate(all_dets):
                det_rows.append({
                    "index": i,
                    "model_role": d.get("model_role", ""),
                    "class": d.get("class_name", ""),
                    "confidence": round(float(d.get("confidence", 0)), 3),
                })
            if det_rows:
                st.dataframe(pd.DataFrame(det_rows), width="stretch", hide_index=True)


# ─────────────────────────────────────────────────────────────────────────────
# TAB 2: Violation Evidence
# ─────────────────────────────────────────────────────────────────────────────
with evidence_tab:
    st.subheader("Latest Violation Evidence")

    if "last_detections" not in st.session_state:
        st.info("Run analysis first to generate violation evidence.")
    else:
        all_dets = st.session_state.get("last_detections", [])
        edited_plates = show_plate_review_ui(
            st.session_state.get("last_plate_results", []),
            st.session_state.get("last_run_id", 0),
        )
        violations = enrich_violations_with_plates(
            detections=all_dets,
            violations=st.session_state.get("last_raw_violations", []),
            plate_results=edited_plates,
        )
        st.session_state["last_reviewed_violations"] = violations

        # ── Gemini AI Evidence Summary (Hybrid mode) ──
        gemini_evidence = st.session_state.get("last_gemini_evidence", "")
        gemini_scene = st.session_state.get("last_gemini_scene", "")
        if gemini_evidence or gemini_scene:
            st.divider()
            st.subheader("Gemini AI Evidence Report")
            if gemini_scene:
                st.markdown(f"**Scene Analysis:** {gemini_scene}")
            if gemini_evidence:
                st.info(f"**Legal Evidence Summary:**\n\n{gemini_evidence}")
            with st.expander("View raw Gemini analysis JSON"):
                st.json(st.session_state.get("last_gemini_raw", {}))

        st.divider()
        st.subheader("Detected Violations")

        if not violations:
            st.success("No violations detected under the current rule settings.")
        else:
            saved_records = st.session_state.get("last_saved_records", [])
            v1, v2, v3 = st.columns(3)
            v1.metric("Violations", len(violations))
            v2.metric(
                "Plates extracted",
                sum(1 for v in violations if v.get("license_plate") not in (None, "", "N/A")),
            )
            v3.metric("Saved records", len(saved_records))

            for i, v in enumerate(violations):
                saved = saved_records[i] if i < len(saved_records) else {}
                render_violation_card(v, i, saved)

                # Show plate crop inline if linked
                plate = next(
                    (
                        p for p in edited_plates
                        if p.get("combined_detection_index", p.get("detection_index")) ==
                           v.get("plate_detection_index")
                    ),
                    None,
                )
                if plate:
                    pc1, pc2, pc3 = st.columns([1, 1, 2])
                    with pc1:
                        st.image(plate.get("plate_crop"), caption="Plate crop", width="stretch")
                    with pc2:
                        st.image(plate.get("enhanced_crop"), caption="Enhanced", width="stretch")
                    with pc3:
                        st.markdown(f"**OCR:** `{plate.get('plate_text', 'N/A')}`")
                        st.markdown(f"**Engine:** {plate.get('ocr_engine', '?')}")
                        st.markdown(f"**OCR confidence:** {plate.get('ocr_confidence', 'N/A')}")
                else:
                    st.caption("No plate crop linked to this violation.")

            st.warning("Review and edit plate text above before saving.")

            if st.button("Save All Violations to Database", type="primary", width="stretch"):
                records = log_violations(
                    image=st.session_state["last_enhanced_image"],
                    original_image_path=Path(st.session_state["last_original_image_path"]),
                    detections=all_dets,
                    violations=st.session_state["last_reviewed_violations"],
                    image_hash=st.session_state.get("last_image_hash", ""),
                    location_name=st.session_state.get("last_location_name", ""),
                    gps_coordinates=st.session_state.get("last_gps_coordinates", ""),
                )
                st.session_state["last_saved_records"] = records
                st.success(f"Saved {len(records)} record(s) to SQLite, annotated/, and crops/.")
                st.rerun()


# ─────────────────────────────────────────────────────────────────────────────
# TAB 3: Review Dashboard
# ─────────────────────────────────────────────────────────────────────────────
with review_tab:
    st.subheader("Human Review Dashboard")
    db_df = get_all_violations(DB_PATH)

    if db_df.empty:
        st.info("No violation records saved yet. Run detection and log violations first.")
    else:
        m1, m2, m3, m4, m5 = st.columns(5)
        m1.metric("Total", len(db_df))
        m2.metric("Pending", int((db_df["review_status"] == "Pending Review").sum()))
        m3.metric("Approved", int((db_df["review_status"] == "Approved").sum()))
        m4.metric("Rejected", int((db_df["review_status"] == "Rejected").sum()))
        m5.metric("Manual Check", int((db_df["review_status"] == "Needs Manual Check").sum()))

        fc1, fc2, fc3 = st.columns(3)
        with fc1:
            sel_status = st.selectbox(
                "Filter by status",
                ["All", "Pending Review", "Approved", "Rejected", "Needs Manual Check"],
            )
        with fc2:
            veh_opts = ["All"] + sorted(db_df["vehicle_type"].dropna().unique().tolist())
            sel_veh = st.selectbox("Filter by vehicle", veh_opts)
        with fc3:
            viol_opts = ["All"] + sorted(db_df["violation_type"].dropna().unique().tolist())
            sel_viol = st.selectbox("Filter by violation", viol_opts)

        fdf = db_df.copy()
        if sel_status != "All":
            fdf = fdf[fdf["review_status"] == sel_status]
        if sel_veh != "All":
            fdf = fdf[fdf["vehicle_type"] == sel_veh]
        if sel_viol != "All":
            fdf = fdf[fdf["violation_type"] == sel_viol]

        display_cols = [
            "id", "violation_id", "timestamp", "location_name",
            "violation_type", "confidence", "ocr_plate_number", "review_status",
        ]
        avail = [c for c in display_cols if c in fdf.columns]
        st.dataframe(fdf[avail], width="stretch", hide_index=True)

        if not fdf.empty:
            st.subheader("Review Selected Record")
            # Pre-build lookup; coalesce None violation_ids from legacy rows
            id_to_vid = {
                row_id: str(vid) if vid is not None else f"Record #{row_id}"
                for row_id, vid in zip(fdf["id"].tolist(), fdf["violation_id"].tolist())
            }
            sel_id = st.selectbox(
                "Select violation",
                fdf["id"].tolist(),
                format_func=lambda rid: id_to_vid.get(rid, f"Record #{rid}"),
            )
            row = fdf[fdf["id"] == sel_id].iloc[0]

            ec, mc = st.columns([2, 1])
            with ec:
                ann_p = row.get("annotated_image_path", "")
                if ann_p and Path(str(ann_p)).exists():
                    st.image(str(ann_p), caption=f"Evidence: {row.get('violation_id')}", width="stretch")
                else:
                    st.warning("Evidence image not found.")
            with mc:
                st.markdown(review_badge(str(row.get("review_status", "Pending Review"))), unsafe_allow_html=True)
                st.markdown(f"**Violation ID:** {row.get('violation_id', '')}")
                st.markdown(f"**Timestamp:** {row.get('timestamp', '')}")
                st.markdown(f"**Location:** {row.get('location_name', 'N/A')}")
                st.markdown(f"**GPS:** {row.get('gps_coordinates', 'N/A')}")
                st.markdown(f"**Violation:** {row.get('violation_type', '')}")
                st.markdown(f"**OCR Plate:** `{row.get('ocr_plate_number', 'N/A')}`")

                # Evidence integrity
                if row.get("image_hash") and isinstance(row.get("image_hash"), str):
                    st.markdown("**SHA-256 hash:**")
                    st.markdown(
                        f'<div class="hash-box">{row["image_hash"][:32]}…</div>',
                        unsafe_allow_html=True,
                    )

                ba, br, bm = st.columns(3)
                with ba:
                    if st.button("Approve", key=f"app_{sel_id}", type="primary"):
                        update_review_status(int(sel_id), "Approved", DB_PATH)
                        st.success("Approved.")
                        st.rerun()
                with br:
                    if st.button("❌ Reject", key=f"rej_{sel_id}"):
                        update_review_status(int(sel_id), "Rejected", DB_PATH)
                        st.success("Rejected.")
                        st.rerun()
                with bm:
                    if st.button("🟦 Manual", key=f"man_{sel_id}"):
                        update_review_status(int(sel_id), "Needs Manual Check", DB_PATH)
                        st.success("Flagged.")
                        st.rerun()

            crop_p = row.get("evidence_path", "")
            if crop_p and Path(str(crop_p)).exists():
                with st.expander("Show vehicle crop"):
                    st.image(str(crop_p), width="stretch")

        st.divider()
        with st.expander("Danger Zone"):
            st.warning("Deletes SQLite records only. Evidence images on disk are preserved.")
            if st.button("Clear All Database Records", type="secondary"):
                delete_all_violations(DB_PATH)
                st.success("All records deleted.")
                st.rerun()


# ─────────────────────────────────────────────────────────────────────────────
# TAB 4: Analytics & Risk
# ─────────────────────────────────────────────────────────────────────────────
with analytics_tab:
    st.subheader("Analytics & Risk Intelligence")

    db_df_a = get_all_violations(DB_PATH)

    if db_df_a.empty:
        st.info("No data yet. Log some violations first to see analytics.")
    else:
        # ── Violation type breakdown ──────────────────────────────────────
        st.markdown("### Violation Type Distribution")
        vtype_counts = db_df_a["violation_type"].value_counts().reset_index()
        vtype_counts.columns = ["Violation Type", "Count"]
        st.bar_chart(vtype_counts.set_index("Violation Type"))

        col_l, col_r = st.columns(2)

        # ── Daily trend ───────────────────────────────────────────────────
        with col_l:
            st.markdown("### Daily Violation Trend")
            stats_df = get_violation_stats(DB_PATH)
            if not stats_df.empty:
                pivot = stats_df.pivot_table(
                    index="date", columns="violation_type", values="count", aggfunc="sum", fill_value=0
                )
                st.line_chart(pivot)
            else:
                st.info("Not enough data for trend chart.")

        # ── Location hotspots ─────────────────────────────────────────────
        with col_r:
            st.markdown("### Location Hotspots")
            loc_df = get_location_stats(DB_PATH)
            if not loc_df.empty:
                st.dataframe(loc_df, width="stretch", hide_index=True)
            else:
                st.info("No location data.")

        st.divider()

        # ── Repeat offender leaderboard ───────────────────────────────────
        st.markdown("### 🚨 Repeat Offender Leaderboard")
        offenders = get_repeat_offenders(DB_PATH, min_violations=1)

        if offenders.empty:
            st.info("No repeat offenders detected yet (requires identified license plates).")
        else:
            # Color-coded risk score
            def color_risk(row):
                score = float(row.get("risk_score", 0))
                cls = risk_color_class(score)
                return f'<span class="{cls}">{score:.1f}</span>'

            offenders_display = offenders.copy()
            st.markdown(
                """
                <small>
                <b>Risk Score</b> = (Violations × 10) + Average Severity &nbsp;|&nbsp;
                <span class="risk-high">■ High (≥70)</span>&nbsp;
                <span class="risk-medium">■ Medium (40–69)</span>&nbsp;
                <span class="risk-low">■ Low (&lt;40)</span>
                </small>
                """,
                unsafe_allow_html=True,
            )

            for _, row in offenders_display.iterrows():
                score = float(row.get("risk_score", 0))
                plate = row.get("license_plate", "N/A")
                total_v = int(row.get("total_violations", 0))
                last = str(row.get("last_seen", ""))[:10]
                vtypes = str(row.get("violation_types", ""))
                avg_sev = row.get("avg_severity", 0)

                risk_cls = risk_color_class(score)
                with st.container(border=True):
                    rc1, rc2, rc3, rc4 = st.columns([2, 1, 1, 2])
                    rc1.markdown(f"**🚗 `{plate}`**")
                    rc2.metric("Violations", total_v)
                    rc3.metric("Avg Severity", f"{avg_sev}/10")
                    rc4.markdown(
                        f'<div style="text-align:right">'
                        f'Risk Score: <span class="{risk_cls}" style="font-size:1.3rem">{score:.1f}</span>'
                        f'<br><small>Last seen: {last}</small>'
                        f'<br><small>{vtypes}</small>'
                        f"</div>",
                        unsafe_allow_html=True,
                    )

        st.divider()

        # ── Review status summary ─────────────────────────────────────────
        st.markdown("### Review Status Summary")
        status_counts = db_df_a["review_status"].value_counts().reset_index()
        status_counts.columns = ["Status", "Count"]
        st.bar_chart(status_counts.set_index("Status"))


# ─────────────────────────────────────────────────────────────────────────────
# TAB 5: System Info
# ─────────────────────────────────────────────────────────────────────────────
with system_tab:
    st.subheader("System Info")
    st.caption("Operational information only. Secrets are never displayed.")

    path_rows = [
        {"Item": "Uploads folder", "Path": str(UPLOADS_DIR.relative_to(PROJECT_DIR)), "Exists": "Yes" if UPLOADS_DIR.exists() else "No"},
        {"Item": "Annotated evidence", "Path": str(ANNOTATED_DIR.relative_to(PROJECT_DIR)), "Exists": "Yes" if ANNOTATED_DIR.exists() else "No"},
        {"Item": "Violation crops", "Path": str(CROPS_DIR.relative_to(PROJECT_DIR)), "Exists": "Yes" if CROPS_DIR.exists() else "No"},
        {"Item": "SQLite database", "Path": str(DB_PATH.relative_to(PROJECT_DIR)), "Exists": "Yes" if DB_PATH.exists() else "No"},
    ]
    st.dataframe(pd.DataFrame(path_rows), width="stretch", hide_index=True)

    st.markdown("### Pre-Trained Models (Local Backend)")
    model_rows = [
        {"Model Key": "base_coco",   "Source": "yolov8n.pt (ultralytics CDN)",                          "Purpose": "Vehicles & persons (COCO)"},
        {"Model Key": "helmet",      "Source": "keremberke/yolov8s-protective-equipment-detection",      "Purpose": "Helmet / No-Helmet detection"},
        {"Model Key": "license_plate","Source": "keremberke/yolov8n-license-plate-detection",            "Purpose": "License plate bounding box"},
    ]
    st.dataframe(pd.DataFrame(model_rows), width="stretch", hide_index=True)

    st.markdown("### Violation Rule Engine")
    rule_rows = [
        {"Violation": k, "Severity Weight": v}
        for k, v in VIOLATION_SEVERITY.items()
    ]
    st.dataframe(pd.DataFrame(rule_rows), width="stretch", hide_index=True)

    db_df_s = get_all_violations(DB_PATH)
    sc1, sc2, sc3 = st.columns(3)
    sc1.metric("SQLite records", len(db_df_s))
    sc2.metric("Annotated images", len(list(ANNOTATED_DIR.glob("*.jpg"))))
    sc3.metric("Crop images", len(list(CROPS_DIR.glob("*.jpg"))))
