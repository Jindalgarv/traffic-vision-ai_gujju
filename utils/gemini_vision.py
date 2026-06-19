"""Gemini Vision traffic scene analyzer for IntelliTraffic AI.

Sends a traffic surveillance image to Google's Gemini Vision API and receives
structured, intelligent analysis including:
  - Vehicle identification and classification
  - Rider count and helmet status (per rider)
  - License plate reading (full-scene context, far superior to crop-based OCR)
  - Violation detection with natural-language evidence descriptions

This replaces the fragile helmet-model + LP-model + geometric-rules pipeline
with a single LLM call that *understands* the scene.

Usage::

    from utils.gemini_vision import analyze_traffic_scene

    result = analyze_traffic_scene(
        image=pil_image,
        api_key="your-gemini-api-key",
        traffic_light_status="RED",
        speed_limit_mph=35,
        location_name="MG Road, Bangalore",
    )
    violations = result["violations"]   # ready for the DB pipeline
    evidence   = result["evidence_summary"]
"""

from __future__ import annotations

import base64
import json
import re
import traceback
from io import BytesIO
from typing import Any, Dict, List, Optional

from PIL import Image


# ── severity mapping ────────────────────────────────────────────────────────
SEVERITY_MAP = {
    "critical": 10,
    "high": 8,
    "medium": 5,
    "low": 3,
}

VIOLATION_TYPE_DISPLAY = {
    "no_helmet": "No Helmet",
    "triple_riding": "Triple Riding",
    "red_light_violation": "Red-Light Violation",
    "stop_line_violation": "Stop-Line Violation",
    "speeding": "Speeding",
    "wrong_way": "Wrong-Way Driving",
    "overloading": "Overloading",
    "no_seatbelt": "No Seatbelt",
    "using_phone": "Using Mobile Phone",
}


# ── prompt ──────────────────────────────────────────────────────────────────

def _build_prompt(
    traffic_light_status: str = "RED",
    speed_limit_mph: int = 35,
    location_name: str = "",
) -> str:
    return f"""You are an expert Indian traffic violation detection system used by traffic police.
Analyze this traffic surveillance image and identify ALL traffic violations with high precision.

**Scene context (provided by the camera system):**
- Traffic signal status reported by sensor: **{traffic_light_status}**
- Posted speed limit: **{speed_limit_mph} mph**
- Camera location: **{location_name or 'Unknown'}**

**Your task:** Analyze the image and return a JSON object with this EXACT structure:

```json
{{
  "scene_description": "One-paragraph description of the traffic scene",
  "traffic_signal": {{
    "visible_in_image": true,
    "apparent_state": "RED or GREEN or YELLOW or NOT_VISIBLE"
  }},
  "vehicles": [
    {{
      "id": 1,
      "type": "motorcycle or scooter or car or truck or bus or auto_rickshaw or bicycle",
      "position_description": "e.g. center-left of frame, heading north",
      "rider_count": 2,
      "helmet_status": [
        {{"rider": "driver", "wearing_helmet": false}},
        {{"rider": "pillion_1", "wearing_helmet": true}}
      ],
      "license_plate": {{
        "visible": true,
        "text": "GJ05AB1234",
        "confidence": "high or medium or low or unreadable"
      }},
      "violations": [
        {{
          "type": "no_helmet",
          "description": "Driver of motorcycle not wearing a helmet",
          "severity": "high",
          "confidence": 0.92
        }}
      ]
    }}
  ],
  "evidence_summary": "A comprehensive paragraph suitable for a legal enforcement report. Describe each vehicle, its license plate, and every violation detected. Be specific and factual."
}}
```

**Violation detection rules:**
1. **No Helmet:** Each rider on a motorcycle/scooter MUST wear a helmet. Report separately for each rider without one.
2. **Triple Riding:** A motorcycle/scooter must carry at most 2 people. 3 or more = triple riding violation.
3. **Red Light Violation:** If traffic signal is RED and a vehicle appears to have crossed the stop line, report this.
4. **Stop-Line Violation:** Vehicle positioned beyond the stop line during a RED signal.
5. **Speeding:** Only report if there is visible evidence (e.g., blur, context).
6. **Wrong Way:** Vehicle clearly traveling against the flow of traffic.

**Critical rules for accuracy:**
- Indian license plates typically follow patterns like: XX 00 XX 0000, XX00XX0000, or similar.
- Do NOT fabricate violations. Only report what you can clearly see in the image.
- If the image quality is poor, note it in the scene description but still try your best.
- Set confidence values between 0.0 and 1.0 based on your certainty.
- For cars, check seatbelts and phone usage if visible.
- For empty or non-traffic images, return an empty vehicles array.
- helmet_status should be an empty list for non-two-wheeler vehicles.

Return ONLY the JSON object, no markdown fences, no explanation."""


# ── image encoding ──────────────────────────────────────────────────────────

def _pil_to_base64(image: Image.Image, max_size: int = 1024) -> str:
    """Resize and encode a PIL image to base64 JPEG for the API."""
    img = image.copy()
    img.thumbnail((max_size, max_size), Image.LANCZOS)
    buf = BytesIO()
    img.save(buf, format="JPEG", quality=85)
    return base64.b64encode(buf.getvalue()).decode("utf-8")


# Models to try in order of preference.  If one returns 503 / 429,
# the next is attempted automatically.
FALLBACK_MODELS = [
    "gemini-2.5-flash",
    "gemini-2.0-flash",
    "gemini-1.5-flash",
]


def _call_gemini(
    image: Image.Image,
    prompt: str,
    api_key: str,
    model_name: str = "gemini-2.5-flash",
) -> dict:
    """Send image + prompt to Gemini and parse the JSON response.

    Automatically falls back through FALLBACK_MODELS when the primary
    model returns 503 UNAVAILABLE or 429 RESOURCE_EXHAUSTED.
    """
    from google import genai

    client = genai.Client(api_key=api_key)

    # Build the ordered list: requested model first, then the remaining fallbacks
    models_to_try = [model_name] + [m for m in FALLBACK_MODELS if m != model_name]

    last_error = None
    for model in models_to_try:
        try:
            response = client.models.generate_content(
                model=model,
                contents=[prompt, image],
                config={
                    "response_mime_type": "application/json",
                    "temperature": 0.1,
                },
            )

            # Parse response
            text = response.text.strip()
            if text.startswith("```"):
                text = re.sub(r"^```(?:json)?\s*", "", text)
                text = re.sub(r"\s*```$", "", text)

            return json.loads(text)

        except Exception as exc:
            error_str = str(exc)
            is_retriable = any(code in error_str for code in ("503", "429", "UNAVAILABLE", "RESOURCE_EXHAUSTED"))
            last_error = exc
            if is_retriable and model != models_to_try[-1]:
                continue  # try the next model
            raise  # non-retriable error or last model failed


# ── result conversion ───────────────────────────────────────────────────────

def _convert_to_violations(analysis: dict) -> List[Dict[str, Any]]:
    """Convert Gemini's structured output into the violation format
    expected by the existing IntelliTraffic pipeline."""
    violations = []

    for vehicle in analysis.get("vehicles", []):
        vehicle_type = vehicle.get("type", "unknown")
        plate_text = vehicle.get("license_plate", {}).get("text", "") or "N/A"
        plate_conf_str = vehicle.get("license_plate", {}).get("confidence", "low")
        plate_conf = {"high": 0.95, "medium": 0.75, "low": 0.50, "unreadable": 0.0}.get(
            plate_conf_str, 0.5
        )

        for v in vehicle.get("violations", []):
            raw_type = v.get("type", "unknown")
            display_type = VIOLATION_TYPE_DISPLAY.get(raw_type, raw_type.replace("_", " ").title())
            severity_str = v.get("severity", "medium")
            severity = SEVERITY_MAP.get(severity_str, 5)
            confidence = float(v.get("confidence", 0.80))

            violations.append({
                "violation_type": display_type,
                "confidence": confidence,
                "vehicle_type": vehicle_type,
                "details": v.get("description", ""),
                "severity": severity,
                "license_plate": plate_text,
                "ocr_confidence": plate_conf,
                "ocr_engine": "gemini_vision",
                "vehicle_id": vehicle.get("id", 0),
                "rider_count": vehicle.get("rider_count", 0),
                "helmet_status": vehicle.get("helmet_status", []),
                # bbox placeholder — Gemini doesn't return pixel coords;
                # YOLO provides the actual bounding boxes.
                "bbox": [0, 0, 0, 0],
                "detection_index": vehicle.get("id", 1) - 1,
            })

    return violations


def _convert_to_plate_results(
    analysis: dict,
    image: Image.Image,
) -> List[Dict[str, Any]]:
    """Convert Gemini's plate readings into the plate_results format
    expected by show_plate_review_ui()."""
    plates = []

    for vehicle in analysis.get("vehicles", []):
        lp = vehicle.get("license_plate", {})
        if not lp.get("visible", False):
            continue

        plate_text = lp.get("text", "").strip()
        if not plate_text:
            continue

        conf_str = lp.get("confidence", "low")
        conf = {"high": 0.95, "medium": 0.75, "low": 0.50, "unreadable": 0.0}.get(conf_str, 0.5)

        # We don't have a tight crop, so use the full image as a placeholder
        # The human reviewer can still edit the text.
        plates.append({
            "plate_text": plate_text,
            "ocr_confidence": conf,
            "ocr_engine": "gemini_vision",
            "bbox": [0, 0, 0, 0],
            "plate_crop": image.copy().resize((200, 60)),  # thumbnail placeholder
            "enhanced_crop": image.copy().resize((200, 60)),
            "detection_index": vehicle.get("id", 1) - 1,
            "detection_confidence": conf,
            "class_name": "license_plate",
            "error": "",
            "vehicle_type": vehicle.get("type", "unknown"),
            "vehicle_id": vehicle.get("id", 0),
        })

    return plates


# ── public API ──────────────────────────────────────────────────────────────

def analyze_traffic_scene(
    image: Image.Image,
    api_key: str,
    traffic_light_status: str = "RED",
    speed_limit_mph: int = 35,
    location_name: str = "",
    model_name: str = "gemini-2.5-flash",
) -> Dict[str, Any]:
    """Analyze a traffic image using Gemini Vision.

    Returns a dict with keys:
        - ``scene_description``  (str)
        - ``evidence_summary``   (str)
        - ``traffic_signal``     (dict)
        - ``vehicles``           (list of vehicle dicts from Gemini)
        - ``violations``         (list — same shape as evaluate_violations output)
        - ``plate_results``      (list — same shape as extract_license_plate_ocr output)
        - ``raw``                (dict — the raw Gemini JSON for debugging)
        - ``error``              (str — empty on success)
    """
    prompt = _build_prompt(traffic_light_status, speed_limit_mph, location_name)

    try:
        raw = _call_gemini(image, prompt, api_key, model_name)
    except Exception as exc:
        return {
            "scene_description": "",
            "evidence_summary": "",
            "traffic_signal": {},
            "vehicles": [],
            "violations": [],
            "plate_results": [],
            "raw": {},
            "error": f"Gemini API error: {exc}\n{traceback.format_exc()}",
        }

    violations = _convert_to_violations(raw)
    plate_results = _convert_to_plate_results(raw, image)

    return {
        "scene_description": raw.get("scene_description", ""),
        "evidence_summary": raw.get("evidence_summary", ""),
        "traffic_signal": raw.get("traffic_signal", {}),
        "vehicles": raw.get("vehicles", []),
        "violations": violations,
        "plate_results": plate_results,
        "raw": raw,
        "error": "",
    }
