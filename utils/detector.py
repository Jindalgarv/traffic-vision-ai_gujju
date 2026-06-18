"""Roboflow API detector helper for TrafficVision AI.

This version does NOT load a local models/best.pt file. Instead, it sends the
uploaded image to Roboflow Hosted/Serverless Inference and converts Roboflow's
prediction format into the same detection dictionaries used by the rest of the
app.

Required environment variables for deployment:
    ROBOFLOW_API_KEY=your_key_here
    ROBOFLOW_HELMET_MODEL_ID=your-helmet-model/1
    ROBOFLOW_PLATE_MODEL_ID=your-license-plate-model/1

Example model id from a Roboflow model URL:
    https://app.roboflow.com/.../models/helmet-detection-traffic-cgg9e/2
    -> ROBOFLOW_HELMET_MODEL_ID=helmet-detection-traffic-cgg9e/2
"""

from __future__ import annotations

import base64
import os
import tempfile
from io import BytesIO
from pathlib import Path
from typing import Any, Dict, List, Optional

import requests
from PIL import Image


TRAFFIC_KEYWORDS = (
    "person",
    "pedestrian",
    "bicycle",
    "bike",
    "car",
    "motorcycle",
    "motorbike",
    "bus",
    "truck",
    "vehicle",
    "traffic light",
    "traffic_light",
    "signal",
    "license_plate",
    "license plate",
    "licence_plate",
    "number_plate",
    "number plate",
    "plate",
    "helmet",
    "no_helmet",
    "no helmet",
    "no-helmet",
)


class TrafficDetector:
    """Small wrapper around Roboflow API inference.

    The rest of the app can still call:
        detections = detector.detect(image)

    Returned detection format:
        {
            "bbox": [x1, y1, x2, y2],
            "confidence": 0.91,
            "class_id": 2,
            "class_name": "motorcycle"
        }
    """

    def __init__(
        self,
        api_key: Optional[str] = None,
        model_id: Optional[str] = None,
        api_url: Optional[str] = None,
        confidence: float = 0.25,
        overlap: float = 0.30,
        max_detections: int = 300,
        filter_traffic_classes: bool = True,
        # Kept only so old app.py code using model_path does not crash.
        model_path: Optional[str | Path] = None,
    ):
        self.api_key = api_key or os.getenv("ROBOFLOW_API_KEY", "").strip()
        self.model_id = model_id or os.getenv("ROBOFLOW_MODEL_ID", "").strip()
        self.api_url = (api_url or os.getenv("ROBOFLOW_API_URL", "https://serverless.roboflow.com")).rstrip("/")
        self.confidence = float(confidence)
        self.overlap = float(overlap)
        self.max_detections = int(max_detections)
        self.filter_traffic_classes = filter_traffic_classes

        if not self.api_key:
            raise ValueError(
                "Missing Roboflow API key. Set ROBOFLOW_API_KEY in your .env file."
            )

        if not self.model_id:
            raise ValueError(
                "Missing Roboflow model id. Example: helmet-detection-traffic-cgg9e/2. Set the model id in your .env file."
            )

    @staticmethod
    def _is_traffic_class(class_name: str) -> bool:
        lower_name = class_name.lower().strip().replace("-", "_")
        return any(keyword in lower_name for keyword in TRAFFIC_KEYWORDS)

    @staticmethod
    def _image_to_jpeg_bytes(image: Any) -> bytes:
        """Convert a PIL/numpy image to JPEG bytes for API upload."""

        if isinstance(image, Image.Image):
            pil_image = image.convert("RGB")
        else:
            pil_image = Image.fromarray(image).convert("RGB")

        buffer = BytesIO()
        pil_image.save(buffer, format="JPEG", quality=92)
        return buffer.getvalue()

    @staticmethod
    def _prediction_to_detection(prediction: Dict[str, Any]) -> Dict[str, Any]:
        """Convert Roboflow center-box prediction into xyxy bbox format."""

        x_center = float(prediction.get("x", 0))
        y_center = float(prediction.get("y", 0))
        width = float(prediction.get("width", 0))
        height = float(prediction.get("height", 0))

        x1 = x_center - width / 2
        y1 = y_center - height / 2
        x2 = x_center + width / 2
        y2 = y_center + height / 2

        return {
            "bbox": [round(x1, 2), round(y1, 2), round(x2, 2), round(y2, 2)],
            "confidence": round(float(prediction.get("confidence", 0.0)), 4),
            "class_id": prediction.get("class_id", prediction.get("class", "")),
            "class_name": str(prediction.get("class", prediction.get("class_name", "unknown"))),
            "roboflow_detection_id": prediction.get("detection_id", ""),
        }

    def _parse_predictions(self, result: Any) -> List[Dict[str, Any]]:
        """Find predictions in common Roboflow model/workflow response shapes."""

        if isinstance(result, list):
            # Some workflow responses return a list with one dictionary inside.
            merged: List[Dict[str, Any]] = []
            for item in result:
                merged.extend(self._parse_predictions(item))
            return merged

        if not isinstance(result, dict):
            return []

        predictions = result.get("predictions")

        # Standard object detection response:
        # {"predictions": [{"x": ..., "y": ..., "class": ...}]}
        if isinstance(predictions, list):
            return predictions

        # Some workflow responses nest predictions:
        # {"predictions": {"predictions": [...]}}
        if isinstance(predictions, dict):
            nested = predictions.get("predictions")
            if isinstance(nested, list):
                return nested

        # Roboflow workflows may use custom output field names.
        for value in result.values():
            if isinstance(value, dict):
                nested_predictions = value.get("predictions")
                if isinstance(nested_predictions, list):
                    return nested_predictions
            if isinstance(value, list) and value and isinstance(value[0], dict):
                if {"x", "y", "width", "height"}.issubset(value[0].keys()):
                    return value

        return []

    def _detect_with_inference_sdk(self, image: Any) -> Dict[str, Any]:
        """Call Roboflow using inference-sdk when installed."""

        try:
            from inference_sdk import InferenceHTTPClient
        except ImportError as exc:
            raise RuntimeError("inference-sdk is not installed") from exc

        # inference-sdk is happiest with a file path, so we write a temp JPEG.
        image_bytes = self._image_to_jpeg_bytes(image)
        with tempfile.NamedTemporaryFile(suffix=".jpg", delete=False) as tmp_file:
            tmp_file.write(image_bytes)
            tmp_path = tmp_file.name

        try:
            client = InferenceHTTPClient(api_url=self.api_url, api_key=self.api_key)
            return client.infer(tmp_path, model_id=self.model_id)
        finally:
            try:
                os.remove(tmp_path)
            except OSError:
                pass

    def _detect_with_rest_api(self, image: Any) -> Dict[str, Any]:
        """Fallback direct REST call if inference-sdk is unavailable."""

        image_bytes = self._image_to_jpeg_bytes(image)
        image_base64 = base64.b64encode(image_bytes).decode("utf-8")

        url = f"{self.api_url}/{self.model_id.strip('/')}"
        params = {
            "api_key": self.api_key,
            "confidence": self.confidence,
            "overlap": self.overlap,
            "format": "json",
            "image_type": "base64",
            "max_detections": self.max_detections,
        }

        # Roboflow legacy hosted endpoints commonly accept raw base64 in the body.
        response = requests.post(
            url,
            params=params,
            data=image_base64,
            headers={"Content-Type": "application/x-www-form-urlencoded"},
            timeout=60,
        )

        if response.status_code >= 400:
            raise RuntimeError(
                f"Roboflow API error {response.status_code}: {response.text[:1000]}"
            )

        return response.json()

    def detect(self, image: Any) -> List[Dict[str, Any]]:
        """Run Roboflow detection and return app-standard detection dictionaries."""

        try:
            result = self._detect_with_inference_sdk(image)
        except Exception:
            # Keep the app usable even if inference-sdk is not installed.
            result = self._detect_with_rest_api(image)

        raw_predictions = self._parse_predictions(result)

        detections: List[Dict[str, Any]] = []
        for prediction in raw_predictions:
            detection = self._prediction_to_detection(prediction)

            if self.filter_traffic_classes and not self._is_traffic_class(detection["class_name"]):
                continue

            detections.append(detection)

        return detections
