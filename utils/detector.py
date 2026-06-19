"""Detector backends for IntelliTraffic AI.

Two backends are available:

LocalYOLODetector (recommended)
    Uses Ultralytics YOLOv8 with pre-trained weights that download
    automatically on first run. No API key required.

    Models used (see utils/model_registry.py for details):
        base_coco     – YOLOv8n COCO  (vehicles, persons)
        helmet        – YOLOv8s protective-equipment (helmet/no_helmet)
        license_plate – YOLOv8n license-plate

TrafficDetector (Roboflow API fallback)
    Sends the image to Roboflow Hosted/Serverless Inference.
    Requires ROBOFLOW_API_KEY and model IDs in .env.

Usage in app.py
---------------
    from utils.detector import LocalYOLODetector, TrafficDetector

    # --- Local (no API key needed) ---
    detector = LocalYOLODetector()
    helmet_dets, plate_dets, vehicle_dets = detector.detect_all(image)

    # --- Roboflow API fallback ---
    detector = TrafficDetector(api_key=..., model_id=..., api_url=...)
    dets = detector.detect(image)
"""

from __future__ import annotations

import base64
import os
import tempfile
from io import BytesIO
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import requests
from PIL import Image

from utils.model_registry import COCO_TRAFFIC_CLASSES, ModelConfig, get_model_config, resolve_model_path


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------

TRAFFIC_KEYWORDS = (
    "person", "pedestrian", "bicycle", "bike", "car", "motorcycle",
    "motorbike", "bus", "truck", "vehicle", "traffic light", "traffic_light",
    "signal", "license_plate", "license plate", "licence_plate", "number_plate",
    "number plate", "plate", "helmet", "no_helmet", "no helmet", "no-helmet",
)


def _image_to_jpeg_bytes(image: Any) -> bytes:
    if isinstance(image, Image.Image):
        pil_image = image.convert("RGB")
    else:
        pil_image = Image.fromarray(image).convert("RGB")
    buffer = BytesIO()
    pil_image.save(buffer, format="JPEG", quality=92)
    return buffer.getvalue()


def _xyxy_detection(
    x1: float, y1: float, x2: float, y2: float,
    confidence: float,
    class_id: int,
    class_name: str,
) -> Dict[str, Any]:
    return {
        "bbox": [round(x1, 2), round(y1, 2), round(x2, 2), round(y2, 2)],
        "confidence": round(float(confidence), 4),
        "class_id": int(class_id),
        "class_name": str(class_name),
    }


# ---------------------------------------------------------------------------
# Local YOLO detector (ultralytics + HuggingFace weights)
# ---------------------------------------------------------------------------

class LocalYOLODetector:
    """Run all three detection pipelines locally using free pre-trained models.

    Models are downloaded automatically on first use and cached to
    ``~/.cache/ultralytics`` (or the Ultralytics default cache).

    Example::

        detector = LocalYOLODetector(confidence=0.30)
        helmet_dets, plate_dets, vehicle_dets = detector.detect_all(pil_image)
    """

    def __init__(
        self,
        confidence: float = 0.25,
        iou_threshold: float = 0.45,
        device: str = "cpu",
    ) -> None:
        self.confidence = float(confidence)
        self.iou = float(iou_threshold)
        self.device = device
        self._models: Dict[str, Any] = {}

    # ------------------------------------------------------------------
    # Lazy model loading
    # ------------------------------------------------------------------

    def _load(self, key: str) -> Any:
        """Lazily load a YOLO model by registry key."""
        if key not in self._models:
            try:
                from ultralytics import YOLO
            except ImportError as exc:
                raise RuntimeError(
                    "ultralytics is not installed. Run: pip install ultralytics"
                ) from exc

            config = get_model_config(key)
            model_path = resolve_model_path(config)
            self._models[key] = YOLO(model_path)
        return self._models[key]

    # ------------------------------------------------------------------
    # Internal inference helper
    # ------------------------------------------------------------------

    def _infer(
        self,
        model_key: str,
        image: Image.Image,
        class_filter: Optional[List[str]] = None,
        override_class_names: Optional[Dict[int, str]] = None,
    ) -> List[Dict[str, Any]]:
        """Run one YOLO model and return normalised detection dicts."""
        model = self._load(model_key)
        results = model.predict(
            source=image,
            conf=self.confidence,
            iou=self.iou,
            device=self.device,
            verbose=False,
        )

        detections: List[Dict[str, Any]] = []
        for result in results:
            boxes = result.boxes
            if boxes is None:
                continue
            names: Dict[int, str] = result.names or {}
            if override_class_names:
                names = {**names, **override_class_names}

            for box in boxes:
                cls_id = int(box.cls[0].item())
                cls_name = names.get(cls_id, str(cls_id))
                conf = float(box.conf[0].item())
                x1, y1, x2, y2 = box.xyxy[0].tolist()

                # Apply class filter
                if class_filter and cls_name.lower() not in [c.lower() for c in class_filter]:
                    continue

                detections.append(
                    _xyxy_detection(x1, y1, x2, y2, conf, cls_id, cls_name)
                )

        return detections

    # ------------------------------------------------------------------
    # Public detection methods
    # ------------------------------------------------------------------

    def detect_vehicles_and_persons(self, image: Image.Image) -> List[Dict[str, Any]]:
        """Detect vehicles and persons using COCO-pretrained YOLOv8n."""
        return self._infer(
            "base_coco",
            image,
            class_filter=list(COCO_TRAFFIC_CLASSES.values()),
            override_class_names=COCO_TRAFFIC_CLASSES,
        )

    def detect_helmets(self, image: Image.Image) -> List[Dict[str, Any]]:
        """Detect helmet / no_helmet using the protective-equipment model."""
        return self._infer("helmet", image, class_filter=["helmet", "no_helmet"])

    def detect_license_plates(self, image: Image.Image) -> List[Dict[str, Any]]:
        """Detect license plate bounding boxes."""
        return self._infer("license_plate", image, class_filter=["license_plate"])

    def detect_all(
        self,
        image: Image.Image,
    ) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]], List[Dict[str, Any]]]:
        """Run all three models and return (helmet_dets, plate_dets, vehicle_dets).

        helmet_dets  – helmet + no_helmet detections (from the helmet model)
        plate_dets   – license_plate detections
        vehicle_dets – car, motorcycle, bus, truck, bicycle, person (COCO)
        """
        vehicle_dets = self.detect_vehicles_and_persons(image)
        helmet_dets = self.detect_helmets(image)
        plate_dets = self.detect_license_plates(image)
        return helmet_dets, plate_dets, vehicle_dets


# ---------------------------------------------------------------------------
# Roboflow API detector (legacy / fallback)
# ---------------------------------------------------------------------------

class TrafficDetector:
    """Roboflow API wrapper (original implementation, kept for compatibility).

    Requires ROBOFLOW_API_KEY, ROBOFLOW_HELMET_MODEL_ID,
    ROBOFLOW_PLATE_MODEL_ID in the .env file.

    The app uses LocalYOLODetector by default; this is used when the user
    explicitly selects "Roboflow API" mode in the sidebar.
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
        model_path: Optional[str | Path] = None,  # legacy arg, unused
    ):
        self.api_key = api_key or os.getenv("ROBOFLOW_API_KEY", "").strip()
        self.model_id = model_id or os.getenv("ROBOFLOW_MODEL_ID", "").strip()
        self.api_url = (
            api_url or os.getenv("ROBOFLOW_API_URL", "https://serverless.roboflow.com")
        ).rstrip("/")
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
                "Missing Roboflow model id. Set the model id in your .env file."
            )

    @staticmethod
    def _is_traffic_class(class_name: str) -> bool:
        lower_name = class_name.lower().strip().replace("-", "_")
        return any(keyword in lower_name for keyword in TRAFFIC_KEYWORDS)

    @staticmethod
    def _prediction_to_detection(prediction: Dict[str, Any]) -> Dict[str, Any]:
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
        if isinstance(result, list):
            merged: List[Dict[str, Any]] = []
            for item in result:
                merged.extend(self._parse_predictions(item))
            return merged
        if not isinstance(result, dict):
            return []
        predictions = result.get("predictions")
        if isinstance(predictions, list):
            return predictions
        if isinstance(predictions, dict):
            nested = predictions.get("predictions")
            if isinstance(nested, list):
                return nested
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
        try:
            from inference_sdk import InferenceHTTPClient
        except ImportError as exc:
            raise RuntimeError("inference-sdk is not installed") from exc

        image_bytes = _image_to_jpeg_bytes(image)
        with tempfile.NamedTemporaryFile(suffix=".jpg", delete=False) as tmp:
            tmp.write(image_bytes)
            tmp_path = tmp.name
        try:
            client = InferenceHTTPClient(api_url=self.api_url, api_key=self.api_key)
            return client.infer(tmp_path, model_id=self.model_id)
        finally:
            try:
                os.remove(tmp_path)
            except OSError:
                pass

    def _detect_with_rest_api(self, image: Any) -> Dict[str, Any]:
        image_bytes = _image_to_jpeg_bytes(image)
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
        try:
            result = self._detect_with_inference_sdk(image)
        except Exception:
            result = self._detect_with_rest_api(image)

        raw_predictions = self._parse_predictions(result)
        detections: List[Dict[str, Any]] = []
        for prediction in raw_predictions:
            detection = self._prediction_to_detection(prediction)
            if self.filter_traffic_classes and not self._is_traffic_class(
                detection["class_name"]
            ):
                continue
            detections.append(detection)
        return detections
