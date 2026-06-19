"""Model registry for IntelliTraffic AI.

Defines the catalogue of pre-trained models used by LocalYOLODetector.
All models are free, publicly available, and download automatically on
first use via huggingface_hub + ultralytics.

Model choices
─────────────
BASE_COCO
    YOLOv8-nano pretrained on MS-COCO (ultralytics official release).
    Detects person(0), bicycle(1), car(2), motorcycle(3), bus(5), truck(7).
    Used for: vehicle counting, person counting, triple-riding.

HELMET_PROTECTIVE
    keremberke/yolov8s-protective-equipment-detection (Hugging Face).
    Detects helmet, no_helmet, mask, no_mask, gloves, etc.
    Used for: helmet non-compliance.

LICENSE_PLATE
    Koushim/yolov8-license-plate-detection (Hugging Face).
    Detects license_plate (single class).
    Used for: plate bounding box → OCR.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional


# ---------------------------------------------------------------------------
# COCO class-id → friendly name (subset relevant to traffic)
# ---------------------------------------------------------------------------
COCO_TRAFFIC_CLASSES: Dict[int, str] = {
    0: "person",
    1: "bicycle",
    2: "car",
    3: "motorcycle",
    5: "bus",
    7: "truck",
}


# ---------------------------------------------------------------------------
# Model descriptors
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class ModelConfig:
    """Immutable descriptor for a downloadable YOLO model."""

    key: str                    # internal registry key
    source: str                 # ultralytics model name (e.g. "yolov8n.pt") for CDN models
    description: str
    classes_of_interest: List[str] = field(default_factory=list)
    # If non-empty, only detections whose class name is in this set are kept.
    class_filter: List[str] = field(default_factory=list)
    # HuggingFace download info (used when source is not a direct ultralytics model)
    hf_repo_id: Optional[str] = None       # e.g. "keremberke/yolov8s-protective-equipment-detection"
    hf_filename: str = "best.pt"           # weights filename in the HF repo


MODELS: Dict[str, ModelConfig] = {
    # ------------------------------------------------------------------ #
    # General traffic scene: vehicles, persons                            #
    # ------------------------------------------------------------------ #
    "base_coco": ModelConfig(
        key="base_coco",
        source="yolov8n.pt",  # ultralytics auto-downloads from their CDN
        description="YOLOv8-nano COCO (vehicles + persons)",
        classes_of_interest=["person", "bicycle", "car", "motorcycle", "bus", "truck"],
        class_filter=["person", "bicycle", "car", "motorcycle", "bus", "truck"],
    ),

    # ------------------------------------------------------------------ #
    # Helmet / protective equipment                                        #
    # Downloaded via huggingface_hub, then loaded as a local .pt file     #
    # ------------------------------------------------------------------ #
    "helmet": ModelConfig(
        key="helmet",
        source="",  # resolved at runtime via hf_repo_id
        description="YOLOv8-small protective-equipment (helmet / no_helmet)",
        classes_of_interest=["helmet", "no_helmet"],
        class_filter=["helmet", "no_helmet"],
        hf_repo_id="keremberke/yolov8s-protective-equipment-detection",
        hf_filename="best.pt",
    ),

    # ------------------------------------------------------------------ #
    # License plate                                                        #
    # ------------------------------------------------------------------ #
    "license_plate": ModelConfig(
        key="license_plate",
        source="",  # resolved at runtime via hf_repo_id
        description="YOLOv8 license-plate detector (Koushim)",
        classes_of_interest=["license_plate"],
        class_filter=["license_plate"],
        hf_repo_id="Koushim/yolov8-license-plate-detection",
        hf_filename="best.pt",
    ),
}


def get_model_config(key: str) -> ModelConfig:
    """Retrieve a ModelConfig by key, raising KeyError on unknown keys."""
    if key not in MODELS:
        raise KeyError(
            f"Unknown model key '{key}'. Available: {list(MODELS.keys())}"
        )
    return MODELS[key]


def resolve_model_path(config: ModelConfig) -> str:
    """Return the local filesystem path to the model weights.

    For ultralytics CDN models (base_coco), returns the model name directly
    (e.g. "yolov8n.pt") — ultralytics handles the download.

    For HuggingFace models, uses huggingface_hub to download the weights
    and returns the cached local path.
    """
    if config.hf_repo_id:
        from huggingface_hub import hf_hub_download
        return hf_hub_download(
            repo_id=config.hf_repo_id,
            filename=config.hf_filename,
        )
    return config.source
