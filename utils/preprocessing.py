"""Adaptive image preprocessing for IntelliTraffic AI.

This module applies traditional computer vision enhancement techniques to
normalize input images before object detection. It handles:

    - Low-light conditions (CLAHE, gamma correction)
    - Motion blur / soft focus (unsharp masking)
    - Haze / fog (dark channel prior dehazing)
    - General noise (bilateral filtering)

Usage:
    from utils.preprocessing import adaptive_preprocess, PreprocessConfig
    enhanced = adaptive_preprocess(pil_image, config=PreprocessConfig())
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Tuple

import numpy as np
from PIL import Image


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

@dataclass
class PreprocessConfig:
    """Runtime configuration for the adaptive preprocessing pipeline."""

    # Low-light enhancement
    apply_clahe: bool = True
    clahe_clip_limit: float = 2.5
    clahe_grid_size: Tuple[int, int] = (8, 8)

    # Gamma correction (< 1.0 brightens, > 1.0 darkens)
    apply_gamma: bool = True
    gamma: float = 0.85

    # Sharpening / deblur
    apply_sharpen: bool = True
    sharpen_strength: float = 1.2  # multiplied against the unsharp mask

    # Dehazing
    apply_dehaze: bool = False  # off by default; expensive on large images

    # Noise removal
    apply_denoise: bool = False  # bilateral filter; useful for low-res cams


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _to_bgr(image: Image.Image) -> np.ndarray:
    """Convert PIL RGB image to OpenCV BGR array."""
    return np.array(image.convert("RGB"))[:, :, ::-1]


def _to_pil(bgr: np.ndarray) -> Image.Image:
    """Convert OpenCV BGR array back to PIL RGB image."""
    return Image.fromarray(bgr[:, :, ::-1])


def _clahe(bgr: np.ndarray, clip_limit: float, grid_size: Tuple[int, int]) -> np.ndarray:
    """Apply CLAHE on the L-channel of the LAB color space."""
    try:
        import cv2

        lab = cv2.cvtColor(bgr, cv2.COLOR_BGR2LAB)
        l_channel, a_channel, b_channel = cv2.split(lab)
        clahe_obj = cv2.createCLAHE(
            clipLimit=clip_limit,
            tileGridSize=grid_size,
        )
        l_enhanced = clahe_obj.apply(l_channel)
        lab_enhanced = cv2.merge((l_enhanced, a_channel, b_channel))
        return cv2.cvtColor(lab_enhanced, cv2.COLOR_LAB2BGR)
    except ImportError:
        # Fallback: histogram stretch on each channel
        result = bgr.copy().astype(np.float32)
        for ch in range(3):
            p2, p98 = np.percentile(result[:, :, ch], [2, 98])
            if p98 > p2:
                result[:, :, ch] = np.clip(
                    (result[:, :, ch] - p2) / (p98 - p2) * 255, 0, 255
                )
        return result.astype(np.uint8)


def _gamma_correction(bgr: np.ndarray, gamma: float) -> np.ndarray:
    """Apply power-law gamma correction."""
    if abs(gamma - 1.0) < 1e-3:
        return bgr
    lut = np.array(
        [((i / 255.0) ** gamma) * 255 for i in range(256)],
        dtype=np.uint8,
    )
    return lut[bgr]


def _sharpen(bgr: np.ndarray, strength: float) -> np.ndarray:
    """Unsharp masking: blurred subtracted from original, then added back."""
    try:
        import cv2

        blur = cv2.GaussianBlur(bgr, (0, 0), sigmaX=3)
        sharpened = cv2.addWeighted(bgr, 1 + strength, blur, -strength, 0)
        return sharpened
    except ImportError:
        # Pure numpy: simple 3x3 laplacian add
        kernel = np.array([
            [0, -0.5, 0],
            [-0.5, 3, -0.5],
            [0, -0.5, 0],
        ])
        from scipy.ndimage import convolve
        result = bgr.astype(np.float32)
        for ch in range(3):
            result[:, :, ch] = convolve(result[:, :, ch], kernel)
        return np.clip(result, 0, 255).astype(np.uint8)


def _dehaze(bgr: np.ndarray) -> np.ndarray:
    """Dark channel prior dehazing (simplified, single-scale version)."""
    try:
        import cv2

        img_float = bgr.astype(np.float32) / 255.0
        # Dark channel: min over patch of size 15x15
        min_channel = np.min(img_float, axis=2)
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (15, 15))
        dark_channel = cv2.erode(min_channel, kernel)

        # Estimate atmospheric light (top 0.1% brightest pixels in dark channel)
        flat_dark = dark_channel.flatten()
        num_pixels = max(1, int(len(flat_dark) * 0.001))
        indices = np.argsort(flat_dark)[-num_pixels:]
        rows, cols = np.unravel_index(indices, dark_channel.shape)
        atm_light = np.mean(img_float[rows, cols, :], axis=0)
        atm_light = np.clip(atm_light, 0.8, 1.0)

        # Estimate transmission map
        omega = 0.85
        normalized = img_float / atm_light
        normalized_dark = np.min(normalized, axis=2)
        normalized_dark_erode = cv2.erode(normalized_dark, kernel)
        transmission = 1 - omega * normalized_dark_erode

        # Refine transmission using guided filter (simple box filter approximation)
        transmission = cv2.blur(transmission, (5, 5))
        transmission = np.clip(transmission, 0.1, 1.0)[:, :, np.newaxis]

        # Recover scene radiance
        t0 = 0.1
        recovered = (img_float - atm_light) / np.maximum(transmission, t0) + atm_light
        recovered = np.clip(recovered * 255, 0, 255).astype(np.uint8)
        return recovered
    except Exception:
        return bgr


def _denoise(bgr: np.ndarray) -> np.ndarray:
    """Bilateral filter for edge-preserving noise reduction."""
    try:
        import cv2

        return cv2.bilateralFilter(bgr, d=9, sigmaColor=75, sigmaSpace=75)
    except ImportError:
        return bgr


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def adaptive_preprocess(
    image: Image.Image,
    config: PreprocessConfig | None = None,
) -> Image.Image:
    """Apply the adaptive enhancement pipeline to a PIL image.

    Steps (applied in order, each controlled by its config flag):
        1. CLAHE on the L channel (luminance equalisation)
        2. Gamma correction (exposure control)
        3. Sharpening / deblur via unsharp masking
        4. Dehazing via dark channel prior (optional, costly)
        5. Bilateral denoising (optional)

    Args:
        image: Input PIL image (any mode; converted to RGB internally).
        config: PreprocessConfig instance. Uses defaults if None.

    Returns:
        Enhanced PIL image in RGB mode.
    """
    if config is None:
        config = PreprocessConfig()

    bgr = _to_bgr(image)

    if config.apply_clahe:
        bgr = _clahe(bgr, config.clahe_clip_limit, config.clahe_grid_size)

    if config.apply_gamma:
        bgr = _gamma_correction(bgr, config.gamma)

    if config.apply_sharpen:
        bgr = _sharpen(bgr, config.sharpen_strength)

    if config.apply_dehaze:
        bgr = _dehaze(bgr)

    if config.apply_denoise:
        bgr = _denoise(bgr)

    return _to_pil(bgr)


def compute_image_hash(image_bytes: bytes) -> str:
    """Compute SHA-256 hash of raw image bytes for evidence integrity.

    Args:
        image_bytes: Raw bytes of the uploaded image file.

    Returns:
        Lowercase hex string of the SHA-256 digest, e.g.
        'a3f2...8c91'
    """
    import hashlib

    return hashlib.sha256(image_bytes).hexdigest()
