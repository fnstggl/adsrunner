"""
Wrapper around the existing greenscreen compositor (compositor_v4_final.py).
Composites uploaded UI screenshots into greenscreen device placeholders.
"""

import sys
import os

import cv2
import numpy as np

# Add project root to path so we can import the compositor
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import compositor_v4_final as compositor


def compose_ui_into_greenscreen(
    scene: np.ndarray,
    ui_screenshot: np.ndarray,
    feather: int = 3,
) -> np.ndarray:
    """
    Composite a UI screenshot into a greenscreen device placeholder in the scene.

    Args:
        scene: BGR image with a greenscreen device placeholder
        ui_screenshot: BGR image of the UI to composite in
        feather: Edge feathering radius (default 3)

    Returns:
        BGR composited image, or the original scene if no greenscreen is found
    """
    # Detect greenscreen corners using the existing CV-based detector
    corners, blend_mask = compositor.detect_green_corners(scene)

    if corners is None:
        print("[COMPOSITE] No greenscreen detected in scene — returning original")
        return scene

    print(f"[COMPOSITE] Greenscreen detected at corners: {corners.tolist()}")

    # Use the existing compositor to warp UI into the screen quad
    result = compositor.composite(
        scene, ui_screenshot, corners,
        feather=feather,
        blend_mask=blend_mask,
    )

    return result


def needs_compositing(scene: np.ndarray) -> bool:
    """Check if a scene image contains a greenscreen placeholder."""
    corners, _ = compositor.detect_green_corners(scene)
    return corners is not None
