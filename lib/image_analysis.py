"""
Deterministic image analysis for ad text placement.

Pure OpenCV + NumPy. No network. No Claude calls. Runs in ~20-40ms per image
on a 1080x1350 source. Returns the `image_analysis` sub-object of
text_design_spec, used downstream to:

    - pick quiet zones for text placement
    - suggest light-vs-dark text color
    - classify dominant hue for color_strategy
    - provide a 5-color palette for brand-accent decisions
"""

from __future__ import annotations

from typing import Any

import cv2
import numpy as np

# 9-cell zone keys (match lib/ad_design_system.py cell zones, minus band zones)
_ZONE_KEYS_3X3 = [
    "top_left",    "top_center",    "top_right",
    "middle_left", "center",        "middle_right",
    "bottom_left", "bottom_center", "bottom_right",
]


def _luminance(bgr: np.ndarray) -> np.ndarray:
    """Return a single-channel float32 luminance image in [0, 1]."""
    yuv = cv2.cvtColor(bgr, cv2.COLOR_BGR2YUV)
    y = yuv[:, :, 0].astype(np.float32) / 255.0
    return y


def _edge_energy(bgr: np.ndarray) -> np.ndarray:
    """Sobel edge magnitude normalized to [0, 1]. A 'busyness' map."""
    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    gx = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)
    mag = cv2.magnitude(gx, gy)
    m = float(mag.max()) if mag.size else 1.0
    if m <= 1e-6:
        return np.zeros_like(mag, dtype=np.float32)
    return (mag / m).astype(np.float32)


def _grid_means(img: np.ndarray) -> list[float]:
    """Mean of each cell of a 3x3 grid. Returns 9 floats in row-major order."""
    h, w = img.shape[:2]
    out: list[float] = []
    for row in range(3):
        for col in range(3):
            y0 = (row * h) // 3
            y1 = ((row + 1) * h) // 3
            x0 = (col * w) // 3
            x1 = ((col + 1) * w) // 3
            cell = img[y0:y1, x0:x1]
            if cell.size == 0:
                out.append(0.0)
            else:
                out.append(float(cell.mean()))
    return out


def _dominant_palette(bgr: np.ndarray, k: int = 5) -> list[str]:
    """Return k dominant colors as #RRGGBB strings via k-means."""
    small = cv2.resize(bgr, (96, 96), interpolation=cv2.INTER_AREA)
    pixels = small.reshape(-1, 3).astype(np.float32)
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0)
    try:
        _, labels, centers = cv2.kmeans(
            pixels, k, None, criteria, 3, cv2.KMEANS_PP_CENTERS
        )
    except cv2.error:
        return ["#888888"] * k

    centers = centers.astype(np.int32)
    counts = np.bincount(labels.flatten(), minlength=k)
    order = np.argsort(-counts)
    palette: list[str] = []
    for idx in order:
        b, g, r = centers[idx]
        palette.append(f"#{int(r):02X}{int(g):02X}{int(b):02X}")
    return palette


def _classify_hue(bgr: np.ndarray) -> str:
    """Classify dominant hue into a coarse bucket."""
    hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
    h, s, v = cv2.split(hsv)
    mask = (s > 25) & (v > 25)
    if not mask.any():
        return "neutral"
    mean_h = float(h[mask].mean())
    mean_s = float(s[mask].mean()) / 255.0
    if mean_s < 0.15:
        return "neutral"
    # OpenCV hue is 0..179
    if mean_h < 10 or mean_h >= 170:
        return "warm_red"
    if mean_h < 22:
        return "warm_orange"
    if mean_h < 38:
        return "warm_yellow"
    if mean_h < 78:
        return "cool_green"
    if mean_h < 100:
        return "cool_teal"
    if mean_h < 130:
        return "cool_blue"
    if mean_h < 150:
        return "cool_violet"
    return "warm_magenta"


def analyze_image(bgr: np.ndarray) -> dict[str, Any]:
    """Return the image_analysis sub-object of text_design_spec.

    Safe on unusual inputs; returns neutral defaults on failure.
    """
    try:
        if bgr is None or bgr.size == 0:
            return _neutral_defaults()

        small = cv2.resize(bgr, (270, 338), interpolation=cv2.INTER_AREA)
        lum = _luminance(small)
        busy = _edge_energy(small)

        brightness = float(lum.mean())
        contrast = float(lum.std())
        dominant_hue = _classify_hue(small)
        palette = _dominant_palette(small, k=5)

        lum_grid = _grid_means(lum)
        busy_grid = _grid_means(busy)

        zone_brightness = {k: round(lum_grid[i], 3) for i, k in enumerate(_ZONE_KEYS_3X3)}
        zone_busyness   = {k: round(busy_grid[i], 3) for i, k in enumerate(_ZONE_KEYS_3X3)}

        # quiet_score = low busyness (invert), ignore extreme-dark/extreme-bright cells
        quiet_scores = {}
        for k in _ZONE_KEYS_3X3:
            q = 1.0 - zone_busyness[k]
            # penalty for mid-luminance (text contrast is harder)
            lb = zone_brightness[k]
            if 0.35 < lb < 0.65:
                q -= 0.15
            quiet_scores[k] = round(max(0.0, q), 3)

        sorted_zones = sorted(quiet_scores.items(), key=lambda kv: kv[1], reverse=True)
        quietest_zones = [k for k, _ in sorted_zones[:3]]
        busiest_zones  = [k for k, _ in sorted_zones[-3:]]

        # Suggested text color: if overall image is brighter than 0.55, dark text
        suggested_text_color = "dark" if brightness > 0.55 else "light"

        # Extract vibrant accent color for eyebrows, emphasis, CTA
        accent_color = _extract_accent_color(small)
        accent_usage = _suggest_accent_usage(dominant_hue, brightness)

        return {
            "brightness":           round(brightness, 3),
            "contrast":             round(contrast, 3),
            "dominant_hue":         dominant_hue,
            "dominant_palette":     palette,
            "zone_brightness":      zone_brightness,
            "zone_busyness":        zone_busyness,
            "quietest_zones":       quietest_zones,
            "busiest_zones":        busiest_zones,
            "suggested_text_color": suggested_text_color,
            "accent_color":         accent_color,
            "accent_usage":         accent_usage,
        }
    except Exception as exc:
        print(f"[IMAGE_ANALYSIS] failed, returning defaults: {exc}")
        return _neutral_defaults()


def _extract_accent_color(bgr: np.ndarray) -> str:
    """Extract a vibrant accent color suitable for eyebrows/emphasis.

    Looks for saturated, high-value pixels. If found, returns a vibrant accent.
    Otherwise returns a muted neutral.
    """
    try:
        hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
        h, s, v = cv2.split(hsv)

        # High saturation + moderate-to-high value = good accent candidates
        mask = (s > 70) & (v > 100)

        if mask.sum() > bgr.size // 50:  # at least 2% of image
            # Mean of saturated pixels
            mean_h = float(h[mask].mean())
            mean_s = float(min(255, s[mask].mean() + 30))  # boost saturation
            mean_v = float(min(255, v[mask].mean()))
        else:
            # Fall back to most-saturated pixels in the image
            idx = np.argsort(s.flat)[-100:]  # top 100 most saturated pixels
            mean_h = float(h.flat[idx].mean())
            mean_s = float(s.flat[idx].mean())
            mean_v = float(v.flat[idx].mean())

        # Convert back to BGR
        pixel_hsv = np.uint8([[[int(mean_h), int(mean_s), int(mean_v)]]])
        pixel_bgr = cv2.cvtColor(pixel_hsv, cv2.COLOR_HSV2BGR)[0][0]
        b, g, r = int(pixel_bgr[0]), int(pixel_bgr[1]), int(pixel_bgr[2])
        return f"#{r:02X}{g:02X}{b:02X}"
    except Exception:
        return "#888888"  # neutral fallback


def _suggest_accent_usage(hue: str, brightness: float) -> str:
    """Suggest where to use the accent color based on image mood."""
    # If image is very bright, use accent sparingly (eyebrow only)
    # If image is dark, can use accent more freely (eyebrow + emphasis)
    if brightness > 0.7:
        return "eyebrow"
    elif brightness > 0.5:
        return "eyebrow_and_emphasis"
    else:
        return "eyebrow_and_cta"


def _neutral_defaults() -> dict[str, Any]:
    return {
        "brightness":           0.5,
        "contrast":             0.25,
        "dominant_hue":         "neutral",
        "dominant_palette":     ["#888888", "#AAAAAA", "#666666", "#CCCCCC", "#444444"],
        "zone_brightness":      {k: 0.5 for k in _ZONE_KEYS_3X3},
        "zone_busyness":        {k: 0.3 for k in _ZONE_KEYS_3X3},
        "quietest_zones":       ["bottom_center", "top_center", "bottom_left"],
        "busiest_zones":        ["center", "middle_left", "middle_right"],
        "suggested_text_color": "light",
        "accent_color":         "#666666",
        "accent_usage":         "eyebrow",
    }
