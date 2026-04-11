"""
Deterministic layout token computation.

Given design intent (layout family, zone, hierarchy scale) and image analysis,
compute ALL px-level layout decisions: widths, font sizes, spacing, colors.

Claude uses only these tokens; never invents layout values.
"""

from __future__ import annotations

from typing import Any

import lib.ad_design_system as ads


def compute_layout_tokens(
    layout_family: str,
    placement_zone: str,
    hierarchy_scale: str,
    text_design_spec: dict[str, Any],
    image_analysis: dict[str, Any],
) -> dict[str, Any]:
    """Deterministically compute all layout px values for a given design context.

    Args:
        layout_family: e.g. "editorial_headline_only", "product_headline_cta"
        placement_zone: e.g. "bottom_center", "center", "top_left"
        hierarchy_scale: "md" | "lg" | "xl" | "xxl"
        text_design_spec: The full text_design_spec dict
        image_analysis: The image_analysis dict with brightness, accent_color, palette

    Returns:
        dict with all layout tokens: zone_rect, usable_width, headline_size_range,
        support_size_range, eyebrow_size, cta_size, spacing_unit, gaps, line_heights,
        colors (headline, support, accent, cta_bg, cta_fg), etc.
    """
    try:
        # ========== Zone Geometry ==========
        zone_rect = ads.zone_bounds(placement_zone)
        safe_margin = 16  # px margin inside zone boundary

        usable_width = max(120, zone_rect["w"] - (2 * safe_margin))
        zone_height = zone_rect["h"]

        # ========== Headline Sizing (word-count driven) ==========
        headline_content = text_design_spec.get("text_elements", {}).get("headline", {}).get("content", "")
        word_count = len(headline_content.split()) if headline_content.strip() else 3

        hl_min_px, hl_max_px = ads.headline_font_size_range(word_count)

        # Clamp to usable_width (rough check: ~5 chars per px at max size)
        hl_max_px = min(hl_max_px, max(100, int(usable_width / 2.5)))
        hl_min_px = min(hl_min_px, hl_max_px)

        # ========== Support Copy Sizing ==========
        support_min_px, support_max_px = ads.support_copy_font_size(hl_max_px)

        # Eyebrow and CTA sizes (fixed from design system)
        eyebrow_size_px = ads.eyebrow_font_size()
        cta_size_px = 44  # standard CTA font size

        # ========== Line Heights ==========
        headline_line_height = 1.0   # tight headlines
        eyebrow_line_height = 1.2
        support_line_height = 1.4    # more spacious body
        cta_line_height = 1.0

        # ========== Spacing Scale (rhythm-based) ==========
        spacing_scales = {
            "md": {"unit": 6, "gaps": {"hl_sup": 10, "sup_cta": 12}, "margin_top": 8, "margin_bottom": 8},
            "lg": {"unit": 8, "gaps": {"hl_sup": 12, "sup_cta": 16}, "margin_top": 12, "margin_bottom": 12},
            "xl": {"unit": 8, "gaps": {"hl_sup": 16, "sup_cta": 20}, "margin_top": 16, "margin_bottom": 16},
            "xxl": {"unit": 8, "gaps": {"hl_sup": 20, "sup_cta": 24}, "margin_top": 20, "margin_bottom": 20},
        }
        spacing_config = spacing_scales.get(hierarchy_scale, spacing_scales["lg"])

        # ========== CTA Padding ==========
        cta_padding_x = 24
        cta_padding_y = 12

        # ========== Element Limits ==========
        max_lines_headline = 3
        max_lines_support = 2

        # ========== Color Validation ==========
        image_brightness = image_analysis.get("brightness", 0.5)
        accent_color_raw = image_analysis.get("accent_color", "#888888")
        dominant_palette = image_analysis.get("dominant_palette", ["#888888"] * 5)
        suggested_text_color = image_analysis.get("suggested_text_color", "light")

        # Validate accent color has good contrast with potential headline colors
        accent_color = _validate_accent_color(accent_color_raw, dominant_palette, image_brightness)

        # Headline color: use suggested_text_color but validate
        if suggested_text_color == "dark":
            headline_color = "#111111"  # near-black
        else:
            headline_color = "#FFFFFF"  # white

        headline_color = _validate_color_contrast(headline_color, "transparent", min_contrast=4.5)

        # Support color: lighter/darker version of headline for hierarchy
        if suggested_text_color == "dark":
            support_color = "#333333"  # dark gray
        else:
            support_color = "#E8E8E8"  # light gray

        support_color = _validate_color_contrast(support_color, "transparent", min_contrast=4.5)

        # CTA colors: pick best contrast for button
        cta_bg_color = _pick_cta_color(accent_color, dominant_palette, headline_color)
        cta_fg_color = "#FFFFFF" if _is_dark_hex(cta_bg_color) else "#111111"

        # Eyebrow color: subtle accent variant
        eyebrow_color = accent_color

        return {
            "zone_rect": zone_rect,
            "safe_margin": safe_margin,
            "usable_width": usable_width,
            "zone_height": zone_height,
            "headline_size_range": (hl_min_px, hl_max_px),
            "headline_line_height": headline_line_height,
            "support_size_range": (support_min_px, support_max_px),
            "support_line_height": support_line_height,
            "eyebrow_size": eyebrow_size_px,
            "eyebrow_line_height": eyebrow_line_height,
            "cta_size": cta_size_px,
            "cta_line_height": cta_line_height,
            "cta_padding_x": cta_padding_x,
            "cta_padding_y": cta_padding_y,
            "spacing_unit": spacing_config["unit"],
            "gap_headline_support": spacing_config["gaps"]["hl_sup"],
            "gap_support_cta": spacing_config["gaps"]["sup_cta"],
            "margin_top": spacing_config["margin_top"],
            "margin_bottom": spacing_config["margin_bottom"],
            "max_lines_headline": max_lines_headline,
            "max_lines_support": max_lines_support,
            "min_font_size": 24,
            "max_font_size": 200,
            "font_size_fallback_cascade": [150, 140, 130, 120, 110, 100, 90, 80, 70, 60, 50, 40],
            "headline_color": headline_color,
            "support_color": support_color,
            "eyebrow_color": eyebrow_color,
            "accent_color": accent_color,
            "cta_bg": cta_bg_color,
            "cta_fg": cta_fg_color,
            "image_brightness": image_brightness,
            "suggested_text_color": suggested_text_color,
        }
    except Exception as exc:
        print(f"[LAYOUT_TOKENS] compute failed, returning defaults: {exc}")
        return _neutral_layout_tokens()


def _validate_accent_color(
    accent: str,
    palette: list[str],
    image_brightness: float,
) -> str:
    """Validate accent color has sufficient contrast with headline colors.

    If accent doesn't contrast well with typical headline color for this image,
    pick the best alternative from the palette.

    Args:
        accent: hex string, e.g. "#FF6B35"
        palette: list of 5 hex strings from image k-means
        image_brightness: 0.0 (dark) to 1.0 (bright)

    Returns:
        Validated hex string.
    """
    try:
        # Determine likely headline color based on image brightness
        headline_color = "#FFFFFF" if image_brightness < 0.55 else "#111111"

        # Check contrast between accent and headline
        accent_contrast = _contrast_ratio(accent, headline_color)

        # If accent has good contrast, use it
        if accent_contrast >= 3.0:
            return accent

        # Otherwise, find best-contrast color from palette
        best_color = accent
        best_contrast = accent_contrast

        for pal_color in palette:
            pal_contrast = _contrast_ratio(pal_color, headline_color)
            if pal_contrast > best_contrast:
                best_color = pal_color
                best_contrast = pal_contrast

        return best_color
    except Exception:
        return accent


def _validate_color_contrast(
    color: str,
    bg_color: str,
    min_contrast: float = 4.5,
) -> str:
    """Validate color has sufficient contrast.

    If contrast is below minimum (WCAG AA), lightens or darkens the color
    to achieve minimum contrast.

    Args:
        color: hex string
        bg_color: background hex string (or "transparent")
        min_contrast: minimum contrast ratio (4.5 for normal, 3.0 for large)

    Returns:
        Validated hex string.
    """
    try:
        # For transparent backgrounds, assume white or black depending on color
        if bg_color.lower() == "transparent":
            if _is_dark_hex(color):
                bg_color = "#FFFFFF"
            else:
                bg_color = "#111111"

        current_contrast = _contrast_ratio(color, bg_color)

        if current_contrast >= min_contrast:
            return color

        # Try lightening first (for dark colors)
        lightened = _lighten_hex(color, 0.3)
        if _contrast_ratio(lightened, bg_color) >= min_contrast:
            return lightened

        # Try darkening (for light colors)
        darkened = _darken_hex(color, 0.3)
        if _contrast_ratio(darkened, bg_color) >= min_contrast:
            return darkened

        # Fallback to pure black or white
        if _is_dark_hex(bg_color):
            return "#FFFFFF"
        else:
            return "#111111"
    except Exception:
        return color


def _pick_cta_color(
    accent: str,
    palette: list[str],
    headline_color: str,
) -> str:
    """Pick CTA background color with best contrast vs headline.

    Tries accent first, then palette colors, returns highest-contrast option.

    Args:
        accent: hex string
        palette: list of 5 hex strings
        headline_color: hex string for headline text

    Returns:
        Hex string for CTA background.
    """
    try:
        candidates = [accent] + palette
        best_color = candidates[0]
        best_contrast = _contrast_ratio(candidates[0], headline_color)

        for candidate in candidates[1:]:
            contrast = _contrast_ratio(candidate, headline_color)
            if contrast > best_contrast:
                best_color = candidate
                best_contrast = contrast

        return best_color
    except Exception:
        return accent


def _contrast_ratio(color1: str, color2: str) -> float:
    """Compute WCAG contrast ratio between two colors.

    Args:
        color1: hex string
        color2: hex string

    Returns:
        Contrast ratio (1.0 to 21.0).
    """
    try:
        lum1 = _relative_luminance(_hex_to_rgb(color1))
        lum2 = _relative_luminance(_hex_to_rgb(color2))

        lighter = max(lum1, lum2)
        darker = min(lum1, lum2)

        return (lighter + 0.05) / (darker + 0.05)
    except Exception:
        return 1.0


def _relative_luminance(rgb: tuple[int, int, int]) -> float:
    """Compute relative luminance of RGB color (WCAG definition).

    Args:
        rgb: (r, g, b) each 0-255

    Returns:
        Relative luminance 0.0-1.0.
    """
    r, g, b = [x / 255.0 for x in rgb]

    r = r / 12.92 if r <= 0.03928 else ((r + 0.055) / 1.055) ** 2.4
    g = g / 12.92 if g <= 0.03928 else ((g + 0.055) / 1.055) ** 2.4
    b = b / 12.92 if b <= 0.03928 else ((b + 0.055) / 1.055) ** 2.4

    return 0.2126 * r + 0.7152 * g + 0.0722 * b


def _hex_to_rgb(hex_str: str) -> tuple[int, int, int]:
    """Convert #RRGGBB to (R, G, B) tuple."""
    hex_str = hex_str.lstrip("#")
    return tuple(int(hex_str[i : i + 2], 16) for i in (0, 2, 4))  # type: ignore


def _rgb_to_hex(r: int, g: int, b: int) -> str:
    """Convert (R, G, B) to #RRGGBB."""
    return f"#{r:02X}{g:02X}{b:02X}"


def _is_dark_hex(hex_str: str) -> bool:
    """Determine if a color is dark (luminance < 0.5)."""
    try:
        rgb = _hex_to_rgb(hex_str)
        lum = _relative_luminance(rgb)
        return lum < 0.5
    except Exception:
        return True


def _lighten_hex(hex_str: str, factor: float) -> str:
    """Lighten a color by interpolating toward white.

    Args:
        hex_str: #RRGGBB
        factor: 0.0 (no change) to 1.0 (pure white)

    Returns:
        Lightened hex string.
    """
    try:
        r, g, b = _hex_to_rgb(hex_str)
        r = int(r + (255 - r) * factor)
        g = int(g + (255 - g) * factor)
        b = int(b + (255 - b) * factor)
        return _rgb_to_hex(r, g, b)
    except Exception:
        return hex_str


def _darken_hex(hex_str: str, factor: float) -> str:
    """Darken a color by interpolating toward black.

    Args:
        hex_str: #RRGGBB
        factor: 0.0 (no change) to 1.0 (pure black)

    Returns:
        Darkened hex string.
    """
    try:
        r, g, b = _hex_to_rgb(hex_str)
        r = int(r * (1.0 - factor))
        g = int(g * (1.0 - factor))
        b = int(b * (1.0 - factor))
        return _rgb_to_hex(r, g, b)
    except Exception:
        return hex_str


def _neutral_layout_tokens() -> dict[str, Any]:
    """Safe defaults for layout tokens."""
    return {
        "zone_rect": {"x": 0, "y": 0, "w": 1080, "h": 600},
        "safe_margin": 16,
        "usable_width": 1048,
        "zone_height": 600,
        "headline_size_range": (110, 160),
        "headline_line_height": 1.0,
        "support_size_range": (36, 48),
        "support_line_height": 1.4,
        "eyebrow_size": 28,
        "eyebrow_line_height": 1.2,
        "cta_size": 44,
        "cta_line_height": 1.0,
        "cta_padding_x": 24,
        "cta_padding_y": 12,
        "spacing_unit": 8,
        "gap_headline_support": 12,
        "gap_support_cta": 16,
        "margin_top": 12,
        "margin_bottom": 12,
        "max_lines_headline": 3,
        "max_lines_support": 2,
        "min_font_size": 24,
        "max_font_size": 200,
        "font_size_fallback_cascade": [150, 140, 130, 120, 110, 100, 90, 80, 70, 60, 50, 40],
        "headline_color": "#FFFFFF",
        "support_color": "#E8E8E8",
        "eyebrow_color": "#888888",
        "accent_color": "#888888",
        "cta_bg": "#6366F1",
        "cta_fg": "#FFFFFF",
        "image_brightness": 0.5,
        "suggested_text_color": "light",
    }
