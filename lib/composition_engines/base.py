"""
Base class for composition engines.

Each family (hero_statement, direct_response_stack, etc.) implements a renderer
that generates deterministic HTML from validated intent + layout tokens.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any


class CompositionEngine(ABC):
    """Base class for deterministic HTML/CSS composition engines.

    Subclasses implement family-specific rendering logic.
    """

    # Family metadata
    family_name: str = ""
    description: str = ""
    required_elements: list[str] = []  # elements that must be present
    optional_elements: list[str] = []  # elements that may be present
    forbidden_elements: list[str] = []  # elements that must not be present
    typical_zones: list[str] = []  # recommended zones for this family
    allowed_alignments: list[str] = []  # allowed alignment values
    allowed_cta_styles: list[str] = []  # allowed CTA styles for this family
    default_container_type: str = "none"  # default container strategy

    def __init__(
        self,
        intent: dict[str, Any],
        layout_tokens: dict[str, Any],
        text_elements: dict[str, Any],
        image_analysis: dict[str, Any],
    ):
        """Initialize the composition engine.

        Args:
            intent: Validated layout intent JSON from Claude
            layout_tokens: Computed layout constraints
            text_elements: Headline, support, CTA, eyebrow content
            image_analysis: Image brightness, colors, zones
        """
        self.intent = intent
        self.tokens = layout_tokens
        self.text_elements = text_elements
        self.image_analysis = image_analysis

    @abstractmethod
    def render(self) -> str:
        """Generate HTML/CSS overlay.

        Returns:
            str: Complete HTML document (<!DOCTYPE html> to </html>)
        """
        pass

    def _get_zone_rect(self) -> dict[str, int]:
        """Get the zone rectangle from tokens."""
        return self.tokens.get("zone_rect", {"x": 0, "y": 0, "w": 1080, "h": 600})

    def _get_safe_inset(self) -> tuple[int, int, int, int]:
        """Get left, top, right, bottom insets based on safe margin."""
        margin = self.tokens.get("safe_margin", 16)
        zone = self._get_zone_rect()
        x, y = zone["x"], zone["y"]
        w, h = zone["w"], zone["h"]
        return (x + margin, y + margin, x + w - margin, y + h - margin)

    def _get_container_css(self) -> str:
        """Generate CSS for the container strategy."""
        container_type = self.intent.get("container", {}).get("type", "none")
        opacity = self.intent.get("container", {}).get("opacity_preference", 0.0)
        blur = self.intent.get("container", {}).get("blur_preference", 0)

        if container_type == "none":
            return ""
        elif container_type == "shadow_only":
            return "text-shadow: 0 2px 8px rgba(0, 0, 0, 0.5), 0 4px 16px rgba(0, 0, 0, 0.3);"
        elif container_type == "translucent_card":
            bg_color = (
                "rgba(0, 0, 0, {})".format(opacity)
                if self.intent.get("color", {}).get("mode") == "light_on_dark_area"
                else "rgba(255, 255, 255, {})".format(opacity)
            )
            return f"background: {bg_color}; border-radius: 8px; padding: 20px;"
        elif container_type == "glass_blur":
            return f"background: rgba(255, 255, 255, {opacity}); backdrop-filter: blur({blur}px); border-radius: 12px; padding: 20px;"
        elif container_type == "solid_chip":
            return "background: rgba(0, 0, 0, 0.8); border-radius: 999px; padding: 12px 20px;"
        elif container_type == "outlined_card":
            return "border: 2px solid rgba(255, 255, 255, 0.3); border-radius: 8px; padding: 20px;"
        else:
            return ""

    def _get_headline_color(self) -> str:
        """Get the headline color from tokens."""
        return self.tokens.get("headline_color", "#FFFFFF")

    def _get_support_color(self) -> str:
        """Get the support copy color from tokens."""
        return self.tokens.get("support_color", "#E8E8E8")

    def _get_accent_color(self) -> str:
        """Get the accent color from tokens."""
        return self.tokens.get("accent_color", "#888888")

    def _get_cta_colors(self) -> tuple[str, str]:
        """Get CTA background and foreground colors from tokens."""
        bg = self.tokens.get("cta_bg", "#6366F1")
        fg = self.tokens.get("cta_fg", "#FFFFFF")
        return bg, fg

    def _build_html_wrapper(self, inner_html: str) -> str:
        """Wrap inner HTML with doctype and head."""
        font_family_css = self._get_font_face_css()
        return f"""<!DOCTYPE html>
<html>
<head>
    <meta charset="UTF-8">
    <style>
        * {{ margin: 0; padding: 0; box-sizing: border-box; }}
        html, body {{
            width: 1080px;
            height: 1350px;
            overflow: hidden;
            margin: 0;
            padding: 0;
            background: transparent;
        }}
        {font_family_css}
    </style>
</head>
<body>
{inner_html}
</body>
</html>"""

    def _get_font_face_css(self) -> str:
        """Return placeholder for font-face CSS (injected by render_svg_overlay)."""
        return "/* @font-face rules will be injected here */"

    def _render_headline(self, headline_content: str, lines: list[str] = None) -> str:
        """Render headline with optional semantic line breaks."""
        scale = self.intent.get("hierarchy", {}).get("headline_scale", "xl")
        line_height = self.tokens.get("headline_line_height", 1.0)
        color = self._get_headline_color()
        role = self.intent.get("typography", {}).get("headline_role", "display_impact")
        font_family = self._get_font_for_role(role)
        hl_min, hl_max = self.tokens.get("headline_size_range", (100, 150))

        # Use max size by default; system will shrink if needed
        font_size = hl_max

        # If lines are provided, use them for structure
        if lines:
            line_html = "<br>".join(f"<span>{line}</span>" for line in lines)
        else:
            line_html = headline_content

        return f"""<div style="
            font-family: {font_family};
            font-size: {font_size}px;
            line-height: {line_height};
            color: {color};
            font-weight: 700;
        ">
            {line_html}
        </div>"""

    def _render_support_copy(self, support_content: str) -> str:
        """Render support copy."""
        if not support_content:
            return ""
        line_height = self.tokens.get("support_line_height", 1.4)
        color = self._get_support_color()
        role = self.intent.get("typography", {}).get("support_role", "modern_sans")
        font_family = self._get_font_for_role(role)
        sup_min, sup_max = self.tokens.get("support_size_range", (32, 48))

        return f"""<div style="
            margin-top: {self.tokens.get('gap_headline_support', 12)}px;
            font-family: {font_family};
            font-size: {sup_max}px;
            line-height: {line_height};
            color: {color};
            font-weight: 400;
        ">
            {support_content}
        </div>"""

    def _render_cta(self, cta_content: str, style: str = "pill_filled") -> str:
        """Render CTA button."""
        if not cta_content:
            return ""
        cta_size = self.tokens.get("cta_size", 44)
        bg, fg = self._get_cta_colors()
        font_family = self._get_font_for_role("modern_sans")

        if style == "none":
            return ""
        elif style == "pill_filled":
            return f"""<button style="
                margin-top: {self.tokens.get('gap_support_cta', 16)}px;
                padding: {self.tokens.get('cta_padding_y', 12)}px {self.tokens.get('cta_padding_x', 24)}px;
                font-family: {font_family};
                font-size: {cta_size}px;
                font-weight: 700;
                color: {fg};
                background: {bg};
                border: none;
                border-radius: 999px;
                cursor: pointer;
                text-transform: uppercase;
            ">{cta_content}</button>"""
        elif style == "ghost_outlined":
            return f"""<button style="
                margin-top: {self.tokens.get('gap_support_cta', 16)}px;
                padding: {self.tokens.get('cta_padding_y', 12)}px {self.tokens.get('cta_padding_x', 24)}px;
                font-family: {font_family};
                font-size: {cta_size}px;
                font-weight: 700;
                color: {bg};
                background: transparent;
                border: 2px solid {bg};
                border-radius: 4px;
                cursor: pointer;
                text-transform: uppercase;
            ">{cta_content}</button>"""
        else:
            return f"""<a style="
                display: inline-block;
                margin-top: {self.tokens.get('gap_support_cta', 16)}px;
                font-family: {font_family};
                font-size: {cta_size}px;
                font-weight: 700;
                color: {bg};
                text-decoration: underline;
                cursor: pointer;
            ">{cta_content}</a>"""

    def _render_eyebrow(self, eyebrow_content: str) -> str:
        """Render eyebrow/badge text."""
        if not eyebrow_content:
            return ""
        size = self.tokens.get("eyebrow_size", 28)
        color = self._get_accent_color()
        font_family = self._get_font_for_role("modern_sans")

        return f"""<div style="
            font-family: {font_family};
            font-size: {size}px;
            font-weight: 600;
            color: {color};
            text-transform: uppercase;
            letter-spacing: 0.05em;
            margin-bottom: {self.tokens.get('gap_headline_support', 12)}px;
        ">
            {eyebrow_content}
        </div>"""

    def _get_font_for_role(self, role: str) -> str:
        """Map typography role to actual font family.

        Returns the first available font in the role's family list.
        """
        from . import TYPOGRAPHY_ROLES

        role_info = TYPOGRAPHY_ROLES.get(role, {})
        fonts = role_info.get("fonts", ["Inter"])
        return f"'{fonts[0]}'"
