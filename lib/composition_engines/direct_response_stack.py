"""Direct response stack composition engine.

Headline + support + CTA stacked vertically.
Perfect for performance marketing, e-commerce, simple offers.
"""

from __future__ import annotations

from typing import Any

from .base import CompositionEngine


class DirectResponseStackEngine(CompositionEngine):
    """Vertically stacked headline, support, CTA."""

    family_name = "direct_response_stack"
    description = "Headline + support + CTA stacked vertically"
    required_elements = ["headline"]
    optional_elements = ["eyebrow", "support_copy", "cta"]
    forbidden_elements = []
    typical_zones = ["bottom_center", "lower_third", "center"]
    allowed_alignments = ["left", "center"]
    allowed_cta_styles = [
        "pill_filled",
        "rectangular_filled",
        "ghost_outlined",
        "text_arrow",
        "none",
    ]
    default_container_type = "none"

    def render(self) -> str:
        """Generate HTML for direct response stack."""
        left, top, right, bottom = self._get_safe_inset()
        zone = self._get_zone_rect()
        usable_width = self.tokens.get("usable_width", 1000)

        alignment = self.intent.get("placement", {}).get("alignment", "center")
        text_align = "center" if alignment == "center" else "left"

        # Build content blocks
        blocks = []

        # Eyebrow
        eyebrow_content = self.text_elements.get("eyebrow", {}).get("content", "")
        if eyebrow_content:
            blocks.append(self._render_eyebrow(eyebrow_content))

        # Headline (required)
        headline_content = self.text_elements.get("headline", {}).get("content", "")
        lines = self.text_elements.get("headline", {}).get("lines", [])
        blocks.append(self._render_headline(headline_content, lines))

        # Support copy
        support_content = self.text_elements.get("support_copy", {}).get("content", "")
        if support_content:
            blocks.append(self._render_support_copy(support_content))

        # CTA
        cta_content = self.text_elements.get("cta", {}).get("content", "")
        cta_style = self.intent.get("cta_intent", {}).get("style", "none")
        if cta_content and cta_style != "none":
            blocks.append(self._render_cta(cta_content, cta_style))

        container_css = self._get_container_css()
        inner_html = "\n".join(blocks)

        body_html = f"""<div style="
            position: absolute;
            left: {left}px;
            top: {top}px;
            width: {usable_width}px;
            text-align: {text_align};
            {container_css}
        ">
            {inner_html}
        </div>"""

        return self._build_html_wrapper(body_html)
