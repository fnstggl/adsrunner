"""Hero statement composition engine.

Large bold headline, no CTA, minimal support.
Maximizes negative space for editorial impact.
"""

from __future__ import annotations

from typing import Any

from .base import CompositionEngine


class HeroStatementEngine(CompositionEngine):
    """Bold headline with maximum negative space."""

    family_name = "hero_statement"
    description = "Large headline, no CTA, max negative space"
    required_elements = ["headline"]
    optional_elements = ["eyebrow", "support_copy"]
    forbidden_elements = ["cta"]
    typical_zones = ["center", "top_center", "bottom_center"]
    allowed_alignments = ["center", "left"]
    allowed_cta_styles = ["none"]
    default_container_type = "none"

    def render(self) -> str:
        """Generate HTML for hero statement."""
        left, top, right, bottom = self._get_safe_inset()
        zone = self._get_zone_rect()
        usable_width = self.tokens.get("usable_width", 1000)

        alignment = self.intent.get("placement", {}).get("alignment", "center")
        text_align = "center" if alignment == "center" else "left"

        blocks = []

        # Eyebrow (optional)
        eyebrow_content = self.text_elements.get("eyebrow", {}).get("content", "")
        if eyebrow_content:
            blocks.append(self._render_eyebrow(eyebrow_content))

        # Headline (required, very large)
        headline_content = self.text_elements.get("headline", {}).get("content", "")
        lines = self.text_elements.get("headline", {}).get("lines", [])
        blocks.append(self._render_headline(headline_content, lines))

        # Support copy (minimal, optional)
        support_content = self.text_elements.get("support_copy", {}).get("content", "")
        if support_content:
            blocks.append(self._render_support_copy(support_content))

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
