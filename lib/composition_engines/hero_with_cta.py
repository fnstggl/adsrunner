"""Hero with CTA composition engine.

Large headline + optional support + prominent CTA.
"""

from __future__ import annotations

from .base import CompositionEngine


class HeroWithCtaEngine(CompositionEngine):
    """Bold headline + optional support + prominent CTA."""

    family_name = "hero_with_cta"
    description = "Headline + optional support + prominent CTA"
    required_elements = ["headline", "cta"]
    optional_elements = ["eyebrow", "support_copy"]
    forbidden_elements = []
    typical_zones = ["bottom_center", "lower_third", "center"]
    allowed_alignments = ["center"]
    allowed_cta_styles = ["pill_filled", "rectangular_filled", "ghost_outlined"]
    default_container_type = "none"

    def render(self) -> str:
        """Generate HTML for hero with CTA."""
        left, top, right, bottom = self._get_safe_inset()
        zone = self._get_zone_rect()
        usable_width = self.tokens.get("usable_width", 1000)

        blocks = []

        # Eyebrow
        eyebrow_content = self.text_elements.get("eyebrow", {}).get("content", "")
        if eyebrow_content:
            blocks.append(self._render_eyebrow(eyebrow_content))

        # Headline (required, large)
        headline_content = self.text_elements.get("headline", {}).get("content", "")
        lines = self.text_elements.get("headline", {}).get("lines", [])
        blocks.append(self._render_headline(headline_content, lines))

        # Support copy (optional)
        support_content = self.text_elements.get("support_copy", {}).get("content", "")
        if support_content:
            blocks.append(self._render_support_copy(support_content))

        # CTA (required)
        cta_content = self.text_elements.get("cta", {}).get("content", "")
        cta_style = self.intent.get("cta_intent", {}).get("style", "pill_filled")
        blocks.append(self._render_cta(cta_content, cta_style))

        container_css = self._get_container_css()
        inner_html = "\n".join(blocks)

        body_html = f"""<div style="
            position: absolute;
            left: {left}px;
            top: {top}px;
            width: {usable_width}px;
            text-align: center;
            {container_css}
        ">
            {inner_html}
        </div>"""

        return self._build_html_wrapper(body_html)
