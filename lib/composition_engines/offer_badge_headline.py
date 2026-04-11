"""Offer badge headline composition engine."""
from .base import CompositionEngine

class OfferBadgeHeadlineEngine(CompositionEngine):
    family_name = "offer_badge_headline"
    description = "Badge + headline + CTA"
    required_elements = ["headline", "cta"]
    optional_elements = ["eyebrow", "support_copy"]
    forbidden_elements = []
    typical_zones = ["bottom_center", "center"]
    allowed_alignments = ["center"]
    allowed_cta_styles = ["pill_filled", "rectangular_filled"]
    default_container_type = "solid_chip"

    def render(self) -> str:
        left, top, _, _ = self._get_safe_inset()
        usable_width = self.tokens.get("usable_width", 1000)
        blocks = []
        eyebrow_content = self.text_elements.get("eyebrow", {}).get("content", "")
        if eyebrow_content:
            blocks.append(self._render_eyebrow(eyebrow_content))
        headline_content = self.text_elements.get("headline", {}).get("content", "")
        blocks.append(self._render_headline(headline_content))
        cta_content = self.text_elements.get("cta", {}).get("content", "")
        blocks.append(self._render_cta(cta_content, "pill_filled"))
        inner_html = "\n".join(blocks)
        body_html = f"""<div style="position: absolute; left: {left}px; top: {top}px; width: {usable_width}px; text-align: center;">
            {inner_html}
        </div>"""
        return self._build_html_wrapper(body_html)
