"""Soft card overlay composition engine."""
from .base import CompositionEngine

class SoftCardOverlayEngine(CompositionEngine):
    family_name = "soft_card_overlay"
    description = "All elements in soft-edged card"
    required_elements = ["headline"]
    optional_elements = ["eyebrow", "support_copy", "cta"]
    forbidden_elements = []
    typical_zones = ["center", "bottom_center"]
    allowed_alignments = ["center"]
    allowed_cta_styles = ["pill_filled", "ghost_outlined", "none"]
    default_container_type = "glass_blur"

    def render(self) -> str:
        left, top, _, _ = self._get_safe_inset()
        usable_width = self.tokens.get("usable_width", 1000)
        blocks = []
        eyebrow_content = self.text_elements.get("eyebrow", {}).get("content", "")
        if eyebrow_content:
            blocks.append(self._render_eyebrow(eyebrow_content))
        headline_content = self.text_elements.get("headline", {}).get("content", "")
        blocks.append(self._render_headline(headline_content))
        support_content = self.text_elements.get("support_copy", {}).get("content", "")
        if support_content:
            blocks.append(self._render_support_copy(support_content))
        cta_content = self.text_elements.get("cta", {}).get("content", "")
        if cta_content:
            blocks.append(self._render_cta(cta_content))
        inner_html = "\n".join(blocks)
        body_html = f"""<div style="position: absolute; left: {left}px; top: {top}px; width: {usable_width}px; text-align: center;">
            {inner_html}
        </div>"""
        return self._build_html_wrapper(body_html)
