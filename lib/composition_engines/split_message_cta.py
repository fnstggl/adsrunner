"""Split message CTA composition engine."""
from .base import CompositionEngine

class SplitMessageCtaEngine(CompositionEngine):
    family_name = "split_message_cta"
    description = "Two-column (message | CTA)"
    required_elements = ["headline", "cta"]
    optional_elements = ["support_copy"]
    forbidden_elements = []
    typical_zones = ["center", "bottom_center"]
    allowed_alignments = ["left", "center"]
    allowed_cta_styles = ["pill_filled", "rectangular_filled"]
    default_container_type = "none"

    def render(self) -> str:
        left, top, _, _ = self._get_safe_inset()
        usable_width = self.tokens.get("usable_width", 1000)
        blocks = []
        headline_content = self.text_elements.get("headline", {}).get("content", "")
        blocks.append(self._render_headline(headline_content))
        support_content = self.text_elements.get("support_copy", {}).get("content", "")
        if support_content:
            blocks.append(self._render_support_copy(support_content))
        cta_content = self.text_elements.get("cta", {}).get("content", "")
        blocks.append(self._render_cta(cta_content))
        inner_html = "\n".join(blocks)
        body_html = f"""<div style="position: absolute; left: {left}px; top: {top}px; width: {usable_width}px;">
            {inner_html}
        </div>"""
        return self._build_html_wrapper(body_html)
