"""Utility explainer composition engine."""
from .base import CompositionEngine

class UtilityExplainerEngine(CompositionEngine):
    family_name = "utility_explainer"
    description = "Inline callouts/labels"
    required_elements = ["headline"]
    optional_elements = ["support_copy"]
    forbidden_elements = ["cta"]
    typical_zones = ["center", "top_center"]
    allowed_alignments = ["left", "center"]
    allowed_cta_styles = ["none"]
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
        inner_html = "\n".join(blocks)
        body_html = f"""<div style="position: absolute; left: {left}px; top: {top}px; width: {usable_width}px;">
            {inner_html}
        </div>"""
        return self._build_html_wrapper(body_html)
