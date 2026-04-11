"""Pain point fragments composition engine."""

from .base import CompositionEngine


class PainPointFragmentsEngine(CompositionEngine):
    """Multiple small text blocks with varied sizes."""

    family_name = "pain_point_fragments"
    description = "Multiple text blocks with varied emphasis"
    required_elements = ["headline"]
    optional_elements = ["eyebrow", "support_copy", "cta"]
    forbidden_elements = []
    typical_zones = ["center", "bottom_center"]
    allowed_alignments = ["left", "center"]
    allowed_cta_styles = ["text_arrow", "none"]
    default_container_type = "shadow_only"

    def render(self) -> str:
        left, top, _, _ = self._get_safe_inset()
        usable_width = self.tokens.get("usable_width", 1000)

        blocks = []
        headline_content = self.text_elements.get("headline", {}).get("content", "")
        lines = self.text_elements.get("headline", {}).get("lines", [])
        blocks.append(self._render_headline(headline_content, lines))

        support_content = self.text_elements.get("support_copy", {}).get("content", "")
        if support_content:
            blocks.append(self._render_support_copy(support_content))

        inner_html = "\n".join(blocks)

        body_html = f"""<div style="position: absolute; left: {left}px; top: {top}px; width: {usable_width}px;">
            {inner_html}
        </div>"""

        return self._build_html_wrapper(body_html)
