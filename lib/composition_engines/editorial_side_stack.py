"""Editorial side stack composition engine."""

from __future__ import annotations

from .base import CompositionEngine


class EditorialSideStackEngine(CompositionEngine):
    """Headline + support stacked tight, CTA smaller to side."""

    family_name = "editorial_side_stack"
    description = "Headline + support stacked, CTA to side"
    required_elements = ["headline"]
    optional_elements = ["eyebrow", "support_copy", "cta"]
    forbidden_elements = []
    typical_zones = ["center", "top_center", "bottom_center"]
    allowed_alignments = ["left", "center"]
    allowed_cta_styles = ["text_arrow", "underlined_text", "tiny_anchor", "none"]
    default_container_type = "none"

    def render(self) -> str:
        """Generate HTML for editorial side stack."""
        left, top, right, bottom = self._get_safe_inset()
        usable_width = self.tokens.get("usable_width", 1000)
        alignment = self.intent.get("placement", {}).get("alignment", "left")
        text_align = "center" if alignment == "center" else alignment

        blocks = []

        eyebrow_content = self.text_elements.get("eyebrow", {}).get("content", "")
        if eyebrow_content:
            blocks.append(self._render_eyebrow(eyebrow_content))

        headline_content = self.text_elements.get("headline", {}).get("content", "")
        lines = self.text_elements.get("headline", {}).get("lines", [])
        blocks.append(self._render_headline(headline_content, lines))

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
