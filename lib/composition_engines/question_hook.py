"""Question hook composition engine."""

from .base import CompositionEngine


class QuestionHookEngine(CompositionEngine):
    """Question + answer format."""

    family_name = "question_hook"
    description = "Question format (large Q, smaller A)"
    required_elements = ["headline"]
    optional_elements = ["support_copy", "cta"]
    forbidden_elements = []
    typical_zones = ["center", "bottom_center"]
    allowed_alignments = ["left", "center"]
    allowed_cta_styles = ["pill_filled", "text_arrow", "none"]
    default_container_type = "none"

    def render(self) -> str:
        left, top, _, _ = self._get_safe_inset()
        usable_width = self.tokens.get("usable_width", 1000)
        alignment = self._get_alignment_css()

        # Get typography values
        hl_min, hl_max = self.tokens.get("headline_size_range", (100, 150))
        sup_min, sup_max = self.tokens.get("support_size_range", (32, 48))
        hl_line_height = self.tokens.get("headline_line_height", 1.0)
        sup_line_height = self.tokens.get("support_line_height", 1.4)
        gap = self.tokens.get("gap_headline_support", 12)
        gap_sup_cta = self.tokens.get("gap_support_cta", 16)

        max_text_width = self._calculate_max_text_width(hl_max, optimal_chars=60)

        # Get colors and fonts
        hl_color = self._get_headline_color()
        sup_color = self._get_support_color()
        hl_role = self.intent.get("typography", {}).get("headline_role", "display_impact")
        sup_role = self.intent.get("typography", {}).get("support_role", "modern_sans")
        hl_font = self._get_font_for_role(hl_role)
        sup_font = self._get_font_for_role(sup_role)

        blocks = []

        # Headline (question, required)
        headline_content = self.text_elements.get("headline", {}).get("content", "")
        lines = self.text_elements.get("headline", {}).get("lines", [])

        if lines:
            line_html = "<br>".join(f"<span>{line}</span>" for line in lines)
        else:
            line_html = headline_content

        # Calculate responsive font size based on headline length
        responsive_hl_size = self._calculate_responsive_font_size(
            headline_content, hl_min, hl_max, lines
        )

        headline_html = f"""<div style="
            font-family: {hl_font};
            font-size: {responsive_hl_size}px;
            font-weight: 700;
            line-height: {hl_line_height};
            color: {hl_color};
            max-width: {max_text_width}px;
            margin-bottom: {gap}px;
            text-align: {alignment};
        ">
            {line_html}
        </div>"""
        blocks.append(headline_html)

        # Support copy (answer, optional)
        support_content = self.text_elements.get("support_copy", {}).get("content", "")
        if support_content:
            support_html = f"""<div style="
                font-family: {sup_font};
                font-size: {int(sup_max * 0.9)}px;
                font-weight: 400;
                line-height: {sup_line_height};
                color: {sup_color};
                max-width: {max_text_width}px;
                margin-bottom: {gap_sup_cta}px;
                text-align: {alignment};
            ">
                {support_content}
            </div>"""
            blocks.append(support_html)

        # CTA (optional)
        cta_content = self.text_elements.get("cta", {}).get("content", "")
        cta_style = self.intent.get("cta_intent", {}).get("style", "none")
        if cta_content and cta_style != "none":
            blocks.append(self._render_cta(cta_content, cta_style))

        inner_html = "\n".join(blocks)
        body_html = f"""<div style="
            position: absolute;
            left: {left}px;
            top: {top}px;
            width: {usable_width}px;
            text-align: {alignment};
        ">
            {inner_html}
        </div>"""

        return self._build_html_wrapper(body_html)
