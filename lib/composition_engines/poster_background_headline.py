"""Poster background headline composition engine."""
from .base import CompositionEngine

class PosterBackgroundHeadlineEngine(CompositionEngine):
    family_name = "poster_background_headline"
    description = "Huge faded text behind small foreground"
    required_elements = ["headline"]
    optional_elements = ["support_copy"]
    forbidden_elements = ["cta"]
    typical_zones = ["center", "bottom_center"]
    allowed_alignments = ["center"]
    allowed_cta_styles = ["none"]
    default_container_type = "background_text_layer"

    def render(self) -> str:
        left, top, _, _ = self._get_safe_inset()
        usable_width = self.tokens.get("usable_width", 1000)

        # Get typography values
        hl_min, hl_max = self.tokens.get("headline_size_range", (100, 150))
        sup_min, sup_max = self.tokens.get("support_size_range", (32, 48))
        hl_line_height = self.tokens.get("headline_line_height", 1.0)
        sup_line_height = self.tokens.get("support_line_height", 1.4)
        gap = self.tokens.get("gap_headline_support", 12)

        # Poster bg uses very constrained max-width for dramatic effect
        max_text_width = self._calculate_max_text_width(hl_max, optimal_chars=40)

        # Get colors and fonts
        hl_color = self._get_headline_color()
        sup_color = self._get_support_color()
        hl_role = self.intent.get("typography", {}).get("headline_role", "display_impact")
        sup_role = self.intent.get("typography", {}).get("support_role", "modern_sans")
        hl_font = self._get_font_for_role(hl_role)
        sup_font = self._get_font_for_role(sup_role)

        blocks = []

        # Background headline (large, faded)
        headline_content = self.text_elements.get("headline", {}).get("content", "")
        lines = self.text_elements.get("headline", {}).get("lines", [])

        if lines:
            line_html = "<br>".join(f"<span>{line}</span>" for line in lines)
        else:
            line_html = headline_content

        # Calculate responsive font size based on headline length, then apply dramatic scaling
        responsive_hl_size = self._calculate_responsive_font_size(
            headline_content, hl_min, hl_max, lines
        )
        background_font_size = int(responsive_hl_size * 1.3)
        headline_html = f"""<div style="
            position: absolute;
            top: 50%;
            left: 50%;
            transform: translate(-50%, -50%);
            font-family: {hl_font};
            font-size: {background_font_size}px;
            font-weight: 700;
            line-height: {hl_line_height};
            color: {hl_color};
            max-width: {max_text_width}px;
            text-align: center;
            opacity: 0.08;
            z-index: 0;
        ">
            {line_html}
        </div>"""
        blocks.append(headline_html)

        # Foreground support copy (if present)
        support_content = self.text_elements.get("support_copy", {}).get("content", "")
        if support_content:
            support_html = f"""<div style="
                position: relative;
                z-index: 10;
                font-family: {sup_font};
                font-size: {sup_max}px;
                font-weight: 400;
                line-height: {sup_line_height};
                color: {sup_color};
                max-width: {max_text_width}px;
                text-align: center;
            ">
                {support_content}
            </div>"""
            blocks.append(support_html)

        inner_html = "\n".join(blocks)
        body_html = f"""<div style="
            position: absolute;
            left: {left}px;
            top: {top}px;
            width: {usable_width}px;
            text-align: center;
        ">
            {inner_html}
        </div>"""
        return self._build_html_wrapper(body_html)
