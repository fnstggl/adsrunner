"""Minimal product-led composition engine."""
from .base import CompositionEngine

class MinimalProductLedEngine(CompositionEngine):
    family_name = "minimal_product_led"
    description = "Tiny text, extreme negative space"
    required_elements = ["headline"]
    optional_elements = ["support_copy"]
    forbidden_elements = ["cta"]
    typical_zones = ["center", "bottom_center", "bottom_right"]
    allowed_alignments = ["left", "right"]
    allowed_cta_styles = ["none"]
    default_container_type = "none"

    def render(self) -> str:
        left, top, _, _ = self._get_safe_inset()
        alignment = self._get_alignment_css()

        # Minimal product-led uses constrained width for extreme negative space
        usable_width = self.tokens.get("usable_width", 400)
        # Override with smaller width for this family
        usable_width = min(usable_width, 400)

        # Get typography values
        hl_min, hl_max = self.tokens.get("headline_size_range", (100, 150))
        sup_min, sup_max = self.tokens.get("support_size_range", (32, 48))
        hl_line_height = self.tokens.get("headline_line_height", 1.0)
        sup_line_height = self.tokens.get("support_line_height", 1.4)
        gap = self.tokens.get("gap_headline_support", 12)

        # Minimal product-led uses smaller, tighter max-width
        max_text_width = self._calculate_max_text_width(int(hl_max * 0.8), optimal_chars=40)

        # Get colors and fonts
        hl_color = self._get_headline_color()
        sup_color = self._get_support_color()
        hl_role = self.intent.get("typography", {}).get("headline_role", "display_impact")
        sup_role = self.intent.get("typography", {}).get("support_role", "modern_sans")
        hl_font = self._get_font_for_role(hl_role)
        sup_font = self._get_font_for_role(sup_role)

        blocks = []

        # Headline (required, small and tight)
        headline_content = self.text_elements.get("headline", {}).get("content", "")
        lines = self.text_elements.get("headline", {}).get("lines", [])

        if lines:
            line_html = "<br>".join(f"<span>{line}</span>" for line in lines)
        else:
            line_html = headline_content

        headline_html = f"""<div style="
            font-family: {hl_font};
            font-size: {int(hl_max * 0.8)}px;
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

        # Support copy (optional, very minimal)
        support_content = self.text_elements.get("support_copy", {}).get("content", "")
        if support_content:
            support_html = f"""<div style="
                font-family: {sup_font};
                font-size: {int(sup_max * 0.7)}px;
                font-weight: 400;
                line-height: {sup_line_height};
                color: {sup_color};
                max-width: {max_text_width}px;
                text-align: {alignment};
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
            text-align: {alignment};
        ">
            {inner_html}
        </div>"""
        return self._build_html_wrapper(body_html)
