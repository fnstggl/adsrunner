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

        # Get typography values
        hl_min, hl_max = self.tokens.get("headline_size_range", (100, 150))
        sup_min, sup_max = self.tokens.get("support_size_range", (32, 48))
        hl_line_height = self.tokens.get("headline_line_height", 1.0)
        sup_line_height = self.tokens.get("support_line_height", 1.4)
        gap_hl_sup = self.tokens.get("gap_headline_support", 12)
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

        # Eyebrow (optional)
        eyebrow_content = self.text_elements.get("eyebrow", {}).get("content", "")
        if eyebrow_content:
            eyebrow_size = self.tokens.get("eyebrow_size", 28)
            accent_color = self._get_accent_color()
            eyebrow_html = f"""<div style="
                font-family: {sup_font};
                font-size: {eyebrow_size}px;
                font-weight: 600;
                color: {accent_color};
                text-transform: uppercase;
                letter-spacing: 0.05em;
                margin-bottom: {gap_hl_sup}px;
                text-align: center;
            ">
                {eyebrow_content}
            </div>"""
            blocks.append(eyebrow_html)

        # Headline (required)
        headline_content = self.text_elements.get("headline", {}).get("content", "")
        lines = self.text_elements.get("headline", {}).get("lines", [])

        if lines:
            line_html = "<br>".join(f"<span>{line}</span>" for line in lines)
        else:
            line_html = headline_content

        headline_html = f"""<div style="
            font-family: {hl_font};
            font-size: {hl_max}px;
            font-weight: 700;
            line-height: {hl_line_height};
            color: {hl_color};
            max-width: {max_text_width}px;
            margin: 0 auto {gap_hl_sup}px;
            text-align: center;
        ">
            {line_html}
        </div>"""
        blocks.append(headline_html)

        # Support copy (optional)
        support_content = self.text_elements.get("support_copy", {}).get("content", "")
        if support_content:
            support_html = f"""<div style="
                font-family: {sup_font};
                font-size: {sup_max}px;
                font-weight: 400;
                line-height: {sup_line_height};
                color: {sup_color};
                max-width: {max_text_width}px;
                margin: 0 auto {gap_sup_cta}px;
                text-align: center;
            ">
                {support_content}
            </div>"""
            blocks.append(support_html)

        # CTA (optional)
        cta_content = self.text_elements.get("cta", {}).get("content", "")
        cta_style = self.intent.get("cta_intent", {}).get("style", "none")
        if cta_content and cta_style != "none":
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
