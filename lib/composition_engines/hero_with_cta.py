"""Hero with CTA composition engine.

Large headline + optional support + prominent CTA.
Balanced composition with clear action.
"""

from __future__ import annotations

from .base import CompositionEngine


class HeroWithCtaEngine(CompositionEngine):
    """Bold headline + optional support + prominent CTA."""

    family_name = "hero_with_cta"
    description = "Headline + optional support + prominent CTA"
    required_elements = ["headline", "cta"]
    optional_elements = ["eyebrow", "support_copy"]
    forbidden_elements = []
    typical_zones = ["bottom_center", "lower_third", "center"]
    allowed_alignments = ["center"]
    allowed_cta_styles = ["pill_filled", "rectangular_filled", "ghost_outlined"]
    default_container_type = "none"

    def render(self) -> str:
        """Generate HTML for hero with CTA using proper typography."""
        left, top, right, bottom = self._get_safe_inset()
        usable_width = self.tokens.get("usable_width", 1000)

        # Typography values
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

        # Eyebrow
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

        # Headline
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

        # Support copy
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

        # CTA
        cta_content = self.text_elements.get("cta", {}).get("content", "")
        cta_style = self.intent.get("cta_intent", {}).get("style", "pill_filled")
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
