"""Direct response stack composition engine.

Headline + support + CTA stacked vertically.
Perfect for performance marketing, e-commerce, simple offers.

Typography mathematics:
- Headline: 150-200px, tight line-height (1.0), max-width controlled
- Support: 32-48px, spacious line-height (1.4), same max-width as headline
- Spacing: Determined by layout tokens (gap_headline_support, gap_support_cta)
- Alignment: Respects intent placement alignment
- Measure: ~65 chars optimal per line (prevents awkward wrapping)
"""

from __future__ import annotations

from typing import Any

from .base import CompositionEngine


class DirectResponseStackEngine(CompositionEngine):
    """Vertically stacked headline, support, CTA."""

    family_name = "direct_response_stack"
    description = "Headline + support + CTA stacked vertically"
    required_elements = ["headline"]
    optional_elements = ["eyebrow", "support_copy", "cta"]
    forbidden_elements = []
    typical_zones = ["bottom_center", "lower_third", "center"]
    allowed_alignments = ["left", "center"]
    allowed_cta_styles = [
        "pill_filled",
        "rectangular_filled",
        "ghost_outlined",
        "text_arrow",
        "none",
    ]
    default_container_type = "none"

    def render(self) -> str:
        """Generate HTML for direct response stack with proper typography."""
        left, top, right, bottom = self._get_safe_inset()
        usable_width = self.tokens.get("usable_width", 1000)
        alignment = self._get_alignment_css()

        # Get typography values from tokens
        hl_min, hl_max = self.tokens.get("headline_size_range", (100, 150))
        sup_min, sup_max = self.tokens.get("support_size_range", (32, 48))
        hl_line_height = self.tokens.get("headline_line_height", 1.0)
        sup_line_height = self.tokens.get("support_line_height", 1.4)
        gap_hl_sup = self.tokens.get("gap_headline_support", 12)
        gap_sup_cta = self.tokens.get("gap_support_cta", 16)

        # Calculate optimal max-width for readable text
        # At hl_max font size, estimate character width and constrain to ~65 chars
        max_text_width = self._calculate_max_text_width(hl_max, optimal_chars=65)

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
                text-align: {alignment};
            ">
                {eyebrow_content}
            </div>"""
            blocks.append(eyebrow_html)

        # Headline (required) WITH PROPER LAYOUT CONTROL
        headline_content = self.text_elements.get("headline", {}).get("content", "")
        lines = self.text_elements.get("headline", {}).get("lines", [])

        if lines:
            # Respect semantic line breaks from intent
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
            margin: 0 auto {gap_hl_sup}px;
            text-align: {alignment};
        ">
            {line_html}
        </div>"""
        blocks.append(headline_html)

        # Support copy (optional) WITH PROPER LAYOUT CONTROL
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

        container_css = self._get_container_css()
        inner_html = "\n".join(blocks)

        body_html = f"""<div style="
            position: absolute;
            left: {left}px;
            top: {top}px;
            width: {usable_width}px;
            text-align: {alignment};
            {container_css}
        ">
            {inner_html}
        </div>"""

        return self._build_html_wrapper(body_html)
