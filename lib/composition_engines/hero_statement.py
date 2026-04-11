"""Hero statement composition engine.

Large bold headline, no CTA, minimal support.
Maximizes negative space for editorial impact.

Typography:
- Headline: Largest possible (xxl scale), tight line-height, constrained width
- Support: Minimal, light weight, spacious
- Maximum negative space (no CTA to compete)
"""

from __future__ import annotations

from .base import CompositionEngine


class HeroStatementEngine(CompositionEngine):
    """Bold headline with maximum negative space."""

    family_name = "hero_statement"
    description = "Large headline, no CTA, max negative space"
    required_elements = ["headline"]
    optional_elements = ["eyebrow", "support_copy"]
    forbidden_elements = ["cta"]
    typical_zones = ["center", "top_center", "bottom_center"]
    allowed_alignments = ["center", "left"]
    allowed_cta_styles = ["none"]
    default_container_type = "none"

    def render(self) -> str:
        """Generate HTML for hero statement with proper typography."""
        left, top, right, bottom = self._get_safe_inset()
        usable_width = self.tokens.get("usable_width", 1000)
        alignment = self._get_alignment_css()

        # Get typography values
        hl_min, hl_max = self.tokens.get("headline_size_range", (100, 150))
        sup_min, sup_max = self.tokens.get("support_size_range", (32, 48))
        hl_line_height = self.tokens.get("headline_line_height", 1.0)
        sup_line_height = self.tokens.get("support_line_height", 1.4)
        gap = self.tokens.get("gap_headline_support", 12)

        # For hero statement, use very tight max-width (dramatic effect)
        max_text_width = self._calculate_max_text_width(hl_max, optimal_chars=50)

        # Get colors and fonts
        hl_color = self._get_headline_color()
        sup_color = self._get_support_color()
        hl_role = self.intent.get("typography", {}).get("headline_role", "display_impact")
        sup_role = self.intent.get("typography", {}).get("support_role", "modern_sans")
        hl_font = self._get_font_for_role(hl_role)
        sup_font = self._get_font_for_role(sup_role)

        blocks = []

        # Eyebrow (optional, very minimal)
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
                letter-spacing: 0.1em;
                margin-bottom: {gap}px;
                text-align: {alignment};
            ">
                {eyebrow_content}
            </div>"""
            blocks.append(eyebrow_html)

        # Headline (required, VERY LARGE, dramatic)
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
            margin: 0 auto {gap}px;
            text-align: {alignment};
            letter-spacing: -0.02em;
        ">
            {line_html}
        </div>"""
        blocks.append(headline_html)

        # Support copy (minimal, optional)
        support_content = self.text_elements.get("support_copy", {}).get("content", "")
        if support_content:
            support_html = f"""<div style="
                font-family: {sup_font};
                font-size: {int(sup_max * 0.75)}px;
                font-weight: 300;
                line-height: {sup_line_height};
                color: {sup_color};
                max-width: {max_text_width}px;
                margin: 0 auto;
                text-align: {alignment};
                opacity: 0.85;
            ">
                {support_content}
            </div>"""
            blocks.append(support_html)

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
