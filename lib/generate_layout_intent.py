"""
Claude generates layout intent as structured JSON (via tools).

Claude decides WHAT (design intent); the system generates HOW (HTML/CSS).
Output is JSON, NOT HTML.
"""

from __future__ import annotations

import json
from typing import Any

import anthropic

# JSON schema for layout intent
_LAYOUT_INTENT_SCHEMA = {
    "type": "object",
    "properties": {
        "layout_family": {
            "type": "string",
            "enum": [
                "hero_statement",
                "hero_with_cta",
                "editorial_side_stack",
                "direct_response_stack",
                "pain_point_fragments",
                "question_hook",
                "testimonial_quote",
                "offer_badge_headline",
                "poster_background_headline",
                "soft_card_overlay",
                "split_message_cta",
                "minimal_product_led",
                "utility_explainer",
            ],
            "description": "Family of composition engines",
        },
        "text_elements": {
            "type": "object",
            "properties": {
                "eyebrow": {
                    "type": "object",
                    "properties": {
                        "content": {"type": "string"},
                        "present": {"type": "boolean"},
                    },
                    "required": ["present"],
                },
                "headline": {
                    "type": "object",
                    "properties": {
                        "content": {"type": "string"},
                        "lines": {
                            "type": "array",
                            "items": {"type": "string"},
                            "description": "semantic line breaks",
                        },
                    },
                    "required": ["content"],
                },
                "support_copy": {
                    "type": "object",
                    "properties": {
                        "content": {"type": "string"},
                        "present": {"type": "boolean"},
                    },
                    "required": ["present"],
                },
                "cta": {
                    "type": "object",
                    "properties": {
                        "content": {"type": "string"},
                        "present": {"type": "boolean"},
                    },
                    "required": ["present"],
                },
            },
            "required": ["headline"],
        },
        "typography": {
            "type": "object",
            "properties": {
                "headline_role": {
                    "type": "string",
                    "enum": [
                        "display_impact",
                        "editorial_serif",
                        "modern_sans",
                        "warm_serif",
                        "handwritten_accent",
                    ],
                },
                "support_role": {
                    "type": "string",
                    "enum": ["modern_sans", "editorial_serif", "warm_serif"],
                },
                "cta_font_role": {
                    "type": "string",
                    "enum": ["modern_sans", "display_impact"],
                },
                "emphasis_spans": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "properties": {
                            "start_char": {"type": "integer"},
                            "end_char": {"type": "integer"},
                            "treatment": {
                                "type": "string",
                                "enum": ["color_accent", "bold", "alternate_font"],
                            },
                        },
                        "required": ["start_char", "end_char", "treatment"],
                    },
                },
            },
            "required": ["headline_role"],
        },
        "placement": {
            "type": "object",
            "properties": {
                "primary_zone": {"type": "string"},
                "alignment": {"type": "string", "enum": ["left", "center", "right"]},
                "vertical_rhythm": {"type": "string", "enum": ["tight", "spacious"]},
            },
            "required": ["primary_zone", "alignment"],
        },
        "hierarchy": {
            "type": "object",
            "properties": {
                "headline_scale": {"type": "string", "enum": ["md", "lg", "xl", "xxl"]},
                "headline_max_lines": {"type": "integer", "minimum": 1, "maximum": 3},
                "support_max_lines": {"type": "integer", "minimum": 1, "maximum": 2},
                "density": {"type": "string", "enum": ["minimal", "moderate", "dense"]},
            },
            "required": ["headline_scale", "density"],
        },
        "cta_intent": {
            "type": "object",
            "properties": {
                "present": {"type": "boolean"},
                "style": {
                    "type": "string",
                    "enum": [
                        "pill_filled",
                        "rectangular_filled",
                        "ghost_outlined",
                        "underlined_text",
                        "text_arrow",
                        "badge_cta",
                        "tiny_anchor",
                        "none",
                    ],
                },
                "prominence": {"type": "string", "enum": ["restrained", "standard", "dominant"]},
            },
            "required": ["present"],
        },
        "container": {
            "type": "object",
            "properties": {
                "type": {
                    "type": "string",
                    "enum": [
                        "none",
                        "shadow_only",
                        "translucent_card",
                        "glass_blur",
                        "solid_chip",
                        "gradient_panel",
                        "outlined_card",
                        "hard_block",
                        "background_text_layer",
                    ],
                },
                "opacity_preference": {"type": "number", "minimum": 0.0, "maximum": 1.0},
                "blur_preference": {"type": "integer", "minimum": 0, "maximum": 20},
            },
            "required": ["type"],
        },
        "color": {
            "type": "object",
            "properties": {
                "mode": {
                    "type": "string",
                    "enum": ["light_on_dark_area", "dark_on_light_area"],
                },
                "use_accent": {"type": "boolean"},
                "accent_usage": {
                    "type": "string",
                    "enum": ["eyebrow", "eyebrow_and_emphasis", "eyebrow_and_cta"],
                },
            },
            "required": ["mode"],
        },
    },
    "required": [
        "layout_family",
        "text_elements",
        "typography",
        "placement",
        "hierarchy",
        "cta_intent",
        "container",
        "color",
    ],
}

_LAYOUT_INTENT_TOOL = {
    "name": "submit_layout_intent",
    "description": "Submit the layout design intent (not HTML). Claude decides WHAT (intent); system generates HOW (HTML).",
    "input_schema": _LAYOUT_INTENT_SCHEMA,
}


def generate_layout_intent(
    image_b64: str,
    image_media_type: str,
    product_description: str,
    tone_mode: str,
    text_design_spec: dict[str, Any],
    image_analysis: dict[str, Any],
    layout_tokens: dict[str, Any],
) -> dict[str, Any]:
    """Call Claude to generate layout intent as structured JSON.

    Claude chooses design intent; system will generate HTML from this intent.

    Args:
        image_b64: Base64-encoded image
        image_media_type: e.g., "image/jpeg"
        product_description: The product/ad copy
        tone_mode: e.g., "performance_ugc"
        text_design_spec: The full spec
        image_analysis: Brightness, colors, zones, etc.
        layout_tokens: Computed layout constraints

    Returns:
        dict: Validated layout intent JSON
    """
    system_prompt = """You are a senior creative director and layout strategist.

Your job: Decide the LAYOUT INTENT for an ad overlay. Do NOT generate HTML.

Return structured JSON that specifies:
- Which composition engine to use (layout_family)
- Which text elements to include
- Typography roles (headline, support, CTA)
- Emphasis strategy (color, weight, font mixing)
- Placement and alignment
- Hierarchy (scale, line limits, density)
- CTA presence/style
- Container strategy
- Color strategy (light-on-dark or dark-on-light)

This intent will be validated against family rules, then the system generates the final HTML/CSS deterministically.

Requirements:
1. Honor the layout families — each has rules about what elements are allowed
2. Match typography roles to the tone (performance_ugc needs snappy, editorial needs thoughtful)
3. Respect zone constraints (all text must fit in chosen zone)
4. Choose headline_scale based on text length and available space
5. CTA is optional; only include if it fits family rules
6. Emphasis is sparse — color or weight, not both
7. Container defaults to "none" for maximum clarity (override only if needed)
8. Use accent color sparingly (eyebrow only, or eyebrow+emphasis, never everything)
9. Alignment: center for hero families, left/right for editorial families
10. Density reflects the amount of text (minimal for 1-2 lines, dense for 5+ lines)

Composition families available:
- hero_statement: Bold headline, no CTA, max negative space
- hero_with_cta: Bold headline + optional support + CTA
- editorial_side_stack: Headline + support stacked, small CTA to side
- direct_response_stack: Headline + support + CTA stacked vertically
- pain_point_fragments: Multiple small text blocks
- question_hook: Q + A format
- testimonial_quote: Quote + attribution
- offer_badge_headline: Badge + headline + CTA
- poster_background_headline: Big faded text behind small foreground
- soft_card_overlay: All in soft-edged card
- split_message_cta: Two-column (message | CTA)
- minimal_product_led: Tiny text, 80% negative space
- utility_explainer: Inline callouts

Submit your layout intent as JSON using the provided tool."""

    user_msg = f"""Product description: {product_description}
Tone mode: {tone_mode}

Image brightness: {image_analysis.get("brightness", 0.5)}
Dominant hue: {image_analysis.get("dominant_hue")}
Accent color: {image_analysis.get("accent_color")}
Suggested text color: {image_analysis.get("suggested_text_color")}

Zone constraints: usable_width={layout_tokens.get("usable_width")}px, headline_size_range={layout_tokens.get("headline_size_range")}

Decide the layout intent now. Submit using the tool."""

    client = anthropic.Anthropic()
    response = client.messages.create(
        model="claude-sonnet-4-20250514",
        max_tokens=2000,
        system=system_prompt,
        tools=[_LAYOUT_INTENT_TOOL],
        messages=[
            {
                "role": "user",
                "content": [
                    {
                        "type": "image",
                        "source": {
                            "type": "base64",
                            "media_type": image_media_type,
                            "data": image_b64,
                        },
                    },
                    {"type": "text", "text": user_msg},
                ],
            }
        ],
    )

    # Extract tool use
    intent = None
    for block in response.content:
        if block.type == "tool_use" and block.name == "submit_layout_intent":
            intent = block.input
            break

    if not intent:
        raise ValueError("Claude did not return layout intent via tool")

    return intent
