"""
Generate HTML/CSS deterministically from validated layout intent + layout tokens.

System generates the final HTML; Claude is not involved here.
This ensures complete control over layout, spacing, overflow prevention.
"""

from __future__ import annotations

from typing import Any

from . import composition_engines as ce


def generate_html_from_intent(
    intent: dict[str, Any],
    layout_tokens: dict[str, Any],
    text_elements: dict[str, Any],
    image_analysis: dict[str, Any],
) -> str:
    """Deterministically generate HTML from validated intent.

    Args:
        intent: Validated layout intent JSON from Claude
        layout_tokens: Computed layout constraints
        text_elements: Headline, support_copy, cta, eyebrow content
        image_analysis: Image brightness, colors, zones, etc.

    Returns:
        str: Complete HTML document (<!DOCTYPE html> to </html>)
    """
    family_name = intent.get("layout_family", "direct_response_stack")

    try:
        engine_class = ce.get_engine_class(family_name)
        engine = engine_class(intent, layout_tokens, text_elements, image_analysis)
        html = engine.render()
        return html
    except Exception as exc:
        print(f"[COMPOSITION] Failed to render {family_name}: {exc}")
        # Fallback to simple direct response stack
        from lib.composition_engines.direct_response_stack import DirectResponseStackEngine

        engine = DirectResponseStackEngine(intent, layout_tokens, text_elements, image_analysis)
        return engine.render()
