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
        import traceback
        traceback.print_exc()
        # Fallback to simple direct response stack
        from lib.composition_engines.direct_response_stack import DirectResponseStackEngine

        try:
            engine = DirectResponseStackEngine(intent, layout_tokens, text_elements, image_analysis)
            return engine.render()
        except Exception as fallback_exc:
            print(f"[COMPOSITION] Fallback also failed: {fallback_exc}")
            traceback.print_exc()
            # Last resort: minimal HTML
            return """<!DOCTYPE html>
<html>
<head><meta charset="UTF-8"><style>* {margin:0;padding:0;box-sizing:border-box;} body {width:1080px;height:1350px;background:transparent;}</style></head>
<body><div style="position:absolute;left:20px;top:20px;color:white;font-size:40px;font-weight:bold;">Composition failed</div></body>
</html>"""
