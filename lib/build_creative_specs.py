"""
Phase 3: Generate 1 editorial-quality creative spec for ad generation.

This stage now:
1. Generates short, punchy headlines with semantic line-breaking
2. Adds intra-headline emphasis (color/style on key words)
3. Enforces eyebrow styling rules (small, tracked, accent color)
4. Defaults to NO container/scrim (text integrated into image)
5. Defaults CTA to none or tiny_anchor for editorial families
6. Uses extracted accent color from image for eyebrows/emphasis

All design decisions are made UPSTREAM here, so render stage is
a faithful implementor, not a creative director.
"""

from __future__ import annotations

import json

from anthropic import Anthropic

from . import ad_design_system as ads
from . import text_design_spec as tds

# ─────────────────────────────────────────────────────────────────────────────
# JSON SCHEMA — enforces structure
# ─────────────────────────────────────────────────────────────────────────────

_LAYOUT_FAMILY_KEYS   = list(ads.LAYOUT_FAMILIES.keys())
_ZONE_KEYS            = list(ads.ZONES.keys())
_CONTAINER_KEYS       = list(ads.CONTAINERS.keys())
_CTA_STYLE_KEYS       = list(ads.CTA_STYLES.keys())
_ROLE_KEYS            = list(ads.TYPOGRAPHY_ROLES.keys())
_TONE_KEYS            = list(ads.TONE_MODES.keys())
_COLOR_MODE_KEYS      = list(ads.COLOR_STRATEGIES.keys())

_EMPHASIS_SPAN_SCHEMA = {
    "type": "object",
    "properties": {
        "start":     {"type": "integer", "minimum": 0},
        "end":       {"type": "integer", "minimum": 0},
        "treatment": {"type": "string", "enum": _ROLE_KEYS},
        "color":     {"type": "string"},  # NEW: hex color for this span
    },
    "required": ["start", "end", "treatment"],
}

_ELEMENT_SCHEMA = {
    "type": "object",
    "properties": {
        "content":        {"type": "string"},
        "lines":          {"type": "array", "items": {"type": "string"}},  # NEW: semantic line breaks
        "case":           {"type": "string", "enum": ["upper", "title", "sentence", "as_written"]},
        "style":          {"type": "string"},
        "structure":      {"type": "string", "enum": ["sentence", "fragments", "bullet"]},
        "emphasis_spans": {"type": "array", "items": _EMPHASIS_SPAN_SCHEMA},
    },
    "required": ["content"],
}

_TEXT_DESIGN_SPEC_SCHEMA = {
    "type": "object",
    "properties": {
        "tone_mode":     {"type": "string", "enum": _TONE_KEYS},
        "layout_family": {"type": "string", "enum": _LAYOUT_FAMILY_KEYS},
        "text_density":  {"type": "string", "enum": ["none", "minimal", "moderate", "dense"]},

        "text_elements": {
            "type": "object",
            "properties": {
                "eyebrow":      _ELEMENT_SCHEMA,
                "badge":        _ELEMENT_SCHEMA,
                "headline":     _ELEMENT_SCHEMA,
                "support_copy": _ELEMENT_SCHEMA,
                "cta":          _ELEMENT_SCHEMA,
                "attribution":  _ELEMENT_SCHEMA,
            },
        },

        "hierarchy_profile": {
            "type": "object",
            "properties": {
                "dominant":       {"type": "string", "enum": ["headline", "badge", "support"]},
                "headline_scale": {"type": "string", "enum": ["md", "lg", "xl", "xxl"]},
                "support_scale":  {"type": ["string", "null"], "enum": ["xs", "sm", "md", None]},
                "cta_scale":      {"type": ["string", "null"], "enum": ["sm", "md", "lg", None]},
                "scale_ratio":    {"type": "number"},
            },
            "required": ["dominant", "headline_scale"],
        },

        "placement": {
            "type": "object",
            "properties": {
                "primary_zone":    {"type": "string", "enum": _ZONE_KEYS},
                "alignment":       {"type": "string", "enum": ["left", "center", "right"]},
                "margin_profile":  {"type": "string", "enum": ["tight", "standard", "generous"]},
                "vertical_rhythm": {"type": "string", "enum": ["compact", "spacious"]},
                "block_anchor":    {"type": "string"},
            },
            "required": ["primary_zone", "alignment"],
        },

        "container_strategy": {
            "type": "object",
            "properties": {
                "type":    {"type": "string", "enum": _CONTAINER_KEYS},
                "opacity": {"type": "number", "minimum": 0, "maximum": 1},
                "blur_px": {"type": "integer", "minimum": 0, "maximum": 40},
                "radius":  {"type": "string", "enum": ["none", "sharp", "soft", "pill"]},
                "padding": {"type": "string", "enum": ["tight", "standard", "generous"]},
            },
            "required": ["type"],
        },

        "typography": {
            "type": "object",
            "properties": {
                "primary_family": {"type": "string"},
                "primary_role":   {"type": "string", "enum": _ROLE_KEYS},
                "accent_family":  {"type": ["string", "null"]},
                "accent_role":    {"type": ["string", "null"], "enum": _ROLE_KEYS + [None]},
                "cta_family":     {"type": "string"},
                "tracking":       {"type": "string", "enum": ["tight", "normal", "loose"]},
                "case_style":     {"type": "string", "enum": ["upper", "title", "sentence", "as_written"]},
                "line_height":    {"type": "string", "enum": ["tight", "normal", "loose"]},
            },
            "required": ["primary_family", "primary_role", "cta_family"],
        },

        "color_strategy": {
            "type": "object",
            "properties": {
                "mode":            {"type": "string", "enum": _COLOR_MODE_KEYS},
                "headline_color":  {"type": "string"},
                "support_color":   {"type": "string"},
                "accent_color":    {"type": "string"},
                "cta_bg":          {"type": ["string", "null"]},
                "cta_fg":          {"type": ["string", "null"]},
            },
            "required": ["mode", "headline_color"],
        },

        "cta_style": {
            "type": "object",
            "properties": {
                "type":       {"type": "string", "enum": _CTA_STYLE_KEYS},
                "prominence": {"type": "string", "enum": ["anchor", "standard", "dominant"]},
            },
            "required": ["type"],
        },

        "scrim": {
            "type": "object",
            "properties": {
                "enabled":     {"type": "boolean"},
                "type":        {"type": ["string", "null"], "enum": ["gradient", "radial", "solid_card", "blur_only", None]},
                "extent":      {"type": ["string", "null"], "enum": ["lower_third", "upper_third", "spotlight", "full", None]},
                "max_opacity": {"type": "number", "minimum": 0, "maximum": 1},
            },
            "required": ["enabled"],
        },
    },
    "required": [
        "tone_mode", "layout_family", "text_density",
        "text_elements", "hierarchy_profile", "placement",
        "container_strategy", "typography", "color_strategy",
        "cta_style", "scrim",
    ],
}

_CREATIVE_SPEC_TOOL = {
    "name": "emit_creative_spec",
    "description": "Emit a single editorial-quality creative spec.",
    "input_schema": {
        "type": "object",
        "properties": {
            "id":    {"type": "string"},
            "angle": {"type": "string"},
            "sceneType":         {"type": "string"},
            "format":            {"type": "string", "const": "4:5"},
            "needsUi":           {"type": "boolean"},
            "uiPlacementType":   {"type": ["string", "null"], "enum": [
                "phone-in-hand", "laptop-on-desk", "tablet-on-table",
                "phone-on-surface", "floating-device", None,
            ]},
            "assetsToUse":       {"type": "array", "items": {"type": "string"}},
            "text_design_spec":  _TEXT_DESIGN_SPEC_SCHEMA,
        },
        "required": [
            "id", "angle", "sceneType", "format", "needsUi",
            "assetsToUse", "text_design_spec",
        ],
    },
}


# ─────────────────────────────────────────────────────────────────────────────
# SYSTEM PROMPT — hyper-editorial, decision-tree driven
# ─────────────────────────────────────────────────────────────────────────────

def _build_system_prompt() -> str:
    return f"""You are a world-class ad creative director and editorial designer. You design ads that sell like $5,000 professional campaigns. You have studied Poppi, Base44, Ridge, Kassable, Real Deals, and StreetEasy. You understand that the best ad text is:
    - Punchy and specific (not generic)
    - Hierarchical (headline dominates by 3-5x visual weight)
    - Integrated into the image (not pasted on top)
    - Editorial and intentional (not performance-marketing default)
    - Sometimes missing elements (a strong design choice is NO CTA, or NO subheadline)

Your job is to design ONE ad. You must emit a complete `text_design_spec` by calling the `emit_creative_spec` tool.

═══════════════════════════════════════════════════════════════════════════════
DESIGN DECISION TREE
═══════════════════════════════════════════════════════════════════════════════

STEP 1: Choose tone_mode
{chr(10).join(f"  - {k}: {v['notes']}" for k, v in ads.TONE_MODES.items())}

STEP 2: Choose layout_family
{ads.describe_all_families()}

STEP 3: Generate the headline
  - Goal: Punchy, specific, memorable. 3-10 words max.
  - Avoid generic filler ("Learn More", "Discover", "Explore").
  - If 5+ words, prepare to add emphasis_spans (1-2 key words get color/style).
  - Examples of good headlines:
    * "One Sip & You're Starstruck" (5 words, potential emphasis on "Starstruck")
    * "Actually affordable apartments. Actually in NYC." (fragmented, 2-3 semantic chunks)
    * "Been scrolling for hours. Find real deals in minutes." (pain-point hook)
    * "Brain fog. Fatigue. Crashes." (fragments, 1 word each)

STEP 4: Break headline into semantic lines
  - Do NOT word-wrap. Break on meaning.
  - Typical: 1-3 lines max.
  - Examples:
    * "One Sip & You're Starstruck" → ["One Sip &", "You're Starstruck"]
    * "Actually affordable apartments. Actually in NYC." → ["Actually affordable", "apartments", "Actually in NYC"]
    * "Been scrolling for hours and found nothing good?" → ["Been scrolling", "for hours and", "found nothing good?"]
  - Put these in headline.lines array.

STEP 5: Add intra-headline emphasis (if 5+ words)
  - Pick 1-2 key words or phrases.
  - Assign them a treatment (role: display_impact, editorial_serif, warm_serif, etc).
  - Optionally assign an accent color.
  - Example:
    * headline.content = "One Sip & You're Starstruck"
    * headline.emphasis_spans = [
        {{"start": 13, "end": 27, "treatment": "editorial_serif", "color": "#FF6A3D"}}
      ]
    * This makes "You're Starstruck" render in editorial serif + accent color.

STEP 6: Generate eyebrow (if appropriate)
  - Only for editorial, premium, or utility families.
  - Keep it SHORT (max 5-6 words).
  - Uppercase, tracked, use the extracted accent color.
  - Example: "BELOW-MARKET HOUSING", "THE #1 PLACE", "SUMMER EDITION"
  - Avoid: eyebrows that are just generic labels.

STEP 7: Decide support_copy
  - Many great ads have NO support copy.
  - When present: 1-2 short sentences max.
  - Support should NOT compete with headline (3-5x smaller).
  - If layout_family forbids it, do NOT add it.

STEP 8: Decide CTA
  - Editorial families (hero_statement, editorial_side_stack, testimonial_quote):
    default to `none` or `tiny_anchor` (optional, restrained).
  - Performance families (direct_response_stack, offer_badge_headline):
    default to `pill_filled` or `rectangular_filled` (present, clear).
  - CTA copy must be specific to the product/offer.
    Bad: "Learn More", "Click Here", "Shop Now"
    Good: "Find Real Deals", "Get Paid Today", "Search Now"

STEP 9: Choose placement.primary_zone
  - Prefer quiet zones (from image_analysis.quietest_zones if provided).
  - Avoid centered-bottom defaults. Use top-left, bottom-right, left_rail, etc.
  - Editorial > center. Performance > can center if needed.

STEP 10: Choose typography
  - Headline: one role (display_impact, editorial_serif, modern_sans, etc).
  - Eyebrow + support: complementary role, NO MORE than 2 families total.
  - Font pairing rules: never two display_impact, never two editorial_serif.

STEP 11: Choose color_strategy
  - Do NOT default to white-on-dark.
  - Use image mood to guide: warm_tonal, cool_tonal, brand_accent, etc.
  - Use the extracted accent_color from image analysis for eyebrow/emphasis.

STEP 12: Choose container_strategy
  - DEFAULT: `none` or `shadow_only`.
  - Only use translucent_card, glass_blur, gradient_panel if image is busy OR layout family requires it.
  - NO generic bottom scrims.

STEP 13: Set scrim.enabled
  - DEFAULT: `false`.
  - Only enable if text readability requires it (very busy background).
  - Prefer shadow-only contrast over scrim.

STEP 14: Choose hierarchy_profile
  - headline_scale = function of word count:
    * 3-5 words → `xxl` (160-200px)
    * 6-9 words → `xl` (110-140px)
    * 10-14 words → `lg` (80-110px)
  - scale_ratio = target visual weight ratio (headline : support). Aim for 4-6x.

═══════════════════════════════════════════════════════════════════════════════
HARD RULES
═══════════════════════════════════════════════════════════════════════════════
- Every headline is <10 words (aim for 3-8).
- Do NOT use quotation marks around headlines.
- If layout_family forbids an element, do NOT include it.
- Eyebrow is optional BUT if present: always 28-32px, uppercase, tracked.
- CTA is optional. If none, cta_style.type = "none".
- Never use generic filler copy.
- No more than 2 font families total (primary + accent).
- Accent color comes from image_analysis.accent_color.
- Container default is "none", NOT "translucent_card" or "gradient_panel".

═══════════════════════════════════════════════════════════════════════════════
REMEMBER
═══════════════════════════════════════════════════════════════════════════════
You are designing editorial-quality ads. Think like a magazine creative director.
Restraint is a virtue. Negative space is design. A missing CTA is sometimes the
right choice. Typography is destiny.

Emit the `emit_creative_spec` tool call now."""


# ─────────────────────────────────────────────────────────────────────────────
# FALLBACK SPEC
# ─────────────────────────────────────────────────────────────────────────────

def _fallback_spec(product_description: str, classification: dict) -> dict:
    needs_ui = bool(classification.get("needs_ui"))
    headline_copy = product_description[:50] or "Ready when you are"
    return {
        "id": "ad_1",
        "angle": "lifestyle",
        "sceneType": "lifestyle-home",
        "format": "4:5",
        "needsUi": needs_ui,
        "uiPlacementType": "phone-in-hand" if needs_ui else None,
        "assetsToUse": [],
        "text_design_spec": {
            "tone_mode": "performance_ugc",
            "layout_family": "hero_with_cta",
            "text_density": "minimal",
            "text_elements": {
                "eyebrow": None,
                "badge": None,
                "headline": {
                    "content": headline_copy,
                    "lines": [headline_copy],
                    "emphasis_spans": [],
                },
                "support_copy": None,
                "cta": {"content": "Get Started"},
                "attribution": None,
            },
            "hierarchy_profile": {
                "dominant": "headline",
                "headline_scale": "xl",
                "cta_scale": "md",
                "scale_ratio": 6.0,
            },
            "placement": {
                "primary_zone": "lower_third",
                "alignment": "center",
                "margin_profile": "standard",
                "vertical_rhythm": "spacious",
                "block_anchor": "bottom-center",
            },
            "container_strategy": {"type": "none", "opacity": 0.0, "blur_px": 0, "radius": "none", "padding": "standard"},
            "typography": {
                "primary_family": "Inter",
                "primary_role": "display_impact",
                "accent_family": None,
                "accent_role": None,
                "cta_family": "Inter",
                "tracking": "tight",
                "case_style": "upper",
                "line_height": "tight",
            },
            "color_strategy": {
                "mode": "light_on_dark_area",
                "headline_color": "#FFFFFF",
                "support_color": "#EBEBEB",
                "accent_color": "#FFFFFF",
                "cta_bg": "#FFFFFF",
                "cta_fg": "#111111",
            },
            "cta_style": {"type": "pill_filled", "prominence": "standard"},
            "scrim": {"enabled": False, "type": None, "extent": None, "max_opacity": 0.0},
        },
    }


# ─────────────────────────────────────────────────────────────────────────────
# PUBLIC API
# ─────────────────────────────────────────────────────────────────────────────

def build_creative_specs(
    product_description: str,
    classification: dict,
    ad_goal: str = "",
    has_logo: bool = False,
    has_product_images: bool = False,
    has_ui_screenshots: bool = False,
) -> list[dict]:
    """Build 1 editorial-quality creative spec."""
    client = Anthropic()

    user_message = f"""Product: {product_description}
Product Type: {classification.get('product_type', 'other')}
Suggested Styles: {json.dumps(classification.get('likely_ad_styles', []))}
Needs UI: {classification.get('needs_ui', False)}
Ad Goal: {ad_goal or 'General awareness / conversion'}
Available Assets: logo={has_logo}, product_images={has_product_images}, ui_screenshots={has_ui_screenshots}

Design ONE editorial-quality ad. Walk through the decision tree in the system prompt. Emit the `emit_creative_spec` tool call."""

    try:
        response = client.messages.create(
            model="claude-sonnet-4-20250514",
            max_tokens=3000,
            system=_build_system_prompt(),
            tools=[_CREATIVE_SPEC_TOOL],
            tool_choice={"type": "tool", "name": "emit_creative_spec"},
            messages=[{"role": "user", "content": user_message}],
        )

        tool_block = None
        for block in response.content:
            if block.type == "tool_use" and block.name == "emit_creative_spec":
                tool_block = block
                break

        if tool_block is None:
            raise ValueError("No emit_creative_spec tool call in response")

        raw_spec: dict = dict(tool_block.input)
    except Exception as exc:
        print(f"[BUILD_SPEC] tool call failed: {exc} — using fallback spec")
        raw_spec = _fallback_spec(product_description, classification)

    # Normalize + validate the text_design_spec
    raw_tds = raw_spec.get("text_design_spec") or {}
    normalized_tds = tds.normalize(raw_tds)
    ok, violations = tds.validate(normalized_tds)
    if not ok:
        print(f"[BUILD_SPEC] spec violations (normalized anyway): {violations}")
    raw_spec["text_design_spec"] = normalized_tds

    # Derive legacy top-level fields for backward compat
    legacy = tds.derive_legacy_fields(normalized_tds)
    raw_spec.update(legacy)

    # Ensure mandatory legacy fields exist
    raw_spec.setdefault("id", "ad_1")
    raw_spec.setdefault("format", "4:5")
    raw_spec.setdefault("assetsToUse", [])
    raw_spec.setdefault("uiPlacementType", None)

    return [raw_spec]
