"""
Generate 1 creative spec for ad generation based on classified input.

The spec now contains both the legacy top-level fields (for backward
compatibility with app.py and generate_prompt.py) AND a rich `text_design_spec`
object that upstream commits the design decisions BEFORE the HTML overlay
stage runs. See lib/ad_design_system.py and lib/text_design_spec.py.

We use the Anthropic `tools` parameter so the model must emit a schema-valid
JSON object — much more reliable than JSON-in-text parsing.
"""

from __future__ import annotations

import json

from anthropic import Anthropic

from . import ad_design_system as ads
from . import text_design_spec as tds

# ─────────────────────────────────────────────────────────────────────────────
# JSON SCHEMA for the tool input — forced structured output
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
    },
    "required": ["start", "end", "treatment"],
}

_ELEMENT_SCHEMA = {
    "type": "object",
    "properties": {
        "content":        {"type": "string"},
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
    "description": "Emit a single structured creative spec + text_design_spec for the ad.",
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
# SYSTEM PROMPT — walks Claude through the decision tree
# ─────────────────────────────────────────────────────────────────────────────

def _build_system_prompt() -> str:
    return f"""You are a senior performance creative director + editorial designer for Meta ads. You direct ads that feel like $5k campaign work — editorial restraint, strong hierarchy, intelligent copy. You have studied the typography and layout of high-performing ads from Poppi, Base44, Ridge, Kassable, AgelessRx, Apple, and classic editorial campaigns.

Your job is to design ONE ad: the creative angle, copy, and a full structured text_design_spec that commits the design decisions upstream. You MUST emit your response by calling the `emit_creative_spec` tool. Do not output free-form text.

DECISION TREE (follow in order):

1) Choose a tone_mode that fits the product and ad goal:
{chr(10).join(f"   - {k}: {v['notes']}" for k, v in ads.TONE_MODES.items())}

2) Choose a layout_family. You have 13 to choose from. Do NOT default to the same family every time:
{ads.describe_all_families()}

3) Decide which text_elements to populate. ONLY populate elements allowed by the chosen layout_family. Leave forbidden elements absent (null). Not every ad needs headline + support_copy + cta. Great ads are often headline-only or headline + cta only.

4) Write the copy. Copy rules:
   - Headlines: short, punchy, specific. Max ~8 words. No quotation marks around the headline.
   - Pain-point fragments: 2-4 period-separated short phrases (e.g. "Brain fog. Fatigue. Crashes.").
   - Questions: rhetorical hooks that imply the reader's state (e.g. "Doing everything right, but still feeling off?").
   - CTA copy: specific and action-oriented ("Find Real Deals", "Start Your Plan"), never generic "Learn More".
   - Do NOT use quotation marks inside copy.
   - You may use intra-headline emphasis via `emphasis_spans` (character ranges of the headline that switch to a different typography role — the Poppi move).

5) Choose placement.primary_zone from the 13 zones. Do not default to bottom_center. Editorial ads often use left_rail, right_rail, upper_third, or top_left.

6) Choose container_strategy. Most premium ads use `none` or `shadow_only`. `glass_blur` and `translucent_card` are good for SaaS. AVOID `gradient_panel` with `extent=lower_third` — that is the generic AI default.

7) Choose typography. Pick ONE primary_role and ONE (optional) accent_role. Never pair two display_impact fonts, never pair two editorial_serif fonts. CTA is almost always `Inter` 700 unless the family is soft_card_overlay with an editorial system.
   Roles:
{ads.describe_all_roles()}

8) Choose color_strategy. Do NOT default to white-on-dark-scrim. Read the image mood and pick a deliberate strategy:
{chr(10).join(f"   - {k}: {v}" for k, v in ads.COLOR_STRATEGIES.items())}

9) Choose cta_style. `none` and `tiny_anchor` are valid choices for editorial/premium layouts. The CTA should NOT always be a pill_filled button.
{ads.describe_all_cta_styles()}

10) Scrim: default disabled. Only enable if the layout family and image mood require it.

HARD RULES:
- If you choose layout_family `hero_statement`, you MUST leave support_copy, eyebrow, badge, and attribution absent.
- If you choose layout_family `minimal_product_led`, you MUST keep the total character count under ~30.
- If you choose cta_style.type = `none`, do NOT put any content in text_elements.cta.
- The `headline.content` is the ONLY required element for most families. Be selective.
- needsUi must come from the classification input.

Emit exactly ONE tool call."""


# ─────────────────────────────────────────────────────────────────────────────
# FALLBACK SPEC — used if the tool call fails
# ─────────────────────────────────────────────────────────────────────────────

def _fallback_spec(product_description: str, classification: dict) -> dict:
    needs_ui = bool(classification.get("needs_ui"))
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
                "headline": {"content": product_description[:60] or "Ready when you are", "emphasis_spans": []},
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
                "primary_family": "Inter", "primary_role": "display_impact",
                "accent_family": None, "accent_role": None,
                "cta_family": "Inter", "tracking": "tight", "case_style": "upper", "line_height": "tight",
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
    """
    Build 1 CreativeSpec object for ad generation.

    The returned spec contains:
        - legacy top-level fields:   headline, subheadline, cta, negativeSpaceZone, textTemplate
        - legacy control fields:     id, angle, sceneType, format, needsUi, uiPlacementType, assetsToUse
        - NEW rich design spec:      spec["text_design_spec"] (see lib/text_design_spec.py)
    """
    client = Anthropic()

    user_message = f"""Product: {product_description}
Product Type: {classification.get('product_type', 'other')}
Suggested Styles: {json.dumps(classification.get('likely_ad_styles', []))}
Needs UI: {classification.get('needs_ui', False)}
Ad Goal: {ad_goal or 'General awareness / conversion'}
Available Assets: logo={has_logo}, product_images={has_product_images}, ui_screenshots={has_ui_screenshots}

Design ONE ad. Walk through the decision tree in the system prompt. Emit the `emit_creative_spec` tool call."""

    try:
        response = client.messages.create(
            model="claude-sonnet-4-20250514",
            max_tokens=2500,
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

    # Derive legacy top-level fields for backward compat with app.py + generate_prompt.py
    legacy = tds.derive_legacy_fields(normalized_tds)
    raw_spec.update(legacy)

    # Ensure mandatory legacy fields exist
    raw_spec.setdefault("id", "ad_1")
    raw_spec.setdefault("format", "4:5")
    raw_spec.setdefault("assetsToUse", [])
    raw_spec.setdefault("uiPlacementType", None)

    return [raw_spec]
