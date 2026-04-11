"""
Ad Design System — structured design grammar for high-converting ad text.

This module is PURE DATA + small helpers. No network, no Claude calls. It is
the single source of truth for:

    - Layout families (compositional archetypes)
    - Placement zones (13 zones on the 1080x1350 canvas)
    - Container strategies
    - CTA styles
    - Typography roles and font pairing rules
    - Tone modes
    - Color strategies
    - Quality targets (what to avoid / require)
    - Cross-field validators and resolvers

It is consumed by:
    - lib/build_creative_specs.py  -> to describe the design grammar to Claude
    - lib/text_design_spec.py      -> to normalize + validate specs
    - lib/render_svg_overlay.py    -> to build the HTML overlay directive
"""

from __future__ import annotations

from typing import Any

# ─────────────────────────────────────────────────────────────────────────────
# CANVAS GEOMETRY  (1080 x 1350, Meta 4:5 feed)
# ─────────────────────────────────────────────────────────────────────────────

CANVAS_W = 1080
CANVAS_H = 1350
SAFE_PAD = 60


def _rect(x: int, y: int, w: int, h: int) -> dict[str, int]:
    return {"x": x, "y": y, "w": w, "h": h}


# 9 cell zones (3x3 grid inside safe padding) + 4 band zones
ZONES: dict[str, dict[str, Any]] = {
    # top row
    "top_left":    {"rect": _rect(60,  60,  320, 410), "anchor": "top-left",    "safe_padding": 24},
    "top_center":  {"rect": _rect(380, 60,  320, 410), "anchor": "top-center",  "safe_padding": 24},
    "top_right":   {"rect": _rect(700, 60,  320, 410), "anchor": "top-right",   "safe_padding": 24},
    # middle row
    "middle_left":  {"rect": _rect(60,  470, 320, 410), "anchor": "middle-left",  "safe_padding": 24},
    "center":       {"rect": _rect(380, 470, 320, 410), "anchor": "center",       "safe_padding": 24},
    "middle_right": {"rect": _rect(700, 470, 320, 410), "anchor": "middle-right", "safe_padding": 24},
    # bottom row
    "bottom_left":   {"rect": _rect(60,  880, 320, 410), "anchor": "bottom-left",   "safe_padding": 24},
    "bottom_center": {"rect": _rect(380, 880, 320, 410), "anchor": "bottom-center", "safe_padding": 24},
    "bottom_right":  {"rect": _rect(700, 880, 320, 410), "anchor": "bottom-right",  "safe_padding": 24},
    # band zones
    "upper_third":   {"rect": _rect(60,  60,  960, 410), "anchor": "top-center",     "safe_padding": 32},
    "lower_third":   {"rect": _rect(60,  880, 960, 410), "anchor": "bottom-center",  "safe_padding": 32},
    "left_rail":     {"rect": _rect(60,  60,  500, 1230), "anchor": "middle-left",   "safe_padding": 32},
    "right_rail":    {"rect": _rect(520, 60,  500, 1230), "anchor": "middle-right",  "safe_padding": 32},
}

# Mapping of new 13-zone vocabulary -> legacy 6-value negativeSpaceZone
# (for back-compat with lib/generate_prompt.py)
LEGACY_ZONE_MAP: dict[str, str] = {
    "top_left":      "top-left",
    "top_center":    "top-center",
    "top_right":     "top-right",
    "middle_left":   "top-left",
    "center":        "top-center",
    "middle_right":  "top-right",
    "bottom_left":   "bottom-left",
    "bottom_center": "bottom-center",
    "bottom_right":  "bottom-right",
    "upper_third":   "top-center",
    "lower_third":   "bottom-center",
    "left_rail":     "top-left",
    "right_rail":    "top-right",
}


def zone_bounds(zone_key: str) -> dict[str, int]:
    """Return the (x, y, w, h) rect for a zone, defaulting to lower_third."""
    return ZONES.get(zone_key, ZONES["lower_third"])["rect"]


def to_legacy_zone(zone_key: str) -> str:
    return LEGACY_ZONE_MAP.get(zone_key, "bottom-center")


# ─────────────────────────────────────────────────────────────────────────────
# TYPOGRAPHY ROLES
# ─────────────────────────────────────────────────────────────────────────────
# Each role lists fonts available to it. The font-family strings MUST match
# the @font-face declarations injected by lib/render_svg_overlay.py.

TYPOGRAPHY_ROLES: dict[str, list[dict[str, str]]] = {
    "display_impact": [
        {"family": "Bebas Neue",       "weight": "400", "style": "normal"},
        {"family": "Anton",            "weight": "400", "style": "normal"},
        {"family": "Oswald",           "weight": "700", "style": "normal"},
        {"family": "Montserrat",       "weight": "900", "style": "normal"},
        {"family": "Inter",            "weight": "900", "style": "normal"},
    ],
    "modern_sans": [
        {"family": "Inter",            "weight": "700", "style": "normal"},
        {"family": "Inter",            "weight": "400", "style": "normal"},
        {"family": "Poppins",          "weight": "700", "style": "normal"},
        {"family": "Poppins",          "weight": "600", "style": "normal"},
        {"family": "Poppins",          "weight": "400", "style": "normal"},
        {"family": "Space Grotesk",    "weight": "700", "style": "normal"},
        {"family": "Montserrat",       "weight": "700", "style": "normal"},
    ],
    "editorial_serif": [
        {"family": "Playfair Display",      "weight": "700", "style": "italic"},
        {"family": "DM Serif Display",      "weight": "400", "style": "italic"},
        {"family": "Libre Baskerville",     "weight": "700", "style": "italic"},
        {"family": "Libre Baskerville",     "weight": "400", "style": "italic"},
        {"family": "Cormorant Garamond",    "weight": "700", "style": "italic"},
    ],
    "warm_serif": [
        {"family": "Lora",                  "weight": "400", "style": "italic"},
        {"family": "Libre Baskerville",     "weight": "400", "style": "italic"},
    ],
    "handwritten_accent": [
        {"family": "Caveat",                "weight": "700", "style": "normal"},
        {"family": "Caveat",                "weight": "400", "style": "normal"},
    ],
}

# Legal pair combinations: primary_role -> allowed accent_role values.
# (Never pair two display_impact. Never pair two editorial_serif.)
ROLE_PAIRS: dict[str, list[str]] = {
    "display_impact":    ["modern_sans", "editorial_serif", "handwritten_accent", None],
    "modern_sans":       ["editorial_serif", "display_impact", "handwritten_accent", None],
    "editorial_serif":   ["modern_sans", "handwritten_accent", None],
    "warm_serif":        ["modern_sans", "handwritten_accent", None],
    "handwritten_accent":["modern_sans", "editorial_serif", None],
}


def resolve_font_pair(primary_role: str, tone_mode: str, layout_family: str) -> dict[str, Any]:
    """Return a safe default font pairing for the chosen role combination."""
    primary_candidates = TYPOGRAPHY_ROLES.get(primary_role, TYPOGRAPHY_ROLES["modern_sans"])
    primary = primary_candidates[0]
    accent_role_candidates = ROLE_PAIRS.get(primary_role, [None])
    accent_role = accent_role_candidates[0] if accent_role_candidates else None
    accent = TYPOGRAPHY_ROLES[accent_role][0] if accent_role else None
    return {
        "primary_family": primary["family"],
        "primary_role":   primary_role,
        "accent_family":  accent["family"] if accent else None,
        "accent_role":    accent_role,
        "cta_family":     "Inter",  # CTA is almost always Inter 700
    }


# ─────────────────────────────────────────────────────────────────────────────
# TONE MODES
# ─────────────────────────────────────────────────────────────────────────────

TONE_MODES: dict[str, dict[str, Any]] = {
    "editorial_premium": {
        "preferred_roles":    ["editorial_serif", "display_impact"],
        "preferred_density":  ["none", "minimal"],
        "cta_bias":           ["none", "tiny_anchor", "underlined_text", "text_arrow"],
        "notes":              "Restraint, dramatic hierarchy, image-led composition.",
    },
    "performance_ugc": {
        "preferred_roles":    ["display_impact", "modern_sans"],
        "preferred_density":  ["minimal", "moderate"],
        "cta_bias":           ["pill_filled", "rectangular_filled", "ghost_outlined"],
        "notes":              "Strong hooks, obvious clarity, fast hierarchy.",
    },
    "direct_response": {
        "preferred_roles":    ["modern_sans", "display_impact"],
        "preferred_density":  ["moderate", "dense"],
        "cta_bias":           ["pill_filled", "rectangular_filled"],
        "notes":              "Offer, outcome, button. Mobile-first readability.",
    },
    "hybrid": {
        "preferred_roles":    ["modern_sans", "editorial_serif"],
        "preferred_density":  ["minimal", "moderate"],
        "cta_bias":           ["pill_filled", "text_arrow", "ghost_outlined"],
        "notes":              "Editorial look, performance bones.",
    },
}


# ─────────────────────────────────────────────────────────────────────────────
# LAYOUT FAMILIES  (13)
# ─────────────────────────────────────────────────────────────────────────────

LAYOUT_FAMILIES: dict[str, dict[str, Any]] = {
    "hero_statement": {
        "description":       "One massive headline. Nothing else. Poppi-style.",
        "required_elements": ["headline"],
        "allowed_elements":  ["headline"],
        "forbidden_elements":["support_copy", "eyebrow", "badge", "attribution"],
        "typical_density":   "minimal",
        "typical_zones":     ["center", "upper_third", "lower_third"],
        "typical_containers":["none", "shadow_only"],
        "cta_behavior":      ["none", "tiny_anchor"],
        "typography_bias":   ["display_impact", "editorial_serif"],
    },
    "hero_with_cta": {
        "description":       "Huge headline + single CTA. Subway-app style.",
        "required_elements": ["headline", "cta"],
        "allowed_elements":  ["headline", "cta"],
        "forbidden_elements":["support_copy", "eyebrow", "badge", "attribution"],
        "typical_density":   "minimal",
        "typical_zones":     ["lower_third", "upper_third", "center"],
        "typical_containers":["none", "shadow_only"],
        "cta_behavior":      ["pill_filled", "rectangular_filled", "text_arrow", "underlined_text"],
        "typography_bias":   ["display_impact", "modern_sans"],
    },
    "editorial_side_stack": {
        "description":       "Eyebrow + headline + support stacked left or right. Magazine feel.",
        "required_elements": ["headline"],
        "allowed_elements":  ["eyebrow", "headline", "support_copy", "cta"],
        "forbidden_elements":["badge", "attribution"],
        "typical_density":   "moderate",
        "typical_zones":     ["left_rail", "right_rail", "bottom_left", "top_left"],
        "typical_containers":["none", "shadow_only", "translucent_card"],
        "cta_behavior":      ["text_arrow", "underlined_text", "ghost_outlined", "none"],
        "typography_bias":   ["editorial_serif", "modern_sans"],
    },
    "direct_response_stack": {
        "description":       "Headline + short support + prominent CTA. Performance classic.",
        "required_elements": ["headline", "cta"],
        "allowed_elements":  ["eyebrow", "headline", "support_copy", "cta"],
        "forbidden_elements":["badge", "attribution"],
        "typical_density":   "moderate",
        "typical_zones":     ["lower_third", "bottom_center", "bottom_left"],
        "typical_containers":["none", "shadow_only", "gradient_panel", "glass_blur"],
        "cta_behavior":      ["pill_filled", "rectangular_filled"],
        "typography_bias":   ["display_impact", "modern_sans"],
    },
    "pain_point_fragments": {
        "description":       "2-4 short fragments period-separated. 'Brain fog. Fatigue. Crashes.'",
        "required_elements": ["headline"],
        "allowed_elements":  ["headline", "support_copy", "cta"],
        "forbidden_elements":["eyebrow", "badge", "attribution"],
        "typical_density":   "minimal",
        "typical_zones":     ["left_rail", "upper_third", "lower_third"],
        "typical_containers":["none", "shadow_only"],
        "cta_behavior":      ["pill_filled", "text_arrow", "none"],
        "typography_bias":   ["display_impact", "modern_sans", "editorial_serif"],
    },
    "question_hook": {
        "description":       "Question headline engages reader. 'Doing everything right, but still feeling off?'",
        "required_elements": ["headline"],
        "allowed_elements":  ["headline", "support_copy", "cta"],
        "forbidden_elements":["eyebrow", "badge", "attribution"],
        "typical_density":   "moderate",
        "typical_zones":     ["upper_third", "lower_third", "center"],
        "typical_containers":["none", "shadow_only", "translucent_card"],
        "cta_behavior":      ["pill_filled", "text_arrow", "ghost_outlined"],
        "typography_bias":   ["editorial_serif", "modern_sans"],
    },
    "testimonial_quote": {
        "description":       "Pull quote + attribution. Social proof.",
        "required_elements": ["headline", "attribution"],
        "allowed_elements":  ["headline", "attribution", "cta"],
        "forbidden_elements":["eyebrow", "badge", "support_copy"],
        "typical_density":   "minimal",
        "typical_zones":     ["center", "lower_third", "upper_third"],
        "typical_containers":["none", "shadow_only", "translucent_card"],
        "cta_behavior":      ["text_arrow", "underlined_text", "none"],
        "typography_bias":   ["editorial_serif", "warm_serif"],
    },
    "offer_badge_headline": {
        "description":       "Prominent offer badge/chip + headline.",
        "required_elements": ["badge", "headline"],
        "allowed_elements":  ["badge", "headline", "support_copy", "cta"],
        "forbidden_elements":["eyebrow", "attribution"],
        "typical_density":   "moderate",
        "typical_zones":     ["lower_third", "bottom_center", "bottom_left", "top_left"],
        "typical_containers":["solid_chip", "none", "shadow_only"],
        "cta_behavior":      ["pill_filled", "rectangular_filled", "text_arrow"],
        "typography_bias":   ["display_impact", "modern_sans"],
    },
    "poster_background_headline": {
        "description":       "Oversized display typography dominating the frame.",
        "required_elements": ["headline"],
        "allowed_elements":  ["headline", "attribution"],
        "forbidden_elements":["eyebrow", "badge", "support_copy", "cta"],
        "typical_density":   "minimal",
        "typical_zones":     ["upper_third", "lower_third", "center", "left_rail", "right_rail"],
        "typical_containers":["background_text_layer", "none", "shadow_only"],
        "cta_behavior":      ["none"],
        "typography_bias":   ["display_impact", "editorial_serif"],
    },
    "soft_card_overlay": {
        "description":       "Glass or translucent card holding all copy. Clean SaaS.",
        "required_elements": ["headline"],
        "allowed_elements":  ["eyebrow", "headline", "support_copy", "cta"],
        "forbidden_elements":["badge", "attribution"],
        "typical_density":   "moderate",
        "typical_zones":     ["bottom_center", "center", "lower_third"],
        "typical_containers":["glass_blur", "translucent_card", "outlined_card"],
        "cta_behavior":      ["pill_filled", "rectangular_filled", "text_arrow"],
        "typography_bias":   ["modern_sans", "editorial_serif"],
    },
    "split_message_cta": {
        "description":       "Headline on one side of frame, CTA on another. Editorial asymmetry.",
        "required_elements": ["headline", "cta"],
        "allowed_elements":  ["eyebrow", "headline", "support_copy", "cta"],
        "forbidden_elements":["badge", "attribution"],
        "typical_density":   "minimal",
        "typical_zones":     ["left_rail", "right_rail", "top_left", "bottom_right"],
        "typical_containers":["none", "shadow_only", "solid_chip"],
        "cta_behavior":      ["pill_filled", "ghost_outlined", "text_arrow"],
        "typography_bias":   ["display_impact", "modern_sans", "editorial_serif"],
    },
    "minimal_product_led": {
        "description":       "Tiny anchor text only. Product does the talking. Apple/Ridge.",
        "required_elements": [],
        "allowed_elements":  ["headline", "cta"],
        "forbidden_elements":["eyebrow", "badge", "support_copy", "attribution"],
        "typical_density":   "minimal",
        "typical_zones":     ["bottom_left", "bottom_right", "bottom_center", "top_left"],
        "typical_containers":["none", "shadow_only"],
        "cta_behavior":      ["none", "tiny_anchor", "underlined_text"],
        "typography_bias":   ["modern_sans", "editorial_serif"],
    },
    "utility_explainer": {
        "description":       "Eyebrow + headline + 1-2 bullet fragments + CTA. SaaS feature ad.",
        "required_elements": ["headline", "cta"],
        "allowed_elements":  ["eyebrow", "headline", "support_copy", "cta"],
        "forbidden_elements":["badge", "attribution"],
        "typical_density":   "dense",
        "typical_zones":     ["lower_third", "bottom_center", "left_rail"],
        "typical_containers":["glass_blur", "translucent_card", "none", "gradient_panel"],
        "cta_behavior":      ["pill_filled", "rectangular_filled", "text_arrow"],
        "typography_bias":   ["modern_sans", "display_impact"],
    },
}


def allowed_elements(layout_family: str) -> list[str]:
    fam = LAYOUT_FAMILIES.get(layout_family)
    if not fam:
        return ["headline", "support_copy", "cta"]
    return fam["allowed_elements"]


def forbidden_elements(layout_family: str) -> list[str]:
    fam = LAYOUT_FAMILIES.get(layout_family)
    if not fam:
        return []
    return fam["forbidden_elements"]


def required_elements(layout_family: str) -> list[str]:
    fam = LAYOUT_FAMILIES.get(layout_family)
    if not fam:
        return ["headline"]
    return fam["required_elements"]


def describe_family_for_prompt(family_key: str) -> str:
    """Compact English description Claude can ingest."""
    fam = LAYOUT_FAMILIES.get(family_key)
    if not fam:
        return family_key
    return (
        f"{family_key}: {fam['description']} "
        f"Required: {fam['required_elements']}. "
        f"Forbidden: {fam['forbidden_elements']}. "
        f"Typical density: {fam['typical_density']}. "
        f"CTA behavior: {fam['cta_behavior']}."
    )


# ─────────────────────────────────────────────────────────────────────────────
# CONTAINER STRATEGIES  (9)
# ─────────────────────────────────────────────────────────────────────────────

CONTAINERS: dict[str, dict[str, Any]] = {
    "none": {
        "description": "Naked text directly on image. Text-shadow optional.",
        "opacity_range": (0.0, 0.0),
        "blur_range":    (0, 0),
        "radius":        "none",
    },
    "shadow_only": {
        "description": "No container. Heavy text-shadow for contrast.",
        "opacity_range": (0.0, 0.0),
        "blur_range":    (0, 0),
        "radius":        "none",
    },
    "translucent_card": {
        "description": "Semi-transparent solid card. rgba(255,255,255,0.8..0.95) or rgba(0,0,0,0.55..0.75).",
        "opacity_range": (0.55, 0.95),
        "blur_range":    (0, 0),
        "radius":        "soft",
    },
    "glass_blur": {
        "description": "backdrop-filter blur card. Modern SaaS / premium.",
        "opacity_range": (0.30, 0.72),
        "blur_range":    (16, 32),
        "radius":        "soft",
    },
    "solid_chip": {
        "description": "Small high-contrast opaque chip. Eyebrows and badges only.",
        "opacity_range": (0.90, 1.0),
        "blur_range":    (0, 0),
        "radius":        "pill",
    },
    "gradient_panel": {
        "description": "linear-gradient strip fading from opaque to transparent.",
        "opacity_range": (0.5, 0.85),
        "blur_range":    (0, 0),
        "radius":        "none",
    },
    "outlined_card": {
        "description": "2px stroke card, transparent interior.",
        "opacity_range": (0.0, 0.0),
        "blur_range":    (0, 0),
        "radius":        "soft",
    },
    "hard_block": {
        "description": "Solid opaque rectangle. Poster look.",
        "opacity_range": (1.0, 1.0),
        "blur_range":    (0, 0),
        "radius":        "sharp",
    },
    "background_text_layer": {
        "description": "Oversized tinted text behind subject (z-index illusion).",
        "opacity_range": (0.12, 0.35),
        "blur_range":    (0, 0),
        "radius":        "none",
    },
}


# ─────────────────────────────────────────────────────────────────────────────
# CTA STYLES  (8)
# ─────────────────────────────────────────────────────────────────────────────

CTA_STYLES: dict[str, dict[str, Any]] = {
    "none":               {"description": "No CTA."},
    "pill_filled":        {"description": "Rounded pill, solid fill, min 56px tall, radius >= 28px."},
    "rectangular_filled": {"description": "Sharp/soft corner filled button."},
    "ghost_outlined":     {"description": "2px stroke, no fill. Transparent interior."},
    "underlined_text":    {"description": "Text with 2-3px underline. No button box."},
    "text_arrow":         {"description": "Text followed by an arrow glyph (→). Editorial."},
    "badge_cta":          {"description": "Small high-contrast badge. Often uppercase and tracked."},
    "tiny_anchor":        {"description": "Bottom-corner tiny text. Extreme restraint."},
}


def resolve_cta_style(layout_family: str, tone_mode: str, cta_content: str | None) -> list[str]:
    """Return the list of CTA styles legal for this family/tone combination."""
    if cta_content is None or cta_content.strip() == "":
        return ["none"]
    fam = LAYOUT_FAMILIES.get(layout_family, {})
    allowed = fam.get("cta_behavior") or ["pill_filled"]
    # Filter by tone bias if tone has a bias list
    tone = TONE_MODES.get(tone_mode, {})
    biased = tone.get("cta_bias") or []
    if biased:
        prioritized = [c for c in allowed if c in biased]
        if prioritized:
            return prioritized
    return allowed


# ─────────────────────────────────────────────────────────────────────────────
# COLOR STRATEGIES
# ─────────────────────────────────────────────────────────────────────────────

COLOR_STRATEGIES: dict[str, str] = {
    "light_on_dark_area":  "White/near-white text placed where the image is dark.",
    "dark_on_light_area":  "Near-black text placed where the image is bright.",
    "brand_accent":        "Headline in a single vivid accent color derived from the image palette.",
    "monochrome_image":    "Image tinted monochrome; text in white or brand accent.",
    "warm_tonal":          "Warm cream/peach text matching a warm-dominant image.",
    "cool_tonal":          "Cool blue/steel text matching a cool-dominant image.",
}


# ─────────────────────────────────────────────────────────────────────────────
# QUALITY TARGETS — injected into every prompt
# ─────────────────────────────────────────────────────────────────────────────

QUALITY_TARGETS: dict[str, list[str]] = {
    "avoid": [
        "generic_blue_pill_cta",
        "centered_default_layout",
        "bottom_scrim_covering_half",
        "three_text_blocks_at_equal_weight",
        "text_directly_over_subject_or_face",
        "two_display_fonts_paired_together",
        "plain_white_rectangle_as_cta",
        "headline_sub_cta_always_present",
        "timid_headline_font_size",
    ],
    "require": [
        "extreme_hierarchy_between_headline_and_everything_else",
        "image_aware_placement_into_quiet_zones",
        "specific_and_non_generic_cta_copy_when_cta_present",
        "intentional_restraint_when_layout_family_allows",
    ],
}


# ─────────────────────────────────────────────────────────────────────────────
# VALIDATOR
# ─────────────────────────────────────────────────────────────────────────────

def validate_spec(spec: dict[str, Any]) -> list[str]:
    """Return a list of violation strings. Empty list means valid."""
    violations: list[str] = []

    lf = spec.get("layout_family")
    if lf not in LAYOUT_FAMILIES:
        violations.append(f"unknown layout_family: {lf!r}")
        return violations

    fam = LAYOUT_FAMILIES[lf]
    elements = spec.get("text_elements") or {}

    # Required elements present
    for req in fam["required_elements"]:
        if not elements.get(req):
            violations.append(f"layout_family {lf} requires element {req!r}")

    # Forbidden elements absent
    for forb in fam["forbidden_elements"]:
        if elements.get(forb):
            violations.append(f"layout_family {lf} forbids element {forb!r}")

    # Zone must exist
    placement = spec.get("placement") or {}
    zone = placement.get("primary_zone")
    if zone not in ZONES:
        violations.append(f"unknown primary_zone: {zone!r}")

    # Container must exist
    container = (spec.get("container_strategy") or {}).get("type")
    if container not in CONTAINERS:
        violations.append(f"unknown container: {container!r}")

    # CTA style must exist
    cta_type = (spec.get("cta_style") or {}).get("type")
    if cta_type not in CTA_STYLES:
        violations.append(f"unknown cta_style.type: {cta_type!r}")

    # CTA element presence matches cta_style type
    has_cta_content = bool((elements.get("cta") or {}).get("content"))
    if cta_type == "none" and has_cta_content:
        violations.append("cta_style.type=none but cta content is present")
    if cta_type != "none" and not has_cta_content:
        violations.append(f"cta_style.type={cta_type} but no cta content")

    # Font roles must exist
    typography = spec.get("typography") or {}
    primary_role = typography.get("primary_role")
    if primary_role and primary_role not in TYPOGRAPHY_ROLES:
        violations.append(f"unknown primary_role: {primary_role!r}")
    accent_role = typography.get("accent_role")
    if accent_role and accent_role not in TYPOGRAPHY_ROLES:
        violations.append(f"unknown accent_role: {accent_role!r}")

    return violations


# ─────────────────────────────────────────────────────────────────────────────
# PROMPT DESCRIPTORS — compact strings fed to Claude
# ─────────────────────────────────────────────────────────────────────────────

def describe_all_families() -> str:
    return "\n".join(f"- {describe_family_for_prompt(k)}" for k in LAYOUT_FAMILIES.keys())


def describe_all_zones() -> str:
    lines = []
    for k, v in ZONES.items():
        r = v["rect"]
        lines.append(f"- {k}: rect=({r['x']},{r['y']},{r['w']}x{r['h']})")
    return "\n".join(lines)


def describe_all_containers() -> str:
    return "\n".join(f"- {k}: {v['description']}" for k, v in CONTAINERS.items())


def describe_all_cta_styles() -> str:
    return "\n".join(f"- {k}: {v['description']}" for k, v in CTA_STYLES.items())


def describe_all_roles() -> str:
    lines = []
    for role, fonts in TYPOGRAPHY_ROLES.items():
        flist = ", ".join(f"{f['family']} {f['weight']}{'i' if f['style']=='italic' else ''}" for f in fonts)
        lines.append(f"- {role}: {flist}")
    return "\n".join(lines)
