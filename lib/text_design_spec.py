"""
text_design_spec — structured handoff between creative spec and HTML renderer.

Responsibilities:
    - DEFAULT_SPEC: safe fallback spec when upstream does not provide one
    - normalize(raw): fill missing keys with defaults, clamp ranges
    - validate(spec): cross-field validation via ad_design_system
    - merge_image_analysis(spec, analysis): attach deterministic image analysis
    - to_prompt_directive(spec): compact text block injected into Claude prompt
    - derive_legacy_fields(spec): back-compat top-level fields for app.py
"""

from __future__ import annotations

from typing import Any

from . import ad_design_system as ads

# ─────────────────────────────────────────────────────────────────────────────
# DEFAULT SPEC — used when upstream fails
# ─────────────────────────────────────────────────────────────────────────────

DEFAULT_SPEC: dict[str, Any] = {
    "tone_mode":      "performance_ugc",
    "layout_family":  "direct_response_stack",
    "text_density":   "moderate",
    "text_elements": {
        "eyebrow":      None,
        "badge":        None,
        "headline":     {
            "content": "",
            "lines": [],  # NEW: semantic line breaks (e.g., ["Actually", "affordable apartments", "Actually in NYC"])
            "emphasis_spans": [],  # Enhanced: now includes "color" field
        },
        "support_copy": None,
        "cta":          None,
        "attribution":  None,
    },
    "hierarchy_profile": {
        "dominant":       "headline",
        "headline_scale": "xl",
        "support_scale":  "sm",
        "cta_scale":      "md",
        "scale_ratio":    5.0,
    },
    "placement": {
        "primary_zone":    "lower_third",
        "alignment":       "center",
        "margin_profile":  "standard",
        "vertical_rhythm": "spacious",
        "block_anchor":    "bottom-center",
    },
    "container_strategy": {
        "type":    "none",
        "opacity": 0.0,
        "blur_px": 0,
        "radius":  "none",
        "padding": "standard",
    },
    "typography": {
        "primary_family": "Inter",
        "primary_role":   "modern_sans",
        "accent_family":  None,
        "accent_role":    None,
        "cta_family":     "Inter",
        "tracking":       "tight",
        "case_style":     "upper",
        "line_height":    "tight",
    },
    "color_strategy": {
        "mode":           "light_on_dark_area",
        "headline_color": "#FFFFFF",
        "support_color":  "#EBEBEB",
        "accent_color":   "#FFFFFF",
        "cta_bg":         "#FFFFFF",
        "cta_fg":         "#111111",
    },
    "cta_style": {
        "type":       "pill_filled",
        "prominence": "standard",
    },
    "scrim": {
        "enabled":     False,
        "type":        None,
        "extent":      None,
        "max_opacity": 0.0,
    },
    "image_analysis":   None,  # filled in by merge_image_analysis
    "quality_targets":  ads.QUALITY_TARGETS,
}


# ─────────────────────────────────────────────────────────────────────────────
# NORMALIZE + VALIDATE
# ─────────────────────────────────────────────────────────────────────────────

def _deep_merge(base: dict, override: dict | None) -> dict:
    if not override:
        return {k: _clone(v) for k, v in base.items()}
    out: dict[str, Any] = {}
    for k, v in base.items():
        if k in override:
            ov = override[k]
            if isinstance(v, dict) and isinstance(ov, dict):
                out[k] = _deep_merge(v, ov)
            else:
                out[k] = ov
        else:
            out[k] = _clone(v)
    # pick up any new keys in override
    for k, v in override.items():
        if k not in out:
            out[k] = v
    return out


def _clone(v: Any) -> Any:
    if isinstance(v, dict):
        return {k: _clone(x) for k, x in v.items()}
    if isinstance(v, list):
        return [_clone(x) for x in v]
    return v


def normalize(raw: dict | None) -> dict[str, Any]:
    """Merge raw into DEFAULT_SPEC. Clean None -> removed element entries."""
    spec = _deep_merge(DEFAULT_SPEC, raw or {})

    # Drop forbidden elements according to layout family
    lf = spec.get("layout_family", "direct_response_stack")
    forbidden = set(ads.forbidden_elements(lf))
    elements = spec.get("text_elements") or {}
    for key in list(elements.keys()):
        if key in forbidden:
            elements[key] = None
    spec["text_elements"] = elements

    # Ensure cta_style matches presence of cta content
    cta_content = (elements.get("cta") or {}).get("content") if elements.get("cta") else None
    cta_style = spec.get("cta_style") or {}
    if not cta_content:
        cta_style["type"] = "none"
    elif cta_style.get("type") == "none" and cta_content:
        # Pick a reasonable default per layout family
        legal = ads.resolve_cta_style(lf, spec.get("tone_mode", "performance_ugc"), cta_content)
        cta_style["type"] = legal[0] if legal else "pill_filled"
    spec["cta_style"] = cta_style

    # Clamp opacity and blur ranges
    cs = spec.get("container_strategy") or {}
    ctype = cs.get("type", "none")
    cmeta = ads.CONTAINERS.get(ctype, ads.CONTAINERS["none"])
    o_lo, o_hi = cmeta["opacity_range"]
    b_lo, b_hi = cmeta["blur_range"]
    try:
        cs["opacity"] = max(o_lo, min(o_hi, float(cs.get("opacity", 0) or 0)))
    except (TypeError, ValueError):
        cs["opacity"] = o_lo
    try:
        cs["blur_px"] = int(max(b_lo, min(b_hi, int(cs.get("blur_px", 0) or 0))))
    except (TypeError, ValueError):
        cs["blur_px"] = b_lo
    spec["container_strategy"] = cs

    return spec


def validate(spec: dict) -> tuple[bool, list[str]]:
    violations = ads.validate_spec(spec)
    return (len(violations) == 0, violations)


def merge_image_analysis(spec: dict, analysis: dict) -> dict:
    spec = dict(spec)
    spec["image_analysis"] = analysis

    # Optionally nudge color mode based on suggested_text_color if the spec
    # is on its defaults (avoid overwriting explicit creative choices).
    color = spec.get("color_strategy") or {}
    accent_color = analysis.get("accent_color", "#888888")
    if color.get("mode") in (None, "light_on_dark_area", "dark_on_light_area"):
        if analysis.get("suggested_text_color") == "dark":
            color["mode"] = "dark_on_light_area"
            color.setdefault("headline_color", "#111111")
            color.setdefault("support_color", "#333333")
        else:
            color["mode"] = "light_on_dark_area"
            color.setdefault("headline_color", "#FFFFFF")
            color.setdefault("support_color", "#EBEBEB")

    # Use the extracted accent color for eyebrows, emphasis, CTA
    color["accent_color"] = accent_color
    spec["color_strategy"] = color

    # Apply accent color to eyebrow if present
    elements = spec.get("text_elements") or {}
    eyebrow = elements.get("eyebrow")
    if eyebrow:
        eyebrow.setdefault("emphasis_spans", [])
        # Eyebrow gets the accent color
        if not eyebrow.get("_accent_applied"):
            eyebrow["_accent_applied"] = True

    # Apply accent color to emphasis_spans in headline if present
    headline = elements.get("headline")
    if headline and headline.get("emphasis_spans"):
        for span in headline["emphasis_spans"]:
            # If span doesn't already have a color, use the accent
            if not span.get("color"):
                span["color"] = accent_color

    # Nudge primary_zone toward the quietest available zone if current pick
    # isn't in the top-3 quiet zones (image-aware placement).
    placement = spec.get("placement") or {}
    quietest = analysis.get("quietest_zones") or []
    lf = spec.get("layout_family", "direct_response_stack")
    typical = ads.LAYOUT_FAMILIES.get(lf, {}).get("typical_zones", [])
    if placement.get("primary_zone") not in quietest:
        # Prefer intersection of typical_zones and quietest_zones
        intersect = [z for z in typical if z in quietest]
        if intersect:
            placement["primary_zone"] = intersect[0]
        elif quietest:
            placement["primary_zone"] = quietest[0]
    spec["placement"] = placement

    return spec


# ─────────────────────────────────────────────────────────────────────────────
# PROMPT DIRECTIVE — compact text block fed to Claude in the user message
# ─────────────────────────────────────────────────────────────────────────────

def to_prompt_directive(spec: dict) -> str:
    lf  = spec.get("layout_family", "direct_response_stack")
    fam = ads.LAYOUT_FAMILIES.get(lf, {})
    elements = spec.get("text_elements") or {}
    placement = spec.get("placement") or {}
    container = spec.get("container_strategy") or {}
    typography = spec.get("typography") or {}
    color = spec.get("color_strategy") or {}
    cta_style = spec.get("cta_style") or {}
    scrim = spec.get("scrim") or {}
    analysis = spec.get("image_analysis") or {}

    zone = placement.get("primary_zone", "lower_third")
    zrect = ads.zone_bounds(zone)

    # Which elements will actually render
    active_elements: list[str] = []
    for key in ["eyebrow", "badge", "headline", "support_copy", "cta", "attribution"]:
        obj = elements.get(key)
        if obj and obj.get("content"):
            active_elements.append(key)

    forbidden = fam.get("forbidden_elements", [])

    def _content(k: str) -> str:
        obj = elements.get(k) or {}
        return (obj.get("content") or "").strip()

    lines: list[str] = []
    lines.append("── TEXT DESIGN SPEC ──")
    lines.append(f"tone_mode:       {spec.get('tone_mode')}")
    lines.append(f"layout_family:   {lf}")
    lines.append(f"  > {fam.get('description','')}")
    lines.append(f"text_density:    {spec.get('text_density')}")
    lines.append(f"active_elements: {active_elements}")
    lines.append(f"forbidden:       {forbidden}")
    lines.append("")
    lines.append("── COPY (use these verbatim, do NOT invent) ──")
    if _content("eyebrow"):
        lines.append(f"  eyebrow:      {_content('eyebrow')!r}")
    if _content("badge"):
        lines.append(f"  badge:        {_content('badge')!r}")
    if _content("headline"):
        lines.append(f"  headline:     {_content('headline')!r}")
        hl = elements.get("headline") or {}
        hl_lines = hl.get("lines") or []
        if hl_lines:
            lines.append(f"  lines:        {hl_lines}")
        spans = hl.get("emphasis_spans") or []
        if spans:
            lines.append(f"  emphasis_spans: {spans}")
    if _content("support_copy"):
        lines.append(f"  support_copy: {_content('support_copy')!r}")
        sc = elements.get("support_copy") or {}
        lines.append(f"  support_structure: {sc.get('structure','sentence')}")
    if _content("attribution"):
        lines.append(f"  attribution:  {_content('attribution')!r}")
    if _content("cta"):
        lines.append(f"  cta:          {_content('cta')!r}")
    lines.append("")
    lines.append("── PLACEMENT ──")
    lines.append(f"  primary_zone: {zone}  rect=(x={zrect['x']}, y={zrect['y']}, w={zrect['w']}, h={zrect['h']})")
    lines.append(f"  alignment:    {placement.get('alignment')}")
    lines.append(f"  margin:       {placement.get('margin_profile')}")
    lines.append(f"  rhythm:       {placement.get('vertical_rhythm')}")
    lines.append(f"  block_anchor: {placement.get('block_anchor')}")
    lines.append("  HARD RULE: All text blocks must be placed INSIDE the rect above.")
    lines.append("")
    lines.append("── CONTAINER STRATEGY ──")
    lines.append(f"  type:    {container.get('type')}")
    lines.append(f"  opacity: {container.get('opacity')}")
    lines.append(f"  blur_px: {container.get('blur_px')}")
    lines.append(f"  radius:  {container.get('radius')}")
    lines.append(f"  padding: {container.get('padding')}")
    ctype_meta = ads.CONTAINERS.get(container.get("type", "none"), {})
    if ctype_meta:
        lines.append(f"  notes:   {ctype_meta.get('description','')}")
    lines.append("")
    lines.append("── TYPOGRAPHY ──")
    lines.append(f"  primary_family: {typography.get('primary_family')}  (role: {typography.get('primary_role')})")
    if typography.get("accent_family"):
        lines.append(f"  accent_family:  {typography.get('accent_family')}  (role: {typography.get('accent_role')})")
    lines.append(f"  cta_family:     {typography.get('cta_family')}")
    lines.append(f"  tracking:       {typography.get('tracking')}")
    lines.append(f"  case_style:     {typography.get('case_style')}")
    lines.append(f"  line_height:    {typography.get('line_height')}")
    lines.append("  HARD RULE: Do not use any font-family not listed above.")
    lines.append("  HARD RULE: Maximum 2 font families total (primary + accent). CTA uses cta_family.")
    lines.append("")
    lines.append("── COLOR ──")
    lines.append(f"  mode:           {color.get('mode')}")
    lines.append(f"  headline_color: {color.get('headline_color')}")
    lines.append(f"  support_color:  {color.get('support_color')}")
    lines.append(f"  accent_color:   {color.get('accent_color')}")
    lines.append(f"  cta_bg:         {color.get('cta_bg')}")
    lines.append(f"  cta_fg:         {color.get('cta_fg')}")
    lines.append("")
    lines.append("── CTA STYLE ──")
    lines.append(f"  type:       {cta_style.get('type')}")
    lines.append(f"  prominence: {cta_style.get('prominence')}")
    cta_meta = ads.CTA_STYLES.get(cta_style.get("type", "none"), {})
    if cta_meta:
        lines.append(f"  notes:      {cta_meta.get('description','')}")
    if cta_style.get("type") == "none":
        lines.append("  HARD RULE: DO NOT render any button, link, or arrow as a CTA.")
    lines.append("")
    lines.append("── SCRIM ──")
    if scrim.get("enabled"):
        lines.append(f"  enabled: True  type={scrim.get('type')} extent={scrim.get('extent')} max_opacity={scrim.get('max_opacity')}")
    else:
        lines.append("  enabled: False (do NOT add a bottom scrim unless explicitly enabled above)")
    lines.append("")
    if analysis:
        lines.append("── IMAGE ANALYSIS (deterministic) ──")
        lines.append(f"  brightness:         {analysis.get('brightness')}")
        lines.append(f"  contrast:           {analysis.get('contrast')}")
        lines.append(f"  dominant_hue:       {analysis.get('dominant_hue')}")
        lines.append(f"  dominant_palette:   {analysis.get('dominant_palette')}")
        lines.append(f"  quietest_zones:     {analysis.get('quietest_zones')}")
        lines.append(f"  busiest_zones:      {analysis.get('busiest_zones')}")
        lines.append(f"  suggested_text_clr: {analysis.get('suggested_text_color')}")
        lines.append("")
    lines.append("── QUALITY TARGETS ──")
    for t in (spec.get("quality_targets") or {}).get("avoid", []):
        lines.append(f"  AVOID: {t}")
    for t in (spec.get("quality_targets") or {}).get("require", []):
        lines.append(f"  REQUIRE: {t}")
    return "\n".join(lines)


# ─────────────────────────────────────────────────────────────────────────────
# BACKWARD COMPATIBILITY — derive legacy top-level spec fields
# ─────────────────────────────────────────────────────────────────────────────

def derive_legacy_fields(text_design_spec: dict) -> dict[str, Any]:
    """Derive legacy headline/subheadline/cta/negativeSpaceZone/textTemplate
    fields from a text_design_spec so app.py and generate_prompt.py continue
    to work without changes.
    """
    elements = text_design_spec.get("text_elements") or {}
    placement = text_design_spec.get("placement") or {}
    color = text_design_spec.get("color_strategy") or {}

    headline = (elements.get("headline") or {}).get("content") or ""
    support  = (elements.get("support_copy") or {}).get("content") or ""
    cta      = (elements.get("cta") or {}).get("content") or ""

    zone_key = placement.get("primary_zone", "lower_third")
    legacy_zone = ads.to_legacy_zone(zone_key)

    mode = color.get("mode", "light_on_dark_area")
    if mode == "dark_on_light_area":
        legacy_template = "dark-on-light"
    else:
        legacy_template = "light-on-dark"

    return {
        "headline":          headline,
        "subheadline":       support,
        "cta":               cta,
        "negativeSpaceZone": legacy_zone,
        "textTemplate":      legacy_template,
    }
