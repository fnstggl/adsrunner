"""
HTML/CSS ad text overlay pipeline.

Claude generates a transparent HTML overlay (text + scrims + CTA).
Playwright/Chromium rasterizes it in ~1-2s (real browser — no font freezes).
PIL alpha_composites the result onto the background photo.

Drop-in replacement for lib/render_text_overlay.render_text_overlay.

Phase 2 upgrade: this stage now accepts a rich `text_design_spec` dict that
pre-commits the layout family, placement, container, typography, color, and
CTA style. The Claude call becomes a structured implementor, not a freestyle
designer. See lib/text_design_spec.py and lib/ad_design_system.py.
"""

from __future__ import annotations

import base64
import io
import re
import traceback
from pathlib import Path

import anthropic
import cv2
import numpy as np
import requests
from PIL import Image

from . import ad_design_system as ads
from . import image_analysis as imganalysis
from . import text_design_spec as tds

# ---------------------------------------------------------------------------
# Module-level font cache — fetched once, reused across calls
# ---------------------------------------------------------------------------
_FONT_CACHE: dict[str, str] | None = None

# Font definitions: key -> Google Fonts CSS URL query
_FONT_SPECS: dict[str, str] = {
    "Inter-400":                   "family=Inter:wght@400",
    "Inter-700":                   "family=Inter:wght@700",
    "Inter-900":                   "family=Inter:wght@900",
    "Montserrat-700":              "family=Montserrat:wght@700",
    "Montserrat-900":              "family=Montserrat:wght@900",
    "BebasNeue-400":               "family=Bebas+Neue",
    "Oswald-700":                  "family=Oswald:wght@700",
    "PlayfairDisplay-700italic":   "family=Playfair+Display:ital,wght@1,700",
    "Lora-400italic":              "family=Lora:ital@1",
    "DMSerifDisplay-400italic":    "family=DM+Serif+Display:ital@1",
    "SpaceGrotesk-700":            "family=Space+Grotesk:wght@700",
    "CormorantGaramond-700italic": "family=Cormorant+Garamond:ital,wght@1,700",
    # Phase 2 additions
    "Poppins-400":                 "family=Poppins:wght@400",
    "Poppins-600":                 "family=Poppins:wght@600",
    "Poppins-700":                 "family=Poppins:wght@700",
    "Anton-400":                   "family=Anton",
    "LibreBaskerville-400italic":  "family=Libre+Baskerville:ital@1",
    "LibreBaskerville-700italic":  "family=Libre+Baskerville:ital,wght@1,700",
    "Caveat-400":                  "family=Caveat:wght@400",
    "Caveat-700":                  "family=Caveat:wght@700",
}

_WOFF2_URL_RE = re.compile(r"url\((https://fonts\.gstatic\.com[^)]+\.woff2)\)")

# Google Fonts requires a browser-like UA to serve woff2
_GFONTS_UA = (
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
    "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
)

_FONT_FACE_MAP: list[dict] = [
    {"key": "Inter-400",                    "family": "Inter",               "weight": "400", "style": "normal"},
    {"key": "Inter-700",                    "family": "Inter",               "weight": "700", "style": "normal"},
    {"key": "Inter-900",                    "family": "Inter",               "weight": "900", "style": "normal"},
    {"key": "Montserrat-700",               "family": "Montserrat",          "weight": "700", "style": "normal"},
    {"key": "Montserrat-900",               "family": "Montserrat",          "weight": "900", "style": "normal"},
    {"key": "BebasNeue-400",                "family": "Bebas Neue",          "weight": "400", "style": "normal"},
    {"key": "Oswald-700",                   "family": "Oswald",              "weight": "700", "style": "normal"},
    {"key": "PlayfairDisplay-700italic",    "family": "Playfair Display",    "weight": "700", "style": "italic"},
    {"key": "Lora-400italic",               "family": "Lora",                "weight": "400", "style": "italic"},
    {"key": "DMSerifDisplay-400italic",     "family": "DM Serif Display",    "weight": "400", "style": "italic"},
    {"key": "SpaceGrotesk-700",             "family": "Space Grotesk",       "weight": "700", "style": "normal"},
    {"key": "CormorantGaramond-700italic",  "family": "Cormorant Garamond",  "weight": "700", "style": "italic"},
    # Phase 2 additions
    {"key": "Poppins-400",                  "family": "Poppins",             "weight": "400", "style": "normal"},
    {"key": "Poppins-600",                  "family": "Poppins",             "weight": "600", "style": "normal"},
    {"key": "Poppins-700",                  "family": "Poppins",             "weight": "700", "style": "normal"},
    {"key": "Anton-400",                    "family": "Anton",               "weight": "400", "style": "normal"},
    {"key": "LibreBaskerville-400italic",   "family": "Libre Baskerville",   "weight": "400", "style": "italic"},
    {"key": "LibreBaskerville-700italic",   "family": "Libre Baskerville",   "weight": "700", "style": "italic"},
    {"key": "Caveat-400",                   "family": "Caveat",              "weight": "400", "style": "normal"},
    {"key": "Caveat-700",                   "family": "Caveat",              "weight": "700", "style": "normal"},
]


# ---------------------------------------------------------------------------
# Font fetching — same as before, WOFF2 base64 encoded
# ---------------------------------------------------------------------------

def _fetch_fonts() -> dict[str, str]:
    """Fetch all fonts from Google Fonts, base64 encode them."""
    global _FONT_CACHE
    if _FONT_CACHE is not None:
        return _FONT_CACHE

    fonts: dict[str, str] = {}
    for key, query in _FONT_SPECS.items():
        try:
            css_url = f"https://fonts.googleapis.com/css2?{query}&display=swap"
            css_resp = requests.get(css_url, headers={"User-Agent": _GFONTS_UA}, timeout=10)
            css_resp.raise_for_status()
            match = _WOFF2_URL_RE.search(css_resp.text)
            if not match:
                print(f"[HTML_RENDER] No woff2 URL found for {key}")
                fonts[key] = ""
                continue
            woff2_url = match.group(1)
            woff2_resp = requests.get(woff2_url, timeout=15)
            woff2_resp.raise_for_status()
            fonts[key] = base64.b64encode(woff2_resp.content).decode()
            print(f"[HTML_RENDER] Fetched {key} ({len(woff2_resp.content)} bytes)")
        except Exception as exc:
            print(f"[HTML_RENDER] Font fetch failed for {key}: {exc}")
            fonts[key] = ""

    _FONT_CACHE = fonts
    return fonts


def _build_font_face_css(fonts: dict[str, str]) -> str:
    """Build @font-face CSS from fetched WOFF2 fonts (embedded as base64)."""
    blocks = []
    for entry in _FONT_FACE_MAP:
        b64 = fonts.get(entry["key"], "")
        if not b64:
            continue
        blocks.append(
            f"@font-face {{ font-family: '{entry['family']}'; "
            f"font-weight: {entry['weight']}; font-style: {entry['style']}; "
            f"src: url('data:font/woff2;base64,{b64}') format('woff2'); }}"
        )
    return "\n".join(blocks)


# ---------------------------------------------------------------------------
# System prompt
# ---------------------------------------------------------------------------

_SYSTEM_PROMPT = """You are a senior front-end typographer and HTML/CSS implementor.

Your job: Render a text_design_spec into a transparent HTML ad overlay using ONLY the layout tokens provided. Do NOT invent layout values. Do NOT freestyle.

LAYOUT TOKENS — Use these exact px values (provided in the spec):

  zone_rect:               {x, y, w, h}           # zone boundary on 1080×1350 canvas
  safe_margin:             16 px                  # inset from zone edge
  usable_width:            1048 px                # w - (2 * margin)
  headline_size_range:     [150, 200]             # (min_px, max_px) from word count
  support_size_range:      [36, 48]               # support copy sizing
  eyebrow_size:            28 px                  # fixed
  cta_size:                44 px                  # fixed
  headline_line_height:    1.0                    # tight
  support_line_height:     1.4                    # spacious
  gap_headline_support:    12 px                  # vertical gap
  gap_support_cta:         16 px                  # vertical gap
  headline_color:          "#FFFFFF"              # exact color
  support_color:           "#E8E8E8"              # exact color
  accent_color:            "#FF6B35"              # exact color (from image)
  cta_bg:                  "#FF6B35"              # CTA button background
  cta_fg:                  "#FFFFFF"              # CTA button foreground

Use these values in your CSS. Do NOT round, adjust, or guess.

OVERFLOW PREVENTION (CRITICAL):
If text overflows the zone bounds, you should aim for 90-95% of max_font_size on the first attempt. The system validates post-render and will shrink if needed. Never let text leave zone bounds.

HTML / CSS TECHNICAL RULES:
- Document is 1080×1350px, transparent background.
- html, body: width: 1080px; height: 1350px; overflow: hidden; margin: 0; padding: 0; background: transparent.
- All text blocks: position: absolute.
- Fonts are injected automatically — do NOT include @font-face.
- No external URLs, images, scripts, background colors.
- Rendered by headless Chromium at 1:1 pixel ratio.

ALLOWED CSS:
- font-family, font-size, font-weight, font-style, letter-spacing, line-height, color, opacity, text-shadow
- linear-gradient(), radial-gradient(), backdrop-filter: blur()
- border-radius, border, box-shadow, flexbox, grid, transforms
- <br> for line breaks, <span> for emphasis

CONTRACT — Honor the spec exactly:
1. Active elements: Only render elements from active_elements list. Do NOT add forbidden elements.
2. Placement zone: ALL text must stay inside zone_rect.
3. Typography: Use ONLY primary_family, accent_family, cta_family. Max 2 total.
4. Colors: Use headline_color, support_color, accent_color, cta_bg, cta_fg exactly as provided.
5. Sizing: Use headline_size_range, support_size_range, eyebrow_size, cta_size from layout_tokens.
6. Container: Apply container_strategy.type (none, shadow_only, translucent_card, etc.).
7. CTA: If cta_style.type=none, do NOT render a button. Otherwise match the style.
8. Alignment: Use placement.alignment (left/center/right) — no default centering.
9. Emphasis spans: Render headline emphasis_spans with their specified colors and font roles.
10. Line limits: headline ≤ max_lines_headline, support ≤ max_lines_support.

Output ONLY HTML: <!DOCTYPE html> to </html>. No explanation, no markdown. Render the overlay now."""


# ---------------------------------------------------------------------------
# User message builder
# ---------------------------------------------------------------------------

def _build_user_content(
    img_b64: str,
    img_media_type: str,
    design_directive: str,
    available_fonts: list[str],
    image_description: str,
    performance_hints: str,
) -> list[dict]:
    fonts_line = ", ".join(available_fonts) if available_fonts else "Inter"
    text_msg = f"""The photo above is your visual reference — use it for fine-grained negative space, subject position, and color palette context. Do not embed or reference the photo in the HTML.

AVAILABLE FONT FAMILIES (these are actually loaded into the renderer right now — do not use any other font):
{fonts_line}

{design_directive}

{"IMAGE DESCRIPTION: " + image_description if image_description else ""}
{"PERFORMANCE HINTS: " + performance_hints if performance_hints else ""}

Render the complete HTML overlay document now. Output only HTML — start with <!DOCTYPE html> and end with </html>."""

    return [
        {
            "type": "image",
            "source": {
                "type": "base64",
                "media_type": img_media_type,
                "data": img_b64,
            },
        },
        {"type": "text", "text": text_msg},
    ]


# ---------------------------------------------------------------------------
# Playwright rasterizer
# ---------------------------------------------------------------------------

def _rasterize_html(html: str, width: int = 1080, height: int = 1350) -> bytes:
    """Render HTML to a transparent PNG using headless Chromium via Playwright."""
    from playwright.sync_api import sync_playwright

    with sync_playwright() as pw:
        browser = pw.chromium.launch(args=["--no-sandbox", "--disable-setuid-sandbox"])
        page = browser.new_page(viewport={"width": width, "height": height})
        # Transparent background so the overlay composites cleanly onto the photo
        page.set_content(html, wait_until="load")
        png_bytes = page.screenshot(
            full_page=False,
            omit_background=True,  # transparent PNG
        )
        browser.close()

    return png_bytes


# ---------------------------------------------------------------------------
# Overflow prevention: validate post-render and shrink if needed
# ---------------------------------------------------------------------------

def _validate_and_fix_overflow(
    html: str,
    spec: dict,
    client: anthropic.Anthropic,
    font_face_css: str,
    max_retries: int = 2,
) -> str:
    """Validate that text stays within zone bounds post-render.

    If potential overflow is detected, request Claude to use smaller font sizes
    from the fallback cascade and re-render.

    Returns:
        Fixed HTML (or original if no overflow detected).
    """
    tokens = spec.get("layout_tokens") or {}
    zone_rect = tokens.get("zone_rect", {})
    headline_min, headline_max = tokens.get("headline_size_range", (100, 150))
    fallback_cascade = tokens.get("font_size_fallback_cascade", [150, 140, 130, 120, 110, 100, 90, 80])

    # Heuristic: extract font-sizes from CSS
    sizes = [int(m.group(1)) for m in re.finditer(r"font-size\s*:\s*(\d+)px", html)]
    if not sizes:
        return html

    max_size_in_html = max(sizes)

    # If max size is already near the minimum, we can't shrink further
    if max_size_in_html <= 60:
        print("[HTML_RENDER] Font size already minimal, cannot shrink further")
        return html

    # Crude overflow detection: if headline font-size is close to the max, and
    # zone_width is small, there's likely overflow. Estimate character count.
    zone_w = zone_rect.get("w", 1080)
    zone_h = zone_rect.get("h", 600)

    # Rough estimate: ~5 chars per 100px at headline size
    max_chars_per_line = max(8, (zone_w - 32) // (max_size_in_html / 20))

    # Extract headline content
    headline_match = re.search(r"<div[^>]*>([^<]+)</div>", html)
    headline_text = headline_match.group(1) if headline_match else ""
    headline_text = headline_text.strip()

    # If headline exceeds ~3 lines worth of characters, we might overflow
    estimated_lines = len(headline_text) / max_chars_per_line
    estimated_height = estimated_lines * (max_size_in_html * 1.1)  # rough with line-height

    if estimated_height > zone_h * 0.8:  # using 80% of zone height as safety threshold
        print(
            f"[HTML_RENDER] Potential overflow detected: "
            f"est. {estimated_height:.0f}px height vs {zone_h}px zone. Requesting shrink..."
        )

        # Request Claude to use smaller font sizes
        next_size = headline_max - 10
        if next_size < 60:
            next_size = 60

        shrink_prompt = f"""Your previous HTML may have text overflowing the zone bounds.

Zone bounds: width={zone_w}px, height={zone_h}px
Current max font-size in headline: {max_size_in_html}px
Headline text: {headline_text!r}

Please revise the HTML to use smaller font sizes:
- Reduce headline from {max_size_in_html}px to {next_size}px or smaller
- Ensure all text stays within the zone bounds
- Keep all other styling intact

Use this fallback cascade if needed: {fallback_cascade}
Stop at min_font_size: 24px

Do not change layout, zones, or fonts. Only adjust font-sizes and spacing to fit.
Output ONLY the revised HTML (<!DOCTYPE html> to </html>)."""

        try:
            revision_response = client.messages.create(
                model="claude-sonnet-4-20250514",
                max_tokens=4000,
                messages=[{"role": "user", "content": shrink_prompt}],
            )
            shrunk_html = revision_response.content[0].text.strip()

            # Strip markdown fences
            if "```" in shrunk_html:
                match = re.search(r"<!DOCTYPE html>[\s\S]*?</html>", shrunk_html, re.IGNORECASE)
                if match:
                    shrunk_html = match.group(0)

            if not re.match(r"<!DOCTYPE html>", shrunk_html, re.IGNORECASE):
                match = re.search(r"<!DOCTYPE html>[\s\S]*?</html>", shrunk_html, re.IGNORECASE)
                if match:
                    shrunk_html = match.group(0)
                else:
                    print("[HTML_RENDER] Shrink response not valid HTML, keeping original")
                    return html

            # Inject font CSS
            if font_face_css:
                font_style_block = f"<style>\n{font_face_css}\n</style>"
                if "</head>" in shrunk_html:
                    shrunk_html = shrunk_html.replace("</head>", f"{font_style_block}\n</head>", 1)
                else:
                    shrunk_html = shrunk_html.replace("<body", f"{font_style_block}\n<body", 1)

            print("[HTML_RENDER] Shrink revision applied")
            return shrunk_html
        except Exception as exc:
            print(f"[HTML_RENDER] Shrink revision failed: {exc}, keeping original")
            return html

    return html


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------

def render_text_overlay(
    image: np.ndarray,
    headline: str = "",
    subheadline: str = "",
    cta: str = "",
    zone: str = "bottom-center",
    template: str = "light-on-dark",
    logo: np.ndarray = None,
    layout_id: str = None,
    image_description: str = "",
    performance_hints: str = "",
    text_design_spec: dict | None = None,
    spec_id: str | None = None,
) -> np.ndarray:
    """Generate an HTML ad overlay using Claude, rasterize with Playwright/Chromium,
    and alpha-composite onto the background photo.

    Phase 2: accepts a rich `text_design_spec` dict. If not provided, one is
    constructed from the legacy headline/subheadline/cta/zone/template args.
    Falls back to the PIL renderer on any error.
    """
    try:
        # Step 0: Build / normalize the text_design_spec.
        spec = _build_or_adapt_spec(
            text_design_spec=text_design_spec,
            headline=headline,
            subheadline=subheadline,
            cta=cta,
            zone=zone,
            template=template,
        )

        # Step 1: Downscale reference image for Claude vision (layout/color context only).
        rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        small = Image.fromarray(rgb).resize((540, 675), Image.LANCZOS)
        buf_small = io.BytesIO()
        small.save(buf_small, format="JPEG", quality=85)
        img_b64 = base64.b64encode(buf_small.getvalue()).decode()
        img_media_type = "image/jpeg"
        print(f"[HTML_RENDER] Reference image: {len(buf_small.getvalue()):,} bytes (540×675 JPEG)")

        # Step 2: Deterministic OpenCV image analysis and merge into spec
        analysis = imganalysis.analyze_image(image)
        spec = tds.merge_image_analysis(spec, analysis)
        ok, violations = tds.validate(spec)
        if not ok:
            print(f"[HTML_RENDER] spec violations (rendering anyway): {violations}")

        # Step 3: Fetch fonts (module-level cache)
        fonts = _fetch_fonts()
        font_face_css = _build_font_face_css(fonts)
        available_families = _available_families(fonts)

        # Step 4: Build the compact design directive and user content
        directive = tds.to_prompt_directive(spec)
        client = anthropic.Anthropic()
        user_content = _build_user_content(
            img_b64=img_b64,
            img_media_type=img_media_type,
            design_directive=directive,
            available_fonts=available_families,
            image_description=image_description,
            performance_hints=performance_hints,
        )

        response = client.messages.create(
            model="claude-sonnet-4-20250514",
            max_tokens=5000,
            system=_SYSTEM_PROMPT,
            messages=[{"role": "user", "content": user_content}],
        )

        html_str = response.content[0].text.strip()

        # Strip accidental markdown fences
        if "```" in html_str:
            match = re.search(r"<!DOCTYPE html>[\s\S]*?</html>", html_str, re.IGNORECASE)
            if match:
                html_str = match.group(0)

        if not re.match(r"<!DOCTYPE html>", html_str, re.IGNORECASE):
            match = re.search(r"<!DOCTYPE html>[\s\S]*?</html>", html_str, re.IGNORECASE)
            if match:
                html_str = match.group(0)
            else:
                raise ValueError("Claude response does not contain valid HTML")

        # Step 5: Inject @font-face CSS into <head> before Playwright renders.
        if font_face_css:
            font_style_block = f"<style>\n{font_face_css}\n</style>"
            if "</head>" in html_str:
                html_str = html_str.replace("</head>", f"{font_style_block}\n</head>", 1)
            else:
                html_str = html_str.replace("<body", f"{font_style_block}\n<body", 1)

        print(f"[HTML_RENDER] HTML payload: {len(html_str):,} bytes — rasterizing with Playwright")

        # Step 5b: Write debug HTML artifact so you can inspect what Claude produced.
        try:
            debug_dir = Path("outputs/html_debug")
            debug_dir.mkdir(parents=True, exist_ok=True)
            debug_name = (spec_id or spec.get("layout_family") or "overlay") + ".html"
            debug_path = debug_dir / debug_name
            debug_path.write_text(html_str, encoding="utf-8")
            print(f"[HTML_RENDER] Debug HTML written -> {debug_path}")
        except Exception as dbg_exc:
            print(f"[HTML_RENDER] Debug write failed (non-fatal): {dbg_exc}")

        # Step 5c: Deterministic scoring + revision if needed
        try:
            score = _score_html(html_str, spec)
            print(f"[HTML_RENDER] score={score['generic_ai_risk']} warnings={score['warnings']}")
            # If score is too high, request a revision pass
            html_str = _revise_html_if_needed(html_str, spec, score, client, font_face_css)
        except Exception as score_exc:
            print(f"[HTML_RENDER] scoring/revision failed (non-fatal): {score_exc}")

        # Step 6: Rasterize with Playwright
        png_bytes = _rasterize_html(html_str, width=1080, height=1350)

        # Step 6b: Validate text bounds and trigger shrink-retry if needed
        try:
            html_str_revised = _validate_and_fix_overflow(
                html_str, spec, client, font_face_css, max_retries=2
            )
            if html_str_revised != html_str:
                # HTML was revised due to overflow; re-rasterize
                print("[HTML_RENDER] Re-rasterizing after overflow fix...")
                png_bytes = _rasterize_html(html_str_revised, width=1080, height=1350)
                html_str = html_str_revised
        except Exception as validate_exc:
            print(f"[HTML_RENDER] overflow validation failed (non-fatal): {validate_exc}")

        # Step 7: Alpha-composite the transparent overlay onto the original photo
        overlay_pil = Image.open(io.BytesIO(png_bytes)).convert("RGBA")
        bg_pil = Image.fromarray(rgb).convert("RGBA")
        if bg_pil.size != (1080, 1350):
            bg_pil = bg_pil.resize((1080, 1350), Image.LANCZOS)

        bg_pil.alpha_composite(overlay_pil)

        rgb_arr = np.array(bg_pil.convert("RGB"))
        bgr_arr = cv2.cvtColor(rgb_arr, cv2.COLOR_RGB2BGR)

        print(f"[HTML_RENDER] Success — output shape {bgr_arr.shape}")
        return bgr_arr

    except Exception as exc:
        print(f"[HTML_RENDER] FAILED: {exc} — falling back to PIL")
        traceback.print_exc()
        from lib.render_text_overlay import render_text_overlay as _fallback
        return _fallback(
            image=image,
            headline=headline,
            subheadline=subheadline,
            cta=cta,
            zone=zone,
            template=template,
            logo=logo,
        )


# ---------------------------------------------------------------------------
# Helpers: spec adaptation from legacy args, available-families extraction,
# deterministic scoring heuristics.
# ---------------------------------------------------------------------------

def _build_or_adapt_spec(
    text_design_spec: dict | None,
    headline: str,
    subheadline: str,
    cta: str,
    zone: str,
    template: str,
) -> dict:
    """If a spec was provided, normalize it. Otherwise synthesize a minimal
    spec from the legacy headline/sub/cta/zone/template arguments so this
    function still works for any caller that hasn't been upgraded."""
    if text_design_spec:
        return tds.normalize(text_design_spec)

    legacy_zone_map = {
        "top-left":      "top_left",
        "top-center":    "upper_third",
        "top-right":     "top_right",
        "bottom-left":   "bottom_left",
        "bottom-center": "lower_third",
        "bottom-right":  "bottom_right",
    }
    zone_key = legacy_zone_map.get(zone, "lower_third")

    if template == "dark-on-light":
        color_mode = "dark_on_light_area"
        headline_color = "#111111"
        support_color = "#333333"
        cta_bg = "#111111"
        cta_fg = "#FFFFFF"
    else:
        color_mode = "light_on_dark_area"
        headline_color = "#FFFFFF"
        support_color = "#EBEBEB"
        cta_bg = "#FFFFFF"
        cta_fg = "#111111"

    raw = {
        "tone_mode": "performance_ugc",
        "layout_family": "direct_response_stack" if cta else "hero_statement",
        "text_density": "moderate" if subheadline else "minimal",
        "text_elements": {
            "eyebrow": None,
            "badge": None,
            "headline": {"content": headline or "", "emphasis_spans": []},
            "support_copy": {"content": subheadline, "structure": "sentence"} if subheadline else None,
            "cta": {"content": cta} if cta else None,
            "attribution": None,
        },
        "hierarchy_profile": {
            "dominant": "headline",
            "headline_scale": "xl",
            "support_scale": "sm" if subheadline else None,
            "cta_scale": "md" if cta else None,
            "scale_ratio": 5.0,
        },
        "placement": {
            "primary_zone": zone_key,
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
            "mode": color_mode,
            "headline_color": headline_color,
            "support_color": support_color,
            "accent_color": headline_color,
            "cta_bg": cta_bg,
            "cta_fg": cta_fg,
        },
        "cta_style": {"type": "pill_filled" if cta else "none", "prominence": "standard"},
        "scrim": {"enabled": False, "type": None, "extent": None, "max_opacity": 0.0},
    }
    return tds.normalize(raw)


def _available_families(fonts: dict[str, str]) -> list[str]:
    """Return the set of font-family strings that actually loaded (non-empty base64)."""
    out: list[str] = []
    seen = set()
    for entry in _FONT_FACE_MAP:
        key = entry["key"]
        if fonts.get(key):
            fam = entry["family"]
            if fam not in seen:
                seen.add(fam)
                out.append(fam)
    return out


def _revise_html_if_needed(
    html: str,
    spec: dict,
    score: dict,
    client: anthropic.Anthropic,
    font_face_css: str,
) -> str:
    """If generic_ai_risk > 40, call Claude again to fix the HTML."""
    if score["generic_ai_risk"] <= 40:
        return html

    print(f"[HTML_RENDER] generic_ai_risk={score['generic_ai_risk']} > 40, requesting revision...")
    warnings_str = "\n".join(f"- {w}" for w in score["warnings"])
    revision_prompt = f"""You previously generated HTML for an ad overlay. The quality review flagged these issues:

{warnings_str}

The HTML risks looking generic or auto-made.

Here is your previous HTML:
<html>
{html}
</html>

Requirements for revision:
1. Honor the text_design_spec exactly (do not change layout, zones, fonts).
2. Fix the flagged issues above.
3. Make the output feel more editorial, intentional, professional.
4. Ensure hierarchy is extreme (headline 4-5x larger than support).
5. Ensure typography contrasts (if headline is large + bold, support is noticeably smaller + lighter).
6. Avoid centered-default patterns unless explicitly required.
7. Use the specified colors, accent color, and emphasis spans exactly.
8. Output ONLY the revised HTML (<!DOCTYPE html> to </html>). No explanation.

Revise the HTML now."""

    revision_response = client.messages.create(
        model="claude-sonnet-4-20250514",
        max_tokens=4000,
        messages=[{"role": "user", "content": revision_prompt}],
    )
    revised_html = revision_response.content[0].text.strip()

    # Strip markdown fences
    if "```" in revised_html:
        match = re.search(r"<!DOCTYPE html>[\s\S]*?</html>", revised_html, re.IGNORECASE)
        if match:
            revised_html = match.group(0)

    if not re.match(r"<!DOCTYPE html>", revised_html, re.IGNORECASE):
        match = re.search(r"<!DOCTYPE html>[\s\S]*?</html>", revised_html, re.IGNORECASE)
        if match:
            revised_html = match.group(0)

    if font_face_css:
        font_style_block = f"<style>\n{font_face_css}\n</style>"
        if "</head>" in revised_html:
            revised_html = revised_html.replace("</head>", f"{font_style_block}\n</head>", 1)
        else:
            revised_html = revised_html.replace("<body", f"{font_style_block}\n<body", 1)

    print(f"[HTML_RENDER] Revision complete, re-scoring...")
    revised_score = _score_html(revised_html, spec)
    print(f"[HTML_RENDER] revised score={revised_score['generic_ai_risk']}")
    return revised_html


def _score_html(html: str, spec: dict) -> dict:
    """Deterministic heuristic critique of the generated HTML. Log-only.

    Returns:
        { "generic_ai_risk": int 0..100, "warnings": [str, ...] }
    """
    warnings: list[str] = []
    risk = 0

    # 1) Element presence vs layout family
    lf = spec.get("layout_family", "")
    fam = ads.LAYOUT_FAMILIES.get(lf, {})
    forbidden = set(fam.get("forbidden_elements", []))
    elements = spec.get("text_elements") or {}

    # Crude detection: if a forbidden element's content string appears in the
    # HTML, that's a violation. (Not perfect but sufficient as a signal.)
    for key in ["eyebrow", "badge", "support_copy", "attribution"]:
        if key in forbidden:
            obj = elements.get(key) or {}
            content = (obj.get("content") or "").strip()
            if content and content[:20] in html:
                warnings.append(f"forbidden element content present: {key}")
                risk += 20

    # 2) CTA presence vs cta_style
    cta_type = (spec.get("cta_style") or {}).get("type", "none")
    has_button_like = bool(re.search(r"<button|<a\b|border-radius\s*:\s*\d{2,}px", html))
    if cta_type == "none" and has_button_like:
        # "button-like" shapes when no CTA was requested
        warnings.append("cta_style=none but button-like element present")
        risk += 15
    if cta_type in ("pill_filled", "rectangular_filled") and not has_button_like:
        warnings.append(f"cta_style={cta_type} but no button-like element found")
        risk += 10

    # 3) Font-family count (max 2 distinct families)
    fams = set(re.findall(r"font-family\s*:\s*['\"]?([A-Za-z ]+?)['\",;]", html))
    fams = {f.strip() for f in fams if f.strip() and f.strip().lower() not in ("sans-serif", "serif", "inherit")}
    if len(fams) > 3:
        warnings.append(f"too many font families: {sorted(fams)}")
        risk += 20

    # 4) Centered-default when alignment != center
    alignment = (spec.get("placement") or {}).get("alignment", "center")
    if alignment != "center":
        if re.search(r"text-align\s*:\s*center", html):
            warnings.append("text-align:center but placement.alignment != center")
            risk += 10
        if re.search(r"left\s*:\s*50%", html) and re.search(r"translate\(-50%", html):
            warnings.append("left:50% + translate(-50%) centering but alignment != center")
            risk += 10

    # 5) Scrim default abuse
    scrim_enabled = (spec.get("scrim") or {}).get("enabled", False)
    if not scrim_enabled:
        # Look for a full-width opaque-ish gradient anchored to bottom
        if re.search(r"linear-gradient\([^)]*\)[\s\S]{0,200}bottom\s*:\s*0", html) and re.search(r"width\s*:\s*(100%|1080px)", html):
            warnings.append("possible bottom scrim but scrim.enabled=False")
            risk += 15

    # 6) Timid headline size
    sizes = [int(m.group(1)) for m in re.finditer(r"font-size\s*:\s*(\d+)px", html)]
    if sizes:
        max_size = max(sizes)
        hl_scale = (spec.get("hierarchy_profile") or {}).get("headline_scale", "xl")
        min_target = {"md": 70, "lg": 88, "xl": 118, "xxl": 148}.get(hl_scale, 100)
        if max_size < min_target:
            warnings.append(f"max font-size {max_size}px below target {min_target}px for scale {hl_scale}")
            risk += 10

    risk = min(risk, 100)
    return {"generic_ai_risk": risk, "warnings": warnings}
