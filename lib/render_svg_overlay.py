"""
SVG-based ad text rendering pipeline.

Replaces the PIL/OpenCV renderer with Claude-generated SVGs rasterized via CairoSVG.
All fonts and images are embedded as base64 — zero network dependencies at render time.
"""

from __future__ import annotations

import base64
import io
import re
import traceback

import anthropic
import cv2
import numpy as np
import requests
from PIL import Image

# ---------------------------------------------------------------------------
# Module-level font cache — fetched once, reused across calls
# ---------------------------------------------------------------------------
_FONT_CACHE: dict[str, str] | None = None

# Font definitions: key -> (Google Fonts CSS URL query)
_FONT_SPECS: dict[str, str] = {
    "Inter-400": "family=Inter:wght@400",
    "Inter-700": "family=Inter:wght@700",
    "Inter-900": "family=Inter:wght@900",
    "Montserrat-700": "family=Montserrat:wght@700",
    "Montserrat-900": "family=Montserrat:wght@900",
    "BebasNeue-400": "family=Bebas+Neue",
    "Oswald-700": "family=Oswald:wght@700",
    "PlayfairDisplay-700italic": "family=Playfair+Display:ital,wght@1,700",
    "Lora-400italic": "family=Lora:ital@1",
    "DMSerifDisplay-400italic": "family=DM+Serif+Display:ital@1",
    "SpaceGrotesk-700": "family=Space+Grotesk:wght@700",
    "CormorantGaramond-700italic": "family=Cormorant+Garamond:ital,wght@1,700",
}

_WOFF2_URL_RE = re.compile(r"url\((https://fonts\.gstatic\.com[^)]+\.woff2)\)")

# Google Fonts requires a browser-like UA to serve woff2 instead of ttf
_GFONTS_UA = "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"


# ---------------------------------------------------------------------------
# Font fetching
# ---------------------------------------------------------------------------

def _fetch_fonts() -> dict[str, str]:
    """Fetch all fonts from Google Fonts, base64 encode them.  Returns dict of key -> b64 string."""
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
                print(f"[SVG_RENDER] No woff2 URL found for {key}")
                fonts[key] = ""
                continue
            woff2_url = match.group(1)
            woff2_resp = requests.get(woff2_url, timeout=15)
            woff2_resp.raise_for_status()
            fonts[key] = base64.b64encode(woff2_resp.content).decode()
            print(f"[SVG_RENDER] Fetched {key} ({len(woff2_resp.content)} bytes)")
        except Exception as exc:
            print(f"[SVG_RENDER] Font fetch failed for {key}: {exc}")
            fonts[key] = ""

    _FONT_CACHE = fonts
    return fonts


# ---------------------------------------------------------------------------
# @font-face CSS generation
# ---------------------------------------------------------------------------

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
]


def _build_font_face_css(fonts: dict[str, str]) -> str:
    """Build @font-face CSS block from fetched fonts, skipping empty ones."""
    blocks = []
    for entry in _FONT_FACE_MAP:
        b64 = fonts.get(entry["key"], "")
        if not b64:
            continue
        blocks.append(
            f"@font-face {{ font-family: '{entry['family']}'; font-weight: {entry['weight']}; "
            f"font-style: {entry['style']}; "
            f"src: url('data:font/woff2;base64,{b64}') format('woff2'); }}"
        )
    return "\n".join(blocks)


# ---------------------------------------------------------------------------
# System prompt for Claude
# ---------------------------------------------------------------------------

_SYSTEM_PROMPT = """You are a world-class Meta/Instagram ad creative director and SVG typographer. Your ads are indistinguishable from those produced by top creative studios. You have studied the typography and layout of high-performing ads from Poppi, Base44, Ridge, and Kassable.

You will receive:
- A reference photo (for visual context — layout, negative space, colors, subject position)
- Pre-embedded font @font-face declarations
- Ad copy: headline, subheadline, CTA
- Image description

You must output a SINGLE complete SVG file. Output ONLY the SVG. No explanation. No markdown. No code fences. The response must start with <svg and end with </svg>.

SVG TECHNICAL REQUIREMENTS:
- viewBox="0 0 1080 1350" width="1080" height="1350"
- The SVG background MUST be fully transparent. Do NOT include any <image> element, background <rect>, or any element that fills the full canvas. You are generating a text/overlay layer only — it will be composited onto the photo in post-processing.
- Do NOT include any <image> element of any kind. No background image. No placeholder. Nothing.
- If @font-face CSS is provided below, include it EXACTLY as-is inside a <defs><style> block. Do NOT modify, regenerate, or invent any base64 font data.
- If no @font-face CSS is provided, use font-family names directly without @font-face declarations. Do NOT generate any base64-encoded data yourself.
- NEVER generate base64-encoded strings of any kind. All base64 data is pre-provided to you.
- Text elements use <text> with explicit x, y positioning. Do NOT use <foreignObject>.
- For multi-line text, use multiple <tspan> elements with x and dy attributes
- For scrim/overlay effects use <rect> with fill opacity or linearGradient (these are fine — they will composite correctly over the photo)
- For the CTA pill: <rect> with rx equal to half its height, plus centered <text>

CRITICAL SVG RULES (CairoSVG compatibility):
- NEVER use CSS classes for styling text. Use inline style="" attributes on every <text> and <tspan> element.
- NEVER use <foreignObject> — it is not supported by CairoSVG.
- Always specify font-family, font-size, font-weight, and fill as inline style on each <text> element.
- Use letter-spacing and word-spacing as inline style attributes.
- For text wrapping, manually break lines using <tspan x="..." dy="..."> for each line.
- Keep headline to 1-3 lines max. Break lines at natural word boundaries.
- The SVG must be self-contained with no external references.

DESIGN REQUIREMENTS — study these rules and follow them exactly:

HIERARCHY: One dominant text element (the headline). One secondary (subheadline). One tertiary (CTA). Never three things at equal visual weight.
NEGATIVE SPACE: Text must live where the photo has open space. Analyze the image description to determine where the subject is NOT — place text there. Never put text directly on a face or detailed subject area.
TYPOGRAPHY CONTRAST: Mix dramatically. If the headline is Inter 900 ultra-bold, the subheadline must be Inter 400 light or Playfair Display italic — not Inter 700 again. The contrast IS the design.
SCALE: The headline should be visually dominant — fontSize between 80px and 160px depending on copy length. Never timid. If the headline is 3 words, go 140px+.
COLOR: Do NOT default to white text on dark scrim every time. Derive a palette from the image mood. Consider: white text no scrim (if image has dark areas), dark text on light card, a single brand accent color for the CTA that contrasts with the image.
CTA PILL: Always premium-looking. Minimum 48px tall, 32px horizontal padding, rx=24 or higher. The button fill should contrast strongly with whatever is behind it.
SCRIM: If needed, use a subtle linearGradient — not a flat black rectangle. Max 70% opacity at darkest point. Often no scrim is better than a heavy one.
LETTER SPACING: Headlines benefit from negative letter-spacing (-0.02em to -0.05em). Subheadlines and CTA: slightly positive (0.05em to 0.1em) for breathing room.
LINE HEIGHT: Tight for big headlines (1.0 to 1.1). Looser for body/sub (1.4 to 1.6).
BRAND-FIRST: If there's a brand name in the copy, it gets a prominent position — either top-center or integrated into the headline treatment.

WHAT MAKES AN AD PROFITABLE VS GENERIC:
- Profitable: The text feels designed INTO the image, not stamped OVER it
- Profitable: One thing catches your eye first, then guides you to the CTA
- Profitable: The CTA is specific ("Find Real Deals") not generic ("Learn More")
- Generic: Centered white text with a dark scrim covering the whole bottom half
- Generic: Three text blocks all the same size
- Generic: A plain white rectangle as the button"""


# ---------------------------------------------------------------------------
# User message builder
# ---------------------------------------------------------------------------

def _build_user_content(
    img_b64: str,
    font_face_css: str,
    headline: str,
    subheadline: str,
    cta: str,
    image_description: str,
    performance_hints: str,
) -> list[dict]:
    """Build the user message content blocks (vision image + text instructions)."""

    text_msg = f"""IMAGE DESCRIPTION: {image_description}

AD COPY:
Headline: {headline}
Subheadline: {subheadline}
CTA: {cta}

{"PERFORMANCE HINTS: " + performance_hints if performance_hints else ""}

EMBEDDED @font-face CSS (include EXACTLY as-is in <defs><style> — do NOT modify or regenerate):
{font_face_css if font_face_css else "(No fonts were pre-embedded. Use font-family names without @font-face. Do NOT generate any base64 data.)"}

BACKGROUND IMAGE: The photo above is shown for your visual reference only — use it to understand negative space, subject position, and color palette. Do NOT include any <image> element in your SVG. The SVG must be transparent where there is no text or overlay element.

AVAILABLE FONTS (all pre-embedded as @font-face — use exact font-family names as shown):

DISPLAY/IMPACT (bold headlines, high-energy brands):
- 'Bebas Neue' weight 400 — ultra-condensed all-caps, maximum impact, Ridge/streetwear energy
- 'Oswald' weight 700 — condensed bold, editorial authority, slightly warmer than Bebas
- 'Montserrat' weight 900 — geometric ultra-bold, modern tech/startup feel
- 'Inter' weight 900 — neutral ultra-bold, clean and universal, works with anything

MODERN SANS (versatile, contemporary):
- 'Inter' weight 700 — clean workhorse bold, great for subheadlines and CTA
- 'Inter' weight 400 — light and readable, supporting text
- 'Space Grotesk' weight 700 — techy geometric with personality, great for SaaS/fintech
- 'Montserrat' weight 700 — geometric bold, premium feel

EDITORIAL SERIF (luxury, fashion, food, lifestyle):
- 'Playfair Display' weight 700 italic — high-contrast elegant serif, Vogue energy
- 'Cormorant Garamond' weight 700 italic — ultra-refined luxury serif, the most elegant option
- 'DM Serif Display' weight 400 italic — approachable editorial serif, food/lifestyle brands
- 'Lora' weight 400 italic — warm literary serif, health/wellness/organic brands

FONT PAIRING RULES — choose one combination that fits the ad's mood:
- High-energy/youth/product: Bebas Neue headline + Inter 400 subheadline
- Premium/luxury/lifestyle: Cormorant Garamond italic headline + Inter 400 subheadline
- Tech/SaaS/startup: Space Grotesk or Montserrat 900 headline + Inter 400 subheadline
- Warm/lifestyle/relatable: DM Serif Display italic headline + Inter 700 subheadline
- Editorial/fashion: Playfair Display italic headline + Inter 300 subheadline
- Bold/universal: Inter 900 headline + Lora italic subheadline
- Never use two display fonts together. Never use two serifs together.
- The CTA button text is always Inter 700, letter-spacing: 0.08em, regardless of headline font.

Generate the complete SVG ad now."""

    return [
        {
            "type": "image",
            "source": {
                "type": "base64",
                "media_type": "image/png",
                "data": img_b64,
            },
        },
        {"type": "text", "text": text_msg},
    ]


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------

def render_text_overlay(
    image: np.ndarray,
    headline: str,
    subheadline: str = "",
    cta: str = "",
    zone: str = "bottom-center",
    template: str = "light-on-dark",
    logo: np.ndarray = None,
    layout_id: str = None,
    image_description: str = "",
    performance_hints: str = "",
) -> np.ndarray:
    """Generate an SVG ad overlay using Claude and rasterize with CairoSVG.

    Same signature as lib/render_text_overlay.render_text_overlay so it's a
    drop-in replacement.  Falls back to the PIL renderer on any error.
    """
    try:
        # Step 1: Encode base image as base64 (for Claude vision only — not embedded in SVG)
        rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        pil_img = Image.fromarray(rgb)
        buf = io.BytesIO()
        pil_img.save(buf, format="PNG")
        img_b64 = base64.b64encode(buf.getvalue()).decode()

        # Step 2: Fetch and embed fonts
        fonts = _fetch_fonts()
        font_face_css = _build_font_face_css(fonts)

        # Step 3: Call Claude to generate transparent SVG overlay
        # The image is sent as a vision block so Claude can analyse layout/colors,
        # but the SVG it produces contains NO embedded image — transparent background only.
        client = anthropic.Anthropic()
        user_content = _build_user_content(
            img_b64=img_b64,
            font_face_css=font_face_css,
            headline=headline,
            subheadline=subheadline,
            cta=cta,
            image_description=image_description,
            performance_hints=performance_hints,
        )

        response = client.messages.create(
            model="claude-sonnet-4-20250514",
            max_tokens=16000,
            system=_SYSTEM_PROMPT,
            messages=[{"role": "user", "content": user_content}],
        )

        svg_str = response.content[0].text.strip()

        # Ensure we only have the SVG (strip any accidental markdown fences)
        if "```" in svg_str:
            match = re.search(r"<svg[\s\S]*?</svg>", svg_str)
            if match:
                svg_str = match.group(0)

        if not svg_str.startswith("<svg"):
            match = re.search(r"<svg[\s\S]*?</svg>", svg_str)
            if match:
                svg_str = match.group(0)
            else:
                raise ValueError("Claude response does not contain valid SVG")

        # Safety net: strip any <image> elements Claude may have hallucinated.
        # These would re-introduce the large-payload freeze we're avoiding.
        svg_str = re.sub(r"<image\b[^>]*/?>", "", svg_str)

        print(f"[SVG_RENDER] SVG payload size: {len(svg_str):,} bytes (no embedded image)")

        # Step 4: Rasterize the transparent overlay SVG with CairoSVG.
        # With no embedded image, the payload is ~200 KB (fonts only) — renders in seconds.
        import cairosvg
        png_bytes = cairosvg.svg2png(
            bytestring=svg_str.encode("utf-8"),
            output_width=1080,
            output_height=1350,
        )

        # Step 5: Alpha-composite the transparent overlay onto the original photo.
        overlay_pil = Image.open(io.BytesIO(png_bytes)).convert("RGBA")

        # Prepare background: resize to exact output dimensions if needed
        bg_pil = Image.fromarray(rgb).convert("RGBA")
        if bg_pil.size != (1080, 1350):
            bg_pil = bg_pil.resize((1080, 1350), Image.LANCZOS)

        # Composite: overlay sits on top of photo, respecting per-pixel alpha
        bg_pil.alpha_composite(overlay_pil)

        # Convert back to BGR numpy array for the rest of the pipeline
        rgb_arr = np.array(bg_pil.convert("RGB"))
        bgr_arr = cv2.cvtColor(rgb_arr, cv2.COLOR_RGB2BGR)

        print(f"[SVG_RENDER] Success — output shape {bgr_arr.shape}")
        return bgr_arr

    except Exception as exc:
        print(f"[SVG_RENDER] FAILED: {exc} — falling back to PIL")
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
