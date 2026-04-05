"""
HTML/CSS ad text overlay pipeline.

Claude generates a transparent HTML overlay (text + scrims + CTA).
Playwright/Chromium rasterizes it in ~1-2s (real browser — no font freezes).
PIL alpha_composites the result onto the background photo.

Drop-in replacement for lib/render_text_overlay.render_text_overlay.
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

_SYSTEM_PROMPT = """You are a world-class Meta/Instagram ad creative director and front-end typographer. Your ads are indistinguishable from those produced by top creative studios. You have studied the typography and layout of high-performing ads from Poppi, Base44, Ridge, and Kassable.

You will receive:
- A reference photo (for visual context — layout, negative space, colors, subject position)
- Ad copy: headline, subheadline, CTA
- Image description

You must output a SINGLE HTML document. Output ONLY the HTML. No explanation. No markdown. No code fences. The response must start with <!DOCTYPE html> and end with </html>.

HTML TECHNICAL REQUIREMENTS:
- The document represents a 1080×1350px ad overlay — transparent background, text/scrims only.
- html and body: width: 1080px; height: 1350px; overflow: hidden; margin: 0; padding: 0; background: transparent;
- All elements use position: absolute for placement.
- Do NOT include any <style> block or @font-face declarations — fonts are injected automatically.
- Do NOT reference any external URLs (no Google Fonts CDN, no images, no scripts).
- Do NOT include any background image or full-canvas background color.
- The document will be screenshotted by a headless browser at 1:1 pixel ratio. What you write is exactly what renders.

WHAT YOU CAN USE (all work perfectly in Chromium):
- Any CSS property: font-family, font-size, font-weight, font-style, letter-spacing, line-height, text-shadow, color, opacity
- Gradients: linear-gradient(), radial-gradient() on any element
- Rounded corners: border-radius on any div (use for CTA pills, cards, scrims)
- Flexbox and CSS Grid for centering/alignment
- Multiple text-shadow values for glow or multi-shadow effects
- mix-blend-mode if useful
- CSS transforms (rotate, scale) for creative angles
- White-space: pre-wrap or <br> tags for explicit line breaks in headlines

DESIGN REQUIREMENTS — study these rules and follow them exactly:

HIERARCHY: One dominant text element (the headline). One secondary (subheadline). One tertiary (CTA). Never three things at equal visual weight.
NEGATIVE SPACE: Text must live where the photo has open space. Analyze the image description to determine where the subject is NOT — place text there. Never put text directly on a face or detailed subject area.
TYPOGRAPHY CONTRAST: Mix dramatically. If the headline is Inter 900 ultra-bold, the subheadline must be Inter 400 light or Playfair Display italic — not Inter 700 again. The contrast IS the design.
SCALE: The headline should be visually dominant — font-size between 80px and 160px depending on copy length. Never timid. If the headline is 3 words, go 140px+.
COLOR: Do NOT default to white text on dark scrim every time. Derive a palette from the image mood. Consider: white text no scrim (if image has dark areas), dark text on light card, a single brand accent color for the CTA that contrasts with the image.
CTA PILL: Always premium-looking. Min 56px tall, 36px horizontal padding, border-radius: 28px or higher. The button fill must contrast strongly with whatever is behind it.
SCRIM: If needed, use a CSS linear-gradient on an absolutely-positioned div — not a flat opaque rectangle. Max 70% opacity at darkest point. Often no scrim is better than a heavy one.
LETTER SPACING: Headlines benefit from letter-spacing: -0.02em to -0.05em. Subheadlines and CTA: slightly positive (0.05em to 0.1em).
LINE HEIGHT: Tight for big headlines (line-height: 1.0 to 1.1). Looser for body/sub (1.4 to 1.6).

AVAILABLE FONTS (pre-embedded — use exact font-family names):

DISPLAY/IMPACT (bold headlines, high-energy brands):
- 'Bebas Neue' weight 400 — ultra-condensed all-caps, maximum impact
- 'Oswald' weight 700 — condensed bold, editorial authority
- 'Montserrat' weight 900 — geometric ultra-bold, modern tech/startup feel
- 'Inter' weight 900 — neutral ultra-bold, clean and universal

MODERN SANS (versatile, contemporary):
- 'Inter' weight 700 — clean workhorse bold, great for subheadlines and CTA
- 'Inter' weight 400 — light and readable, supporting text
- 'Space Grotesk' weight 700 — techy geometric, great for SaaS/fintech
- 'Montserrat' weight 700 — geometric bold, premium feel

EDITORIAL SERIF (luxury, fashion, food, lifestyle):
- 'Playfair Display' weight 700 italic — high-contrast elegant serif
- 'Cormorant Garamond' weight 700 italic — ultra-refined luxury serif
- 'DM Serif Display' weight 400 italic — approachable editorial serif
- 'Lora' weight 400 italic — warm literary serif, health/wellness

FONT PAIRING RULES:
- High-energy/youth/product: 'Bebas Neue' headline + 'Inter' 400 subheadline
- Premium/luxury/lifestyle: 'Cormorant Garamond' italic headline + 'Inter' 400 subheadline
- Tech/SaaS/startup: 'Space Grotesk' or 'Montserrat' 900 headline + 'Inter' 400 subheadline
- Warm/lifestyle/relatable: 'DM Serif Display' italic headline + 'Inter' 700 subheadline
- Editorial/fashion: 'Playfair Display' italic headline + 'Inter' 400 subheadline
- Bold/universal: 'Inter' 900 headline + 'Lora' italic subheadline
- Never two display fonts together. Never two serifs together.
- CTA button text: always 'Inter' font-weight 700, letter-spacing: 0.08em.

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
    img_media_type: str,
    headline: str,
    subheadline: str,
    cta: str,
    image_description: str,
    performance_hints: str,
) -> list[dict]:
    text_msg = f"""IMAGE DESCRIPTION: {image_description}

AD COPY:
Headline: {headline}
Subheadline: {subheadline}
CTA: {cta}

{"PERFORMANCE HINTS: " + performance_hints if performance_hints else ""}

The photo above is your visual reference — analyse negative space, subject position, and color palette to inform layout and color decisions. Do not embed or reference it in the HTML. Do not include any <style> block or font declarations — fonts are injected automatically after your response.

Generate the complete HTML overlay document now."""

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
    """Generate an HTML ad overlay using Claude, rasterize with Playwright/Chromium,
    and alpha-composite onto the background photo.

    Drop-in replacement for lib/render_text_overlay.render_text_overlay.
    Falls back to the PIL renderer on any error.
    """
    try:
        # Step 1: Encode a downscaled reference image for Claude vision.
        # Claude only needs layout/color context — 540×675 JPEG ~50KB vs 4MB full PNG.
        rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        small = Image.fromarray(rgb).resize((540, 675), Image.LANCZOS)
        buf_small = io.BytesIO()
        small.save(buf_small, format="JPEG", quality=85)
        img_b64 = base64.b64encode(buf_small.getvalue()).decode()
        img_media_type = "image/jpeg"
        print(f"[HTML_RENDER] Reference image: {len(buf_small.getvalue()):,} bytes (540×675 JPEG)")

        # Step 2: Fetch fonts (module-level cache — one network round-trip per process lifetime)
        fonts = _fetch_fonts()
        font_face_css = _build_font_face_css(fonts)

        # Step 3: Ask Claude to generate the HTML overlay.
        # Font CSS is NOT sent to Claude — it's injected into the HTML after generation.
        # This removes ~172 KB of base64 from the prompt, cutting API time dramatically.
        client = anthropic.Anthropic()
        user_content = _build_user_content(
            img_b64=img_b64,
            img_media_type=img_media_type,
            headline=headline,
            subheadline=subheadline,
            cta=cta,
            image_description=image_description,
            performance_hints=performance_hints,
        )

        response = client.messages.create(
            model="claude-sonnet-4-20250514",
            max_tokens=4000,
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

        # Step 4: Inject @font-face CSS into the HTML before Playwright sees it.
        # Chromium handles WOFF2 natively in milliseconds — this is where fonts live.
        if font_face_css:
            font_style_block = f"<style>\n{font_face_css}\n</style>"
            if "</head>" in html_str:
                html_str = html_str.replace("</head>", f"{font_style_block}\n</head>", 1)
            else:
                # Fallback: insert after <body> opening tag
                html_str = html_str.replace("<body", f"{font_style_block}\n<body", 1)

        print(f"[HTML_RENDER] HTML payload: {len(html_str):,} bytes — rasterizing with Playwright")

        # Step 5: Rasterize with Playwright (headless Chromium — handles WOFF2 natively, ~1-2s)
        png_bytes = _rasterize_html(html_str, width=1080, height=1350)

        # Step 6: Alpha-composite the transparent overlay onto the original photo
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
