#!/usr/bin/env python3
"""
Patched render_svg_overlay.py that uses pyppeteer instead of Playwright.
This bypasses the browser installation issues in sandboxed environments.
"""

import asyncio
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
from pyppeteer import launch

from lib import ad_design_system as ads
from lib import image_analysis as imganalysis
from lib import text_design_spec as tds

# Font cache and specs (same as original)
_FONT_CACHE: dict[str, str] | None = None

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

_GFONTS_UA = (
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
    "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
)


async def _rasterize_html_pyppeteer(html: str, width: int = 1080, height: int = 1350) -> bytes:
    """Render HTML to PNG using pyppeteer (instead of Playwright)."""
    print("[HTML_RENDER] Launching Chromium via pyppeteer...")

    try:
        browser = await launch(
            headless=True,
            args=["--no-sandbox", "--disable-setuid-sandbox", "--single-process"],
        )
        page = await browser.newPage()
        await page.setViewport({"width": width, "height": height})

        print("[HTML_RENDER] Setting page content...")
        await page.setContent(html, waitUntil="load")

        print("[HTML_RENDER] Taking screenshot...")
        png_bytes = await page.screenshot({"omitBackground": True})

        await browser.close()
        print("[HTML_RENDER] ✓ Pyppeteer rendering successful")
        return png_bytes

    except Exception as e:
        print(f"[HTML_RENDER] Pyppeteer failed: {e}")
        raise


def render_text_overlay_patched(
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
    """
    Patched version that uses pyppeteer instead of Playwright.
    Same functionality, but works in sandboxed environments.
    """
    try:
        # Step 0: Build / normalize the text_design_spec
        spec = tds.build_spec(
            headline=headline,
            subheadline=subheadline,
            cta=cta,
            zone=zone,
            template=template,
        ) if not text_design_spec else text_design_spec

        # Step 1: Downscale reference image for Claude vision
        rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        small = Image.fromarray(rgb).resize((540, 675), Image.LANCZOS)
        buf_small = io.BytesIO()
        small.save(buf_small, format="JPEG", quality=85)
        img_b64 = base64.b64encode(buf_small.getvalue()).decode()
        img_media_type = "image/jpeg"
        print(f"[HTML_RENDER] Reference image: {len(buf_small.getvalue()):,} bytes (540×675 JPEG)")

        # Step 2: Image analysis
        analysis = imganalysis.analyze_image(image)
        spec = tds.merge_image_analysis(spec, analysis)

        # Step 3: Get layout intent from Claude
        from lib.generate_layout_intent import generate_layout_intent
        from lib.generate_html_from_intent import generate_html_from_intent

        product_description = headline or subheadline or cta or "Product"
        intent = generate_layout_intent(
            image_b64=img_b64,
            image_media_type=img_media_type,
            product_description=product_description,
            tone_mode=spec.get("tone_mode", "performance_ugc"),
            text_design_spec=spec,
            image_analysis=analysis,
            layout_tokens=spec.get("layout_tokens", {}),
        )
        print(f"[HTML_RENDER] Layout intent: family={intent.get('layout_family')}")

        # Step 4: Generate HTML from intent
        text_elements = intent.get("text_elements", spec.get("text_elements", {}))
        layout_tokens = spec.get("layout_tokens", {})
        html = generate_html_from_intent(intent, layout_tokens, text_elements, analysis)

        # Step 5: Render HTML with pyppeteer instead of Playwright
        print("[HTML_RENDER] Rendering HTML with pyppeteer...")
        png_bytes = asyncio.run(_rasterize_html_pyppeteer(html, width=1080, height=1350))

        # Step 6: Composite onto background
        png_img = Image.open(io.BytesIO(png_bytes)).convert("RGBA")
        bg_img = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB)).convert("RGBA")

        # Composite
        bg_img.paste(png_img, (0, 0), png_img)
        result = cv2.cvtColor(np.array(bg_img), cv2.COLOR_RGB2BGR)

        print("[HTML_RENDER] ✓ Rendering complete (pyppeteer)")
        return result

    except Exception as e:
        print(f"[HTML_RENDER] FAILED: {e}")
        traceback.print_exc()
        return image  # Return original image as fallback


if __name__ == "__main__":
    print("Patched render module loaded. Use this instead of lib.render_svg_overlay")
