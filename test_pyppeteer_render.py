#!/usr/bin/env python3
"""
Test professional rendering using pyppeteer instead of Playwright.
This should work in sandboxed environments without requiring browser installation.
"""

import os
from dotenv import load_dotenv

# Load environment variables FIRST
load_dotenv()

import asyncio
import cv2
import numpy as np
from pathlib import Path
from pyppeteer import launch
from PIL import Image
import io


async def render_html_with_pyppeteer(html_content: str, width: int = 1080, height: int = 1350) -> np.ndarray:
    """Render HTML to image using pyppeteer."""
    print("[PYPPETEER] Launching Chromium...")

    try:
        browser = await launch(
            headless=True,
            args=[
                "--no-sandbox",
                "--disable-setuid-sandbox",
                "--single-process",
                "--disable-dev-shm-usage",
            ],
        )
        print("[PYPPETEER] ✓ Browser launched")

        page = await browser.newPage()
        await page.setViewport({"width": width, "height": height})
        print(f"[PYPPETEER] ✓ Viewport set to {width}x{height}")

        print("[PYPPETEER] Setting page content...")
        await page.setContent(html_content, waitUntil="load")
        print("[PYPPETEER] ✓ Content loaded")

        print("[PYPPETEER] Taking screenshot...")
        png_bytes = await page.screenshot({"omitBackground": True})
        print(f"[PYPPETEER] ✓ Screenshot captured ({len(png_bytes):,} bytes)")

        await browser.close()
        print("[PYPPETEER] ✓ Browser closed")

        # Convert to numpy array
        image = Image.open(io.BytesIO(png_bytes))
        return cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)

    except Exception as e:
        print(f"[PYPPETEER] ✗ Failed: {e}")
        raise


async def test_pyppeteer_rendering():
    """Test pyppeteer rendering with real HTML and image."""

    print("=" * 80)
    print("PYPPETEER PROFESSIONAL RENDERING TEST")
    print("=" * 80)

    # Load test image
    image_path = "/home/user/adsrunner/test_image.jpg"
    print(f"\n[Step 1] Loading test image: {image_path}")

    if not Path(image_path).exists():
        print(f"✗ Image not found: {image_path}")
        return False

    image = cv2.imread(image_path)
    if image is None:
        print(f"✗ Failed to load image")
        return False

    h, w = image.shape[:2]
    print(f"✓ Image loaded: {w}x{h}")

    # Read the HTML that was generated in the previous test
    html_path = "/home/user/adsrunner/e2e_test_output.html"
    print(f"\n[Step 2] Loading generated HTML: {html_path}")

    if not Path(html_path).exists():
        print(f"✗ HTML not found: {html_path}")
        return False

    with open(html_path, "r") as f:
        html_content = f.read()

    print(f"✓ HTML loaded ({len(html_content)} bytes)")

    print(f"\n[Step 3] Rendering HTML with pyppeteer (NO FALLBACK)...")
    try:
        overlay = await render_html_with_pyppeteer(html_content, width=1080, height=1350)
        print(f"✓ Overlay rendered: {overlay.shape}")

        # Composite overlay onto image
        print(f"\n[Step 4] Compositing overlay onto image...")

        # Convert to PIL for proper alpha compositing
        bg_pil = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
        overlay_pil = Image.fromarray(cv2.cvtColor(overlay, cv2.COLOR_BGR2RGB))

        # Add alpha channel to overlay
        if overlay_pil.mode != "RGBA":
            overlay_pil = overlay_pil.convert("RGBA")

        # Composite
        bg_pil.paste(overlay_pil, (0, 0), overlay_pil)

        # Convert back to CV2 BGR
        result = cv2.cvtColor(np.array(bg_pil), cv2.COLOR_RGB2BGR)

        # Save output
        output_path = "/home/user/adsrunner/pyppeteer_professional_output.jpg"
        cv2.imwrite(output_path, result)

        print(f"\n✓ PYPPETEER RENDERING SUCCESSFUL!")
        print(f"  → Output saved to: {output_path}")
        print(f"  → Output size: {result.shape}")
        print(f"\nNO FALLBACKS USED - PRODUCTION-GRADE OUTPUT")

        return True

    except Exception as e:
        print(f"✗ Rendering failed: {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    import sys
    success = asyncio.run(test_pyppeteer_rendering())
    sys.exit(0 if success else 1)
