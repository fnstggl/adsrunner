#!/usr/bin/env python3
"""
Composite HTML text overlay onto image.
Renders HTML to PNG and overlays on background image.
"""

import asyncio
import base64
import io
from pathlib import Path

import cv2
import numpy as np
from PIL import Image


async def render_html_to_image(html_content: str, width: int = 1080, height: int = 1350) -> np.ndarray:
    """Render HTML to image using Playwright."""
    try:
        from playwright.async_api import async_playwright
    except ImportError:
        print("Installing playwright...")
        import subprocess
        subprocess.run(["pip", "install", "playwright"], check=True)
        from playwright.async_api import async_playwright

    async with async_playwright() as p:
        browser = await p.chromium.launch()
        page = await browser.new_page(viewport={"width": width, "height": height})

        # Set content and wait for render
        await page.set_content(html_content)
        await page.wait_for_load_state("networkidle")

        # Take screenshot
        screenshot_bytes = await page.screenshot(omit_background=False)

        await browser.close()

        # Convert to numpy array
        image = Image.open(io.BytesIO(screenshot_bytes))
        return cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)


def composite_overlay(background_image: np.ndarray, overlay_image: np.ndarray) -> np.ndarray:
    """
    Composite overlay image on top of background image.
    Overlay should have transparency (RGBA or with alpha channel).
    """
    # Ensure overlay has alpha channel
    if overlay_image.shape[2] == 3:
        # Add alpha channel (fully opaque)
        overlay_image = cv2.cvtColor(overlay_image, cv2.COLOR_BGR2BGRA)
    elif overlay_image.shape[2] != 4:
        print(f"Unexpected overlay shape: {overlay_image.shape}")
        return background_image

    # Convert background to have same height/width as overlay if needed
    if background_image.shape[:2] != overlay_image.shape[:2]:
        bg_h, bg_w = background_image.shape[:2]
        ov_h, ov_w = overlay_image.shape[:2]

        # Resize overlay to match background
        overlay_image = cv2.resize(overlay_image, (bg_w, bg_h), interpolation=cv2.INTER_LINEAR)

    # Extract alpha channel
    overlay_bgr = overlay_image[:, :, :3]
    alpha = overlay_image[:, :, 3].astype(float) / 255.0

    # Blend
    for c in range(3):
        background_image[:, :, c] = (
            overlay_bgr[:, :, c] * alpha + background_image[:, :, c] * (1 - alpha)
        ).astype(np.uint8)

    return background_image


async def create_composite(html_path: str, image_path: str, output_path: str) -> bool:
    """Create composite image from HTML overlay and background image."""

    print(f"Loading HTML from: {html_path}")
    with open(html_path, "r") as f:
        html_content = f.read()

    print(f"Loading background image from: {image_path}")
    background = cv2.imread(image_path)
    if background is None:
        print(f"Failed to load background image")
        return False

    print(f"Background size: {background.shape}")

    print("Rendering HTML to image...")
    try:
        overlay = await render_html_to_image(html_content, width=1080, height=1350)
        print(f"Overlay size: {overlay.shape}")
    except Exception as e:
        print(f"Failed to render HTML: {e}")
        import traceback
        traceback.print_exc()
        return False

    print("Compositing overlay onto background...")
    result = composite_overlay(background.copy(), overlay)

    print(f"Saving composite to: {output_path}")
    cv2.imwrite(output_path, result)
    print("✓ Composite created successfully!")

    return True


if __name__ == "__main__":
    import io
    import sys

    html_file = "/home/user/adsrunner/e2e_test_output.html"
    image_file = "/home/user/adsrunner/test_image.jpg"
    output_file = "/home/user/adsrunner/composite_output.jpg"

    success = asyncio.run(
        create_composite(html_file, image_file, output_file)
    )

    if success:
        print(f"\n✓ Composite saved to: {output_file}")
        sys.exit(0)
    else:
        print("\n✗ Failed to create composite")
        sys.exit(1)
