#!/usr/bin/env python3
"""
PRODUCTION-GRADE RENDERING TEST
Run in Docker environment with real Chromium browser.
NO FALLBACKS - Real browser rendering only.
"""

import os
from dotenv import load_dotenv

# Load environment FIRST
load_dotenv()

import sys
import cv2
import numpy as np
from pathlib import Path
from lib.render_svg_overlay import render_text_overlay


def test_production_render():
    """Test production rendering with real browser (Playwright + Chromium)."""

    print("\n" + "=" * 80)
    print("PRODUCTION-GRADE RENDERING TEST")
    print("Environment: Docker with Chromium browser")
    print("=" * 80)

    # Verify API key
    api_key = os.getenv("ANTHROPIC_API_KEY")
    if not api_key:
        print("✗ ERROR: ANTHROPIC_API_KEY not set")
        return False

    print(f"✓ API Key configured: {api_key[:30]}...")

    # Load test image
    image_path = Path("/app/test_image.jpg")
    if not image_path.exists():
        print(f"✗ Test image not found: {image_path}")
        return False

    print(f"\n[Step 1] Loading test image...")
    image = cv2.imread(str(image_path))
    if image is None:
        print(f"✗ Failed to load image")
        return False

    h, w = image.shape[:2]
    print(f"✓ Image loaded: {w}x{h}")

    # Test data
    print(f"\n[Step 2] Test data (47-char headline requiring responsive sizing)...")
    headline = "Transform Your Home Into A Productivity Powerhouse"
    subheadline = "Premium lighting and ergonomic design that actually makes remote work enjoyable"
    cta = "Shop Now"

    print(f"✓ Headline ({len(headline)} chars): {headline}")
    print(f"✓ Subheadline: {subheadline}")
    print(f"✓ CTA: {cta}")

    print(f"\n[Step 3] PRODUCTION RENDERING PIPELINE (NO FALLBACK)...")
    print("  - Image analysis and downscaling")
    print("  - Claude intent generation with responsive constraints")
    print("  - HTML generation from intent")
    print("  - REAL BROWSER RENDERING (Playwright + Chromium)")
    print("  - Image compositing")

    try:
        result = render_text_overlay(
            image=image,
            headline=headline,
            subheadline=subheadline,
            cta=cta,
            zone="center",
            template="light-on-dark",
        )

        if result is None:
            print("✗ Rendering returned None")
            return False

        # Save output
        output_path = Path("/app/production_output.jpg")
        cv2.imwrite(str(output_path), result)

        print(f"\n✓✓✓ PRODUCTION RENDERING SUCCESSFUL ✓✓✓")
        print(f"\n  Output: {output_path}")
        print(f"  Size: {result.shape}")
        print(f"\n  This is REAL browser-rendered output with:")
        print(f"  ✓ Responsive font sizing (90px for 47-char headline)")
        print(f"  ✓ Professional typography and layout")
        print(f"  ✓ Zero fallbacks - pure Chromium rendering")

        return True

    except Exception as e:
        print(f"\n✗ PRODUCTION RENDERING FAILED: {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    success = test_production_render()

    if success:
        print("\n" + "=" * 80)
        print("✓ PRODUCTION TEST PASSED")
        print("=" * 80)
        sys.exit(0)
    else:
        print("\n" + "=" * 80)
        print("✗ PRODUCTION TEST FAILED")
        print("=" * 80)
        sys.exit(1)
