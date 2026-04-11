#!/usr/bin/env python3
"""
Test the full professional rendering pipeline with responsive font sizing.
Uses render_text_overlay to generate high-quality ad creative.
"""

import os
from dotenv import load_dotenv

# Load environment variables FIRST
load_dotenv()

import cv2
import numpy as np
from pathlib import Path
from lib.render_svg_overlay import render_text_overlay


def test_professional_render():
    """Test the full rendering pipeline with responsive sizing."""

    print("=" * 80)
    print("PROFESSIONAL RENDERING PIPELINE TEST")
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

    # Test case: Home office product with long headline
    print(f"\n[Step 2] Preparing test data...")

    headline = "Transform Your Home Into A Productivity Powerhouse"
    subheadline = "Premium lighting and ergonomic design that actually makes remote work enjoyable"
    cta = "Shop Now"

    print(f"✓ Headline (47 chars): {headline}")
    print(f"✓ Subheadline: {subheadline}")
    print(f"✓ CTA: {cta}")

    print(f"\n[Step 3] Running full professional rendering pipeline...")
    print("  - Downscaling image for Claude vision")
    print("  - Analyzing image (brightness, colors, zones)")
    print("  - Generating layout intent with Claude")
    print("  - Building composition engine with responsive sizing")
    print("  - Rendering HTML with professional fonts")
    print("  - Compositing overlay onto image")

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
        output_path = "/home/user/adsrunner/professional_output.jpg"
        cv2.imwrite(output_path, result)

        print(f"\n✓ RENDERING SUCCESSFUL!")
        print(f"  → Output saved to: {output_path}")
        print(f"  → Output size: {result.shape}")

        return True

    except Exception as e:
        print(f"✗ Rendering failed: {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    import sys
    success = test_professional_render()
    sys.exit(0 if success else 1)
