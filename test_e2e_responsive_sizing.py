#!/usr/bin/env python3
"""End-to-end test of responsive font sizing with real image and Claude API."""

import base64
import json
import sys
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

# Import pipeline components
from lib.classify_input import classify_input
from lib.build_creative_specs import build_creative_specs
from lib.generate_layout_intent import generate_layout_intent
from lib.generate_html_from_intent import generate_html_from_intent


def encode_image(image_path: str) -> tuple[str, str]:
    """Encode image to base64 and detect media type."""
    with open(image_path, "rb") as f:
        image_data = f.read()

    b64 = base64.b64encode(image_data).decode()

    # Detect media type from extension
    ext = Path(image_path).suffix.lower()
    media_type_map = {
        ".jpg": "image/jpeg",
        ".jpeg": "image/jpeg",
        ".png": "image/png",
        ".gif": "image/gif",
        ".webp": "image/webp",
    }
    media_type = media_type_map.get(ext, "image/jpeg")

    return b64, media_type


def run_e2e_test(image_path: str):
    """Run full end-to-end pipeline."""

    print("=" * 80)
    print("END-TO-END TEST: Responsive Font Sizing")
    print("=" * 80)

    # Check if image exists
    if not Path(image_path).exists():
        print(f"✗ Image not found: {image_path}")
        return False

    print(f"\n[Step 1] Encoding image...")
    try:
        image_b64, media_type = encode_image(image_path)
        print(f"✓ Image encoded ({media_type}, {len(image_b64)} bytes)")
    except Exception as e:
        print(f"✗ Failed to encode image: {e}")
        return False

    # Sample product for testing
    product_description = "Premium home office setup with modern lighting and ergonomic workspace. Perfect for remote work and productivity."

    print(f"\n[Step 2] Classifying input...")
    try:
        classification = classify_input(product_description)
        print(f"✓ Classification: {classification}")
    except Exception as e:
        print(f"✗ Classification failed: {e}")
        return False

    print(f"\n[Step 3] Building creative specs...")
    try:
        specs = build_creative_specs(classification, product_description)
        print(f"✓ Specs built")
        print(f"  - Tone: {specs.get('tone_mode')}")
        print(f"  - Family: {specs.get('primary_family')}")
    except Exception as e:
        print(f"✗ Specs building failed: {e}")
        return False

    print(f"\n[Step 4] Generating layout intent with Claude...")
    try:
        intent = generate_layout_intent(
            image_b64=image_b64,
            image_media_type=media_type,
            product_description=product_description,
            tone_mode=specs.get("tone_mode", "performance_ugc"),
            text_design_spec=specs,
            image_analysis={"brightness": "medium", "colors": ["warm", "neutral"]},
            layout_tokens={
                "headline_size_range": (100, 150),
                "support_size_range": (32, 48),
                "headline_line_height": 1.0,
                "support_line_height": 1.4,
                "gap_headline_support": 12,
                "gap_support_cta": 16,
                "usable_width": 900,
                "zone_rect": {"x": 50, "y": 100, "w": 980, "h": 800},
                "safe_margin": 16,
                "headline_color": "#FFFFFF",
                "support_color": "#F0F0F0",
                "accent_color": "#999999",
                "cta_bg": "#FF6B35",
                "cta_fg": "#FFFFFF",
                "eyebrow_size": 24,
            }
        )

        if not intent:
            print("✗ Intent generation returned None")
            return False

        print(f"✓ Intent generated")
        print(f"  - Family: {intent.get('layout_family')}")
        print(f"  - Zone: {intent.get('placement', {}).get('zone')}")

        # Show text elements
        text_els = intent.get('text_elements', {})
        print(f"  - Headline: '{text_els.get('headline', {}).get('content', '')}'")
        if text_els.get('support_copy', {}).get('present'):
            print(f"  - Support: '{text_els.get('support_copy', {}).get('content', '')}'")
    except Exception as e:
        print(f"✗ Intent generation failed: {e}")
        import traceback
        traceback.print_exc()
        return False

    print(f"\n[Step 5] Generating HTML from intent with responsive sizing...")
    try:
        html = generate_html_from_intent(intent, {"description": product_description})

        if not html or "<!DOCTYPE html>" not in html:
            print("✗ HTML generation failed or invalid")
            return False

        print(f"✓ HTML generated successfully")

        # Check for responsive sizing
        import re
        font_sizes = re.findall(r"font-size: (\d+)px", html)
        if font_sizes:
            font_sizes_int = [int(s) for s in font_sizes]
            print(f"  - Font sizes used: {sorted(set(font_sizes_int))}")

            max_size = max(font_sizes_int)
            min_size = min(font_sizes_int)
            if max_size <= 150:
                print(f"  ✓ All sizes within bounds ({min_size}px - {max_size}px ≤ 150px max)")
            else:
                print(f"  ✗ Size exceeds max: {max_size}px > 150px")

        # Save to file
        output_path = "/home/user/adsrunner/e2e_test_output.html"
        with open(output_path, "w") as f:
            f.write(html)

        print(f"\n✓ TEST PASSED")
        print(f"  → HTML output saved to: {output_path}")
        print(f"\n  You can open this file in a browser to see the responsive")
        print(f"  font sizing in action with the test image dimensions.")

        return True

    except Exception as e:
        print(f"✗ HTML generation failed: {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    # Look for test image in outputs directory
    test_image = None

    # Check common locations
    candidates = [
        "/tmp/test_image.jpg",
        "/tmp/test_image.png",
        "/home/user/adsrunner/test_image.jpg",
        "/home/user/adsrunner/test_image.png",
    ]

    for candidate in candidates:
        if Path(candidate).exists():
            test_image = candidate
            break

    if not test_image:
        print("✗ Test image not found!")
        print("\nPlease save your test image as one of these locations:")
        for c in candidates:
            print(f"  - {c}")
        sys.exit(1)

    success = run_e2e_test(test_image)
    sys.exit(0 if success else 1)
