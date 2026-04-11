#!/usr/bin/env python3
"""
AUTONOMOUS PRODUCTION VALIDATION SYSTEM
Tests all responsive sizing and rendering components without Docker.
Generates detailed validation report for production deployment.
"""

import os
from dotenv import load_dotenv

load_dotenv()

import sys
import json
from pathlib import Path
from datetime import datetime
import cv2
import numpy as np


def validate_responsive_sizing():
    """Validate responsive font sizing algorithm."""
    print("\n" + "=" * 80)
    print("VALIDATION 1: Responsive Font Sizing Algorithm")
    print("=" * 80)

    from lib.composition_engines.direct_response_stack import DirectResponseStackEngine

    test_cases = [
        ("Sale", 4, 150),
        ("Limited Time Offer", 19, 135),
        ("Check Out Our Latest Collection", 31, 112),
        ("Been scrollin for hours. Found nothing good.", 44, 90),
        ("This is a very long headline that spans multiple words and should scale down significantly", 91, 67),
    ]

    engine = DirectResponseStackEngine(
        intent={},
        layout_tokens={"headline_size_range": (100, 150)},
        text_elements={},
        image_analysis={}
    )

    all_pass = True
    print("\n{:<50} {:<10} {:<12} {:<12} {}".format(
        "Headline", "Chars", "Expected", "Actual", "Status"
    ))
    print("-" * 90)

    for text, expected_chars, expected_size in test_cases:
        actual_size = engine._calculate_responsive_font_size(text, 100, 150)
        passed = actual_size == expected_size
        all_pass = all_pass and passed

        status = "✓ PASS" if passed else "✗ FAIL"
        print("{:<50} {:<10} {:<12} {:<12} {}".format(
            text[:50], expected_chars, expected_size, actual_size, status
        ))

    return all_pass


def validate_all_engines():
    """Validate all 13 composition engines load and render."""
    print("\n" + "=" * 80)
    print("VALIDATION 2: All 13 Composition Engines")
    print("=" * 80)

    engines = [
        "direct_response_stack",
        "editorial_side_stack",
        "hero_statement",
        "hero_with_cta",
        "offer_badge_headline",
        "minimal_product_led",
        "question_hook",
        "soft_card_overlay",
        "testimonial_quote",
        "pain_point_fragments",
        "utility_explainer",
        "poster_background_headline",
        "split_message_cta",
    ]

    base_intent = {
        "typography": {
            "headline_role": "display_impact",
            "support_role": "modern_sans"
        },
        "placement": {"alignment": "center"},
        "cta_intent": {"style": "pill_filled"}
    }

    base_tokens = {
        "headline_size_range": (100, 150),
        "support_size_range": (32, 48),
        "headline_line_height": 1.0,
        "support_line_height": 1.4,
        "gap_headline_support": 12,
        "gap_support_cta": 16,
        "usable_width": 1000,
        "zone_rect": {"x": 0, "y": 0, "w": 1080, "h": 600},
        "safe_margin": 16,
        "headline_color": "#FFFFFF",
        "support_color": "#E8E8E8",
        "accent_color": "#888888",
        "cta_bg": "#6366F1",
        "cta_fg": "#FFFFFF",
    }

    text_elements = {
        "headline": {"content": "Transform Your Home Into A Productivity Powerhouse", "lines": []},
        "support_copy": {"content": "Premium design for remote work", "lines": []},
        "cta": {"content": "SHOP NOW", "lines": []},
        "eyebrow": {"content": "LIMITED TIME", "lines": []}
    }

    print("\nEngine Status:")
    print("-" * 90)

    all_pass = True
    for engine_name in engines:
        try:
            # Import engine
            module_name = engine_name.replace("_", " ").title().replace(" ", "")
            exec(f"from lib.composition_engines.{engine_name} import {module_name}Engine")

            # Get the class
            engine_class = eval(f"{module_name}Engine")

            # Instantiate
            engine = engine_class(base_intent, base_tokens, text_elements, {})

            # Try to render
            html = engine.render()

            if html and "<!DOCTYPE html>" in html and "</html>" in html:
                print(f"✓ {engine_name:<40} PASS - Renders valid HTML")
            else:
                print(f"✗ {engine_name:<40} FAIL - Invalid HTML")
                all_pass = False

        except Exception as e:
            print(f"✗ {engine_name:<40} FAIL - {str(e)[:40]}")
            all_pass = False

    return all_pass


def validate_api_and_pipeline():
    """Validate API configuration and pipeline components."""
    print("\n" + "=" * 80)
    print("VALIDATION 3: API Configuration & Pipeline Components")
    print("=" * 80)

    checks = {
        "ANTHROPIC_API_KEY": {
            "check": bool(os.getenv("ANTHROPIC_API_KEY")),
            "description": "API key configured"
        },
        "test_image.jpg": {
            "check": Path("test_image.jpg").exists(),
            "description": "Test image available"
        },
        "lib.classify_input": {
            "check": lambda: __import__("lib.classify_input", fromlist=[""]),
            "description": "Input classification module"
        },
        "lib.generate_layout_intent": {
            "check": lambda: __import__("lib.generate_layout_intent", fromlist=[""]),
            "description": "Layout intent generation module"
        },
        "lib.generate_html_from_intent": {
            "check": lambda: __import__("lib.generate_html_from_intent", fromlist=[""]),
            "description": "HTML generation from intent"
        },
        "lib.render_svg_overlay": {
            "check": lambda: __import__("lib.render_svg_overlay", fromlist=[""]),
            "description": "SVG overlay rendering module"
        },
    }

    all_pass = True
    print("\nComponent Status:")
    print("-" * 90)

    for check_name, check_info in checks.items():
        try:
            if callable(check_info["check"]):
                result = check_info["check"]()
            else:
                result = check_info["check"]

            status = "✓" if result else "✗"
            print(f"{status} {check_name:<40} {check_info['description']}")
            all_pass = all_pass and result
        except Exception as e:
            print(f"✗ {check_name:<40} ERROR: {str(e)[:40]}")
            all_pass = False

    return all_pass


def validate_font_rendering():
    """Validate font system for production rendering."""
    print("\n" + "=" * 80)
    print("VALIDATION 4: Font System & Rendering Infrastructure")
    print("=" * 80)

    try:
        from lib.composition_engines.direct_response_stack import DirectResponseStackEngine

        base_intent = {
            "typography": {"headline_role": "display_impact", "support_role": "modern_sans"},
            "placement": {"alignment": "center"},
            "cta_intent": {"style": "pill_filled"}
        }

        engine = DirectResponseStackEngine(
            base_intent,
            {"headline_size_range": (100, 150)},
            {"headline": {"content": "Test", "lines": []}, "support_copy": {"content": "", "lines": []}},
            {}
        )

        fonts = engine.get_required_fonts()
        print(f"\n✓ Font System Operational")
        print(f"  Required fonts: {fonts}")
        print(f"  Font filtering: {len(fonts)} unique fonts needed (max 3)")

        return True
    except Exception as e:
        print(f"✗ Font system error: {e}")
        return False


def generate_validation_report(results):
    """Generate comprehensive validation report."""
    print("\n" + "=" * 80)
    print("AUTONOMOUS VALIDATION REPORT")
    print("=" * 80)

    timestamp = datetime.now().isoformat()
    print(f"\nGenerated: {timestamp}")
    print(f"Environment: Python {sys.version.split()[0]}")

    passed = sum(1 for v in results.values() if v)
    total = len(results)

    print(f"\nValidation Results: {passed}/{total} PASSED")
    print("-" * 80)

    for check_name, result in results.items():
        status = "✓ PASS" if result else "✗ FAIL"
        print(f"{status} {check_name}")

    print("\n" + "=" * 80)

    if all(results.values()):
        print("✓✓✓ ALL VALIDATIONS PASSED ✓✓✓")
        print("\nProduction readiness: 100%")
        print("Ready for Docker deployment with real browser rendering")
        return 0
    else:
        print("✗ Some validations failed")
        print("Please review errors above")
        return 1


def main():
    """Run all validations."""
    print("\n")
    print("╔════════════════════════════════════════════════════════════╗")
    print("║  AUTONOMOUS PRODUCTION VALIDATION SYSTEM                    ║")
    print("║  Tests responsive sizing and pipeline without Docker       ║")
    print("╚════════════════════════════════════════════════════════════╝")

    results = {
        "Responsive Font Sizing": validate_responsive_sizing(),
        "All 13 Composition Engines": validate_all_engines(),
        "API & Pipeline Components": validate_api_and_pipeline(),
        "Font System & Rendering": validate_font_rendering(),
    }

    exit_code = generate_validation_report(results)
    sys.exit(exit_code)


if __name__ == "__main__":
    main()
