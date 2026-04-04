"""
Generate 1 creative spec for ad generation based on classified input.
"""

import json

from anthropic import Anthropic


def build_creative_specs(
    product_description: str,
    classification: dict,
    ad_goal: str = "",
    has_logo: bool = False,
    has_product_images: bool = False,
    has_ui_screenshots: bool = False,
) -> list[dict]:
    """
    Build 1 CreativeSpec object for ad generation.
    """
    client = Anthropic()

    prompt = f"""You are a senior performance creative director for Meta ads.

Generate exactly 1 creative specification for an ad image.

Product: {product_description}
Product Type: {classification['product_type']}
Suggested Styles: {json.dumps(classification['likely_ad_styles'])}
Needs UI: {classification['needs_ui']}
Ad Goal: {ad_goal or "General awareness / conversion"}
Available Assets: logo={has_logo}, product_images={has_product_images}, ui_screenshots={has_ui_screenshots}

Return a JSON array of exactly 1 object with these fields:
- id: string ("ad_1")
- angle: string - the creative hook/angle (e.g., "pain-point", "aspirational", "social-proof", "comparison", "feature-spotlight")
- sceneType: string - scene description type (e.g., "lifestyle-home", "office-desk", "outdoor-casual", "minimal-studio", "flat-lay")
- format: string - always "4:5" for Meta feed
- needsUi: boolean - whether this ad shows product UI in a device screen
- uiPlacementType: string or null - if needsUi, one of "phone-in-hand", "laptop-on-desk", "tablet-on-table", "phone-on-surface", "floating-device"; null if not needed
- headline: string - short, punchy headline (max 8 words, no quotes)
- subheadline: string - supporting text (max 15 words)
- cta: string - call to action (e.g., "Try Free", "Get Started", "Learn More", "Download Now")
- negativeSpaceZone: string - where to place text overlay, one of "top-left", "top-right", "bottom-left", "bottom-right", "top-center", "bottom-center"
- assetsToUse: array of strings - which uploaded assets to incorporate (from: "logo", "product_image", "ui_screenshot")
- textTemplate: string - one of "dark-on-light", "light-on-dark", "card-overlay", "gradient-overlay"

IMPORTANT RULES:
- If needs_ui is true, set needsUi=true
- If needs_ui is false, set needsUi=false
- The headline must be compelling, specific to the product, and NOT generic
- Do NOT use quotation marks in headlines or subheadlines

Return ONLY the JSON array. No markdown fences."""

    response = client.messages.create(
        model="claude-sonnet-4-20250514",
        max_tokens=1000,
        messages=[{"role": "user", "content": prompt}],
    )

    text = response.content[0].text.strip()
    if text.startswith("```"):
        text = text.split("\n", 1)[1]
        text = text.rsplit("```", 1)[0].strip()

    specs = json.loads(text)
    assert isinstance(specs, list) and len(specs) == 1, "Expected exactly 1 spec"
    return specs
