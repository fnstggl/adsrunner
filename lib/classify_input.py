"""
Lightweight LLM classifier to determine product type and whether UI is needed.
"""

import json
import os

from anthropic import Anthropic


def classify_input(product_description: str, ad_goal: str = "") -> dict:
    """
    Classify product input to determine ad requirements.

    Returns:
        {
            "product_type": "saas" | "app" | "ecommerce" | "service" | "marketplace" | "other",
            "likely_ad_styles": ["lifestyle", ...],
            "needs_ui": bool,
            "reason": str
        }
    """
    client = Anthropic()

    prompt = f"""You are a product classifier for an ad generation system.

Analyze this product/company description and determine the ad requirements.

Product Description: {product_description}
Ad Goal/Style: {ad_goal or "Not specified"}

Return a JSON object with exactly these fields:
- product_type: one of "saas", "app", "ecommerce", "service", "marketplace", "other"
- likely_ad_styles: array of 3-5 ad style strings (e.g., "lifestyle", "product-focused", "testimonial", "comparison", "feature-highlight", "minimal", "social-proof", "before-after")
- needs_ui: boolean
- reason: string explaining the needs_ui decision

needs_ui rules:
- TRUE when the product is meaningfully experienced through a product interface, app screen, dashboard, workflow, map, listing feed, or software UI
- TRUE for SaaS tools, mobile apps, web platforms, marketplaces with digital interfaces
- FALSE for purely physical products unless the ad explicitly includes app/device usage
- FALSE for services without a meaningful digital interface (restaurants, cleaning services, etc.)

Return ONLY valid JSON. No markdown fences, no explanation outside the JSON."""

    response = client.messages.create(
        model="claude-sonnet-4-20250514",
        max_tokens=500,
        messages=[{"role": "user", "content": prompt}],
    )

    text = response.content[0].text.strip()
    if text.startswith("```"):
        text = text.split("\n", 1)[1]
        text = text.rsplit("```", 1)[0].strip()

    return json.loads(text)
