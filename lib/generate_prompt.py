"""
Generate detailed Nano Banana image-generation prompts from creative specs.
"""

import json

from anthropic import Anthropic

GREENSCREEN_RULE = """[GREENSCREEN SYSTEM RULE — NON-NEGOTIABLE]
Any phone/device screen in this image is a compositing placeholder.
The screen fill is pure chroma key green (#00B140).
It is MATTE. It is FLAT. It has ZERO reflection, ZERO glare, ZERO specularity, ZERO gradient.
It does not interact with light in any way.
It behaves like a piece of flat colored paper — not glass, not a display.
Glare and reflections will be added in post-production.
Violating this rule makes the composite unusable."""


def generate_prompt(spec: dict, product_description: str) -> str:
    """
    Generate a single detailed Nano Banana prompt from a creative spec.
    Returns the full prompt string for image generation.
    """
    client = Anthropic()

    needs_ui = spec.get("needsUi", False)
    ui_placement = spec.get("uiPlacementType", "phone-in-hand")

    system_prompt = f"""You are a senior performance creative director specializing in high-converting Meta ads.
Your job is to generate EXTREMELY DETAILED image generation prompts that produce ads indistinguishable from real, high-budget ads currently running on Meta.

CRITICAL REQUIREMENTS:
1. DO NOT generate generic prompts.
2. DO NOT describe aesthetics loosely.
3. You must DIRECT the scene like a film director + product designer.

GOAL: Generate prompts that produce:
- Scroll-stopping Meta ads
- High realism (not AI-looking)
- Strong psychological triggers
- Believable, imperfect environments
- Product-level UI realism where applicable

STYLE RULES:
- Always specify camera type (iPhone, Sony, Canon, etc.)
- Always include imperfections (smudges, wrinkles, dust, scuffs)
- Always include real-world messiness
- Always prioritize realism over beauty
- DO NOT include any text, words, letters, or typography in the image
- DO NOT include logos, brand names, or watermarks
- Leave intentional negative space in the "{spec.get('negativeSpaceZone', 'bottom-center')}" area for text overlays added later
- The image must be photorealistic, 4:5 aspect ratio for Meta feed

{"DEVICE SCREEN RULE: " + GREENSCREEN_RULE if needs_ui else "This ad does NOT include any device screens or product UI."}
{"The device placement type is: " + str(ui_placement) + ". Ensure the device is realistically proportioned — not oversized." if needs_ui else ""}"""

    user_prompt = f"""Generate an extremely detailed image generation prompt for this ad concept:

Product: {product_description}
Creative Angle: {spec.get('angle', 'lifestyle')}
Scene Type: {spec.get('sceneType', 'lifestyle-home')}
Format: 4:5 (1080x1350 pixels)
Needs Device with UI: {needs_ui}
{f"Device Placement: {ui_placement}" if needs_ui else ""}
Negative Space Zone: {spec.get('negativeSpaceZone', 'bottom-center')} (leave clean area here for text overlay)

Follow this EXACT structure for your prompt (be EXTREMELY DETAILED for each section):

[SUBJECT]
(Detailed description of the person/people — age, ethnicity, expression, clothing, accessories, micro-expressions)

[ACTION]
(What they are doing — body language, hand positions, specific gestures, movement)

[CONTEXT]
(Environment details — room type, objects, evidence of real life, location specifics, ambient details)
(IMPORTANT: Specify that the {spec.get('negativeSpaceZone', 'bottom-center')} area of the frame should have clean negative space for text overlay)

[COMPOSITION]
(Camera type, lens, angle, distance, framing, aspect ratio 4:5)

[LIGHTING]
(Specific light sources, color temperature, shadows, contrast)

[REALISM DETAILS]
(Imperfections — smudges, wear, dust, wrinkles, scuffs, real-world messiness)

[COLOR GRADING]
(Color palette, saturation, grain, film style reference)

[STYLE]
(Overall feeling, who this targets, why it works as an ad)

{"[UI / DEVICE REALISM]" if needs_ui else ""}
{"(Device model, how it's held/placed, screen showing pure chroma key green #00B140 — MATTE, FLAT, ZERO reflection — the screen is a compositing placeholder)" if needs_ui else ""}

CRITICAL: Output ONLY the prompt text. No explanations, no markdown headers, no preamble. Just the raw, detailed prompt ready to send to an image generation model.
CRITICAL: Do NOT include ANY text, words, letters, typography, logos, or brand names in the prompt. The image must be purely visual."""

    response = client.messages.create(
        model="claude-sonnet-4-20250514",
        max_tokens=2000,
        system=system_prompt,
        messages=[{"role": "user", "content": user_prompt}],
    )

    return response.content[0].text.strip()


def generate_all_prompts(specs: list[dict], product_description: str) -> list[str]:
    """Generate prompts for all 5 creative specs."""
    prompts = []
    for spec in specs:
        prompt = generate_prompt(spec, product_description)
        prompts.append(prompt)
    return prompts
