"""Composition engines for deterministic HTML/CSS rendering."""

# Typography roles mapping to font families
TYPOGRAPHY_ROLES = {
    "display_impact": {
        "fonts": ["Bebas Neue", "Anton", "Oswald", "Montserrat", "Inter"],
        "weights": [400, 700, 900],
    },
    "modern_sans": {
        "fonts": ["Inter", "Poppins", "Space Grotesk", "Montserrat"],
        "weights": [400, 600, 700],
    },
    "editorial_serif": {
        "fonts": ["Playfair Display", "DM Serif Display", "Libre Baskerville", "Cormorant Garamond"],
        "weights": [400, 700],
        "style": "italic",
    },
    "warm_serif": {
        "fonts": ["Lora", "Libre Baskerville"],
        "weights": [400, 700],
        "style": "italic",
    },
    "handwritten_accent": {
        "fonts": ["Caveat"],
        "weights": [400, 700],
    },
}

# Register composition engines
def get_engine_class(family_name: str):
    """Get the composition engine class for a given family.

    Args:
        family_name: e.g., "hero_statement", "direct_response_stack"

    Returns:
        CompositionEngine subclass
    """
    engines = {
        "hero_statement": lambda: __import__(
            "lib.composition_engines.hero_statement", fromlist=["HeroStatementEngine"]
        ).HeroStatementEngine,
        "hero_with_cta": lambda: __import__(
            "lib.composition_engines.hero_with_cta", fromlist=["HeroWithCtaEngine"]
        ).HeroWithCtaEngine,
        "editorial_side_stack": lambda: __import__(
            "lib.composition_engines.editorial_side_stack", fromlist=["EditorialSideStackEngine"]
        ).EditorialSideStackEngine,
        "direct_response_stack": lambda: __import__(
            "lib.composition_engines.direct_response_stack", fromlist=["DirectResponseStackEngine"]
        ).DirectResponseStackEngine,
        "pain_point_fragments": lambda: __import__(
            "lib.composition_engines.pain_point_fragments", fromlist=["PainPointFragmentsEngine"]
        ).PainPointFragmentsEngine,
        "question_hook": lambda: __import__(
            "lib.composition_engines.question_hook", fromlist=["QuestionHookEngine"]
        ).QuestionHookEngine,
        "testimonial_quote": lambda: __import__(
            "lib.composition_engines.testimonial_quote", fromlist=["TestimonialQuoteEngine"]
        ).TestimonialQuoteEngine,
        "offer_badge_headline": lambda: __import__(
            "lib.composition_engines.offer_badge_headline", fromlist=["OfferBadgeHeadlineEngine"]
        ).OfferBadgeHeadlineEngine,
        "poster_background_headline": lambda: __import__(
            "lib.composition_engines.poster_background_headline",
            fromlist=["PosterBackgroundHeadlineEngine"],
        ).PosterBackgroundHeadlineEngine,
        "soft_card_overlay": lambda: __import__(
            "lib.composition_engines.soft_card_overlay", fromlist=["SoftCardOverlayEngine"]
        ).SoftCardOverlayEngine,
        "split_message_cta": lambda: __import__(
            "lib.composition_engines.split_message_cta", fromlist=["SplitMessageCtaEngine"]
        ).SplitMessageCtaEngine,
        "minimal_product_led": lambda: __import__(
            "lib.composition_engines.minimal_product_led", fromlist=["MinimalProductLedEngine"]
        ).MinimalProductLedEngine,
        "utility_explainer": lambda: __import__(
            "lib.composition_engines.utility_explainer", fromlist=["UtilityExplainerEngine"]
        ).UtilityExplainerEngine,
    }

    if family_name not in engines:
        raise ValueError(f"Unknown composition family: {family_name}")

    return engines[family_name]()

__all__ = ["TYPOGRAPHY_ROLES", "get_engine_class"]
