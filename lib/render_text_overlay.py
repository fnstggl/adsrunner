"""
Deterministic text overlay renderer for ad images.
Renders headline, subheadline, and CTA after image generation.
All sizing is proportional to canvas height for mobile-ready Meta ad scale.
"""

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

# Font paths (Liberation Sans is available on most Linux systems)
FONT_PATHS = [
    "/usr/share/fonts/truetype/liberation/LiberationSans-Bold.ttf",
    "/usr/share/fonts/truetype/liberation/LiberationSans-Regular.ttf",
    "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
    "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
    "/usr/share/fonts/truetype/freefont/FreeSansBold.ttf",
]


def _find_font(bold: bool = True) -> str:
    """Find available font, preferring bold for headlines."""
    import os
    preferred = [p for p in FONT_PATHS if ("Bold" in p) == bold]
    fallback = [p for p in FONT_PATHS if ("Bold" in p) != bold]
    for path in preferred + fallback:
        if os.path.exists(path):
            return path
    return None


def _wrap_text(text: str, font: ImageFont.FreeTypeFont, max_width: int) -> list[str]:
    """Word-wrap text to fit within max_width pixels."""
    words = text.split()
    lines = []
    current_line = ""
    for word in words:
        test_line = f"{current_line} {word}".strip()
        bbox = font.getbbox(test_line)
        if bbox[2] - bbox[0] <= max_width:
            current_line = test_line
        else:
            if current_line:
                lines.append(current_line)
            current_line = word
    if current_line:
        lines.append(current_line)
    return lines or [text]


def _draw_text_with_shadow(draw, pos, text, font, text_color, shadow_offsets=((2, 4), (3, 6), (0, 6))):
    """Draw text with multi-pass shadow for depth."""
    x, y = pos
    shadow_color = (0, 0, 0, 140)
    for ox, oy in shadow_offsets:
        draw.text((x + ox, y + oy), text, font=font, fill=shadow_color)
    draw.text((x, y), text, font=font, fill=text_color)


def render_text_overlay(
    image: np.ndarray,
    headline: str,
    subheadline: str = "",
    cta: str = "",
    zone: str = "bottom-center",
    template: str = "light-on-dark",
    logo: np.ndarray = None,
) -> np.ndarray:
    """
    Render deterministic text overlay on an ad image.

    Args:
        image: BGR numpy array
        headline: Main headline text (rendered ALL CAPS)
        subheadline: Supporting text
        cta: Call-to-action text
        zone: Placement zone (top-left, top-right, bottom-left, bottom-right, top-center, bottom-center)
        template: Text style template (dark-on-light, light-on-dark, card-overlay, gradient-overlay)
        logo: Optional BGR logo image to include

    Returns:
        BGR numpy array with text overlay
    """
    h, w = image.shape[:2]

    # ── Font sizes: scale by HEIGHT with min/max clamps ──────────────────────
    headline_size    = max(64,  min(180, int(h * 0.08)))
    subheadline_size = max(36,  min(90,  int(h * 0.045)))
    cta_size         = max(28,  min(70,  int(h * 0.035)))

    # ── Layout constants ──────────────────────────────────────────────────────
    margin           = int(w * 0.06)          # 6% side margin
    max_text_width   = int(w * 0.85)
    cta_pad_x        = int(w * 0.06)          # horizontal pill padding
    cta_pad_y        = int(h * 0.015)         # vertical pill padding

    # Convert BGR to RGB for PIL
    rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    pil_img = Image.fromarray(rgb)
    overlay = Image.new("RGBA", pil_img.size, (0, 0, 0, 0))
    draw = ImageDraw.Draw(overlay)

    bold_font_path    = _find_font(bold=True)
    regular_font_path = _find_font(bold=False)

    if bold_font_path:
        headline_font = ImageFont.truetype(bold_font_path, headline_size)
        cta_font      = ImageFont.truetype(bold_font_path, cta_size)
    else:
        headline_font = ImageFont.load_default()
        cta_font      = ImageFont.load_default()

    if regular_font_path:
        subheadline_font = ImageFont.truetype(regular_font_path, subheadline_size)
    else:
        subheadline_font = ImageFont.load_default()

    # ── Color scheme ──────────────────────────────────────────────────────────
    if template in ("light-on-dark", "gradient-overlay"):
        text_color    = (255, 255, 255, 255)
        sub_color     = (235, 235, 235, 230)
        cta_bg        = (10,  10,  10,  220)
        cta_text      = (255, 255, 255, 255)
    else:  # dark-on-light, card-overlay
        text_color    = (15,  15,  15,  255)
        sub_color     = (30,  30,  30,  230)
        cta_bg        = (15,  15,  15,  220)
        cta_text      = (255, 255, 255, 255)

    # ── Headline: uppercase for scroll-stopping impact ────────────────────────
    headline_upper = headline.upper()

    # ── Wrap text ─────────────────────────────────────────────────────────────
    headline_lines = _wrap_text(headline_upper, headline_font, max_text_width)
    sub_lines      = _wrap_text(subheadline, subheadline_font, max_text_width) if subheadline else []

    # ── Measure headline block height ─────────────────────────────────────────
    line_gap      = int(headline_size * 0.12)   # tight 1.12 effective line height
    sub_gap       = int(h * 0.02)               # gap between headline and sub
    sub_line_gap  = int(subheadline_size * 0.2)

    headline_block_h = sum(
        headline_font.getbbox(ln)[3] - headline_font.getbbox(ln)[1] + line_gap
        for ln in headline_lines
    )

    # ── Text alignment ────────────────────────────────────────────────────────
    if "left" in zone:
        x_anchor   = margin
        text_align = "left"
    elif "right" in zone:
        x_anchor   = w - margin
        text_align = "right"
    else:
        x_anchor   = w // 2
        text_align = "center"

    # ── Y positions ───────────────────────────────────────────────────────────
    headline_y = int(h * 0.08) if "top" in zone else int(h * 0.08)
    # For bottom zones, anchor CTA at 88% and work upward
    if "bottom" in zone:
        cta_bbox       = cta_font.getbbox(cta) if cta else (0, 0, 0, 0)
        cta_h_px       = cta_bbox[3] - cta_bbox[1]
        cta_btn_height = cta_h_px + cta_pad_y * 2
        cta_y          = int(h * 0.88)

        # Work upward: sub → headline
        sub_block_h = sum(
            subheadline_font.getbbox(ln)[3] - subheadline_font.getbbox(ln)[1] + sub_line_gap
            for ln in sub_lines
        ) if sub_lines else 0

        if cta:
            sub_bottom = cta_y - int(h * 0.025)
        else:
            sub_bottom = int(h * 0.88)

        sub_y       = sub_bottom - sub_block_h
        headline_y  = sub_y - sub_gap - headline_block_h if sub_lines else sub_bottom - sub_gap - headline_block_h
        headline_y  = max(int(h * 0.05), headline_y)
    else:
        # Top zone: headline at top, sub below, CTA further below
        headline_y = int(h * 0.08)
        sub_y      = headline_y + headline_block_h + sub_gap
        sub_block_h = sum(
            subheadline_font.getbbox(ln)[3] - subheadline_font.getbbox(ln)[1] + sub_line_gap
            for ln in sub_lines
        ) if sub_lines else 0
        cta_y = sub_y + sub_block_h + int(h * 0.025) if sub_lines else headline_y + headline_block_h + int(h * 0.025)

    # ── Draw gradient background if needed ───────────────────────────────────
    if template == "gradient-overlay":
        gradient = Image.new("RGBA", pil_img.size, (0, 0, 0, 0))
        grad_draw = ImageDraw.Draw(gradient)
        band = int(h * 0.50)
        if "bottom" in zone:
            for i in range(band):
                alpha = int(190 * (i / band))
                y_pos = h - band + i
                grad_draw.line([(0, y_pos), (w, y_pos)], fill=(0, 0, 0, alpha))
        else:
            for i in range(band):
                alpha = int(190 * (1 - i / band))
                grad_draw.line([(0, i), (w, i)], fill=(0, 0, 0, alpha))
        overlay = Image.alpha_composite(overlay, gradient)
        draw = ImageDraw.Draw(overlay)
    elif template == "card-overlay":
        card_pad = int(margin * 0.7)
        if text_align == "center":
            card_x1 = max(0, x_anchor - max_text_width // 2 - card_pad)
        else:
            card_x1 = max(0, x_anchor - card_pad)
        card_y1 = max(0, headline_y - card_pad)
        card_x2 = min(w, card_x1 + max_text_width + card_pad * 2)
        card_y2 = min(h, (cta_y + 60) + card_pad)
        draw.rounded_rectangle([card_x1, card_y1, card_x2, card_y2], radius=16, fill=(255, 255, 255, 200))

    # ── Draw logo ─────────────────────────────────────────────────────────────
    if logo is not None:
        logo_rgb = cv2.cvtColor(logo, cv2.COLOR_BGR2RGB)
        logo_pil = Image.fromarray(logo_rgb)
        logo_max_h = int(h * 0.06)
        logo_max_w = int(w * 0.25)
        logo_w_px, logo_h_px = logo_pil.size
        scale  = min(logo_max_w / logo_w_px, logo_max_h / logo_h_px)
        new_w  = int(logo_w_px * scale)
        new_h  = int(logo_h_px * scale)
        logo_pil = logo_pil.resize((new_w, new_h), Image.LANCZOS)
        if logo_pil.mode != "RGBA":
            logo_pil = logo_pil.convert("RGBA")
        logo_x = (w - new_w) // 2 if text_align == "center" else (x_anchor - new_w if text_align == "right" else x_anchor)
        logo_y = max(0, headline_y - new_h - int(h * 0.02))
        overlay.paste(logo_pil, (logo_x, logo_y), logo_pil)
        draw = ImageDraw.Draw(overlay)

    # ── Draw headline ─────────────────────────────────────────────────────────
    y_cursor = headline_y
    for line in headline_lines:
        bbox   = headline_font.getbbox(line)
        line_w = bbox[2] - bbox[0]
        line_h = bbox[3] - bbox[1]

        if text_align == "center":
            x_pos = x_anchor - line_w // 2
        elif text_align == "right":
            x_pos = x_anchor - line_w
        else:
            x_pos = x_anchor

        _draw_text_with_shadow(draw, (x_pos, y_cursor), line, headline_font, text_color)
        y_cursor += line_h + line_gap

    # ── Draw subheadline ──────────────────────────────────────────────────────
    if sub_lines:
        y_cursor = sub_y if "bottom" in zone else y_cursor + sub_gap
        for line in sub_lines:
            bbox   = subheadline_font.getbbox(line)
            line_w = bbox[2] - bbox[0]
            line_h = bbox[3] - bbox[1]

            if text_align == "center":
                x_pos = x_anchor - line_w // 2
            elif text_align == "right":
                x_pos = x_anchor - line_w
            else:
                x_pos = x_anchor

            # Lighter shadow for subheadline
            draw.text((x_pos + 1, y_cursor + 2), line, font=subheadline_font, fill=(0, 0, 0, 100))
            draw.text((x_pos, y_cursor), line, font=subheadline_font, fill=sub_color)
            y_cursor += line_h + sub_line_gap

    # ── Draw CTA pill button ──────────────────────────────────────────────────
    if cta:
        cta_upper = cta.upper()
        bbox      = cta_font.getbbox(cta_upper)
        cta_w_px  = bbox[2] - bbox[0]
        cta_h_px  = bbox[3] - bbox[1]

        btn_w = cta_w_px + cta_pad_x * 2
        btn_h = cta_h_px + cta_pad_y * 2
        pill_r = btn_h // 2  # full pill

        if text_align == "center":
            btn_x = x_anchor - btn_w // 2
        elif text_align == "right":
            btn_x = x_anchor - btn_w
        else:
            btn_x = x_anchor

        btn_y = cta_y

        # Button shadow
        draw.rounded_rectangle(
            [btn_x + 3, btn_y + 6, btn_x + btn_w + 3, btn_y + btn_h + 6],
            radius=pill_r,
            fill=(0, 0, 0, 90),
        )
        # Button background
        draw.rounded_rectangle(
            [btn_x, btn_y, btn_x + btn_w, btn_y + btn_h],
            radius=pill_r,
            fill=cta_bg,
        )
        # Button text
        draw.text(
            (btn_x + cta_pad_x, btn_y + cta_pad_y),
            cta_upper,
            font=cta_font,
            fill=cta_text,
        )

    # ── Composite and return ──────────────────────────────────────────────────
    pil_img = pil_img.convert("RGBA")
    result  = Image.alpha_composite(pil_img, overlay)
    result  = result.convert("RGB")
    return cv2.cvtColor(np.array(result), cv2.COLOR_RGB2BGR)
