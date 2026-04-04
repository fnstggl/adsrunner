"""
Deterministic text overlay renderer for ad images.
Renders headline, subheadline, and CTA after image generation.
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
        headline: Main headline text
        subheadline: Supporting text
        cta: Call-to-action text
        zone: Placement zone (top-left, top-right, bottom-left, bottom-right, top-center, bottom-center)
        template: Text style template (dark-on-light, light-on-dark, card-overlay, gradient-overlay)
        logo: Optional BGR logo image to include

    Returns:
        BGR numpy array with text overlay
    """
    h, w = image.shape[:2]

    # Convert BGR to RGB for PIL
    rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    pil_img = Image.fromarray(rgb)
    overlay = Image.new("RGBA", pil_img.size, (0, 0, 0, 0))
    draw = ImageDraw.Draw(overlay)

    # Font sizes relative to image width
    headline_size = max(28, int(w * 0.055))
    subheadline_size = max(18, int(w * 0.032))
    cta_size = max(16, int(w * 0.028))

    bold_font_path = _find_font(bold=True)
    regular_font_path = _find_font(bold=False)

    if bold_font_path:
        headline_font = ImageFont.truetype(bold_font_path, headline_size)
        cta_font = ImageFont.truetype(bold_font_path, cta_size)
    else:
        headline_font = ImageFont.load_default()
        cta_font = ImageFont.load_default()

    if regular_font_path:
        subheadline_font = ImageFont.truetype(regular_font_path, subheadline_size)
    else:
        subheadline_font = ImageFont.load_default()

    # Color scheme based on template
    if template in ("light-on-dark", "gradient-overlay"):
        text_color = (255, 255, 255, 255)
        shadow_color = (0, 0, 0, 160)
        cta_bg = (255, 255, 255, 230)
        cta_text = (0, 0, 0, 255)
    else:  # dark-on-light, card-overlay
        text_color = (20, 20, 20, 255)
        shadow_color = (255, 255, 255, 160)
        cta_bg = (20, 20, 20, 230)
        cta_text = (255, 255, 255, 255)

    # Calculate text zone position
    padding = int(w * 0.06)
    max_text_width = int(w * 0.85)

    # Wrap text
    headline_lines = _wrap_text(headline, headline_font, max_text_width)
    sub_lines = _wrap_text(subheadline, subheadline_font, max_text_width) if subheadline else []

    # Calculate total text block height
    line_spacing = int(headline_size * 0.3)
    sub_spacing = int(subheadline_size * 0.3)

    total_height = 0
    for line in headline_lines:
        bbox = headline_font.getbbox(line)
        total_height += (bbox[3] - bbox[1]) + line_spacing
    if sub_lines:
        total_height += int(headline_size * 0.4)  # gap between headline and sub
        for line in sub_lines:
            bbox = subheadline_font.getbbox(line)
            total_height += (bbox[3] - bbox[1]) + sub_spacing
    if cta:
        total_height += int(headline_size * 0.6)  # gap before CTA
        bbox = cta_font.getbbox(cta)
        total_height += (bbox[3] - bbox[1]) + int(cta_size * 1.5)

    # Determine anchor position based on zone
    if "top" in zone:
        y_start = padding
    elif "bottom" in zone:
        y_start = h - total_height - padding * 2
    else:
        y_start = (h - total_height) // 2

    if "left" in zone:
        x_anchor = padding
        text_align = "left"
    elif "right" in zone:
        x_anchor = w - padding
        text_align = "right"
    else:
        x_anchor = w // 2
        text_align = "center"

    # Draw gradient/card background if needed
    if template == "gradient-overlay":
        gradient = Image.new("RGBA", pil_img.size, (0, 0, 0, 0))
        grad_draw = ImageDraw.Draw(gradient)
        if "bottom" in zone:
            for i in range(int(h * 0.45)):
                alpha = int(180 * (i / (h * 0.45)))
                y_pos = h - int(h * 0.45) + i
                grad_draw.line([(0, y_pos), (w, y_pos)], fill=(0, 0, 0, alpha))
        else:
            for i in range(int(h * 0.45)):
                alpha = int(180 * (1 - i / (h * 0.45)))
                grad_draw.line([(0, i), (w, i)], fill=(0, 0, 0, alpha))
        overlay = Image.alpha_composite(overlay, gradient)
        draw = ImageDraw.Draw(overlay)
    elif template == "card-overlay":
        card_padding = int(padding * 0.7)
        card_x1 = max(0, x_anchor - max_text_width // 2 - card_padding) if text_align == "center" else max(0, x_anchor - card_padding)
        card_y1 = max(0, y_start - card_padding)
        card_x2 = min(w, card_x1 + max_text_width + card_padding * 2)
        card_y2 = min(h, y_start + total_height + card_padding * 2)
        draw.rounded_rectangle(
            [card_x1, card_y1, card_x2, card_y2],
            radius=12,
            fill=(255, 255, 255, 200),
        )

    # Draw logo if provided
    if logo is not None:
        logo_rgb = cv2.cvtColor(logo, cv2.COLOR_BGR2RGB)
        logo_pil = Image.fromarray(logo_rgb)
        logo_max_h = int(h * 0.06)
        logo_max_w = int(w * 0.25)
        logo_w, logo_h = logo_pil.size
        scale = min(logo_max_w / logo_w, logo_max_h / logo_h)
        new_w, new_h = int(logo_w * scale), int(logo_h * scale)
        logo_pil = logo_pil.resize((new_w, new_h), Image.LANCZOS)
        if text_align == "center":
            logo_x = (w - new_w) // 2
        elif text_align == "right":
            logo_x = x_anchor - new_w
        else:
            logo_x = x_anchor
        logo_y = y_start
        # Convert to RGBA if needed
        if logo_pil.mode != "RGBA":
            logo_pil = logo_pil.convert("RGBA")
        overlay.paste(logo_pil, (logo_x, logo_y), logo_pil)
        draw = ImageDraw.Draw(overlay)
        y_start += new_h + int(padding * 0.4)

    # Draw headline
    y_cursor = y_start
    for line in headline_lines:
        bbox = headline_font.getbbox(line)
        line_w = bbox[2] - bbox[0]
        line_h = bbox[3] - bbox[1]

        if text_align == "center":
            x_pos = x_anchor - line_w // 2
        elif text_align == "right":
            x_pos = x_anchor - line_w
        else:
            x_pos = x_anchor

        # Shadow
        draw.text((x_pos + 2, y_cursor + 2), line, font=headline_font, fill=shadow_color)
        draw.text((x_pos, y_cursor), line, font=headline_font, fill=text_color)
        y_cursor += line_h + line_spacing

    # Draw subheadline
    if sub_lines:
        y_cursor += int(headline_size * 0.4)
        for line in sub_lines:
            bbox = subheadline_font.getbbox(line)
            line_w = bbox[2] - bbox[0]
            line_h = bbox[3] - bbox[1]

            if text_align == "center":
                x_pos = x_anchor - line_w // 2
            elif text_align == "right":
                x_pos = x_anchor - line_w
            else:
                x_pos = x_anchor

            draw.text((x_pos + 1, y_cursor + 1), line, font=subheadline_font, fill=shadow_color)
            draw.text((x_pos, y_cursor), line, font=subheadline_font, fill=text_color)
            y_cursor += line_h + sub_spacing

    # Draw CTA button
    if cta:
        y_cursor += int(headline_size * 0.6)
        bbox = cta_font.getbbox(cta)
        cta_w = bbox[2] - bbox[0]
        cta_h = bbox[3] - bbox[1]
        btn_pad_x = int(cta_size * 1.2)
        btn_pad_y = int(cta_size * 0.6)

        if text_align == "center":
            btn_x = x_anchor - (cta_w + btn_pad_x * 2) // 2
        elif text_align == "right":
            btn_x = x_anchor - cta_w - btn_pad_x * 2
        else:
            btn_x = x_anchor

        draw.rounded_rectangle(
            [btn_x, y_cursor, btn_x + cta_w + btn_pad_x * 2, y_cursor + cta_h + btn_pad_y * 2],
            radius=8,
            fill=cta_bg,
        )
        draw.text(
            (btn_x + btn_pad_x, y_cursor + btn_pad_y),
            cta,
            font=cta_font,
            fill=cta_text,
        )

    # Composite overlay onto image
    pil_img = pil_img.convert("RGBA")
    result = Image.alpha_composite(pil_img, overlay)
    result = result.convert("RGB")

    # Convert back to BGR for OpenCV
    return cv2.cvtColor(np.array(result), cv2.COLOR_RGB2BGR)
