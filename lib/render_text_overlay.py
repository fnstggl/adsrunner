"""
Deterministic text overlay renderer for ad images.

Implements a layout engine with:
- canvas-relative safe zones
- measured text boxes
- font shrinking until text fits its allocated box
- collision prevention between headline/sub stack and CTA
- independently anchored CTA pill button
"""

import os
import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

FONT_PATHS_BOLD = [
    "/usr/share/fonts/truetype/liberation/LiberationSans-Bold.ttf",
    "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
    "/usr/share/fonts/truetype/freefont/FreeSansBold.ttf",
]
FONT_PATHS_REGULAR = [
    "/usr/share/fonts/truetype/liberation/LiberationSans-Regular.ttf",
    "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
    "/usr/share/fonts/truetype/freefont/FreeSans.ttf",
]


# ─── Font helpers ─────────────────────────────────────────────────────────────

def _find_font(bold: bool = True) -> str | None:
    paths = FONT_PATHS_BOLD if bold else FONT_PATHS_REGULAR
    for p in paths:
        if os.path.exists(p):
            return p
    # last resort: any available font
    for p in FONT_PATHS_BOLD + FONT_PATHS_REGULAR:
        if os.path.exists(p):
            return p
    return None


def _load_font(path: str | None, size: int) -> ImageFont.FreeTypeFont:
    if path:
        return ImageFont.truetype(path, size)
    return ImageFont.load_default()


# ─── Layout helpers ────────────────────────────────────────────────────────────

def wrap_text_to_width(
    text: str,
    font: ImageFont.FreeTypeFont,
    max_width: int,
) -> list[str]:
    """Word-wrap text using actual font metrics. Never estimates from char count."""
    words = text.split()
    lines: list[str] = []
    current = ""
    for word in words:
        candidate = (current + " " + word).strip()
        w = font.getbbox(candidate)[2] - font.getbbox(candidate)[0]
        if w <= max_width:
            current = candidate
        else:
            if current:
                lines.append(current)
            current = word
    if current:
        lines.append(current)
    return lines or [text]


def measure_text_block(
    lines: list[str],
    font: ImageFont.FreeTypeFont,
    line_height_ratio: float,
) -> tuple[int, int]:
    """
    Return (max_line_width, total_block_height) for a list of wrapped lines.
    Uses actual glyph bounding boxes — no estimates.
    """
    if not lines:
        return 0, 0
    max_w = 0
    total_h = 0
    for i, line in enumerate(lines):
        bbox = font.getbbox(line)
        lw = bbox[2] - bbox[0]
        lh = bbox[3] - bbox[1]
        max_w = max(max_w, lw)
        if i < len(lines) - 1:
            total_h += int(lh * line_height_ratio)
        else:
            total_h += lh   # last line: no extra leading
    return max_w, total_h


def fit_text_block(
    text: str,
    font_path: str | None,
    start_size: int,
    min_size: int,
    max_size: int,
    max_width: int,
    max_height: int,
    max_lines: int,
    line_height_ratio: float,
) -> tuple[int, ImageFont.FreeTypeFont, list[str], int, int]:
    """
    Shrink font size until the wrapped text block fits within (max_width, max_height).

    Returns: (final_size, font, lines, block_width, block_height)
    """
    size = max(min_size, min(max_size, start_size))
    while True:
        font = _load_font(font_path, size)
        lines = wrap_text_to_width(text, font, max_width)
        lines = lines[:max_lines]  # hard cap on line count
        bw, bh = measure_text_block(lines, font, line_height_ratio)
        if bh <= max_height or size <= min_size:
            return size, font, lines, bw, bh
        # Shrink by ~5% per iteration
        size = max(min_size, int(size * 0.95))


# ─── Drawing helpers ───────────────────────────────────────────────────────────

def _shadow(draw, x, y, text, font):
    """Multi-pass drop shadow for headline readability."""
    sc = (0, 0, 0, 145)
    for ox, oy in ((2, 3), (3, 5), (1, 6)):
        draw.text((x + ox, y + oy), text, font=font, fill=sc)


def _x_for_align(x_anchor: int, line_w: int, align: str) -> int:
    if align == "center":
        return x_anchor - line_w // 2
    if align == "right":
        return x_anchor - line_w
    return x_anchor  # left


# ─── Main renderer ─────────────────────────────────────────────────────────────

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
    Render headline, subheadline, and CTA onto an ad image.

    All sizing and positioning is derived from canvas dimensions.
    Font sizes shrink to fit allocated boxes. Collision prevention
    prevents any overlap between the headline/sub stack and the CTA.

    Args:
        image:      BGR numpy array
        headline:   Main headline text (auto-uppercased)
        subheadline: Supporting line
        cta:        Call-to-action label
        zone:       One of top-left/top-center/top-right/bottom-left/bottom-center/bottom-right
        template:   light-on-dark | dark-on-light | card-overlay | gradient-overlay
        logo:       Optional BGR logo array

    Returns:
        BGR numpy array with overlay composited
    """
    h, w = image.shape[:2]

    # ══ Safe-area constants (all canvas-relative) ══════════════════════════════
    h_margin        = int(w  * 0.08)    # horizontal margin both sides
    hl_box_top      = int(h  * 0.06)    # headline box top edge
    hl_box_height   = int(h  * 0.18)    # max height for headline block
    sub_gap         = int(h  * 0.02)    # gap between headline bottom and sub top
    sub_box_height  = int(h  * 0.10)    # max height for subheadline block
    cta_center_y    = int(h  * 0.88)    # CTA button vertical center
    cta_safe_buffer = int(h  * 0.03)    # clearance above CTA for upper stack

    hl_max_w  = int(w * 0.84)
    sub_max_w = int(w * 0.80)

    cta_pad_x = int(w * 0.06)
    cta_pad_y = int(h * 0.015)

    # ══ Font paths ═════════════════════════════════════════════════════════════
    bold_path    = _find_font(bold=True)
    regular_path = _find_font(bold=False)

    # ══ Step 1: Fit headline into its box ══════════════════════════════════════
    hl_text = (headline or "").upper()
    hl_start = max(64, min(180, int(h * 0.08)))

    hl_size, hl_font, hl_lines, hl_bw, hl_bh = fit_text_block(
        text=hl_text,
        font_path=bold_path,
        start_size=hl_start,
        min_size=64,
        max_size=180,
        max_width=hl_max_w,
        max_height=hl_box_height,
        max_lines=4,
        line_height_ratio=1.06,
    )

    # ══ Step 2: Fit subheadline into its box ══════════════════════════════════
    sub_text  = subheadline or ""
    sub_start = max(32, min(90, int(h * 0.045)))

    if sub_text:
        sub_size, sub_font, sub_lines, sub_bw, sub_bh = fit_text_block(
            text=sub_text,
            font_path=regular_path,
            start_size=sub_start,
            min_size=32,
            max_size=90,
            max_width=sub_max_w,
            max_height=sub_box_height,
            max_lines=3,
            line_height_ratio=1.20,
        )
    else:
        sub_size  = sub_start
        sub_font  = _load_font(regular_path, sub_start)
        sub_lines = []
        sub_bh    = 0

    # ══ Step 3: Compute CTA geometry (independently anchored) ═════════════════
    cta_size  = max(28, min(70, int(h * 0.035)))
    cta_font  = _load_font(bold_path, cta_size)
    cta_upper = (cta or "").upper()

    if cta_upper:
        cb = cta_font.getbbox(cta_upper)
        cta_txt_w = cb[2] - cb[0]
        cta_txt_h = cb[3] - cb[1]
        cta_btn_w = cta_txt_w + cta_pad_x * 2
        cta_btn_h = cta_txt_h + cta_pad_y * 2
        cta_btn_top = cta_center_y - cta_btn_h // 2
        # Top edge of the "forbidden zone" above the CTA
        cta_safe_top = cta_btn_top - cta_safe_buffer
    else:
        cta_btn_w = cta_btn_h = cta_btn_top = 0
        cta_safe_top = int(h * 0.88)

    # ══ Step 4: Collision prevention — shrink both until stack clears CTA ═════
    #
    # Stack layout (top zone logic; CTA is always independent):
    #   hl_y            = hl_box_top
    #   sub_y           = hl_y + hl_bh + sub_gap
    #   stack_bottom    = sub_y + sub_bh  (or hl_y + hl_bh if no sub)
    #
    # Require: stack_bottom <= cta_safe_top
    #
    for _ in range(30):  # safety limit
        hl_y         = hl_box_top
        sub_y        = hl_y + hl_bh + sub_gap
        stack_bottom = sub_y + sub_bh if sub_lines else hl_y + hl_bh

        if stack_bottom <= cta_safe_top:
            break   # ✓ no collision

        if hl_size <= 64 and sub_size <= 32:
            break   # already at minimum — accept whatever layout we have

        # Shrink both by 5%
        new_hl_size  = max(64, int(hl_size  * 0.95))
        new_sub_size = max(32, int(sub_size * 0.95))

        if new_hl_size == hl_size and new_sub_size == sub_size:
            break   # no progress possible

        hl_size  = new_hl_size
        sub_size = new_sub_size

        hl_font  = _load_font(bold_path, hl_size)
        hl_lines = wrap_text_to_width(hl_text, hl_font, hl_max_w)
        hl_lines = hl_lines[:4]
        _, hl_bh = measure_text_block(hl_lines, hl_font, 1.06)

        if sub_text:
            sub_font  = _load_font(regular_path, sub_size)
            sub_lines = wrap_text_to_width(sub_text, sub_font, sub_max_w)
            sub_lines = sub_lines[:3]
            _, sub_bh = measure_text_block(sub_lines, sub_font, 1.20)

    # Final positions after collision resolution
    hl_y  = hl_box_top
    sub_y = hl_y + hl_bh + sub_gap

    # ══ Step 5: Determine x-alignment from zone ════════════════════════════════
    if "left" in zone:
        x_anchor = h_margin
        align    = "left"
    elif "right" in zone:
        x_anchor = w - h_margin
        align    = "right"
    else:
        x_anchor = w // 2
        align    = "center"

    # ══ Step 6: Color scheme ═══════════════════════════════════════════════════
    if template in ("light-on-dark", "gradient-overlay"):
        text_clr  = (255, 255, 255, 255)
        sub_clr   = (235, 235, 235, 230)
        cta_bg    = (10,  10,  10,  220)
        cta_fg    = (255, 255, 255, 255)
    else:   # dark-on-light / card-overlay
        text_clr  = (15,  15,  15,  255)
        sub_clr   = (35,  35,  35,  230)
        cta_bg    = (15,  15,  15,  220)
        cta_fg    = (255, 255, 255, 255)

    # ══ Step 7: Build PIL overlay ══════════════════════════════════════════════
    rgb     = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    pil_img = Image.fromarray(rgb)
    overlay = Image.new("RGBA", pil_img.size, (0, 0, 0, 0))
    draw    = ImageDraw.Draw(overlay)

    # ── Background treatments ──────────────────────────────────────────────────
    if template == "gradient-overlay":
        grad    = Image.new("RGBA", pil_img.size, (0, 0, 0, 0))
        gd      = ImageDraw.Draw(grad)
        band    = int(h * 0.48)
        # top fade (for headline area)
        for i in range(band):
            alpha = int(170 * (1 - i / band))
            gd.line([(0, i), (w, i)], fill=(0, 0, 0, alpha))
        # bottom fade (for CTA area)
        for i in range(band):
            alpha = int(170 * (i / band))
            gd.line([(0, h - band + i), (w, h - band + i)], fill=(0, 0, 0, alpha))
        overlay = Image.alpha_composite(overlay, grad)
        draw    = ImageDraw.Draw(overlay)

    elif template == "card-overlay":
        cp      = int(h_margin * 0.5)
        stack_h = (sub_y + sub_bh) if sub_lines else (hl_y + hl_bh)
        cx1 = max(0, (x_anchor - hl_max_w // 2 - cp) if align == "center" else h_margin - cp)
        cy1 = max(0, hl_y - cp)
        cx2 = min(w, cx1 + hl_max_w + cp * 2)
        cy2 = min(h, stack_h + cp)
        draw.rounded_rectangle([cx1, cy1, cx2, cy2], radius=16, fill=(255, 255, 255, 200))

    # ── Logo ───────────────────────────────────────────────────────────────────
    if logo is not None:
        logo_rgb = cv2.cvtColor(logo, cv2.COLOR_BGR2RGB)
        lp = Image.fromarray(logo_rgb).convert("RGBA")
        lmw, lmh = int(w * 0.22), int(h * 0.055)
        lw, lh   = lp.size
        sc       = min(lmw / lw, lmh / lh)
        lp       = lp.resize((int(lw * sc), int(lh * sc)), Image.LANCZOS)
        nlw, nlh = lp.size
        lx = _x_for_align(x_anchor, nlw, align)
        ly = max(2, hl_y - nlh - int(h * 0.012))
        overlay.paste(lp, (lx, ly), lp)
        draw = ImageDraw.Draw(overlay)

    # ── Headline ───────────────────────────────────────────────────────────────
    y = hl_y
    for line in hl_lines:
        bbox  = hl_font.getbbox(line)
        lw    = bbox[2] - bbox[0]
        lh    = bbox[3] - bbox[1]
        x     = _x_for_align(x_anchor, lw, align)
        _shadow(draw, x, y, line, hl_font)
        draw.text((x, y), line, font=hl_font, fill=text_clr)
        y += int(lh * 1.06)

    # ── Subheadline ────────────────────────────────────────────────────────────
    if sub_lines:
        y = sub_y
        for line in sub_lines:
            bbox = sub_font.getbbox(line)
            lw   = bbox[2] - bbox[0]
            lh   = bbox[3] - bbox[1]
            x    = _x_for_align(x_anchor, lw, align)
            draw.text((x + 1, y + 2), line, font=sub_font, fill=(0, 0, 0, 95))
            draw.text((x, y), line, font=sub_font, fill=sub_clr)
            y += int(lh * 1.20)

    # ── CTA pill ───────────────────────────────────────────────────────────────
    if cta_upper:
        pill_r = cta_btn_h // 2
        bx     = _x_for_align(x_anchor, cta_btn_w, align)
        by     = cta_btn_top

        # shadow
        draw.rounded_rectangle(
            [bx + 3, by + 5, bx + cta_btn_w + 3, by + cta_btn_h + 5],
            radius=pill_r, fill=(0, 0, 0, 80),
        )
        # button fill
        draw.rounded_rectangle(
            [bx, by, bx + cta_btn_w, by + cta_btn_h],
            radius=pill_r, fill=cta_bg,
        )
        # centered text inside button
        draw.text(
            (bx + (cta_btn_w - cta_txt_w) // 2, by + (cta_btn_h - cta_txt_h) // 2),
            cta_upper, font=cta_font, fill=cta_fg,
        )

    # ══ Composite and return ═══════════════════════════════════════════════════
    pil_img = pil_img.convert("RGBA")
    result  = Image.alpha_composite(pil_img, overlay).convert("RGB")
    return cv2.cvtColor(np.array(result), cv2.COLOR_RGB2BGR)
