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
from typing import Optional
import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

# ── Font priority lists ────────────────────────────────────────────────────────
FONT_PATHS_BOLD = [
    "/usr/share/fonts/truetype/google-fonts/Poppins-Bold.ttf",
    "/usr/share/fonts/truetype/liberation/LiberationSans-Bold.ttf",
    "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
    "/usr/share/fonts/truetype/freefont/FreeSansBold.ttf",
]
FONT_PATHS_REGULAR = [
    "/usr/share/fonts/truetype/google-fonts/Poppins-Medium.ttf",
    "/usr/share/fonts/truetype/google-fonts/Poppins-Regular.ttf",
    "/usr/share/fonts/truetype/liberation/LiberationSans-Regular.ttf",
    "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
    "/usr/share/fonts/truetype/freefont/FreeSans.ttf",
]


# ─── Font helpers ──────────────────────────────────────────────────────────────

def _find_font(bold: bool = True) -> Optional[str]:
    paths = FONT_PATHS_BOLD if bold else FONT_PATHS_REGULAR
    for p in paths:
        if os.path.exists(p):
            return p
    for p in FONT_PATHS_BOLD + FONT_PATHS_REGULAR:
        if os.path.exists(p):
            return p
    return None


def _load_font(path: Optional[str], size: int) -> ImageFont.FreeTypeFont:
    if path and os.path.exists(path):
        return ImageFont.truetype(path, size)
    return ImageFont.load_default()


# ─── Layout helpers ────────────────────────────────────────────────────────────

def wrap_text_to_width(
    text: str,
    font: ImageFont.FreeTypeFont,
    max_width: int,
) -> list[str]:
    """Word-wrap using actual font metrics."""
    words = text.split()
    lines: list[str] = []
    current = ""
    for word in words:
        candidate = (current + " " + word).strip()
        bb = font.getbbox(candidate)
        w = bb[2] - bb[0]
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
    """Return (max_line_width, total_block_height) using actual glyph metrics."""
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
            total_h += lh
    return max_w, total_h


def fit_text_block(
    text: str,
    font_path: Optional[str],
    start_size: int,
    min_size: int,
    max_size: int,
    max_width: int,
    max_height: int,
    max_lines: int,
    line_height_ratio: float,
) -> tuple[int, ImageFont.FreeTypeFont, list[str], int, int]:
    """Shrink font until block fits. Returns (size, font, lines, bw, bh)."""
    size = max(min_size, min(max_size, start_size))
    while True:
        font = _load_font(font_path, size)
        lines = wrap_text_to_width(text, font, max_width)
        lines = lines[:max_lines]
        bw, bh = measure_text_block(lines, font, line_height_ratio)
        if (bh <= max_height and bw <= max_width) or size <= min_size:
            return size, font, lines, bw, bh
        size = max(min_size, int(size * 0.93))


# ─── Drawing helpers ───────────────────────────────────────────────────────────

def _draw_shadow(draw, x, y, text, font, strength=3):
    """Layered drop-shadow for readability over any background."""
    for ox in range(1, strength + 1):
        alpha = int(200 - ox * 20)
        draw.text((x + ox, y + ox), text, font=font, fill=(0, 0, 0, alpha))


def _x_for_align(x_anchor: int, line_w: int, align: str) -> int:
    if align == "center":
        return x_anchor - line_w // 2
    if align == "right":
        return x_anchor - line_w
    return x_anchor


def _draw_text_with_outline(draw, x, y, text, font, fill, outline_color=(0,0,0,200), outline_width=3):
    """Draw text with a strong outline for maximum readability."""
    for dx in range(-outline_width, outline_width + 1):
        for dy in range(-outline_width, outline_width + 1):
            if dx != 0 or dy != 0:
                draw.text((x + dx, y + dy), text, font=font, fill=outline_color)
    draw.text((x, y), text, font=font, fill=fill)


# ─── Dark scrim helper ─────────────────────────────────────────────────────────

def _apply_bottom_scrim(overlay: Image.Image, scrim_top: int, canvas_h: int, canvas_w: int) -> Image.Image:
    """
    Paint a dark gradient scrim from scrim_top to bottom of canvas.
    This ensures text always has contrast regardless of image content.
    """
    scrim = Image.new("RGBA", (canvas_w, canvas_h), (0, 0, 0, 0))
    sd = ImageDraw.Draw(scrim)
    band = canvas_h - scrim_top
    for i in range(band):
        progress = i / max(band - 1, 1)
        # Ease in: light at top, dark at bottom
        alpha = int(200 * (progress ** 0.6))
        y = scrim_top + i
        sd.line([(0, y), (canvas_w, y)], fill=(0, 0, 0, alpha))
    return Image.alpha_composite(overlay, scrim)


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

    Layout strategy (bottom-anchored):
      1. Reserve CTA at bottom (10% from bottom edge)
      2. Reserve subheadline above CTA
      3. Place headline above subheadline
      4. Apply dark scrim behind text for guaranteed readability
      5. All text is large — minimum sizes are enforced

    Args:
        image:       BGR numpy array
        headline:    Main headline text (auto-uppercased)
        subheadline: Supporting line
        cta:         Call-to-action label
        zone:        Ignored — layout is always bottom-centered for clarity
        template:    light-on-dark | dark-on-light | card-overlay | gradient-overlay
        logo:        Optional BGR logo array

    Returns:
        BGR numpy array with overlay composited
    """
    h, w = image.shape[:2]

    # ══ Canvas metrics ═════════════════════════════════════════════════════════
    H_MARGIN      = int(w * 0.06)       # left/right padding
    TEXT_MAX_W    = w - H_MARGIN * 2    # usable text width
    BOTTOM_PAD    = int(h * 0.05)       # gap from very bottom
    CTA_PAD_X     = int(w * 0.07)       # CTA horizontal padding
    CTA_PAD_Y     = int(h * 0.020)      # CTA vertical padding
    BLOCK_GAP     = int(h * 0.025)      # gap between headline and sub
    CTA_SUB_GAP   = int(h * 0.028)      # gap between sub and CTA
    SCRIM_FADE    = int(h * 0.35)       # how far up scrim reaches

    # ══ Font paths ══════════════════════════════════════════════════════════════
    bold_path    = _find_font(bold=True)
    regular_path = _find_font(bold=False)

    # ══ Step 1: CTA geometry (anchor from bottom up) ════════════════════════════
    cta_upper = (cta or "").upper()
    cta_size  = max(int(h * 0.038), 36)
    cta_font  = _load_font(bold_path, cta_size)

    if cta_upper:
        cb        = cta_font.getbbox(cta_upper)
        cta_txt_w = cb[2] - cb[0]
        cta_txt_h = cb[3] - cb[1]
        cta_btn_w = cta_txt_w + CTA_PAD_X * 2
        cta_btn_h = cta_txt_h + CTA_PAD_Y * 2
        # Bottom edge of CTA button
        cta_btn_bottom = h - BOTTOM_PAD
        cta_btn_top    = cta_btn_bottom - cta_btn_h
    else:
        cta_btn_w = cta_btn_h = cta_txt_w = cta_txt_h = 0
        cta_btn_top = cta_btn_bottom = h - BOTTOM_PAD

    # ══ Step 2: Subheadline geometry (above CTA) ════════════════════════════════
    sub_text = subheadline or ""
    # Start sub at large size: ~5% of canvas height, min 40px
    sub_start = max(int(h * 0.052), 40)
    sub_max_h = int(h * 0.22)   # subheadline block can use up to 22% of height

    if sub_text:
        sub_top_limit = cta_btn_top - CTA_SUB_GAP - sub_max_h
        sub_size, sub_font, sub_lines, sub_bw, sub_bh = fit_text_block(
            text=sub_text,
            font_path=regular_path,
            start_size=sub_start,
            min_size=32,
            max_size=int(h * 0.07),
            max_width=TEXT_MAX_W,
            max_height=sub_max_h,
            max_lines=2,
            line_height_ratio=1.25,
        )
        sub_bottom = cta_btn_top - CTA_SUB_GAP
        sub_top    = sub_bottom - sub_bh
    else:
        sub_size = sub_start
        sub_font = _load_font(regular_path, sub_start)
        sub_lines = []
        sub_bw = sub_bh = 0
        sub_top = sub_bottom = cta_btn_top - CTA_SUB_GAP

    # ══ Step 3: Headline geometry (above subheadline) ══════════════════════════
    hl_text  = (headline or "").upper()
    # Available height for headline: from top of sub upward, minus gap
    hl_bottom   = sub_top - BLOCK_GAP
    hl_max_h    = hl_bottom - int(h * 0.08)   # don't go above 8% from top
    hl_max_h    = max(hl_max_h, int(h * 0.15)) # at least 15% height available
    # Start very large: 12% of canvas height, minimum 80px
    hl_start = max(int(h * 0.12), 80)

    hl_size, hl_font, hl_lines, hl_bw, hl_bh = fit_text_block(
        text=hl_text,
        font_path=bold_path,
        start_size=hl_start,
        min_size=60,
        max_size=int(h * 0.18),
        max_width=TEXT_MAX_W,
        max_height=hl_max_h,
        max_lines=3,
        line_height_ratio=1.15,
    )
    hl_top = hl_bottom - hl_bh

    # ══ Step 4: Scrim region (covers all text) ══════════════════════════════════
    scrim_top = max(0, hl_top - int(h * 0.04))

    # ══ Step 5: Color scheme ════════════════════════════════════════════════════
    if template in ("light-on-dark", "gradient-overlay", "card-overlay"):
        text_clr    = (255, 255, 255, 255)
        sub_clr     = (235, 235, 235, 245)
        cta_bg      = (255, 255, 255, 245)
        cta_fg      = (15,  15,  15,  255)
        outline_clr = (0, 0, 0, 180)
    else:  # dark-on-light
        text_clr    = (15,  15,  15,  255)
        sub_clr     = (40,  40,  40,  245)
        cta_bg      = (15,  15,  15,  235)
        cta_fg      = (255, 255, 255, 255)
        outline_clr = (255, 255, 255, 120)

    # ══ Step 6: Build PIL canvas ════════════════════════════════════════════════
    rgb     = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    pil_img = Image.fromarray(rgb).convert("RGBA")
    overlay = Image.new("RGBA", pil_img.size, (0, 0, 0, 0))

    # ── Apply dark scrim behind text area ──────────────────────────────────────
    overlay = _apply_bottom_scrim(overlay, scrim_top, h, w)
    draw    = ImageDraw.Draw(overlay)

    # ── Logo (top of text block) ───────────────────────────────────────────────
    if logo is not None:
        logo_rgb = cv2.cvtColor(logo, cv2.COLOR_BGR2RGB)
        lp   = Image.fromarray(logo_rgb).convert("RGBA")
        lmw  = int(w * 0.25)
        lmh  = int(h * 0.06)
        lw, lh_l = lp.size
        sc   = min(lmw / lw, lmh / lh_l)
        lp   = lp.resize((int(lw * sc), int(lh_l * sc)), Image.LANCZOS)
        nlw, nlh = lp.size
        lx   = w // 2 - nlw // 2
        ly   = max(4, hl_top - nlh - int(h * 0.015))
        overlay.paste(lp, (lx, ly), lp)
        draw = ImageDraw.Draw(overlay)

    # ── Headline ───────────────────────────────────────────────────────────────
    x_anchor = w // 2
    y = hl_top
    for line in hl_lines:
        bbox = hl_font.getbbox(line)
        lw   = bbox[2] - bbox[0]
        lh   = bbox[3] - bbox[1]
        x    = _x_for_align(x_anchor, lw, "center")
        _draw_text_with_outline(draw, x, y, line, hl_font, text_clr, outline_clr, outline_width=3)
        y += int(lh * 1.15)

    # ── Subheadline ────────────────────────────────────────────────────────────
    if sub_lines:
        y = sub_top
        for line in sub_lines:
            bbox = sub_font.getbbox(line)
            lw   = bbox[2] - bbox[0]
            lh   = bbox[3] - bbox[1]
            x    = _x_for_align(x_anchor, lw, "center")
            # Lighter outline for sub
            _draw_text_with_outline(draw, x, y, line, sub_font, sub_clr, (0,0,0,140), outline_width=2)
            y += int(lh * 1.25)

    # ── CTA pill ───────────────────────────────────────────────────────────────
    if cta_upper:
        pill_r = cta_btn_h // 2
        bx     = x_anchor - cta_btn_w // 2
        by     = cta_btn_top

        # Drop shadow
        draw.rounded_rectangle(
            [bx + 3, by + 5, bx + cta_btn_w + 3, by + cta_btn_h + 5],
            radius=pill_r, fill=(0, 0, 0, 140),
        )
        # Button fill
        draw.rounded_rectangle(
            [bx, by, bx + cta_btn_w, by + cta_btn_h],
            radius=pill_r, fill=cta_bg,
        )
        # CTA text centered in button
        tx = bx + (cta_btn_w - cta_txt_w) // 2
        ty = by + (cta_btn_h - cta_txt_h) // 2
        draw.text((tx, ty), cta_upper, font=cta_font, fill=cta_fg)

    # ══ Composite ══════════════════════════════════════════════════════════════
    result = Image.alpha_composite(pil_img, overlay).convert("RGB")
    return cv2.cvtColor(np.array(result), cv2.COLOR_RGB2BGR)
