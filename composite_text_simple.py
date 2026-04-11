#!/usr/bin/env python3
"""
Create composite by drawing text directly on image.
Simpler approach without needing browser rendering.
"""

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont


def create_composite_with_text(image_path: str, output_path: str) -> bool:
    """Create composite by drawing text directly on image."""

    print(f"Loading image from: {image_path}")
    img = Image.open(image_path)
    img_w, img_h = img.size

    print(f"Image size: {img_w}x{img_h}")

    # Convert to RGB if needed
    if img.mode == "RGBA":
        img = img.convert("RGB")

    # Create a copy for drawing
    canvas = img.copy()
    draw = ImageDraw.Draw(canvas, "RGBA")

    # Define text and styling based on our responsive sizing test
    texts = [
        {
            "text": "PREMIUM WORKSPACE",
            "size": 24,
            "color": (153, 153, 153, 255),  # accent color
            "y_offset": 100,
            "weight": "bold",
        },
        {
            "text": "Transform Your Home Into\nA Productivity Powerhouse",
            "size": 90,
            "color": (255, 255, 255, 255),  # white
            "y_offset": 150,
            "weight": "bold",
        },
        {
            "text": "Modern lighting and ergonomic design\nthat actually makes remote work enjoyable",
            "size": 48,
            "color": (240, 240, 240, 255),  # light gray
            "y_offset": 450,
            "weight": "normal",
        },
    ]

    # Try to use a nice font, fall back to default
    try:
        # Try different font paths
        font_paths = [
            "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
            "/usr/share/fonts/truetype/liberation/LiberationSans-Bold.ttf",
            "/System/Library/Fonts/Helvetica.ttc",
            "C:\\Windows\\Fonts\\arial.ttf",
        ]

        font = None
        for font_path in font_paths:
            try:
                font = ImageFont.truetype(font_path, 24)
                print(f"Using font: {font_path}")
                break
            except Exception:
                continue

        if font is None:
            font = ImageFont.load_default()
            print("Using default font")
    except Exception as e:
        print(f"Font error: {e}")
        font = ImageFont.load_default()

    # Draw semi-transparent dark background for text readability
    padding = 20
    bg_alpha = 180  # 70% opacity dark overlay

    # Draw background box
    bg_y_start = texts[0]["y_offset"] - 30
    bg_y_end = texts[2]["y_offset"] + 150
    draw.rectangle(
        [(0, bg_y_start), (img_w, bg_y_end)],
        fill=(0, 0, 0, bg_alpha),
    )

    # Draw text elements
    for text_info in texts:
        font_size = text_info["size"]
        try:
            text_font = ImageFont.truetype(font_paths[0], font_size)
        except:
            text_font = font

        y = text_info["y_offset"]
        text = text_info["text"]
        color = text_info["color"]

        # Center align text
        lines = text.split("\n")
        line_height = int(font_size * 1.2)

        for i, line in enumerate(lines):
            try:
                bbox = draw.textbbox((0, 0), line, font=text_font)
                line_width = bbox[2] - bbox[0]
            except:
                line_width = len(line) * font_size * 0.6  # Estimate

            x = (img_w - line_width) // 2

            # Draw text with slight shadow for better readability
            draw.text((x + 2, y + 2), line, fill=(0, 0, 0, 100), font=text_font)
            # Draw main text
            draw.text((x, y), line, fill=color, font=text_font)

            y += line_height

    # Add CTA button
    button_y = 600
    button_width = 300
    button_height = 60
    button_x = (img_w - button_width) // 2
    button_color = (255, 107, 53, 255)  # #FF6B35 in RGBA

    # Draw button rounded rectangle
    draw.rectangle(
        [
            (button_x, button_y),
            (button_x + button_width, button_y + button_height),
        ],
        fill=button_color,
    )

    # Draw button text "Shop Now"
    button_text = "SHOP NOW"
    try:
        button_font = ImageFont.truetype(font_paths[0], 44)
    except:
        button_font = text_font

    try:
        bbox = draw.textbbox((0, 0), button_text, font=button_font)
        text_width = bbox[2] - bbox[0]
        text_height = bbox[3] - bbox[1]
    except:
        text_width = len(button_text) * 44 * 0.6
        text_height = 44

    text_x = button_x + (button_width - text_width) // 2
    text_y = button_y + (button_height - text_height) // 2

    draw.text(
        (text_x, text_y),
        button_text,
        fill=(255, 255, 255, 255),
        font=button_font,
    )

    # Save output
    print(f"Saving composite to: {output_path}")
    canvas.save(output_path, quality=95)

    print("✓ Composite created successfully!")
    return True


if __name__ == "__main__":
    import sys

    image_file = "/home/user/adsrunner/test_image.jpg"
    output_file = "/home/user/adsrunner/composite_output.jpg"

    success = create_composite_with_text(image_file, output_file)

    if success:
        print(f"\n✓ Composite saved to: {output_file}")
        sys.exit(0)
    else:
        print("\n✗ Failed to create composite")
        sys.exit(1)
