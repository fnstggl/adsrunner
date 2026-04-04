"""
Image generation via fal.ai API.
"""

import io
import os
import requests

import numpy as np
import cv2


def generate_image(prompt: str, width: int = 1080, height: int = 1350) -> np.ndarray:
    """
    Generate a single image from a text prompt using fal.ai nano-banana-2.

    Returns: BGR numpy array (OpenCV format)
    """
    try:
        import fal_client
    except ImportError:
        raise RuntimeError(
            "fal-client is not installed. Run: pip install fal-client"
        )

    fal_key = os.environ.get("FAL_KEY")
    if not fal_key:
        raise RuntimeError("FAL_KEY environment variable is not set")

    model = os.environ.get("FAL_MODEL", "fal-ai/nano-banana-2")

    result = fal_client.subscribe(
        model,
        arguments={
            "prompt": prompt,
            "image_size": {"width": width, "height": height},
            "num_images": 1,
        },
    )

    image_url = result["images"][0]["url"]

    # Download the image
    resp = requests.get(image_url, timeout=30)
    resp.raise_for_status()

    # Decode to OpenCV BGR array
    img_array = np.frombuffer(resp.content, np.uint8)
    img = cv2.imdecode(img_array, cv2.IMREAD_COLOR)
    if img is None:
        raise RuntimeError("Failed to decode generated image")

    return img


def generate_all_images(prompts: list[str]) -> list[np.ndarray]:
    """Generate images for all prompts. Returns list of BGR numpy arrays."""
    images = []
    for i, prompt in enumerate(prompts):
        print(f"[IMAGE GEN] Generating image {i + 1}/{len(prompts)}...")
        img = generate_image(prompt)
        images.append(img)
    return images
