"""
AI Ad Generator — Flask Backend
Wraps the existing greenscreen compositor with an AI-powered ad generation pipeline.
"""

import base64
import hashlib
import io
import json
import os
import tempfile
import time
from pathlib import Path

import cv2
import numpy as np
from dotenv import load_dotenv
from flask import Flask, Response, jsonify, render_template, request, stream_with_context

load_dotenv()

import compositor_v4_final as compositor
import vision_corner_detector as vcd

from lib.classify_input import classify_input
from lib.build_creative_specs import build_creative_specs
from lib.generate_prompt import generate_prompt, generate_all_prompts
from lib.generate_images import generate_image, generate_all_images
from lib.composite_ad import compose_ui_into_greenscreen
from lib.render_text_overlay import render_text_overlay

CACHE_DIR = Path("/tmp/corners_cache")
CACHE_DIR.mkdir(parents=True, exist_ok=True)

app = Flask(__name__)
app.config["MAX_CONTENT_LENGTH"] = 50 * 1024 * 1024  # 50 MB (increased for multi-upload)


def _encode_png(arr: np.ndarray) -> str:
    """Encode a BGR numpy array as base64 PNG string."""
    ok, buf = cv2.imencode(".png", arr)
    if not ok:
        raise RuntimeError("cv2.imencode failed")
    return base64.b64encode(buf.tobytes()).decode()


def _decode_upload(file_storage) -> np.ndarray:
    """Decode an uploaded image file to BGR numpy array."""
    data = file_storage.read()
    arr = cv2.imdecode(np.frombuffer(data, np.uint8), cv2.IMREAD_COLOR)
    if arr is None:
        raise ValueError(f"Could not decode image: {file_storage.filename}")
    return arr


# ═══════════════════════════════════════════════════════════════════════════
# MAIN PAGE — Ad Generator
# ═══════════════════════════════════════════════════════════════════════════

@app.route("/")
def index():
    return render_template("index.html")


# ═══════════════════════════════════════════════════════════════════════════
# LEGACY — Compositor Tool (still accessible)
# ═══════════════════════════════════════════════════════════════════════════

@app.route("/compositor")
def compositor_page():
    return render_template("compositor.html")


@app.route("/composite", methods=["POST"])
def do_composite():
    """Legacy compositor endpoint — unchanged from original."""
    scene_file = request.files.get("scene")
    ui_file = request.files.get("ui")

    if not scene_file or not ui_file:
        return jsonify({"error": "Both 'scene' and 'ui' images are required."}), 400

    scene_bytes = scene_file.read()
    ui_bytes = ui_file.read()

    if not scene_bytes or not ui_bytes:
        return jsonify({"error": "One or both uploaded files are empty."}), 400

    scene_arr = cv2.imdecode(np.frombuffer(scene_bytes, np.uint8), cv2.IMREAD_COLOR)
    ui_arr = cv2.imdecode(np.frombuffer(ui_bytes, np.uint8), cv2.IMREAD_COLOR)

    if scene_arr is None:
        return jsonify({"error": "Could not decode the scene image."}), 400
    if ui_arr is None:
        return jsonify({"error": "Could not decode the UI image."}), 400

    sh, sw = scene_arr.shape[:2]

    cv_corners, blend_mask = compositor.detect_green_corners(scene_arr)

    if cv_corners is not None:
        corners = cv_corners
        corners_data = {
            "corners": cv_corners.tolist(),
            "image_size": [sw, sh],
            "confidence": "high",
            "notes": "Detected via HSV chroma key analysis (pixel-perfect)",
            "method": "cv_green",
        }
        cached = False
        detection_method = "cv_green"
    else:
        blend_mask = None
        detection_method = "vision_llm"
        scene_hash = hashlib.sha256(scene_bytes).hexdigest()
        cache_path = CACHE_DIR / f"{scene_hash}.json"
        cached = cache_path.exists()

        if cached:
            try:
                with open(cache_path) as f:
                    corners_data = json.load(f)
            except (json.JSONDecodeError, OSError):
                cached = False
                corners_data = None

        if not cached:
            suffix = Path(scene_file.filename or "scene.png").suffix or ".png"
            with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as tf:
                tf.write(scene_bytes)
                scene_tmp = tf.name

            try:
                corners_data = vcd.detect_corners_with_vision(scene_tmp)
            except Exception as exc:
                os.unlink(scene_tmp)
                return jsonify({"error": f"Vision API error: {exc}"}), 502
            finally:
                try:
                    os.unlink(scene_tmp)
                except OSError:
                    pass

            with open(cache_path, "w") as f:
                json.dump(corners_data, f, indent=2)

        stored_w, stored_h = corners_data.get("image_size", [sw, sh])
        if sw != stored_w or sh != stored_h:
            sx, sy = sw / stored_w, sh / stored_h
            corners_data["corners"] = [
                [pt[0] * sx, pt[1] * sy] for pt in corners_data["corners"]
            ]
            corners_data["image_size"] = [sw, sh]

        corners = np.array(corners_data["corners"], dtype=np.float32)

    debug_arr = scene_arr.copy()
    quad = corners.astype(np.int32)
    cv2.polylines(debug_arr, [quad], isClosed=True, color=(0, 255, 0), thickness=3)
    for pt, label in zip(quad, ["TL", "TR", "BR", "BL"]):
        cv2.circle(debug_arr, tuple(pt), 10, (0, 0, 255), -1)
        cv2.putText(debug_arr, label, (int(pt[0]) + 12, int(pt[1]) + 6),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 255), 2)

    try:
        result = compositor.composite(scene_arr, ui_arr, corners,
                                      blend_mask=blend_mask)
    except Exception as exc:
        return jsonify({"error": f"Compositor error: {exc}"}), 500

    stored_w, stored_h = corners_data.get("image_size", [sw, sh])
    return jsonify(
        {
            "image": _encode_png(result),
            "debug_image": _encode_png(debug_arr),
            "corners": corners_data["corners"],
            "detection_size": [stored_w, stored_h],
            "composite_size": [sw, sh],
            "confidence": corners_data.get("confidence", "unknown"),
            "notes": corners_data.get("notes", ""),
            "cached": cached,
            "detection_method": detection_method,
        }
    )


# ═══════════════════════════════════════════════════════════════════════════
# NEW — Ad Generator API Routes
# ═══════════════════════════════════════════════════════════════════════════

@app.route("/api/analyze", methods=["POST"])
def api_analyze():
    """
    Step 1: Classify the product and determine if UI screenshots are needed.
    Returns classification + creative specs preview.
    """
    product_description = request.form.get("product_description", "").strip()
    if not product_description:
        return jsonify({"error": "product_description is required"}), 400

    ad_goal = request.form.get("ad_goal", "").strip()
    has_ui = request.form.get("has_ui_screenshots") == "true"

    try:
        classification = classify_input(product_description, ad_goal)
    except Exception as exc:
        return jsonify({"error": f"Classification failed: {exc}"}), 500

    needs_ui = classification.get("needs_ui", False)

    return jsonify({
        "classification": classification,
        "needs_ui": needs_ui and not has_ui,
        "product_type": classification.get("product_type", "other"),
        "likely_ad_styles": classification.get("likely_ad_styles", []),
    })


@app.route("/api/generate", methods=["POST"])
def api_generate():
    """
    Step 2: Full ad generation pipeline.
    Returns Server-Sent Events for progress, then final results.
    """
    product_description = request.form.get("product_description", "").strip()
    if not product_description:
        return jsonify({"error": "product_description is required"}), 400

    ad_goal = request.form.get("ad_goal", "").strip()
    analysis_raw = request.form.get("analysis", "{}")

    try:
        analysis = json.loads(analysis_raw)
    except json.JSONDecodeError:
        analysis = {}

    classification = analysis.get("classification", {})

    # Collect uploaded files
    logo_arr = None
    if "logo" in request.files:
        try:
            logo_arr = _decode_upload(request.files["logo"])
        except Exception:
            pass

    product_images = []
    for f in request.files.getlist("product_images"):
        try:
            product_images.append(_decode_upload(f))
        except Exception:
            pass

    ui_screenshots = []
    for f in request.files.getlist("ui_screenshots"):
        try:
            ui_screenshots.append(_decode_upload(f))
        except Exception:
            pass

    has_ui = len(ui_screenshots) > 0

    def generate_stream():
        def send_event(data):
            return f"data: {json.dumps(data)}\n\n"

        try:
            # Step 1: Build creative specs
            yield send_event({"type": "progress", "percent": 20, "message": "Building creative concepts..."})

            specs = build_creative_specs(
                product_description=product_description,
                classification=classification,
                ad_goal=ad_goal,
                has_logo=logo_arr is not None,
                has_product_images=len(product_images) > 0,
                has_ui_screenshots=has_ui,
            )

            yield send_event({"type": "progress", "percent": 30, "message": "Creative specs ready. Generating prompts..."})

            # Step 2: Generate prompts
            prompts = generate_all_prompts(specs, product_description)

            yield send_event({"type": "progress", "percent": 40, "message": "Prompts ready. Generating images (this takes a moment)..."})

            # Step 3: Generate images
            ads = []
            for i, (spec, prompt) in enumerate(zip(specs, prompts)):
                pct = 40 + int((i / len(specs)) * 40)
                yield send_event({"type": "progress", "percent": pct, "message": f"Generating image {i + 1} of {len(specs)}..."})

                try:
                    raw_image = generate_image(prompt)
                except Exception as exc:
                    print(f"[ERROR] Image generation failed for ad {i + 1}: {exc}")
                    yield send_event({"type": "progress", "percent": pct, "message": f"Image {i + 1} failed, skipping..."})
                    continue

                # Step 4: Composite UI if needed
                if spec.get("needsUi", False) and has_ui:
                    yield send_event({"type": "progress", "percent": pct + 5, "message": f"Compositing UI into ad {i + 1}..."})
                    # Use first UI screenshot (or cycle through them)
                    ui_img = ui_screenshots[i % len(ui_screenshots)]
                    try:
                        raw_image = compose_ui_into_greenscreen(raw_image, ui_img)
                    except Exception as exc:
                        print(f"[WARN] Compositing failed for ad {i + 1}: {exc}")

                # Step 5: Render text overlay
                yield send_event({"type": "progress", "percent": pct + 8, "message": f"Adding text overlay to ad {i + 1}..."})
                try:
                    final_image = render_text_overlay(
                        image=raw_image,
                        headline=spec.get("headline", ""),
                        subheadline=spec.get("subheadline", ""),
                        cta=spec.get("cta", ""),
                        zone=spec.get("negativeSpaceZone", "bottom-center"),
                        template=spec.get("textTemplate", "light-on-dark"),
                        logo=logo_arr,
                    )
                except Exception as exc:
                    print(f"[WARN] Text overlay failed for ad {i + 1}: {exc}")
                    final_image = raw_image

                ads.append({
                    "image": _encode_png(final_image),
                    "spec_id": spec.get("id", f"ad_{i + 1}"),
                    "angle": spec.get("angle", ""),
                    "headline": spec.get("headline", ""),
                    "subheadline": spec.get("subheadline", ""),
                    "cta": spec.get("cta", ""),
                    "needs_ui": spec.get("needsUi", False),
                    "scene_type": spec.get("sceneType", ""),
                })

            yield send_event({"type": "progress", "percent": 95, "message": "Finalizing..."})

            if not ads:
                yield send_event({"type": "error", "message": "No ads could be generated. Check your FAL_KEY and try again."})
                return

            yield send_event({
                "type": "result",
                "ads": ads,
                "total": len(ads),
            })

        except Exception as exc:
            print(f"[ERROR] Generation pipeline failed: {exc}")
            import traceback
            traceback.print_exc()
            yield send_event({"type": "error", "message": str(exc)})

    return Response(
        stream_with_context(generate_stream()),
        mimetype="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "X-Accel-Buffering": "no",
        },
    )


if __name__ == "__main__":
    missing = []
    if not os.environ.get("ANTHROPIC_API_KEY"):
        missing.append("ANTHROPIC_API_KEY")
    if not os.environ.get("FAL_KEY"):
        missing.append("FAL_KEY")
    if missing:
        print(f"[WARN] Missing env vars: {', '.join(missing)} — some features will fail.")
    app.run(debug=True, port=5000)
