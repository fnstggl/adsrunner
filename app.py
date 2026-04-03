"""
Green Screen Phone Compositor — Flask Backend
Accepts two images, detects phone screen corners via CV green screen detection,
warps the UI screenshot into the screen quad, returns composited PNG.
"""

import hashlib
import io
import json
import os
import tempfile
from pathlib import Path

import cv2
import numpy as np
from dotenv import load_dotenv
from flask import Flask, jsonify, render_template, request, send_file

load_dotenv()

import compositor_v4_final as compositor

CACHE_DIR = Path("/tmp/corners_cache")
CACHE_DIR.mkdir(parents=True, exist_ok=True)

app = Flask(__name__)
app.config["MAX_CONTENT_LENGTH"] = 20 * 1024 * 1024  # 20 MB


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/composite", methods=["POST"])
def do_composite():
    scene_file = request.files.get("scene")
    ui_file = request.files.get("ui")

    if not scene_file or not ui_file:
        return jsonify({"error": "Both 'scene' and 'ui' images are required."}), 400

    scene_bytes = scene_file.read()
    ui_bytes = ui_file.read()

    if not scene_bytes or not ui_bytes:
        return jsonify({"error": "One or both uploaded files are empty."}), 400

    # ── Decode images ─────────────────────────────────────────────────────────
    scene_arr = cv2.imdecode(np.frombuffer(scene_bytes, np.uint8), cv2.IMREAD_COLOR)
    ui_arr    = cv2.imdecode(np.frombuffer(ui_bytes,    np.uint8), cv2.IMREAD_COLOR)

    if scene_arr is None:
        return jsonify({"error": "Could not decode the scene image."}), 400
    if ui_arr is None:
        return jsonify({"error": "Could not decode the UI image."}), 400

    sh, sw = scene_arr.shape[:2]

    # ── Corner detection — CV green screen (no API, no file I/O needed) ───────
    # Cache key: hash of scene bytes so same scene skips re-detection
    scene_hash = hashlib.sha256(scene_bytes).hexdigest()
    cache_path = CACHE_DIR / f"{scene_hash}.json"

    blend_mask = None
    cached = False

    if cache_path.exists():
        # Load cached corners (from a previous request with the same scene)
        try:
            with open(cache_path) as f:
                corners_data = json.load(f)
            corners = np.array(corners_data["corners"], dtype=np.float32)

            # Scale if scene dimensions changed (shouldn't happen, but safe)
            stored_w, stored_h = corners_data.get("image_size", [sw, sh])
            if sw != stored_w or sh != stored_h:
                print(f"[WARN] Dimension mismatch: stored={stored_w}x{stored_h}, "
                      f"current={sw}x{sh}. Scaling corners.")
                sx, sy = sw / stored_w, sh / stored_h
                corners = np.array(
                    [[pt[0] * sx, pt[1] * sy] for pt in corners_data["corners"]],
                    dtype=np.float32
                )

            # Re-detect blend mask (it's cheap and not stored in cache)
            _, blend_mask = compositor.get_clean_mask(scene_arr)
            cached = True
            detection_method = corners_data.get("method", "cached")
            print(f"[DETECT] Using cached corners from {cache_path.name}")

        except (json.JSONDecodeError, OSError, KeyError) as e:
            print(f"[WARN] Cache read failed ({e}), re-detecting...")
            cached = False

    if not cached:
        # Auto-detect via CV green screen — uses high-saturation HSV mask,
        # extreme_quad_corners (convex hull diagonal projection), and
        # RANSAC sub-pixel line fitting per edge.
        print(f"[DETECT] Running CV green screen detection...")
        corners, blend_mask = compositor.detect_green_corners(scene_arr)

        if corners is None:
            return jsonify({
                "error": (
                    "No green screen region detected in the scene image. "
                    "Ensure the phone screen shows a solid chroma green (#00B140) "
                    "with high saturation and no glare or gradients."
                )
            }), 422

        corners_data = {
            "corners": corners.tolist(),
            "image_size": [sw, sh],
            "confidence": "high",
            "notes": "Detected via HSV chroma key + RANSAC sub-pixel line fitting",
            "method": "cv_ransac_subpixel",
        }

        # Save to cache — keyed by scene image hash so same scene is instant next time
        with open(cache_path, "w") as f:
            json.dump(corners_data, f, indent=2)
        print(f"[DETECT] Corners cached → {cache_path}")
        detection_method = "cv_ransac_subpixel"

    # ── Debug visualization: draw detected quad on scene copy ─────────────────
    debug_arr = scene_arr.copy()
    quad = corners.astype(np.int32)
    cv2.polylines(debug_arr, [quad], isClosed=True, color=(0, 255, 0), thickness=3)
    for pt, label in zip(quad, ["TL", "TR", "BR", "BL"]):
        cv2.circle(debug_arr, tuple(pt), 10, (0, 0, 255), -1)
        cv2.putText(
            debug_arr, label,
            (int(pt[0]) + 12, int(pt[1]) + 6),
            cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 255), 2
        )

    # ── Composite ────────────────────────────────────────────────────────────
    try:
        result = compositor.composite(
            scene_arr, ui_arr, corners,
            feather=3,
            blend_mask=blend_mask
        )
    except Exception as exc:
        return jsonify({"error": f"Compositor error: {exc}"}), 500

    # ── Encode and return ────────────────────────────────────────────────────
    import base64

    def _encode_png(arr: np.ndarray) -> str:
        ok, buf = cv2.imencode(".png", arr)
        if not ok:
            raise RuntimeError("cv2.imencode failed")
        return base64.b64encode(buf.tobytes()).decode()

    stored_w, stored_h = corners_data.get("image_size", [sw, sh])

    return jsonify({
        "image":            _encode_png(result),
        "debug_image":      _encode_png(debug_arr),
        "corners":          corners_data["corners"],
        "detection_size":   [stored_w, stored_h],
        "composite_size":   [sw, sh],
        "confidence":       corners_data.get("confidence", "unknown"),
        "notes":            corners_data.get("notes", ""),
        "cached":           cached,
        "detection_method": detection_method,
    })


if __name__ == "__main__":
    app.run(debug=True, port=5000)
