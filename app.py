"""
Green Screen Phone Compositor — Flask Backend
Accepts two images, detects phone screen corners via Claude Vision,
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
import vision_corner_detector as vcd

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

    # ── Corner detection (cached per scene image) ────────────────────────────
    scene_hash = hashlib.sha256(scene_bytes).hexdigest()
    cache_path = CACHE_DIR / f"{scene_hash}.json"

    cached = cache_path.exists()

    if cached:
        try:
            with open(cache_path) as f:
                corners_data = json.load(f)
        except (json.JSONDecodeError, OSError) as exc:
            cached = False  # corrupt cache — re-detect
            corners_data = None

    if not cached:
        # Write scene to a temp file so vision_corner_detector can read it
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

        # Persist to cache
        with open(cache_path, "w") as f:
            json.dump(corners_data, f, indent=2)

    # ── Load images ──────────────────────────────────────────────────────────
    scene_arr = cv2.imdecode(np.frombuffer(scene_bytes, np.uint8), cv2.IMREAD_COLOR)
    ui_arr = cv2.imdecode(np.frombuffer(ui_bytes, np.uint8), cv2.IMREAD_COLOR)

    if scene_arr is None:
        return jsonify({"error": "Could not decode the scene image."}), 400
    if ui_arr is None:
        return jsonify({"error": "Could not decode the UI image."}), 400

    # ── Dimension sanity check ───────────────────────────────────────────────
    # Corners were detected on the full-resolution original; verify the scene
    # uploaded now is the same resolution so coordinates are in the right space.
    sh, sw = scene_arr.shape[:2]
    stored_w, stored_h = corners_data.get("image_size", [sw, sh])
    if sw != stored_w or sh != stored_h:
        print(f"[WARN] Dimension mismatch: detection was on {stored_w}x{stored_h}, "
              f"composite scene is {sw}x{sh}. Scaling corners.")
        sx = sw / stored_w
        sy = sh / stored_h
        corners_data["corners"] = [
            [int(pt[0] * sx), int(pt[1] * sy)]
            for pt in corners_data["corners"]
        ]

    corners = np.array(corners_data["corners"], dtype=np.float32)

    # ── Debug quad: draw detected polygon on scene copy ──────────────────────
    debug_arr = scene_arr.copy()
    quad = corners.astype(np.int32)
    cv2.polylines(debug_arr, [quad], isClosed=True, color=(0, 255, 0), thickness=3)
    for pt, label in zip(quad, ["TL", "TR", "BR", "BL"]):
        cv2.circle(debug_arr, tuple(pt), 10, (0, 0, 255), -1)
        cv2.putText(debug_arr, label, (int(pt[0]) + 12, int(pt[1]) + 6),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 255), 2)

    # ── Composite ────────────────────────────────────────────────────────────
    try:
        result = compositor.composite(scene_arr, ui_arr, corners)
    except Exception as exc:
        return jsonify({"error": f"Compositor error: {exc}"}), 500

    # ── Encode results ───────────────────────────────────────────────────────
    import base64

    def _encode_png(arr: np.ndarray) -> str:
        ok, buf = cv2.imencode(".png", arr)
        if not ok:
            raise RuntimeError("cv2.imencode failed")
        return base64.b64encode(buf.tobytes()).decode()

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
        }
    )


if __name__ == "__main__":
    if not os.environ.get("ANTHROPIC_API_KEY"):
        print("[WARN] ANTHROPIC_API_KEY is not set — Vision API calls will fail.")
    app.run(debug=True, port=5000)
