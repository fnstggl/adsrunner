"""
Realer Estate — Smart Compositor v4
=====================================
FIRST PRINCIPLES APPROACH (how Placeit actually works):

Placeit does NOT use green screen keying at all.
They use PRE-DEFINED corner coordinates stored per-template.
A human (or tool) marks the exact 4 corners of the screen ONCE.
Every future composite just does: warpPerspective(ui, corners) + blend.

This script does the same thing in two modes:

MODE 1 — Interactive (run once per scene):
    python compositor_v4.py --scene scene.png --ui ui.png --output out.png --pick
    → Opens the scene, you click the 4 screen corners (TL, TR, BR, BL)
    → Saves corners to scene_corners.json for reuse
    → Composites and saves output

MODE 2 — Automated (production, reuse saved corners):
    python compositor_v4.py --scene scene.png --ui ui.png --output out.png
    → Loads scene_corners.json automatically
    → Instant composite, no interaction needed

WHY THIS IS CORRECT:
  - Green screen keying is for VIDEO where you don't know the screen position
  - For PHOTOS you shot yourself, you always know the screen position
  - Manual corner picking = pixel-perfect, zero fringe, handles any tilt
  - After picking once, it's fully automated for every new UI you want to composite

DEPENDENCIES: pip install opencv-python numpy
"""

import cv2
import numpy as np
import json
import argparse
import sys
from pathlib import Path


# ── GLOBALS for interactive corner picking ──────────────────────────────────
_corners = []
_display_img = None
_window = "Click 4 screen corners: TL → TR → BR → BL  (r=reset, Enter=confirm)"


def _mouse_callback(event, x, y, flags, param):
    global _corners, _display_img
    if event == cv2.EVENT_LBUTTONDOWN and len(_corners) < 4:
        _corners.append([x, y])
        # Draw dot + label
        vis = param["vis"]
        labels = ["TL", "TR", "BR", "BL"]
        cv2.circle(vis, (x, y), 8, (0, 255, 0), -1)
        cv2.putText(vis, labels[len(_corners)-1], (x+10, y-10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        if len(_corners) > 1:
            pts = np.array(_corners, np.int32)
            cv2.polylines(vis, [pts], len(_corners)==4, (0,255,0), 2)
        cv2.imshow(_window, vis)
        print(f"  Corner {len(_corners)}: ({x}, {y}) — "
              f"{['TL','TR','BR','BL'][len(_corners)-1]}")


def pick_corners_interactively(scene: np.ndarray, corners_path: str) -> np.ndarray:
    """
    Opens an interactive window to click 4 corners.
    Saves to JSON. Returns np.float32 array shape (4,2).
    """
    global _corners

    # Fit scene to screen for picking (max 1000px wide)
    h, w = scene.shape[:2]
    scale = min(1.0, 1000 / w)
    disp_w, disp_h = int(w * scale), int(h * scale)
    display = cv2.resize(scene, (disp_w, disp_h))
    vis = display.copy()

    print(f"\n{'─'*60}")
    print(f"CORNER PICKER")
    print(f"{'─'*60}")
    print(f"Image displayed at {scale:.0%} scale ({disp_w}x{disp_h})")
    print(f"Click the 4 corners of the PHONE SCREEN in this order:")
    print(f"  1. TOP-LEFT")
    print(f"  2. TOP-RIGHT")
    print(f"  3. BOTTOM-RIGHT")
    print(f"  4. BOTTOM-LEFT")
    print(f"Keys: r = reset corners | Enter = confirm & save | q = quit")
    print(f"{'─'*60}\n")

    cv2.namedWindow(_window, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(_window, disp_w, disp_h)
    cv2.setMouseCallback(_window, _mouse_callback, {"vis": vis})
    cv2.imshow(_window, vis)

    while True:
        key = cv2.waitKey(20) & 0xFF
        if key == ord('r'):
            _corners = []
            vis[:] = display[:]
            cv2.imshow(_window, vis)
            print("  [RESET] Click corners again.")
        elif key in [13, 10] and len(_corners) == 4:  # Enter
            break
        elif key == ord('q'):
            cv2.destroyAllWindows()
            sys.exit("Cancelled.")

    cv2.destroyAllWindows()

    # Scale corners back to original resolution
    actual_corners = [[int(x / scale), int(y / scale)] for x, y in _corners]

    # Save
    with open(corners_path, 'w') as f:
        json.dump({"corners": actual_corners, "image_size": [w, h]}, f, indent=2)
    print(f"\n[SAVED] Corners saved to {corners_path}")
    print(f"  TL:{actual_corners[0]}  TR:{actual_corners[1]}")
    print(f"  BR:{actual_corners[2]}  BL:{actual_corners[3]}")

    return np.array(actual_corners, dtype=np.float32)


def load_corners(corners_path: str) -> np.ndarray:
    with open(corners_path) as f:
        data = json.load(f)
    corners = np.array(data["corners"], dtype=np.float32)
    print(f"[CORNERS] Loaded from {corners_path}")
    print(f"  TL:{corners[0].astype(int)}  TR:{corners[1].astype(int)}")
    print(f"  BR:{corners[2].astype(int)}  BL:{corners[3].astype(int)}")
    return corners


def composite(scene: np.ndarray, ui: np.ndarray,
              corners: np.ndarray, feather: int = 3) -> np.ndarray:
    """
    THE CORE — exactly how Placeit works:
    1. corners define the destination quad in scene space
    2. getPerspectiveTransform maps flat UI → that quad
    3. fillConvexPoly makes a clean mask for the exact screen shape
    4. Blend with soft edge
    """
    sh, sw = scene.shape[:2]
    uh, uw = ui.shape[:2]

    # Source corners (flat UI rectangle)
    src = np.array([
        [0,    0   ],
        [uw-1, 0   ],
        [uw-1, uh-1],
        [0,    uh-1],
    ], dtype=np.float32)

    # Destination corners (the phone screen in scene space)
    # Order: TL, TR, BR, BL
    dst = corners

    # Compute perspective transform and warp UI into scene space
    M = cv2.getPerspectiveTransform(src, dst)
    warped = cv2.warpPerspective(
        ui, M, (sw, sh),
        flags=cv2.INTER_LANCZOS4,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=(0, 0, 0)
    )

    # Build a hard mask from the exact screen quad
    mask_hard = np.zeros((sh, sw), dtype=np.uint8)
    quad = corners.astype(np.int32)
    cv2.fillConvexPoly(mask_hard, quad, 255)

    # Optionally feather edges for sub-pixel softness
    if feather > 0:
        mask_f = mask_hard.astype(np.float32) / 255.0
        br = feather * 2 + 1
        mask_f = cv2.GaussianBlur(mask_f, (br, br), feather * 0.5)
        mask = (mask_f * 255).astype(np.uint8)
    else:
        mask = mask_hard

    # Alpha blend
    alpha = mask.astype(np.float32)[..., np.newaxis] / 255.0
    result = (scene.astype(np.float32) * (1 - alpha) +
              warped.astype(np.float32) * alpha)
    return np.clip(result, 0, 255).astype(np.uint8)


def add_screen_glare(result: np.ndarray, corners: np.ndarray,
                     opacity: float = 0.08) -> np.ndarray:
    """
    Optional: add a subtle specular glare over the screen
    to make the composite feel like real glass.
    """
    if opacity <= 0:
        return result
    h, w = result.shape[:2]
    tl, tr, br, bl = corners

    glare = np.zeros((h, w), dtype=np.float32)
    cx = int(tl[0] * 0.6 + tr[0] * 0.4)
    cy = int(tl[1] * 0.6 + bl[1] * 0.4)
    ax = max(int(np.linalg.norm(tr - tl) * 0.45), 1)
    ay = max(int(np.linalg.norm(bl - tl) * 0.3),  1)

    cv2.ellipse(glare, (cx, cy), (ax, ay), -20, 0, 360, 1.0, -1)
    glare = cv2.GaussianBlur(glare, (0, 0), ax * 0.6)

    # Only apply glare inside the screen quad
    screen_mask = np.zeros((h, w), dtype=np.float32)
    cv2.fillConvexPoly(screen_mask, corners.astype(np.int32), 1.0)
    glare *= screen_mask

    white = np.ones_like(result, dtype=np.float32) * 255
    a = (glare * opacity)[..., np.newaxis]
    out = result.astype(np.float32) * (1 - a) + white * a
    return np.clip(out, 0, 255).astype(np.uint8)


def main():
    ap = argparse.ArgumentParser(description="Smart Compositor v4 — Homography-based")
    ap.add_argument("--scene",   required=True, help="Scene image (phone photo)")
    ap.add_argument("--ui",      required=True, help="UI screenshot to composite")
    ap.add_argument("--output",  required=True, help="Output path")
    ap.add_argument("--corners", default=None,
                    help="Path to corners JSON (default: <scene>_corners.json)")
    ap.add_argument("--pick",    action="store_true",
                    help="Force interactive corner picking (even if JSON exists)")
    ap.add_argument("--feather", type=int, default=3,
                    help="Edge feather px (default 3)")
    ap.add_argument("--glare",   type=float, default=0.0,
                    help="Screen glare opacity 0-1 (default 0, add in post)")
    args = ap.parse_args()

    # Load images
    scene = cv2.imread(args.scene)
    ui    = cv2.imread(args.ui)
    if scene is None: sys.exit(f"[ERROR] Cannot load scene: {args.scene}")
    if ui    is None: sys.exit(f"[ERROR] Cannot load UI: {args.ui}")

    sh, sw = scene.shape[:2]
    uh, uw = ui.shape[:2]
    print(f"\n── Smart Compositor v4 ──")
    print(f"[LOAD] Scene: {sw}x{sh} | UI: {uw}x{uh}")
    if uw > uh:
        print(f"[WARN] UI appears to be landscape — expected portrait for phone")

    # Corners file
    corners_path = args.corners or str(Path(args.scene).with_suffix("")) + "_corners.json"

    # Pick or load corners
    if args.pick or not Path(corners_path).exists():
        print(f"\n[MODE] Interactive corner picking")
        corners = pick_corners_interactively(scene, corners_path)
    else:
        print(f"\n[MODE] Using saved corners")
        corners = load_corners(corners_path)

    # Composite
    print(f"\n[COMPOSITE] Warping UI into screen quad...")
    result = composite(scene, ui, corners, args.feather)

    # Optional glare
    if args.glare > 0:
        print(f"[GLARE] Adding {args.glare:.0%} screen glare...")
        result = add_screen_glare(result, corners, args.glare)

    # Save
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(args.output, result)
    print(f"\n[SAVED] {args.output}")
    print(f"\n── Workflow ──────────────────────────────────────────────")
    print(f"  Next run (same scene, new UI):")
    print(f"  python compositor_v4.py --scene {args.scene} --ui NEW_UI.png --output new_out.png")
    print(f"  (corners auto-loaded from {corners_path})")
    print(f"──────────────────────────────────────────────────────────\n")


if __name__ == "__main__":
    main()
