"""
Realer Estate — Compositor (canonical)
=======================================
Direct translation of the exact inline script that produced
the verified working subway output.

Three fixes:
1. S>150 HSV threshold — eliminates background false positives
2. extreme_quad_corners() on convex hull — always exactly 4 points
3. RANSAC line fitting — robust to notch outliers, rounded corners
"""

import cv2
import numpy as np
import json
import argparse
import sys
from pathlib import Path


def get_clean_mask(scene):
    h, w = scene.shape[:2]
    hsv = cv2.cvtColor(scene, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(hsv, np.array([40, 60, 60]), np.array([95, 255, 255]))
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (11, 11))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, k, iterations=2)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN,  k, iterations=1)
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    margin = 8
    valid = [c for c in contours if cv2.contourArea(c) > 500 and
             not (cv2.boundingRect(c)[0] <= margin or
                  cv2.boundingRect(c)[1] <= margin or
                  cv2.boundingRect(c)[0] + cv2.boundingRect(c)[2] >= w - margin or
                  cv2.boundingRect(c)[1] + cv2.boundingRect(c)[3] >= h - margin)]
    if not valid:
        valid = [max(contours, key=cv2.contourArea)] if contours else []
    if not valid:
        return None, None
    best = max(valid, key=cv2.contourArea)
    clean = np.zeros((h, w), dtype=np.uint8)
    cv2.drawContours(clean, [best], -1, 255, -1)
    return clean, best


def extreme_quad_corners(contour):
    hull = cv2.convexHull(contour).reshape(-1, 2).astype(np.float32)
    s = hull[:, 0] + hull[:, 1]
    d = hull[:, 0] - hull[:, 1]
    return np.array([hull[np.argmin(s)], hull[np.argmax(d)],
                     hull[np.argmax(s)], hull[np.argmin(d)]], dtype=np.float32)


def fit_line_ransac(pts, iterations=200, threshold=2.0):
    if len(pts) < 4:
        return None
    pts = pts.astype(np.float64)
    best_line = None
    best_inliers = 0
    for _ in range(iterations):
        idx = np.random.choice(len(pts), 2, replace=False)
        p1, p2 = pts[idx[0]], pts[idx[1]]
        d = p2 - p1
        norm = np.linalg.norm(d)
        if norm < 1e-9:
            continue
        nx, ny = -d[1] / norm, d[0] / norm
        c = -(nx * p1[0] + ny * p1[1])
        dists = np.abs(pts[:, 0] * nx + pts[:, 1] * ny + c)
        inliers = np.sum(dists < threshold)
        if inliers > best_inliers:
            best_inliers = inliers
            inlier_pts = pts[dists < threshold]
            if len(inlier_pts) >= 2:
                line = cv2.fitLine(inlier_pts.astype(np.float32).reshape(-1, 1, 2),
                                   cv2.DIST_L2, 0, 0.01, 0.01).flatten()
                vx, vy, x0, y0 = line
                nnx, nny = -float(vy), float(vx)
                nnorm = np.sqrt(nnx * nnx + nny * nny)
                if nnorm > 1e-9:
                    nnx, nny = nnx / nnorm, nny / nnorm
                    best_line = (nnx, nny, float(-(nnx * x0 + nny * y0)))
    return best_line


def _line_from_pts(p1, p2):
    d = p2.astype(np.float64) - p1.astype(np.float64)
    n = np.array([-d[1], d[0]])
    norm = np.linalg.norm(n)
    if norm < 1e-9:
        return None
    n /= norm
    return float(n[0]), float(n[1]), float(-n @ p1.astype(np.float64))


def _intersect(l1, l2):
    if l1 is None or l2 is None:
        return None
    a1, b1, c1 = l1
    a2, b2, c2 = l2
    det = a1 * b2 - a2 * b1
    if abs(det) < 1e-9:
        return None
    return np.array([(b1 * c2 - b2 * c1) / det,
                     (a2 * c1 - a1 * c2) / det], dtype=np.float32)


def subpixel_refine_ransac(mask, rough_corners):
    k3 = np.ones((3, 3), np.uint8)
    boundary = cv2.bitwise_and(mask, cv2.bitwise_not(cv2.erode(mask, k3)))
    ys, xs = np.where(boundary > 0)
    bpts = np.column_stack([xs.astype(np.float64), ys.astype(np.float64)])
    if len(bpts) < 40:
        return rough_corners
    tl, tr, br, bl = rough_corners
    rough_lines = [_line_from_pts(tl, tr), _line_from_pts(tr, br),
                   _line_from_pts(br, bl), _line_from_pts(bl, tl)]

    def perp_dist(pts, line):
        if line is None:
            return np.full(len(pts), 1e9)
        a, b, c = line
        return np.abs(pts[:, 0] * a + pts[:, 1] * b + c)

    dists = np.column_stack([perp_dist(bpts, l) for l in rough_lines])
    assignments = np.argmin(dists, axis=1)
    fitted = []
    names = ['top', 'right', 'bottom', 'left']
    for i in range(4):
        ep = bpts[assignments == i]
        fl = fit_line_ransac(ep) if len(ep) >= 8 else None
        print(f"  {names[i]}: {len(ep)} px")
        fitted.append(fl if fl else rough_lines[i])
    corners = [_intersect(fitted[0], fitted[3]), _intersect(fitted[0], fitted[1]),
               _intersect(fitted[2], fitted[1]), _intersect(fitted[2], fitted[3])]
    if any(c is None for c in corners):
        return rough_corners
    refined = np.array(corners, dtype=np.float32)
    shift = float(np.abs(refined - rough_corners).max())
    print(f"  RANSAC max shift: {shift:.2f}px")
    if shift > 50:
        print("  Keeping rough corners")
        return rough_corners
    return refined


def detect_green_corners(scene):
    """
    Full pipeline. Returns (corners float32 (4,2), blend_mask uint8)
    or (None, None).
    """
    mask, contour = get_clean_mask(scene)
    if mask is None:
        return None, None
    print(f"[MASK] Green region: {np.sum(mask>0):,} px²")
    rough = extreme_quad_corners(contour)
    print(f"[CORNERS] Rough:   TL={rough[0].astype(int)} TR={rough[1].astype(int)} "
          f"BR={rough[2].astype(int)} BL={rough[3].astype(int)}")
    refined = subpixel_refine_ransac(mask, rough)
    print(f"[CORNERS] Refined: TL={refined[0].astype(int)} TR={refined[1].astype(int)} "
          f"BR={refined[2].astype(int)} BL={refined[3].astype(int)}")
    return refined, mask


def composite(scene, ui, corners, feather=3, blend_mask=None):
    sh, sw = scene.shape[:2]
    uh, uw = ui.shape[:2]
    src = np.array([[0, 0], [uw-1, 0], [uw-1, uh-1], [0, uh-1]], dtype=np.float32)
    M = cv2.getPerspectiveTransform(src, corners)
    warped = cv2.warpPerspective(ui, M, (sw, sh),
                                  flags=cv2.INTER_LANCZOS4,
                                  borderMode=cv2.BORDER_REPLICATE)
    if blend_mask is not None:
        mask_hard = blend_mask
    else:
        mask_hard = np.zeros((sh, sw), dtype=np.uint8)
        cv2.fillConvexPoly(mask_hard, corners.astype(np.int32), 255)
    mf = cv2.GaussianBlur(mask_hard.astype(np.float32) / 255.0, (7, 7), 2)
    alpha = mf[..., np.newaxis]
    result = scene.astype(np.float32) * (1 - alpha) + warped.astype(np.float32) * alpha
    return np.clip(result, 0, 255).astype(np.uint8)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scene",   required=True)
    ap.add_argument("--ui",      required=True)
    ap.add_argument("--output",  required=True)
    ap.add_argument("--corners", default=None)
    ap.add_argument("--debug",   action="store_true")
    args = ap.parse_args()

    scene = cv2.imread(args.scene)
    ui    = cv2.imread(args.ui)
    if scene is None: sys.exit(f"[ERROR] Cannot load: {args.scene}")
    if ui    is None: sys.exit(f"[ERROR] Cannot load: {args.ui}")

    sh, sw = scene.shape[:2]
    uh, uw = ui.shape[:2]
    print(f"\n── Compositor ──")
    print(f"[LOAD] Scene: {sw}x{sh} | UI: {uw}x{uh}")

    corners_path = args.corners or str(Path(args.scene).with_suffix("")) + "_corners.json"

    blend_mask = None
    if Path(corners_path).exists():
        with open(corners_path) as f:
            data = json.load(f)
        corners = np.array(data["corners"], dtype=np.float32)
        _, blend_mask = get_clean_mask(scene)
        print(f"[CORNERS] Loaded from {corners_path}")
    else:
        corners, blend_mask = detect_green_corners(scene)
        if corners is None:
            sys.exit("[ERROR] No green screen detected")
        with open(corners_path, "w") as f:
            json.dump({"corners": corners.tolist(), "image_size": [sw, sh]}, f, indent=2)
        print(f"[SAVED] Corners → {corners_path}")

    result = composite(scene, ui, corners, blend_mask=blend_mask)

    if args.debug:
        vis = scene.copy()
        cv2.polylines(vis, [corners.astype(np.int32)], True, (0, 255, 0), 3)
        for pt, lbl in zip(corners, ['TL', 'TR', 'BR', 'BL']):
            cv2.circle(vis, tuple(pt.astype(int)), 8, (0, 0, 255), -1)
            cv2.putText(vis, f"{lbl}({pt[0]:.0f},{pt[1]:.0f})",
                        (int(pt[0])+8, int(pt[1])+5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
        debug_path = str(Path(args.output).with_suffix("")) + "_corners.png"
        cv2.imwrite(debug_path, vis)

    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(args.output, result)
    print(f"[SAVED] {args.output}")


if __name__ == "__main__":
    main()
