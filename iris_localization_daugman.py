#!/usr/bin/env python3
"""
iris_localization_daugman.py

Use Daugman's integro-differential operator (brute-force) to find pupil and iris circles.

Dependencies:
    pip install numpy opencv-python

Example:
    python iris_localization_daugman.py \
        --image eye.jpg \
        --inner-r-min 20 --inner-r-max 60 \
        --outer-r-min 60 --outer-r-max 120 \
        --center-range 30 --step 2 --res 360 \
        --output overlay.png
"""
import cv2
import numpy as np
import argparse

def compute_gradient_magnitude(gray):
    gx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
    gy = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
    return np.hypot(gx, gy)

def find_best_circle(grad, cx0, cy0, crange, rmin, rmax, step, num_angles):
    thetas = np.linspace(0, 2*np.pi, num_angles, endpoint=False)
    best_val, best = -1, None
    h, w = grad.shape

    for x in range(cx0-step, cx0+step+1, step):
        if x < 0 or x >= w: continue
        for y in range(cy0-step, cy0+step+1, step):
            if y < 0 or y >= h: continue
            for r in range(rmin, rmax+1):
                # sample points on the circle
                xs = (x + r * np.cos(thetas)).astype(int)
                ys = (y + r * np.sin(thetas)).astype(int)
                # mask out-of-bounds
                valid = (xs>=0)&(xs<w)&(ys>=0)&(ys<h)
                val = grad[ys[valid], xs[valid]].sum()
                if val > best_val:
                    best_val, best = val, (x, y, r)
    return best

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--image",      required=True, help="Input eye image")
    p.add_argument("--inner-r-min", type=int, default=20, help="min pupil radius")
    p.add_argument("--inner-r-max", type=int, default=60, help="max pupil radius")
    p.add_argument("--outer-r-min", type=int, default=60, help="min iris radius")
    p.add_argument("--outer-r-max", type=int, default=120,help="max iris radius")
    p.add_argument("--center-range",type=int, default=30, help="±px around center to search")
    p.add_argument("--step",        type=int, default=2,  help="search step for center coords")
    p.add_argument("--res",         type=int, default=360,help="number of angles to sample on each circle")
    p.add_argument("--output",      default="overlay.png", help="where to save overlay")
    args = p.parse_args()

    # load & preprocess
    img = cv2.imread(args.image)
    if img is None:
        raise FileNotFoundError(f"Couldn’t load {args.image}")
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (7,7), 0)

    # gradient magnitude
    grad = compute_gradient_magnitude(gray)

    # assume center ~ image center
    cy0, cx0 = gray.shape[0]//2, gray.shape[1]//2

    # find pupil (inner) circle
    print("Searching for pupil boundary...")
    xi, yi, ri = find_best_circle(
        grad, cx0, cy0,
        args.center_range,
        args.inner_r_min,
        args.inner_r_max,
        args.step,
        args.res
    )
    print(f"  → pupil:  center=({xi},{yi}), radius={ri}")

    # find iris (outer) circle
    print("Searching for iris boundary...")
    xo, yo, ro = find_best_circle(
        grad, xi, yi,               # refine around pupil center
        args.center_range,
        args.outer_r_min,
        args.outer_r_max,
        args.step,
        args.res
    )
    print(f"  → iris:   center=({xo},{yo}), radius={ro}")

    # draw results
    overlay = img.copy()
    cv2.circle(overlay, (xi, yi), ri, (0,255,0), 2)  # pupil in green
    cv2.circle(overlay, (xo, yo), ro, (0,0,255), 2)  # iris in red

    cv2.imwrite(args.output, overlay)
    print(f"Overlay saved to {args.output}")

if __name__ == "__main__":
    main()
