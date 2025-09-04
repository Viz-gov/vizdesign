# dotpad_video.py
# A halftone "dot pad" effect: the frame is sampled on a grid; each cell draws a
# circle whose radius depends on brightness. Supports color or monochrome dots,
# inversion, gamma, and optional grid lines.

import cv2
import numpy as np
import sys
from pathlib import Path

# Get paths from command line arguments
if len(sys.argv) < 3:
    print("Usage: python dotpad.py <input_video> <output_video>")
    sys.exit(1)

INPUT_PATH = sys.argv[1]
OUTPUT_PATH = sys.argv[2]

# -------------------- Tunables --------------------
CELL_PX        = 12     # size of a grid cell in pixels (bigger = fewer dots = faster)
DOT_FILL       = True   # True=filled circles, False=outlined
DOT_THICK      = -1     # -1 = filled; else pixel thickness for outline if DOT_FILL=False
DOT_SCALE      = 0.95   # max radius as fraction of half the cell (0..1)
GAMMA          = 1.0    # >1 emphasizes darks (smaller dots); <1 emphasizes brights

COLOR_MODE     = "color"   # 'color' (sampled per-cell) or 'mono'
MONO_DOT_COLOR = (255, 255, 255)  # BGR when COLOR_MODE='mono'
BG_COLOR       = (0, 0, 0)        # canvas background color

INVERT_BRIGHT  = False    # True -> bright areas become *small* dots (invert mapping)
DRAW_GRID      = False    # thin grid lines over the dots
GRID_COLOR     = (40, 40, 40)
GRID_THICK     = 1
# --------------------------------------------------

def main():
    if not Path(INPUT_PATH).exists():
        raise SystemExit(f"Input not found: {INPUT_PATH}")

    cap = cv2.VideoCapture(INPUT_PATH)
    if not cap.isOpened():
        raise SystemExit("Could not open input video.")

    W = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    H = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    if fps <= 1e-3:
        fps = 30.0

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(OUTPUT_PATH, fourcc, fps, (W, H))
    if not writer.isOpened():
        raise SystemExit("Could not open VideoWriter for output.")

    # Precompute grid centers
    cell = max(2, int(CELL_PX))
    xs = np.arange(cell // 2, W, cell, dtype=np.int32)
    ys = np.arange(cell // 2, H, cell, dtype=np.int32)
    grid_w, grid_h = len(xs), len(ys)
    cx, cy = np.meshgrid(xs, ys)    # shape (grid_h, grid_w)

    # Precompute max radius in pixels for each cell
    r_max = int(min(cell, cell) * 0.5 * DOT_SCALE)
    r_max = max(1, r_max)

    # For sampling, we’ll downscale to grid size
    sample_size = (grid_w, grid_h)

    while True:
        ok, frame = cap.read()
        if not ok:
            break

        # Prepare the blank canvas
        canvas = np.zeros_like(frame)
        canvas[:] = BG_COLOR

        # Downsample for per-cell stats (fast)
        small = cv2.resize(frame, sample_size, interpolation=cv2.INTER_AREA)

        # Luminance (0..1)
        gray = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY).astype(np.float32) / 255.0
        if GAMMA != 1.0:
            gray = np.power(gray, GAMMA)

        # Map luminance -> radius fraction
        if INVERT_BRIGHT:
            frac = gray                    # bright -> large radius if not inverted
        else:
            frac = 1.0 - gray              # bright -> small radius (classic halftone)

        # clamp
        frac = np.clip(frac, 0.0, 1.0)

        # Convert to integer radii
        radii = (r_max * frac).astype(np.int32)

        # Dot colors
        if COLOR_MODE == "color":
            colors = small                 # use sampled color per cell
        else:
            # all dots same color
            colors = np.empty_like(small)
            colors[:] = np.array(MONO_DOT_COLOR, dtype=np.uint8)

        # Draw dots (loop over grid; fast enough for moderate grids)
        # We iterate over rows to keep Python overhead reasonable.
        for j in range(grid_h):
            y = int(ys[j])
            r_row = radii[j]
            col_row = colors[j]
            for i in range(grid_w):
                r = int(r_row[i])
                if r <= 0:
                    continue
                x = int(xs[i])
                color = tuple(int(c) for c in col_row[i])
                thickness = DOT_THICK if DOT_FILL else max(1, DOT_THICK)
                cv2.circle(canvas, (x, y), r, color, thickness, lineType=cv2.LINE_AA)

        # Optional grid overlay
        if DRAW_GRID:
            for x in range(0, W, cell):
                cv2.line(canvas, (x, 0), (x, H), GRID_COLOR, GRID_THICK)
            for y in range(0, H, cell):
                cv2.line(canvas, (0, y), (W, y), GRID_COLOR, GRID_THICK)

        writer.write(canvas)

    cap.release()
    writer.release()
    print(f"Done. Wrote {OUTPUT_PATH} ({W}x{H} @ {fps:.2f} fps).")

if __name__ == "__main__":
    main()
