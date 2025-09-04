# line_tracing_to_bboxes.py
import cv2
import numpy as np
import sys
from pathlib import Path

# Get paths from command line arguments
if len(sys.argv) < 3:
    print("Usage: python blob_detect.py <input_video> <output_video>")
    sys.exit(1)

INPUT_PATH = sys.argv[1]
OUTPUT_PATH = sys.argv[2]

# -------- Tunables (kept close to your line-tracing setup) --------
USE_CLAHE      = False
CLAHE_CLIP     = 2.0
CLAHE_TILE     = 8
BLUR_KSIZE     = 3

# Canny (auto thresholds from image statistics)
CANNY_SIGMA    = 0.33
CANNY_MIN_ADD  = 0
CANNY_MAX_ADD  = 0

# Edge → box post-processing
DILATE_KSIZE   = 3        # thicken edges a bit (odd). 0/1 to disable.
CLOSE_KSIZE    = 3        # slightly smaller close to avoid giant loops
RETR_MODE      = "external"
MIN_CONTOUR_AREA = 20

# New: outlier/border filters
BORDER_MARGIN_PX     = 6     # drop contours (and boxes) touching the frame border within this margin
MAX_FRAME_AREA_FRAC  = 0.25  # drop boxes bigger than 25% of the frame area

# Merge overlapping boxes
MERGE_OVERLAP  = True
IOU_THRESHOLD  = 0.20

# Drawing
BOX_COLOR      = (255, 255, 255)
BOX_THICKNESS  = 2
SHOW_AREA      = True
TEXT_SCALE     = 0.6
TEXT_THICK     = 2
# ---------------------------------------------------------------

def auto_canny_thresholds(gray, sigma=CANNY_SIGMA):
    v = np.median(gray)
    lower = int(max(0, (1.0 - sigma) * v)) + CANNY_MIN_ADD
    upper = int(min(255, (1.0 + sigma) * v)) + CANNY_MAX_ADD
    lower = max(0, min(upper - 1, lower))
    upper = max(lower + 1, min(255, upper))
    return lower, upper

def preprocess_gray(frame):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    if USE_CLAHE:
        clahe = cv2.createCLAHE(clipLimit=CLAHE_CLIP, tileGridSize=(CLAHE_TILE, CLAHE_TILE))
        gray = clahe.apply(gray)
    if BLUR_KSIZE and BLUR_KSIZE >= 3 and BLUR_KSIZE % 2 == 1:
        gray = cv2.medianBlur(gray, BLUR_KSIZE)
    return gray

def odd_or_disable(k):
    if not k or k <= 1:
        return 0
    return k if k % 2 == 1 else k + 1

def postprocess_edges(edges):
    k = odd_or_disable(DILATE_KSIZE)
    if k:
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k))
        edges = cv2.dilate(edges, kernel, iterations=1)
    k = odd_or_disable(CLOSE_KSIZE)
    if k:
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k))
        edges = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, kernel, iterations=1)
    return edges

def contours_mode():
    m = RETR_MODE.lower()
    if m == "list": return cv2.RETR_LIST
    return cv2.RETR_EXTERNAL

def iou(a, b):
    ax, ay, aw, ah = a
    bx, by, bw, bh = b
    ax2, ay2 = ax + aw, ay + ah
    bx2, by2 = bx + bw, by + bh
    inter_x1, inter_y1 = max(ax, bx), max(ay, by)
    inter_x2, inter_y2 = min(ax2, bx2), min(ay2, by2)
    iw, ih = max(0, inter_x2 - inter_x1), max(0, inter_y2 - inter_y1)
    inter = iw * ih
    union = aw * ah + bw * bh - inter
    return inter / union if union > 0 else 0.0

def is_border_contour(cnt, W, H, margin=BORDER_MARGIN_PX):
    # If any contour point lies within 'margin' px of any image border, treat as border contour
    x = cnt[:, 0, 0]
    y = cnt[:, 0, 1]
    return (x <= margin).any() or (y <= margin).any() or (x >= W - 1 - margin).any() or (y >= H - 1 - margin).any()

def merge_boxes(boxes, iou_thr, frame_area):
    if not boxes:
        return []
    boxes = boxes[:]
    merged = True
    while merged:
        merged = False
        out = []
        used = [False]*len(boxes)
        for i in range(len(boxes)):
            if used[i]:
                continue
            mx, my, mw, mh = boxes[i]
            # don't let huge boxes act as magnets in merging
            if (mw * mh) > MAX_FRAME_AREA_FRAC * frame_area:
                used[i] = True
                out.append((mx, my, mw, mh))
                continue
            for j in range(i+1, len(boxes)):
                if used[j]:
                    continue
                if iou((mx, my, mw, mh), boxes[j]) > iou_thr:
                    bx, by, bw, bh = boxes[j]
                    x1, y1 = min(mx, bx), min(my, by)
                    x2, y2 = max(mx+mw, bx+bw), max(my+mh, by+bh)
                    mx, my, mw, mh = x1, y1, x2 - x1, y2 - y1
                    used[j] = True
                    merged = True
            used[i] = True
            out.append((mx, my, mw, mh))
        boxes = out
    return boxes

def main():
    cap = cv2.VideoCapture(INPUT_PATH)
    if not cap.isOpened():
        raise RuntimeError(f"Could not open input: {INPUT_PATH}")

    W  = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    H  = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    if fps <= 1e-3: fps = 30.0
    frame_area = W * H

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(OUTPUT_PATH, fourcc, fps, (W, H))
    if not writer.isOpened():
        raise RuntimeError("Could not open VideoWriter. Try 'avc1' or 'H264'.")

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        gray = preprocess_gray(frame)
        lo, hi = auto_canny_thresholds(gray, CANNY_SIGMA)
        edges = cv2.Canny(gray, lo, hi, apertureSize=3, L2gradient=True)
        proc  = postprocess_edges(edges)

        # Contours
        cnts, _ = cv2.findContours(proc, contours_mode(), cv2.CHAIN_APPROX_NONE)

        # Build boxes with filtering
        boxes = []
        for c in cnts:
            if cv2.contourArea(c) < MIN_CONTOUR_AREA:
                continue
            # Drop contours that touch the image border (within margin)
            if is_border_contour(c, W, H, BORDER_MARGIN_PX):
                continue
            x, y, w, h = cv2.boundingRect(c)
            # Drop absurdly large boxes
            if (w * h) > MAX_FRAME_AREA_FRAC * frame_area:
                continue
            boxes.append((x, y, w, h))

        # Merge overlaps after filtering (and never merge into huge outliers)
        if MERGE_OVERLAP:
            boxes = merge_boxes(boxes, IOU_THRESHOLD, frame_area)

        # Draw
        for (x, y, w, h) in boxes:
            cv2.rectangle(frame, (x, y), (x + w, y + h), BOX_COLOR, BOX_THICKNESS)
            if SHOW_AREA:
                area_est = w * h
                cv2.putText(frame, str(int(area_est)), (x, max(0, y - 8)),
                            cv2.FONT_HERSHEY_SIMPLEX, TEXT_SCALE, BOX_COLOR, TEXT_THICK, cv2.LINE_AA)

        writer.write(frame)

    cap.release()
    writer.release()
    print(f"Done. Wrote {OUTPUT_PATH} ({W}x{H} @ {fps:.3f} fps).")

if __name__ == "__main__":
    if not Path(INPUT_PATH).exists():
        raise SystemExit(f"Input file not found: {INPUT_PATH}")
    main()
