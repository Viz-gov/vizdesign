# splinterframes_selective_glitch.py
# Detect objects (edge→contour→bbox) and invert/glitch only inside each box.

import cv2
import numpy as np
import sys
from pathlib import Path
import random

# Get paths from command line arguments
if len(sys.argv) < 3:
    print("Usage: python splinterframes.py <input_video> <output_video>")
    sys.exit(1)

INPUT_PATH = sys.argv[1]
OUTPUT_PATH = sys.argv[2]

# ---------- Detection (same spirit as your line-tracing settings) ----------
USE_CLAHE        = False
CLAHE_CLIP       = 2.0
CLAHE_TILE       = 8
BLUR_KSIZE       = 3          # 0/1 to disable; keep small to preserve edges
CANNY_SIGMA      = 0.33       # lower => more edges
CANNY_MIN_ADD    = 0
CANNY_MAX_ADD    = 0
DILATE_KSIZE     = 3          # thicken edges just a bit
CLOSE_KSIZE      = 3          # close tiny gaps so contours are closed
RETR_MODE        = "external" # or "list"
MIN_CONTOUR_AREA = 20         # drop tiny specks
BORDER_MARGIN_PX = 6          # drop contours touching frame edge
MAX_AREA_FRAC    = 0.25       # drop absurd giants (>25% of frame)
MERGE_OVERLAP    = True
IOU_THRESHOLD    = 0.20

# ---------- Effect inside boxes ----------
INVERT_ONLY      = True       # True = just invert colors in ROI
INCLUDE_GLITCH   = False      # set True to add light glitch within ROI
RGB_SHIFT_MAX    = 5          # px shift per color channel (ROI only)
DISPLACE_STRENGTH= 8          # ROI noise-warp strength (px)
DRAW_BOXES       = False      # outline boxes for debugging
BOX_COLOR        = (255,255,255)
BOX_THICKNESS    = 2
# --------------------------------------------------------------------------

def auto_canny_thresholds(gray, sigma):
    v = np.median(gray)
    lo = int(max(0, (1.0 - sigma) * v)) + CANNY_MIN_ADD
    hi = int(min(255,(1.0 + sigma) * v)) + CANNY_MAX_ADD
    lo = max(0, min(hi - 1, lo))
    hi = max(lo + 1, min(255, hi))
    return lo, hi

def preprocess_gray(frame):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    if USE_CLAHE:
        clahe = cv2.createCLAHE(clipLimit=CLAHE_CLIP, tileGridSize=(CLAHE_TILE, CLAHE_TILE))
        gray = clahe.apply(gray)
    if BLUR_KSIZE and BLUR_KSIZE >= 3 and BLUR_KSIZE % 2 == 1:
        gray = cv2.medianBlur(gray, BLUR_KSIZE)
    return gray

def odd_or_zero(k):
    if not k or k <= 1: return 0
    return k if k % 2 == 1 else k + 1

def postprocess_edges(edges):
    k = odd_or_zero(DILATE_KSIZE)
    if k:
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k))
        edges = cv2.dilate(edges, kernel, iterations=1)
    k = odd_or_zero(CLOSE_KSIZE)
    if k:
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k))
        edges = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, kernel, iterations=1)
    return edges

def contours_mode():
    return cv2.RETR_LIST if RETR_MODE.lower()=="list" else cv2.RETR_EXTERNAL

def touches_border(cnt, W, H, m):
    x = cnt[:,0,0]; y = cnt[:,0,1]
    return (x <= m).any() or (y <= m).any() or (x >= W-1-m).any() or (y >= H-1-m).any()

def iou(a, b):
    ax, ay, aw, ah = a; bx, by, bw, bh = b
    ax2, ay2 = ax+aw, ay+ah; bx2, by2 = bx+bw, by+bh
    iw = max(0, min(ax2,bx2)-max(ax,bx)); ih = max(0, min(ay2,by2)-max(ay,by))
    inter = iw*ih; union = aw*ah + bw*bh - inter
    return inter/union if union>0 else 0.0

def merge_boxes(boxes, iou_thr):
    if not boxes: return []
    merged = True
    boxes = boxes[:]
    while merged:
        merged = False
        out, used = [], [False]*len(boxes)
        for i in range(len(boxes)):
            if used[i]: continue
            x,y,w,h = boxes[i]
            for j in range(i+1, len(boxes)):
                if used[j]: continue
                if iou((x,y,w,h), boxes[j]) > iou_thr:
                    bx,by,bw,bh = boxes[j]
                    x1,y1 = min(x,bx), min(y,by)
                    x2,y2 = max(x+w, bx+bw), max(y+h, by+bh)
                    x,y,w,h = x1,y1,x2-x1,y2-y1
                    used[j] = True; merged = True
            used[i] = True
            out.append((x,y,w,h))
        boxes = out
    return boxes

# ---- ROI effects ----
def rgb_shift(img, max_shift):
    h,w = img.shape[:2]
    out = img.copy()
    for c in range(3):
        dx = np.random.randint(-max_shift, max_shift+1)
        dy = np.random.randint(-max_shift, max_shift+1)
        M = np.float32([[1,0,dx],[0,1,dy]])
        out[...,c] = cv2.warpAffine(img[...,c], M, (w,h), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT)
    return out

def smooth_noise(h, w, k=51):
    n = np.random.rand(h,w).astype(np.float32)
    k = k if k%2==1 else k+1
    return cv2.GaussianBlur(n,(k,k),0)

def noise_displace(img, strength=8):
    h,w = img.shape[:2]
    n1 = smooth_noise(h,w, max(31, (h//6)|1)); n2 = smooth_noise(h,w, max(31, (w//6)|1))
    flow_x = (n1-0.5)*2*strength; flow_y = (n2-0.5)*2*strength
    gx,gy = np.meshgrid(np.arange(w), np.arange(h))
    map_x = (gx+flow_x).astype(np.float32); map_y = (gy+flow_y).astype(np.float32)
    return cv2.remap(img, map_x, map_y, interpolation=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT)

def apply_effect_in_roi(frame, box):
    x,y,w,h = box
    roi = frame[y:y+h, x:x+w]
    if INVERT_ONLY and not INCLUDE_GLITCH:
        frame[y:y+h, x:x+w] = 255 - roi
        return
    # mild glitch inside ROI
    work = roi
    if INCLUDE_GLITCH:
        work = rgb_shift(work, RGB_SHIFT_MAX)
        work = noise_displace(work, DISPLACE_STRENGTH)
    work = 255 - work if INVERT_ONLY else work
    frame[y:y+h, x:x+w] = work

def main():
    cap = cv2.VideoCapture(INPUT_PATH)
    if not cap.isOpened():
        raise RuntimeError(f"Could not open input: {INPUT_PATH}")

    W  = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    H  = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    if fps <= 1e-3: fps = 30.0

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(OUTPUT_PATH, fourcc, fps, (W, H))
    if not writer.isOpened():
        raise RuntimeError("Could not open VideoWriter.")

    frame_area = W*H

    while True:
        ret, frame = cap.read()
        if not ret: break

        # --- detect boxes (edge → contour) ---
        gray = preprocess_gray(frame)
        lo, hi = auto_canny_thresholds(gray, CANNY_SIGMA)
        edges = cv2.Canny(gray, lo, hi, apertureSize=3, L2gradient=True)
        proc  = postprocess_edges(edges)
        cnts,_ = cv2.findContours(proc, contours_mode(), cv2.CHAIN_APPROX_NONE)

        boxes = []
        for c in cnts:
            if cv2.contourArea(c) < MIN_CONTOUR_AREA: continue
            if touches_border(c, W, H, BORDER_MARGIN_PX): continue
            x,y,w,h = cv2.boundingRect(c)
            if (w*h) > MAX_AREA_FRAC * frame_area: continue
            boxes.append((x,y,w,h))

        if MERGE_OVERLAP:
            boxes = merge_boxes(boxes, IOU_THRESHOLD)

        # --- apply selective effect inside each box ---
        for b in boxes:
            apply_effect_in_roi(frame, b)
            if DRAW_BOXES:
                x,y,w,h = b
                cv2.rectangle(frame, (x,y), (x+w,y+h), BOX_COLOR, BOX_THICKNESS)

        writer.write(frame)

    cap.release()
    writer.release()
    print(f"Done. Wrote {OUTPUT_PATH} ({W}x{H} @ {fps:.3f} fps).")

if __name__ == "__main__":
    if not Path(INPUT_PATH).exists():
        raise SystemExit(f"Input file not found: {INPUT_PATH}")
    main()
