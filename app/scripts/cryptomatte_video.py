# pseudo_cryptomatte_video.py
# Creates a Cryptomatte-style ID matte from regular footage via color clustering.

import cv2
import numpy as np
import sys
from pathlib import Path

# Get paths from command line arguments
if len(sys.argv) < 3:
    print("Usage: python cryptomatte_video.py <input_video> <output_video>")
    sys.exit(1)

INPUT_PATH = sys.argv[1]
OUTPUT_PATH = sys.argv[2]

# -------- Tunables --------
K_CLUSTERS = 6          # number of IDs (increase for more regions)
BLUR_KSIZE = 3          # small denoise before clustering (0/1 disables)
MAX_ITERS  = 5          # Lloyd iterations per frame (keep small for speed)
DOWNSCALE  = 1.0        # e.g., 0.75 for faster processing
COLORSPACE = "lab"      # 'lab' (recommended) or 'hsv' or 'rgb'
DRAW_LABELS = True      # overlay cluster indices
FONT_SCALE  = 0.6
# --------------------------

# Nice distinct colors (BGR)
PALETTE = np.array([
    ( 40, 180, 240), ( 80, 220, 120), (240, 180,  55),
    (240, 120, 200), (120, 120, 240), (120, 200, 240),
    (100, 100, 100), ( 30, 220, 220), ( 40,  80, 240),
    (240, 220, 100), (220, 160,  40), ( 40, 160, 220)
], dtype=np.uint8)

def to_space(img_bgr):
    if COLORSPACE == "lab":
        return cv2.cvtColor(img_bgr, cv2.COLOR_BGR2LAB)
    if COLORSPACE == "hsv":
        return cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV)
    return img_bgr.copy()

def from_space(img_space):
    if COLORSPACE == "lab":
        return cv2.cvtColor(img_space, cv2.COLOR_LAB2BGR)
    if COLORSPACE == "hsv":
        return cv2.cvtColor(img_space, cv2.COLOR_HSV2BGR)
    return img_space

def lloyd_iter(X, centers):
    """
    One Lloyd iteration:
      - assign labels by nearest center
      - recompute centers
    X: (N, C) float32
    centers: (K, C) float32
    """
    # distances: (N, K)
    # compute in chunks to save RAM on large frames
    K = centers.shape[0]
    N = X.shape[0]
    chunk = 200_000
    labels = np.empty(N, np.int32)
    for s in range(0, N, chunk):
        e = min(N, s + chunk)
        d = np.linalg.norm(X[s:e, None, :] - centers[None, :, :], axis=2)
        labels[s:e] = np.argmin(d, axis=1).astype(np.int32)

    # recompute centers
    new_centers = centers.copy()
    for k in range(K):
        idx = np.where(labels == k)[0]
        if idx.size > 0:
            new_centers[k] = X[idx].mean(axis=0)
    return labels, new_centers

def cluster_frame(img_bgr, centers_prev=None):
    img = img_bgr
    if BLUR_KSIZE and BLUR_KSIZE >= 3 and BLUR_KSIZE % 2 == 1:
        img = cv2.medianBlur(img, BLUR_KSIZE)

    Xspace = to_space(img).reshape(-1, 3).astype(np.float32)

    # init centers: k-means++ on the first frame; else reuse previous
    if centers_prev is None:
        # OpenCV kmeans++ init for first frame
        criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 0.1)
        _compact, labels0, centers = cv2.kmeans(
            Xspace, K_CLUSTERS, None, criteria, 1, cv2.KMEANS_PP_CENTERS
        )
        labels = labels0.flatten().astype(np.int32)
    else:
        centers = centers_prev.copy()
        for _ in range(MAX_ITERS):
            labels, new_centers = lloyd_iter(Xspace, centers)
            if np.allclose(new_centers, centers):
                break
            centers = new_centers

    labels = labels.reshape(img.shape[:2]) if centers_prev is not None else labels.reshape(img.shape[:2])

    # color the labels
    colors = PALETTE[:K_CLUSTERS]
    matte = colors[labels % len(colors)]
    return matte, centers, labels

def draw_labels_on_matte(matte, labels):
    # place small indices at superpixel grid to reduce clutter
    h, w = labels.shape
    out = matte.copy()
    step = max(32, min(h, w)//20)
    for y in range(step//2, h, step):
        for x in range(step//2, w, step):
            k = int(labels[y, x])
            cv2.putText(out, str(k), (x, y),
                        cv2.FONT_HERSHEY_SIMPLEX, FONT_SCALE, (255,255,255), 2, cv2.LINE_AA)
    return out

def main():
    if not Path(INPUT_PATH).exists():
        raise SystemExit(f"Input not found: {INPUT_PATH}")

    cap = cv2.VideoCapture(INPUT_PATH)
    if not cap.isOpened():
        raise SystemExit("Could not open input video.")

    W_in  = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    H_in  = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps   = cap.get(cv2.CAP_PROP_FPS) or 30.0
    if fps <= 1e-3: fps = 30.0

    if DOWNSCALE != 1.0:
        W = int(W_in * DOWNSCALE)
        H = int(H_in * DOWNSCALE)
    else:
        W, H = W_in, H_in

    writer = cv2.VideoWriter(OUTPUT_PATH,
                             cv2.VideoWriter_fourcc(*"mp4v"),
                             fps, (W_in, H_in))
    if not writer.isOpened():
        raise SystemExit("Could not open VideoWriter for output.")

    centers = None
    while True:
        ok, frame = cap.read()
        if not ok:
            break

        if DOWNSCALE != 1.0:
            small = cv2.resize(frame, (W, H), interpolation=cv2.INTER_AREA)
        else:
            small = frame

        matte_small, centers, labels = cluster_frame(small, centers_prev=centers)

        # upscale matte back to input size for output
        matte = cv2.resize(matte_small, (W_in, H_in), interpolation=cv2.INTER_NEAREST)

        if DRAW_LABELS:
            # upscale labels with nearest neighbor to keep indices intact
            lab_up = cv2.resize(labels.astype(np.int32), (W_in, H_in), interpolation=cv2.INTER_NEAREST)
            matte = draw_labels_on_matte(matte, lab_up)

        writer.write(matte)

    cap.release()
    writer.release()
    print(f"Done. Wrote {OUTPUT_PATH} ({W_in}x{H_in} @ {fps:.2f} fps).")

if __name__ == "__main__":
    main()
