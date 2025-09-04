# object_connectors_video.py
# Detect objects (YOLOv8), draw bounding boxes, then connect them with curvy dotted lines.

import math
import random
import sys
from pathlib import Path

import cv2
import numpy as np

try:
    from ultralytics import YOLO
except Exception as e:
    raise SystemExit(
        "Ultralytics YOLO not installed.\n"
        "Install with:  pip install ultralytics\n"
        f"Original error: {e}"
    )

# Get paths from command line arguments
if len(sys.argv) < 3:
    print("Usage: python object_connectors_video.py <input_video> <output_video>")
    sys.exit(1)

INPUT_PATH = sys.argv[1]
OUTPUT_PATH = sys.argv[2]

MODEL_PATH  = "yolov8n.pt"         # auto-downloads on first run
CONF_THRESH = 0.35                  # object confidence threshold
IOU_THRESH  = 0.45                  # NMS IoU threshold

# Box drawing
BOX_COLOR   = (255, 255, 255)      # white
BOX_THICK   = 2
SHOW_LABELS = False                # set True to draw class names

# Connectors
K_NEIGHBORS = 2                    # connect each box to K nearest others
MAX_CONNECT_DIST = 0.45            # normalized (0..√2) max distance in diag units
CURVE_BULGE = 0.18                 # 0..~0.5: how curvy the Bezier is
JITTER_PX   = 10                   # random jitter on control points (px)
DOT_SPACING = 10                   # px between dots
DOT_RADIUS  = 2                    # dot size
DOT_COLOR   = (255, 255, 255)      # white
# ------------------------------------------------


def bezier_points(p0, p1, p2, p3, step=5):
    """Sample a cubic Bezier in ~`step` pixel increments (arc-length approx)."""
    # parametric t spacing; small step -> more dots
    num = max(2, int(max(
        np.hypot(*(np.subtract(p1, p0))),
        np.hypot(*(np.subtract(p2, p1))),
        np.hypot(*(np.subtract(p3, p2)))
    ) // step))
    ts = np.linspace(0.0, 1.0, num)
    pts = []
    for t in ts:
        u = 1 - t
        x = (u**3)*p0[0] + 3*(u**2)*t*p1[0] + 3*u*(t**2)*p2[0] + (t**3)*p3[0]
        y = (u**3)*p0[1] + 3*(u**2)*t*p1[1] + 3*u*(t**2)*p2[1] + (t**3)*p3[1]
        pts.append((int(x), int(y)))
    return pts


def draw_dotted_bezier(img, a, b, bulge=0.18, jitter=8, spacing=10, color=(255, 255, 255), r=2):
    """
    Draw a dotted, curved connection between points a and b.
    a,b: (x,y)
    bulge: 0..~0.5 curve amount
    jitter: px jitter on control points for organic feel
    spacing: px between dots
    """
    ax, ay = a; bx, by = b
    mx, my = (ax + bx) / 2.0, (ay + by) / 2.0
    dx, dy = bx - ax, by - ay
    dist = math.hypot(dx, dy)
    if dist < 1e-3:
        return

    # unit perpendicular
    nx, ny = (-dy / dist, dx / dist)

    # control points offset from the straight line
    bulge_px = dist * bulge
    c1 = (ax + dx * 0.33 + nx * bulge_px + random.uniform(-jitter, jitter),
          ay + dy * 0.33 + ny * bulge_px + random.uniform(-jitter, jitter))
    c2 = (ax + dx * 0.66 + nx * -bulge_px + random.uniform(-jitter, jitter),
          ay + dy * 0.66 + ny * -bulge_px + random.uniform(-jitter, jitter))

    pts = bezier_points(a, c1, c2, b, step=spacing)
    for p in pts:
        cv2.circle(img, p, r, color, -1, lineType=cv2.LINE_AA)


def centers_from_boxes(boxes):
    """Convert XYWH boxes to centers."""
    centers = []
    for (x, y, w, h) in boxes:
        centers.append((x + w // 2, y + h // 2))
    return centers


def k_nearest_pairs(centers, k, max_norm_dist, w, h):
    """
    Build a set of undirected pairs (i,j) connecting k nearest neighbors per node,
    discarding long edges. Distance normalized by image diagonal.
    """
    if len(centers) < 2:
        return set()

    diag = math.hypot(w, h)
    pairs = set()
    for i, ci in enumerate(centers):
        # distances to others
        ds = []
        for j, cj in enumerate(centers):
            if j == i:
                continue
            d = math.hypot(ci[0] - cj[0], ci[1] - cj[1]) / diag
            ds.append((d, j))
        ds.sort(key=lambda x: x[0])
        for t in ds[:k]:
            d, j = t
            if d <= max_norm_dist:
                a, b = min(i, j), max(i, j)
                pairs.add((a, b))
    return pairs


def run():
    # Video I/O
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

    writer = cv2.VideoWriter(
        OUTPUT_PATH,
        cv2.VideoWriter_fourcc(*"mp4v"),
        fps, (W, H)
    )
    if not writer.isOpened():
        raise SystemExit("Could not open VideoWriter for output.")

    # Model
    model = YOLO(MODEL_PATH)

    while True:
        ok, frame = cap.read()
        if not ok:
            break

        # YOLO inference
        results = model.predict(
            frame, conf=CONF_THRESH, iou=IOU_THRESH, verbose=False
        )

        boxes_xywh = []
        labels = []

        for r in results:
            if r.boxes is None:
                continue
            for b in r.boxes:
                x1, y1, x2, y2 = b.xyxy[0].cpu().numpy()
                conf = float(b.conf[0])
                if conf < CONF_THRESH:
                    continue
                x = int(max(0, x1))
                y = int(max(0, y1))
                w = int(max(1, x2 - x1))
                h = int(max(1, y2 - y1))
                boxes_xywh.append((x, y, w, h))
                if SHOW_LABELS:
                    cls_id = int(b.cls[0])
                    labels.append((cls_id, conf))

        # Draw boxes
        for idx, (x, y, w, h) in enumerate(boxes_xywh):
            cv2.rectangle(frame, (x, y), (x + w, y + h), BOX_COLOR, BOX_THICK)
            if SHOW_LABELS and idx < len(labels):
                cls_id, conf = labels[idx]
                name = model.names.get(cls_id, str(cls_id))
                txt = f"{name} {conf:.2f}"
                (tw, th), bl = cv2.getTextSize(txt, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
                cv2.rectangle(frame, (x, y - th - bl - 2), (x + tw + 4, y), BOX_COLOR, -1)
                cv2.putText(frame, txt, (x + 2, y - 3),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1, cv2.LINE_AA)

        # Build connectors
        centers = centers_from_boxes(boxes_xywh)
        pairs = k_nearest_pairs(
            centers, K_NEIGHBORS, MAX_CONNECT_DIST, W, H
        )

        # Draw dotted, curvy connectors
        for (i, j) in pairs:
            a = centers[i]
            b = centers[j]
            draw_dotted_bezier(
                frame, a, b,
                bulge=CURVE_BULGE,
                jitter=JITTER_PX,
                spacing=DOT_SPACING,
                color=DOT_COLOR,
                r=DOT_RADIUS
            )

        writer.write(frame)

    cap.release()
    writer.release()
    print(f"Done. Wrote {OUTPUT_PATH} ({W}x{H} @ {fps:.2f} fps).")


if __name__ == "__main__":
    run()
