# ruttra_video.py
import cv2, numpy as np, sys
from pathlib import Path

# Get paths from command line arguments
if len(sys.argv) < 3:
    print("Usage: python ruttra_video.py <input_video> <output_video>")
    sys.exit(1)

INPUT, OUTPUT = sys.argv[1], sys.argv[2]
STEP_X = 4   # horizontal sampling step
STEP_Y = 4   # vertical sampling step
HEIGHT_GAIN = 0.35  # how much luminance lifts scanline

def main():
    cap = cv2.VideoCapture(INPUT)
    if not cap.isOpened(): raise SystemExit(f"Missing {INPUT}")
    W, H = int(cap.get(3)), int(cap.get(4))
    fps = cap.get(5) or 30.0
    writer = cv2.VideoWriter(OUTPUT, cv2.VideoWriter_fourcc(*"mp4v"), fps, (W, H))
    while True:
        ok, frame = cap.read()
        if not ok: break
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY).astype(np.float32)/255.0
        canvas = np.zeros_like(frame)
        for y in range(0, H-STEP_Y, STEP_Y):
            pts = []
            for x in range(0, W, STEP_X):
                h = int((gray[y, x] - 0.5) * 2 * HEIGHT_GAIN * STEP_Y * 10)
                pts.append((x, np.clip(y - h, 0, H-1)))
            for i in range(len(pts)-1):
                cv2.line(canvas, pts[i], pts[i+1], (255,255,255), 1, cv2.LINE_AA)
        writer.write(canvas)
    cap.release(); writer.release()
if __name__ == "__main__": main()
