# effects_video.py
import cv2, numpy as np, random, argparse
from collections import deque, namedtuple

# ------------ Common: simple motion detector (for particles & mesh) ------------
def moving_boxes(frame, subtractor, min_area=800):
    mask = subtractor.apply(frame, learningRate=0.01)
    _, bw = cv2.threshold(mask, 200, 255, cv2.THRESH_BINARY)
    bw = cv2.morphologyEx(bw, cv2.MORPH_OPEN, np.ones((3,3), np.uint8), iterations=1)
    bw = cv2.dilate(bw, np.ones((3,3), np.uint8), iterations=2)
    cnts, _ = cv2.findContours(bw, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    boxes, centers = [], []
    for c in cnts:
        if cv2.contourArea(c) < min_area: 
            continue
        x, y, w, h = cv2.boundingRect(c)
        boxes.append((x,y,w,h))
        centers.append((x + w//2, y + h//2))
    return boxes, centers, bw

# ------------ 1) Trail Effect (motion-only trails) ------------
class TrailEffect:
    def __init__(self, decay=0.92):
        self.decay = decay
        self.canvas = None
        self.bg = cv2.createBackgroundSubtractorMOG2(history=300, varThreshold=25, detectShadows=False)

    def update(self, frame):
        h, w = frame.shape[:2]
        if self.canvas is None:
            self.canvas = np.zeros((h,w,3), np.float32)
        # motion mask
        _, _, bw = moving_boxes(frame, self.bg)
        m = (bw > 0).astype(np.float32)
        m3 = np.dstack([m, m, m])
        # fade previous trails, add current motion colors
        self.canvas *= self.decay
        self.canvas += (frame.astype(np.float32) * 0.15) * m3
        out = cv2.addWeighted(frame, 1.0, np.clip(self.canvas,0,255).astype(np.uint8), 0.8, 0)
        return out

# ------------ 2) Echo Frames (temporal blend) ------------
class EchoEffect:
    def __init__(self, history=6):
        self.history = history
        self.buf = deque(maxlen=history)

    def update(self, frame):
        self.buf.appendleft(frame.copy())
        acc = np.zeros_like(frame, np.float32)
        # exponentially decaying weights
        weights = [0.55] + [0.25, 0.12, 0.05, 0.02, 0.01][:self.history-1]
        weights = weights[:len(self.buf)]
        for w, f in zip(weights, self.buf):
            acc += f.astype(np.float32) * w
        acc = np.clip(acc, 0, 255).astype(np.uint8)
        return acc

# ------------ 3) Flow Fields (optical-flow warp) ------------
class FlowFieldEffect:
    def __init__(self, strength=0.8, step=16, show_vectors=False):
        self.prev_gray = None
        self.strength = strength
        self.step = step
        self.show_vectors = show_vectors

    def update(self, frame):
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        if self.prev_gray is None:
            self.prev_gray = gray
            return frame
        flow = cv2.calcOpticalFlowFarneback(self.prev_gray, gray, None,
                                            0.5, 3, 21, 5, 5, 1.2, 0)
        h, w = gray.shape
        # build remap grid
        fx, fy = flow[...,0], flow[...,1]
        map_x, map_y = np.meshgrid(np.arange(w), np.arange(h))
        map_x = (map_x + fx * self.strength).astype(np.float32)
        map_y = (map_y + fy * self.strength).astype(np.float32)
        warped = cv2.remap(frame, map_x, map_y, interpolation=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT)
        out = cv2.addWeighted(frame, 0.6, warped, 0.7, 0)

        if self.show_vectors:
            vis = out.copy()
            for y in range(0, h, self.step):
                for x in range(0, w, self.step):
                    dx, dy = flow[y, x].astype(np.int32)
                    cv2.line(vis, (x, y), (x+dx, y+dy), (255,255,255), 1, cv2.LINE_AA)
            out = vis

        self.prev_gray = gray
        return out

# ------------ 4) ASCII Art Overlay ------------
ASCII_CHARS = np.asarray(list(" .,:;irsXA253hMHGS#9B&@"))
class AsciiOverlay:
    def __init__(self, cell=8, alpha=0.85):
        self.cell = cell
        self.alpha = alpha  # blend weight for ASCII layer

    def update(self, frame):
        h, w = frame.shape[:2]
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        # downsample to grid
        gh, gw = h // self.cell, w // self.cell
        small_gray = cv2.resize(gray, (gw, gh), interpolation=cv2.INTER_AREA)
        small_col = cv2.resize(frame, (gw, gh), interpolation=cv2.INTER_AREA)

        # map intensities -> chars
        idx = (small_gray / 255.0 * (len(ASCII_CHARS)-1)).astype(np.int32)

        # render text into an empty canvas
        ascii_img = np.zeros((h, w, 3), np.uint8)
        for j in range(gh):
            for i in range(gw):
                ch = ASCII_CHARS[idx[j, i]]  # ASCII_CHARS is already a string array, no need for chr()
                color = tuple(int(x) for x in small_col[j, i])        # BGR for cv2
                org = (i*self.cell, (j+1)*self.cell-2)
                cv2.putText(ascii_img, ch, org, cv2.FONT_HERSHEY_SIMPLEX,
                            fontScale=self.cell/16.0, color=color, thickness=1, lineType=cv2.LINE_AA)
        return cv2.addWeighted(frame, 1.0 - self.alpha, ascii_img, self.alpha, 0)

# ------------ 5) Object Trails with Particles ------------
Particle = namedtuple("Particle", ["x","y","vx","vy","life","color","r"])

class ParticleTrails:
    def __init__(self, spawn=25):
        self.sub = cv2.createBackgroundSubtractorMOG2(history=300, varThreshold=25, detectShadows=False)
        self.particles = []
        self.spawn = spawn  # number of particles per detected center per frame

    def update(self, frame):
        boxes, centers, _ = moving_boxes(frame, self.sub)
        h, w = frame.shape[:2]

        # spawn particles at each moving center
        for (cx, cy) in centers:
            base_color = frame[ max(cy-1,0):min(cy+1,h), max(cx-1,0):min(cx+1,w) ].mean(axis=(0,1)).astype(int).tolist()
            for _ in range(self.spawn):
                ang = random.uniform(0, 2*np.pi)
                spd = random.uniform(0.5, 3.0)
                vx, vy = np.cos(ang)*spd, np.sin(ang)*spd
                life = random.randint(18, 40)
                r = random.uniform(1.0, 3.0)
                self.particles.append(Particle(float(cx), float(cy), vx, vy, life, base_color, r))

        # update & draw
        canvas = frame.copy()
        new_particles = []
        for p in self.particles:
            x, y = p.x + p.vx, p.y + p.vy
            vx, vy = p.vx * 0.98, p.vy * 0.98
            life = p.life - 1
            if 0 <= x < w and 0 <= y < h and life > 0:
                alpha = max(life/40.0, 0.1)
                cv2.circle(canvas, (int(x), int(y)), int(max(p.r*alpha,1)), p.color, thickness=-1, lineType=cv2.LINE_AA)
                new_particles.append(Particle(x,y,vx,vy,life,p.color,p.r))
        self.particles = new_particles

        # light blur to blend the particles
        canvas = cv2.GaussianBlur(canvas, (0,0), 0.8)
        return cv2.addWeighted(frame, 0.7, canvas, 0.6, 0)

# ------------ 6) Bounding Box to Mesh ------------
class BoxMesh:
    def __init__(self, grid=(8,8), color=(0,255,255), alpha=0.85):
        self.sub = cv2.createBackgroundSubtractorMOG2(history=300, varThreshold=25, detectShadows=False)
        self.grid = grid
        self.color = color
        self.alpha = alpha

    def update(self, frame):
        boxes, _, _ = moving_boxes(frame, self.sub)
        overlay = frame.copy()
        for (x,y,w,h) in boxes:
            gx, gy = self.grid
            # vertical lines
            for i in range(gx+1):
                xi = x + int(i * w / gx)
                cv2.line(overlay, (xi, y), (xi, y+h), self.color, 1, cv2.LINE_AA)
            # horizontal lines
            for j in range(gy+1):
                yj = y + int(j * h / gy)
                cv2.line(overlay, (x, yj), (x+w, yj), self.color, 1, cv2.LINE_AA)
            # outer border a bit thicker
            cv2.rectangle(overlay, (x,y), (x+w, y+h), self.color, 2, cv2.LINE_AA)
        return cv2.addWeighted(frame, 1.0 - self.alpha, overlay, self.alpha, 0)

# ------------ Runner ------------
EFFECTS = {
    "trail": TrailEffect,
    "echo": EchoEffect,
    "flow": FlowFieldEffect,
    "ascii": AsciiOverlay,
    "particles": ParticleTrails,
    "bbox_mesh": BoxMesh,
}

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", "-i", required=True)
    ap.add_argument("--output", "-o", default="out.mp4")
    ap.add_argument("--effect", "-e", choices=list(EFFECTS.keys()), required=True)
    # a few optional knobs:
    ap.add_argument("--history", type=int, default=6)          # echo
    ap.add_argument("--flow_strength", type=float, default=0.8)# flow
    ap.add_argument("--ascii_cell", type=int, default=8)       # ascii
    ap.add_argument("--spawn", type=int, default=20)           # particles
    args = ap.parse_args()

    cap = cv2.VideoCapture(args.input)
    if not cap.isOpened():
        raise SystemExit("Could not open input video")

    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out = cv2.VideoWriter(args.output, fourcc, fps, (w,h))

    # build effect instance
    if args.effect == "echo":
        eff = EchoEffect(history=args.history)
    elif args.effect == "flow":
        eff = FlowFieldEffect(strength=args.flow_strength, show_vectors=False)
    elif args.effect == "ascii":
        eff = AsciiOverlay(cell=args.ascii_cell, alpha=0.9)
    elif args.effect == "particles":
        eff = ParticleTrails(spawn=args.spawn)
    else:
        eff = EFFECTS[args.effect]()

    while True:
        ok, frame = cap.read()
        if not ok: break
        out_frame = eff.update(frame)
        out.write(out_frame)

    cap.release()
    out.release()
    print("Saved:", args.output)

if __name__ == "__main__":
    main()
