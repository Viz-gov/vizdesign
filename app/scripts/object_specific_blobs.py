# object_specific_blobs.py
# Combines object detection/classification with blob detection for specific object types
import cv2
import numpy as np
import sys
from pathlib import Path
try:
    from ultralytics import YOLO
except ImportError:
    print("ERROR: ultralytics not installed. Run: pip install ultralytics")
    exit(1)

# -------- Configuration --------
# Get paths and target object from command line arguments
if len(sys.argv) < 3:
    print("Usage: python object_specific_blobs.py <input_video> <output_video> [target_object]")
    sys.exit(1)

INPUT_PATH = sys.argv[1]
OUTPUT_PATH = sys.argv[2]
TARGET_OBJECT = sys.argv[3] if len(sys.argv) > 3 else "fireworks"

# Object detection settings
MODEL_PATH = "yolov8n.pt"  # Will auto-download if not present (nano version for speed)

# Build target objects list based on user input
def get_target_objects(target_object):
    """Get list of objects to detect based on user input"""
    target_lower = target_object.lower()
    
    # If it's a direct COCO class, use it
    coco_names = get_coco_names()
    if target_object in coco_names:
        return [target_object]
    
    # Handle common aliases and related objects
    if target_lower in ["fireworks", "firework"]:
        return [
            "kite",             # Closest COCO class to fireworks (flying colorful objects)
            "sports ball",      # Round bright objects
            "frisbee",          # Flying disc-like objects
            "airplane",         # Flying objects
            "bird",             # Small flying objects
            "fire hydrant",     # Sometimes picks up bright flashes
            "traffic light"     # Bright colored objects
        ]
    elif target_lower in ["snow", "snowflake", "snowflakes"]:
        return ["person", "snowboard", "skis"]  # Objects commonly found in snow scenes
    elif target_lower in ["rain", "water", "drops"]:
        return ["umbrella", "person"]
    else:
        # Try to find similar objects or just use the input directly
        return [target_object]

# This will be set later in main()
TARGET_OBJECTS = []

# Multi-level fireworks detection settings
DETECT_FIREWORKS_BY_BRIGHTNESS = True

# Bright fireworks (active explosions)
BRIGHT_THRESHOLD = 180           # Lowered from 200 for better detection
BRIGHT_MIN_AREA = 30             # Smaller minimum for sparks
BRIGHT_MAX_AREA = 8000           # Larger maximum for big explosions

# Dim fireworks (fading sparks)
DIM_THRESHOLD = 120              # Much lower threshold for fading fireworks
DIM_MIN_AREA = 15                # Very small sparks
DIM_MAX_AREA = 2000              # Medium-sized fading areas

# Color-based detection (for colorful fireworks)
DETECT_BY_COLOR_SATURATION = True
SATURATION_THRESHOLD = 80        # High saturation indicates colorful fireworks
COLOR_MIN_AREA = 20
COLOR_MAX_AREA = 3000

# Motion-based detection (for moving sparks)
DETECT_BY_MOTION = True
MOTION_THRESHOLD = 30            # Pixel difference threshold
MOTION_MIN_AREA = 10
MOTION_MAX_AREA = 1500

# Detection confidence threshold
CONFIDENCE_THRESHOLD = 0.3
IOU_THRESHOLD = 0.5

# Blob detection settings (similar to original script)
USE_CLAHE = False
CLAHE_CLIP = 2.0
CLAHE_TILE = 8
BLUR_KSIZE = 3

# Canny edge detection
CANNY_SIGMA = 0.33
CANNY_MIN_ADD = 0
CANNY_MAX_ADD = 0

# Morphological operations
DILATE_KSIZE = 3
CLOSE_KSIZE = 3
MIN_CONTOUR_AREA = 20

# Blob filtering
BORDER_MARGIN_PX = 6
MAX_FRAME_AREA_FRAC = 0.25
MERGE_OVERLAP = True
MERGE_IOU_THRESHOLD = 0.20

# Drawing settings
BOX_COLOR = (0, 255, 0)      # Green for object boxes
BLOB_COLOR = (255, 255, 255) # White for blob boxes
BOX_THICKNESS = 2
BLOB_THICKNESS = 1
SHOW_LABELS = True
SHOW_CONFIDENCE = True
TEXT_SCALE = 0.6
TEXT_THICK = 2

# -------- Helper Functions --------
def load_model():
    """Load YOLO model, downloading if necessary"""
    print(f"Loading YOLO model: {MODEL_PATH}")
    model = YOLO(MODEL_PATH)
    return model

def get_coco_names():
    """Get COCO dataset class names"""
    return [
        'person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus', 'train', 'truck', 'boat',
        'traffic light', 'fire hydrant', 'stop sign', 'parking meter', 'bench', 'bird', 'cat',
        'dog', 'horse', 'sheep', 'cow', 'elephant', 'bear', 'zebra', 'giraffe', 'backpack',
        'umbrella', 'handbag', 'tie', 'suitcase', 'frisbee', 'skis', 'snowboard', 'sports ball',
        'kite', 'baseball bat', 'baseball glove', 'skateboard', 'surfboard', 'tennis racket',
        'bottle', 'wine glass', 'cup', 'fork', 'knife', 'spoon', 'bowl', 'banana', 'apple',
        'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog', 'pizza', 'donut', 'cake',
        'chair', 'couch', 'potted plant', 'bed', 'dining table', 'toilet', 'tv', 'laptop',
        'mouse', 'remote', 'keyboard', 'cell phone', 'microwave', 'oven', 'toaster', 'sink',
        'refrigerator', 'book', 'clock', 'vase', 'scissors', 'teddy bear', 'hair drier',
        'toothbrush'
    ]

def auto_canny_thresholds(gray, sigma=CANNY_SIGMA):
    """Calculate automatic Canny thresholds based on image statistics"""
    v = np.median(gray)
    lower = int(max(0, (1.0 - sigma) * v)) + CANNY_MIN_ADD
    upper = int(min(255, (1.0 + sigma) * v)) + CANNY_MAX_ADD
    lower = max(0, min(upper - 1, lower))
    upper = max(lower + 1, min(255, upper))
    return lower, upper

def preprocess_gray(frame):
    """Preprocess frame for edge detection"""
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    if USE_CLAHE:
        clahe = cv2.createCLAHE(clipLimit=CLAHE_CLIP, tileGridSize=(CLAHE_TILE, CLAHE_TILE))
        gray = clahe.apply(gray)
    if BLUR_KSIZE and BLUR_KSIZE >= 3 and BLUR_KSIZE % 2 == 1:
        gray = cv2.medianBlur(gray, BLUR_KSIZE)
    return gray

def postprocess_edges(edges):
    """Apply morphological operations to edge image"""
    k = DILATE_KSIZE if DILATE_KSIZE > 1 and DILATE_KSIZE % 2 == 1 else 0
    if k:
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k))
        edges = cv2.dilate(edges, kernel, iterations=1)
    
    k = CLOSE_KSIZE if CLOSE_KSIZE > 1 and CLOSE_KSIZE % 2 == 1 else 0
    if k:
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k))
        edges = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, kernel, iterations=1)
    return edges

def is_border_contour(cnt, W, H, margin=BORDER_MARGIN_PX):
    """Check if contour touches image border"""
    x = cnt[:, 0, 0]
    y = cnt[:, 0, 1]
    return (x <= margin).any() or (y <= margin).any() or (x >= W - 1 - margin).any() or (y >= H - 1 - margin).any()

def iou(box1, box2):
    """Calculate Intersection over Union of two boxes"""
    x1, y1, w1, h1 = box1
    x2, y2, w2, h2 = box2
    
    # Calculate intersection
    xi1, yi1 = max(x1, x2), max(y1, y2)
    xi2, yi2 = min(x1 + w1, x2 + w2), min(y1 + h1, y2 + h2)
    
    if xi2 <= xi1 or yi2 <= yi1:
        return 0.0
    
    inter_area = (xi2 - xi1) * (yi2 - yi1)
    box1_area = w1 * h1
    box2_area = w2 * h2
    union_area = box1_area + box2_area - inter_area
    
    return inter_area / union_area if union_area > 0 else 0.0

def merge_boxes(boxes, iou_threshold, frame_area):
    """Merge overlapping boxes"""
    if not boxes:
        return []
    
    boxes = boxes[:]
    merged = True
    
    while merged:
        merged = False
        out = []
        used = [False] * len(boxes)
        
        for i in range(len(boxes)):
            if used[i]:
                continue
                
            mx, my, mw, mh = boxes[i]
            
            # Skip merging for very large boxes
            if (mw * mh) > MAX_FRAME_AREA_FRAC * frame_area:
                used[i] = True
                out.append((mx, my, mw, mh))
                continue
            
            for j in range(i + 1, len(boxes)):
                if used[j]:
                    continue
                    
                if iou((mx, my, mw, mh), boxes[j]) > iou_threshold:
                    bx, by, bw, bh = boxes[j]
                    x1, y1 = min(mx, bx), min(my, by)
                    x2, y2 = max(mx + mw, bx + bw), max(my + mh, by + bh)
                    mx, my, mw, mh = x1, y1, x2 - x1, y2 - y1
                    used[j] = True
                    merged = True
            
            used[i] = True
            out.append((mx, my, mw, mh))
        
        boxes = out
    
    return boxes

def point_in_box(point, box):
    """Check if point is inside box"""
    px, py = point
    bx, by, bw, bh = box
    return bx <= px <= bx + bw and by <= py <= by + bh

def detect_fireworks_multi_method(frame, prev_frame=None):
    """Detect fireworks using multiple detection methods"""
    all_firework_boxes = []
    
    if not DETECT_FIREWORKS_BY_BRIGHTNESS:
        return all_firework_boxes
    
    # Convert to different color spaces
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    
    # Method 1: Bright regions (active fireworks)
    _, bright_mask = cv2.threshold(gray, BRIGHT_THRESHOLD, 255, cv2.THRESH_BINARY)
    bright_contours, _ = cv2.findContours(bright_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    for contour in bright_contours:
        area = cv2.contourArea(contour)
        if BRIGHT_MIN_AREA <= area <= BRIGHT_MAX_AREA:
            x, y, w, h = cv2.boundingRect(contour)
            all_firework_boxes.append((x, y, w, h, "bright"))
    
    # Method 2: Dim regions (fading fireworks/sparks)
    _, dim_mask = cv2.threshold(gray, DIM_THRESHOLD, 255, cv2.THRESH_BINARY)
    # Apply morphological opening to remove noise
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    dim_mask = cv2.morphologyEx(dim_mask, cv2.MORPH_OPEN, kernel)
    
    dim_contours, _ = cv2.findContours(dim_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    for contour in dim_contours:
        area = cv2.contourArea(contour)
        if DIM_MIN_AREA <= area <= DIM_MAX_AREA:
            x, y, w, h = cv2.boundingRect(contour)
            # Check if this region is not already covered by bright detection
            overlap = any(boxes_overlap((x, y, w, h), (bx, by, bw, bh)) 
                         for bx, by, bw, bh, _ in all_firework_boxes)
            if not overlap:
                all_firework_boxes.append((x, y, w, h, "dim"))
    
    # Method 3: Color saturation (colorful fireworks)
    if DETECT_BY_COLOR_SATURATION:
        _, sat_mask = cv2.threshold(hsv[:,:,1], SATURATION_THRESHOLD, 255, cv2.THRESH_BINARY)
        # Combine with brightness to avoid false positives
        combined_mask = cv2.bitwise_and(sat_mask, cv2.threshold(gray, 60, 255, cv2.THRESH_BINARY)[1])
        
        color_contours, _ = cv2.findContours(combined_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        for contour in color_contours:
            area = cv2.contourArea(contour)
            if COLOR_MIN_AREA <= area <= COLOR_MAX_AREA:
                x, y, w, h = cv2.boundingRect(contour)
                overlap = any(boxes_overlap((x, y, w, h), (bx, by, bw, bh)) 
                             for bx, by, bw, bh, _ in all_firework_boxes)
                if not overlap:
                    all_firework_boxes.append((x, y, w, h, "color"))
    
    # Method 4: Motion detection (moving sparks)
    if DETECT_BY_MOTION and prev_frame is not None:
        prev_gray = cv2.cvtColor(prev_frame, cv2.COLOR_BGR2GRAY)
        
        # Calculate frame difference
        diff = cv2.absdiff(gray, prev_gray)
        _, motion_mask = cv2.threshold(diff, MOTION_THRESHOLD, 255, cv2.THRESH_BINARY)
        
        # Clean up motion mask
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        motion_mask = cv2.morphologyEx(motion_mask, cv2.MORPH_OPEN, kernel)
        motion_mask = cv2.morphologyEx(motion_mask, cv2.MORPH_CLOSE, kernel)
        
        motion_contours, _ = cv2.findContours(motion_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        for contour in motion_contours:
            area = cv2.contourArea(contour)
            if MOTION_MIN_AREA <= area <= MOTION_MAX_AREA:
                x, y, w, h = cv2.boundingRect(contour)
                # Only keep motion regions that are also somewhat bright
                roi_gray = gray[y:y+h, x:x+w]
                if roi_gray.size > 0 and np.mean(roi_gray) > 40:  # Some minimum brightness
                    overlap = any(boxes_overlap((x, y, w, h), (bx, by, bw, bh)) 
                                 for bx, by, bw, bh, _ in all_firework_boxes)
                    if not overlap:
                        all_firework_boxes.append((x, y, w, h, "motion"))
    
    # Convert back to simple format (x, y, w, h) and merge overlapping boxes
    simple_boxes = [(x, y, w, h) for x, y, w, h, _ in all_firework_boxes]
    return merge_firework_boxes(simple_boxes)

def boxes_overlap(box1, box2, threshold=0.3):
    """Check if two boxes overlap significantly"""
    return iou(box1, box2) > threshold

def merge_firework_boxes(boxes, iou_threshold=0.4):
    """Merge overlapping firework boxes"""
    if not boxes:
        return []
    
    boxes = boxes[:]
    merged = True
    
    while merged:
        merged = False
        out = []
        used = [False] * len(boxes)
        
        for i in range(len(boxes)):
            if used[i]:
                continue
                
            mx, my, mw, mh = boxes[i]
            
            for j in range(i + 1, len(boxes)):
                if used[j]:
                    continue
                    
                if iou((mx, my, mw, mh), boxes[j]) > iou_threshold:
                    bx, by, bw, bh = boxes[j]
                    # Merge boxes
                    x1, y1 = min(mx, bx), min(my, by)
                    x2, y2 = max(mx + mw, bx + bw), max(my + mh, by + bh)
                    mx, my, mw, mh = x1, y1, x2 - x1, y2 - y1
                    used[j] = True
                    merged = True
            
            used[i] = True
            out.append((mx, my, mw, mh))
        
        boxes = out
    
    return boxes

def get_blobs_in_objects(frame, object_boxes):
    """Extract blobs only within detected object regions"""
    H, W = frame.shape[:2]
    frame_area = W * H
    
    # Create mask for object regions
    mask = np.zeros((H, W), dtype=np.uint8)
    for box in object_boxes:
        x, y, w, h = box
        mask[y:y+h, x:x+w] = 255
    
    # Preprocess frame for edge detection
    gray = preprocess_gray(frame)
    
    # Apply mask to limit processing to object regions
    gray_masked = cv2.bitwise_and(gray, mask)
    
    # Edge detection
    lo, hi = auto_canny_thresholds(gray_masked, CANNY_SIGMA)
    edges = cv2.Canny(gray_masked, lo, hi, apertureSize=3, L2gradient=True)
    proc = postprocess_edges(edges)
    
    # Find contours
    cnts, _ = cv2.findContours(proc, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    
    # Filter and build blob boxes
    blob_boxes = []
    for c in cnts:
        if cv2.contourArea(c) < MIN_CONTOUR_AREA:
            continue
        
        if is_border_contour(c, W, H, BORDER_MARGIN_PX):
            continue
        
        x, y, w, h = cv2.boundingRect(c)
        
        if (w * h) > MAX_FRAME_AREA_FRAC * frame_area:
            continue
        
        # Only keep blobs that are within object regions
        center = (x + w // 2, y + h // 2)
        in_object = any(point_in_box(center, obj_box) for obj_box in object_boxes)
        
        if in_object:
            blob_boxes.append((x, y, w, h))
    
    # Merge overlapping blobs
    if MERGE_OVERLAP:
        blob_boxes = merge_boxes(blob_boxes, MERGE_IOU_THRESHOLD, frame_area)
    
    return blob_boxes

def main():
    global TARGET_OBJECTS
    
    # Set target objects based on user input
    TARGET_OBJECTS = get_target_objects(TARGET_OBJECT)
    
    # Load YOLO model
    model = load_model()
    coco_names = get_coco_names()
    
    print(f"User requested target: '{TARGET_OBJECT}'")
    print(f"Mapped to COCO objects: {TARGET_OBJECTS}")
    
    # Convert target object names to indices
    target_indices = []
    for obj_name in TARGET_OBJECTS:
        if obj_name in coco_names:
            target_indices.append(coco_names.index(obj_name))
        else:
            print(f"Warning: Object '{obj_name}' not found in COCO classes")
    
    if not target_indices:
        print(f"No valid COCO objects found for '{TARGET_OBJECT}'. Available COCO classes:")
        print(", ".join(coco_names[:20]) + "...")  # Show first 20 classes
        raise ValueError("No valid target objects specified!")
    
    print(f"Final targeting objects: {TARGET_OBJECTS}")
    print(f"Target indices: {target_indices}")
    
    # Open video
    cap = cv2.VideoCapture(INPUT_PATH)
    if not cap.isOpened():
        raise RuntimeError(f"Could not open input: {INPUT_PATH}")
    
    W = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    H = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    if fps <= 1e-3:
        fps = 30.0
    
    print(f"Video: {W}x{H} @ {fps:.2f} fps")
    
    # Setup output video
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(OUTPUT_PATH, fourcc, fps, (W, H))
    if not writer.isOpened():
        raise RuntimeError("Could not open VideoWriter")
    
    frame_count = 0
    prev_frame = None
    
    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            frame_count += 1
            if frame_count % 30 == 0:
                print(f"Processing frame {frame_count}...")
            
            # Run YOLO detection
            results = model(frame, conf=CONFIDENCE_THRESHOLD, iou=IOU_THRESHOLD, verbose=False)
            
            # Extract target object boxes
            object_boxes = []
            for result in results:
                boxes = result.boxes
                if boxes is not None:
                    for box in boxes:
                        cls_id = int(box.cls[0])
                        if cls_id in target_indices:
                            # Convert from YOLO format (x1, y1, x2, y2) to (x, y, w, h)
                            x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                            x, y, w, h = int(x1), int(y1), int(x2 - x1), int(y2 - y1)
                            confidence = float(box.conf[0])
                            
                            object_boxes.append((x, y, w, h))
                            
                            # Draw object detection box
                            cv2.rectangle(frame, (x, y), (x + w, y + h), BOX_COLOR, BOX_THICKNESS)
                            
                            if SHOW_LABELS:
                                label = coco_names[cls_id]
                                if SHOW_CONFIDENCE:
                                    label += f" {confidence:.2f}"
                                
                                # Calculate text size for background
                                (text_w, text_h), baseline = cv2.getTextSize(
                                    label, cv2.FONT_HERSHEY_SIMPLEX, TEXT_SCALE, TEXT_THICK
                                )
                                
                                # Draw text background
                                cv2.rectangle(frame, (x, y - text_h - baseline - 5), 
                                            (x + text_w, y), BOX_COLOR, -1)
                                
                                # Draw text
                                cv2.putText(frame, label, (x, y - 5),
                                          cv2.FONT_HERSHEY_SIMPLEX, TEXT_SCALE, 
                                          (0, 0, 0), TEXT_THICK, cv2.LINE_AA)
            
            # Add custom multi-method fireworks detection
            firework_boxes = detect_fireworks_multi_method(frame, prev_frame)
            for (x, y, w, h) in firework_boxes:
                object_boxes.append((x, y, w, h))
                
                # Draw fireworks detection box in white
                cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 255, 255), BOX_THICKNESS)  # White for fireworks
                
                # Show area number instead of label
                area = w * h
                area_text = str(area)
                cv2.putText(frame, area_text, (x, max(0, y - 8)),
                          cv2.FONT_HERSHEY_SIMPLEX, TEXT_SCALE, 
                          (255, 255, 255), TEXT_THICK, cv2.LINE_AA)
            
            # Get blobs only within detected objects
            if object_boxes:
                blob_boxes = get_blobs_in_objects(frame, object_boxes)
                
                # Draw blob boxes
                for (x, y, w, h) in blob_boxes:
                    cv2.rectangle(frame, (x, y), (x + w, y + h), BLOB_COLOR, BLOB_THICKNESS)
                    
                    # Optionally show blob area
                    area = w * h
                    cv2.putText(frame, str(area), (x, max(0, y - 8)),
                              cv2.FONT_HERSHEY_SIMPLEX, TEXT_SCALE * 0.8, 
                              BLOB_COLOR, 1, cv2.LINE_AA)
            
            writer.write(frame)
            
            # Store current frame for next iteration's motion detection
            prev_frame = frame.copy()
    
    except KeyboardInterrupt:
        print("\nInterrupted by user")
    
    finally:
        cap.release()
        writer.release()
        print(f"Done! Output saved to: {OUTPUT_PATH}")
        print(f"Processed {frame_count} frames")

if __name__ == "__main__":
    if not Path(INPUT_PATH).exists():
        raise SystemExit(f"Input file not found: {INPUT_PATH}")
    main()
