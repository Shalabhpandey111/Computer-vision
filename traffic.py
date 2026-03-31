"""
vehicle_detector.py

Simple vehicle detector + counter using YOLOv8 (ultralytics).
Detects vehicles (car, motorcycle, bus, truck), draws boxes, tracks IDs,
and counts vehicles crossing a counting line.

Usage:
    python vehicle_detector.py --source 0                # webcam
    python vehicle_detector.py --source input.mp4        # video file
    python vehicle_detector.py --source "rtsp://..."     # RTSP stream
"""

import cv2
import numpy as np
from ultralytics import YOLO
import time
from collections import OrderedDict

# --- Simple centroid tracker (lightweight) ---
class CentroidTracker:
    def __init__(self, max_disappeared=30, max_distance=50):
        self.next_object_id = 0
        self.objects = OrderedDict()     # object_id -> centroid
        self.disappeared = OrderedDict() # object_id -> disappeared count
        self.max_disappeared = max_disappeared
        self.max_distance = max_distance

    def register(self, centroid):
        self.objects[self.next_object_id] = centroid
        self.disappeared[self.next_object_id] = 0
        self.next_object_id += 1

    def deregister(self, object_id):
        del self.objects[object_id]
        del self.disappeared[object_id]

    def update(self, input_centroids):
        if len(input_centroids) == 0:
            for object_id in list(self.disappeared.keys()):
                self.disappeared[object_id] += 1
                if self.disappeared[object_id] > self.max_disappeared:
                    self.deregister(object_id)
            return self.objects

        if len(self.objects) == 0:
            for c in input_centroids:
                self.register(c)
            return self.objects

        object_ids = list(self.objects.keys())
        object_centroids = list(self.objects.values())

        D = np.linalg.norm(
            np.array(object_centroids)[:, None] - np.array(input_centroids)[None, :], axis=2
        )
        rows = D.min(axis=1).argsort()
        cols = D.argmin(axis=1)[rows]
        used_rows, used_cols = set(), set()

        for r, c in zip(rows, cols):
            if r in used_rows or c in used_cols:
                continue
            if D[r, c] > self.max_distance:
                continue
            object_id = object_ids[r]
            self.objects[object_id] = input_centroids[c]
            self.disappeared[object_id] = 0
            used_rows.add(r)
            used_cols.add(c)

        unused_rows = set(range(D.shape[0])) - used_rows
        for r in unused_rows:
            object_id = object_ids[r]
            self.disappeared[object_id] += 1
            if self.disappeared[object_id] > self.max_disappeared:
                self.deregister(object_id)

        unused_cols = set(range(D.shape[1])) - used_cols
        for c in unused_cols:
            self.register(input_centroids[c])

        return self.objects

# --- Helper functions ---
def get_centroid(box):
    x1, y1, x2, y2 = box
    cx = int((x1 + x2) / 2)
    cy = int((y1 + y2) / 2)
    return (cx, cy)

def intersects_line(p1, p2, y_line):
    (x1, y1), (x2, y2) = p1, p2
    return (y1 < y_line and y2 >= y_line) or (y1 >= y_line and y2 < y_line)

def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=str, default="0", help="Video source (0 for webcam, or path/URL)")
    parser.add_argument("--model", type=str, default="yolov8n.pt", help="YOLOv8 model to use ('yolov8n.pt', etc.)")
    parser.add_argument("--output", type=str, default="output.mp4", help="Output video file (set empty to disable)")
    parser.add_argument("--imgsz", type=int, default=640, help="Inference image size")
    parser.add_argument("--confidence", type=float, default=0.3, help="Confidence threshold")
    parser.add_argument("--iou", type=float, default=0.45, help="NMS IoU threshold")
    parser.add_argument("--line_position", type=float, default=0.5, help="Relative vertical position for counting line (0-1)")
    args = parser.parse_args()

    try:
        args.source = int(args.source)
    except:
        pass

    model = YOLO(args.model)  # Load YOLOv8 model

    vehicle_class_ids = {2: 'car', 3: 'motorcycle', 5: 'bus', 7: 'truck'}

    cap = cv2.VideoCapture(args.source)
    if not cap.isOpened():
        print(f"ERROR: cannot open source {args.source}")
        return

    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS) if cap.get(cv2.CAP_PROP_FPS) > 1 else 25.0

    out = None
    if args.output:
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(args.output, fourcc, fps, (width, height))

    y_line = int(height * args.line_position)
    tracker = CentroidTracker(max_disappeared=40, max_distance=80)

    tracks_history = {}
    counted_ids = set()
    counts = {'car': 0, 'motorcycle': 0, 'bus': 0, 'truck': 0}
    frame_num = 0
    start_time = time.time()

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame_num += 1

        # Run prediction (wrap frame in list for batch=1)
        results = model.predict(source=[frame], imgsz=args.imgsz, conf=args.confidence, iou=args.iou, verbose=False)
        res = results[0]

        detections = []
        detection_classes = []

        if res.boxes is not None and len(res.boxes) > 0:
            boxes = res.boxes.xyxy.cpu().numpy()
            classes = res.boxes.cls.cpu().numpy().astype(int)
            confs = res.boxes.conf.cpu().numpy()
            for box, cls_id, conf in zip(boxes, classes, confs):
                if cls_id in vehicle_class_ids and conf >= args.confidence:
                    x1, y1, x2, y2 = map(int, box)
                    detections.append((x1, y1, x2, y2))
                    detection_classes.append(cls_id)

        centroids = [get_centroid(b) for b in detections]
        objects = tracker.update(centroids)

        # Draw counting line
        cv2.line(frame, (0, y_line), (width, y_line), (0, 255, 255), 2)
        cv2.putText(frame, f"Counts - Car:{counts['car']}  Motorcycle:{counts['motorcycle']}  Bus:{counts['bus']}  Truck:{counts['truck']}",
                    (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

        centroid_to_det_index = {tuple(get_centroid(d)): i for i, d in enumerate(detections)}

        for object_id, centroid in objects.items():
            cv2.circle(frame, centroid, 4, (0, 0, 255), -1)
            cv2.putText(frame, f"ID {object_id}", (centroid[0] - 10, centroid[1] - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)

            det_index = centroid_to_det_index.get(centroid, None)
            if det_index is not None:
                box = detections[det_index]
                cls_id = detection_classes[det_index]
                label = vehicle_class_ids.get(cls_id, str(cls_id))
                x1, y1, x2, y2 = box
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(frame, f"{label}", (x1, y1 - 6), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)

            prev_centroid = tracks_history.get(object_id, None)
            if prev_centroid is not None and object_id not in counted_ids:
                if intersects_line(prev_centroid, centroid, y_line):
                    if det_index is not None:
                        cls_id = detection_classes[det_index]
                        label = vehicle_class_ids.get(cls_id, None)
                        if label:
                            counts[label] += 1
                            counted_ids.add(object_id)

            tracks_history[object_id] = centroid

        elapsed = time.time() - start_time
        fps_now = frame_num / elapsed if elapsed > 0 else 0
        cv2.putText(frame, f"FPS: {fps_now:.1f}", (width - 120, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

        cv2.imshow("Vehicle Detector", frame)

        if out is not None:
            out.write(frame)

        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break

    cap.release()
    if out is not None:
        out.release()
    cv2.destroyAllWindows()
    print("Final counts:", counts)


if __name__ == "__main__":
    main()
