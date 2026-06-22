#!/usr/bin/env python3
"""
Offline benchmark: runs both the app's EfficientDet-Lite0 (via MediaPipe)
and YOLOv10 (via Ultralytics) on the same video, frame by frame,
then compares detections per-frame using IoU matching.

This eliminates all timestamp/sync/resolution issues.
"""

import argparse
import cv2
import numpy as np
from dataclasses import dataclass, field

# Alias tflite_runtime to ai_edge_litert if tflite_runtime is not installed.
# This allows Ultralytics YOLO to load TFLite models on Python 3.12+
# without requiring the full tensorflow package or a deprecated tflite-runtime wheel.
try:
    import tflite_runtime
except ImportError:
    try:
        import sys
        import ai_edge_litert
        sys.modules['tflite_runtime'] = ai_edge_litert
        import ai_edge_litert.interpreter as interpreter
        sys.modules['tflite_runtime.interpreter'] = interpreter
    except ImportError:
        pass

from ultralytics import YOLO
import mediapipe as mp
from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python import vision as mp_vision

VEHICLE_LABELS = {"car", "bus", "truck", "motorcycle", "motorbike"}

@dataclass
class Detection:
    label: str
    confidence: float
    x_min: float  # normalized 0-1
    y_min: float
    x_max: float
    y_max: float

@dataclass
class FrameMetrics:
    tp: int = 0
    fp: int = 0
    fn: int = 0

def calculate_iou(a: Detection, b: Detection) -> float:
    x1 = max(a.x_min, b.x_min)
    y1 = max(a.y_min, b.y_min)
    x2 = min(a.x_max, b.x_max)
    y2 = min(a.y_max, b.y_max)

    inter = max(0, x2 - x1) * max(0, y2 - y1)
    area_a = (a.x_max - a.x_min) * (a.y_max - a.y_min)
    area_b = (b.x_max - b.x_min) * (b.y_max - b.y_min)
    union = area_a + area_b - inter
    return inter / (union + 1e-6)

def run_mediapipe(frame_rgb: np.ndarray, detector, timestamp_ms: int) -> list[Detection]:
    """Run the same EfficientDet-Lite0 model the Android app uses."""
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame_rgb)
    result = detector.detect_for_video(mp_image, timestamp_ms)

    detections = []
    h, w = frame_rgb.shape[:2]
    for det in result.detections:
        if not det.categories:
            continue
        cat = max(det.categories, key=lambda c: c.score)
        label = (cat.display_name or cat.category_name).lower()
        if label not in VEHICLE_LABELS:
            continue
        bb = det.bounding_box
        detections.append(Detection(
            label=label,
            confidence=cat.score,
            x_min=bb.origin_x / w,
            y_min=bb.origin_y / h,
            x_max=(bb.origin_x + bb.width) / w,
            y_max=(bb.origin_y + bb.height) / h,
        ))
    return detections

def run_yolo(frame_bgr: np.ndarray, model, conf: float = None) -> list[Detection]:
    """Run YOLO model."""
    h, w = frame_bgr.shape[:2]
    kwargs = {"verbose": False}
    if conf is not None:
        kwargs["conf"] = conf
    results = model(frame_bgr, **kwargs)[0]
    detections = []
    for box in results.boxes:
        label = model.names[int(box.cls[0])]
        if label not in VEHICLE_LABELS:
            continue
        coords = box.xyxy[0].tolist()
        detections.append(Detection(
            label=label,
            confidence=float(box.conf[0]),
            x_min=coords[0] / w,
            y_min=coords[1] / h,
            x_max=coords[2] / w,
            y_max=coords[3] / h,
        ))
    return detections

def match_frame(online: list[Detection], gt: list[Detection], iou_thresh: float) -> FrameMetrics:
    """Greedy IoU matching for a single frame."""
    matched_online = set()
    matched_gt = set()

    # Build IoU matrix and greedily match
    pairs = []
    for gi, g in enumerate(gt):
        for oi, o in enumerate(online):
            iou = calculate_iou(g, o)
            if iou >= iou_thresh:
                pairs.append((iou, gi, oi))
    pairs.sort(reverse=True)

    for iou, gi, oi in pairs:
        if gi in matched_gt or oi in matched_online:
            continue
        matched_gt.add(gi)
        matched_online.add(oi)

    tp = len(matched_gt)
    fp = len(online) - len(matched_online)
    fn = len(gt) - len(matched_gt)
    return FrameMetrics(tp=tp, fp=fp, fn=fn)

def main():
    parser = argparse.ArgumentParser(description="Offline detector benchmark")
    parser.add_argument("--video", required=True, help="Path to test video")
    parser.add_argument("--model", default="efficientdet-lite0.tflite",
                        help="Path to the TFLite model used by the app")
    parser.add_argument("--yolo", default="yolov10n.pt", help="YOLO model")
    parser.add_argument("--iou", type=float, default=0.3, help="IoU threshold")
    parser.add_argument("--score", type=float, default=0.45,
                        help="Score threshold for the online detector")
    parser.add_argument("--skip", type=int, default=1,
                        help="Process every Nth frame (1=all)")
    args = parser.parse_args()

    # ── Initialise detectors ──────────────────────────────────────────
    is_yolo_online = False
    online_detector = None

    if "yolo" in args.model.lower():
        is_yolo_online = True
    else:
        try:
            base_opts = mp_python.BaseOptions(model_asset_path=args.model)
            mp_opts = mp_vision.ObjectDetectorOptions(
                base_options=base_opts,
                running_mode=mp_vision.RunningMode.VIDEO,
                score_threshold=args.score,
                max_results=10,
            )
            online_detector = mp_vision.ObjectDetector.create_from_options(mp_opts)
        except ValueError as e:
            if "Metadata" in str(e):
                print(f"MediaPipe failed to load '{args.model}': {e}")
                print("Falling back to YOLO loader for the online model...")
                is_yolo_online = True
            else:
                raise e

    if is_yolo_online:
        print(f"Loading '{args.model}' as YOLO online model...")
        online_detector = YOLO(args.model)

    yolo_model = YOLO(args.yolo)

    # ── Process video ─────────────────────────────────────────────────
    import time
    inference_times = []
    
    cap = cv2.VideoCapture(args.video)
    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    total_tp, total_fp, total_fn = 0, 0, 0
    frames_processed = 0

    frame_idx = 0
    while cap.isOpened():
        ret, frame_bgr = cap.read()
        if not ret:
            break

        if frame_idx % args.skip != 0:
            frame_idx += 1
            continue

        timestamp_ms = int((frame_idx / fps) * 1000)

        t0 = time.perf_counter()
        if is_yolo_online:
            online_dets = run_yolo(frame_bgr, online_detector, conf=args.score)
        else:
            frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
            online_dets = run_mediapipe(frame_rgb, online_detector, timestamp_ms)
        inference_times.append(time.perf_counter() - t0)

        gt_dets = run_yolo(frame_bgr, yolo_model)

        m = match_frame(online_dets, gt_dets, args.iou)
        total_tp += m.tp
        total_fp += m.fp
        total_fn += m.fn
        frames_processed += 1

        if frames_processed % 100 == 0:
            print(f"  Processed {frames_processed} frames "
                  f"(frame {frame_idx}/{total_frames})  "
                  f"TP={total_tp} FP={total_fp} FN={total_fn}")

        frame_idx += 1

    cap.release()
    if online_detector is not None and hasattr(online_detector, "close"):
        online_detector.close()

    # ── Report ────────────────────────────────────────────────────────
    precision = total_tp / (total_tp + total_fp) if (total_tp + total_fp) else 0
    recall    = total_tp / (total_tp + total_fn) if (total_tp + total_fn) else 0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0

    mean_latency = np.mean(inference_times) if inference_times else 0
    mean_latency_ms = mean_latency * 1000
    throughput_fps = 1.0 / mean_latency if mean_latency > 0 else 0

    print()
    print("═══════════════════════════════════════")
    print("  Offline Detector Benchmark Report")
    print("═══════════════════════════════════════")
    print(f"  Video:            {args.video}")
    print(f"  Frames processed: {frames_processed}")
    print(f"  Online model:     {args.model}")
    print(f"  GT model:         {args.yolo}")
    print(f"  IoU threshold:    {args.iou}")
    print(f"  Score threshold:  {args.score}")
    print("───────────────────────────────────────")
    print(f"  True Positives:   {total_tp}")
    print(f"  False Positives:  {total_fp}")
    print(f"  False Negatives:  {total_fn}")
    print(f"  Precision:        {precision:.4f}")
    print(f"  Recall:           {recall:.4f}")
    print(f"  F1-Score:         {f1:.4f}")
    print(f"  Avg Latency:      {mean_latency_ms:.2f} ms")
    print(f"  FPS:              {throughput_fps:.2f}")
    print("═══════════════════════════════════════")

if __name__ == "__main__":
    main()
