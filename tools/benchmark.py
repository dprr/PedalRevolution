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
import os
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

try:
    import rfdetr
    from rfdetr.assets.coco_classes import COCO_CLASSES
    RFDETR_AVAILABLE = True
except ImportError:
    RFDETR_AVAILABLE = False

try:
    import onnxruntime as ort
    ONNX_AVAILABLE = True
except ImportError:
    ONNX_AVAILABLE = False

VEHICLE_LABELS = {"car", "bus", "truck", "motorcycle", "motorbike", "bicycle"}

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

ASSETS_DIR = os.path.join(os.path.dirname(__file__), "../app/src/main/assets")

def resolve_model_path(path: str) -> str:
    if "rfdetr" in path.lower():
        return path
    if path.endswith(".onnx") or path.endswith(".pt") or path.endswith(".tflite"):
        if os.path.exists(path):
            return path
        assets_path = os.path.join(ASSETS_DIR, path)
        if os.path.exists(assets_path):
            return assets_path
    return path

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

def run_rfdetr(frame_bgr: np.ndarray, model, conf: float = None) -> list[Detection]:
    """Run RF-DETR model."""
    h, w = frame_bgr.shape[:2]
    # rfdetr's predict usually takes a path or an image.
    # It returns a supervision.Detections object.
    # Note: run_rfdetr assumes frame_bgr is BGR but RF-DETR might expect RGB internally.
    # Most modern libs handle this, but let's be safe if we need to convert.
    frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)

    # Passing the numpy array directly
    results = model.predict(frame_rgb, threshold=conf or 0.25)

    detections = []
    for i in range(len(results)):
        class_id = results.class_id[i]
        label = COCO_CLASSES[class_id].lower()
        if label not in VEHICLE_LABELS:
            continue

        box = results.xyxy[i]
        detections.append(Detection(
            label=label,
            confidence=float(results.confidence[i]),
            x_min=box[0] / w,
            y_min=box[1] / h,
            x_max=box[2] / w,
            y_max=box[3] / h,
        ))
    return detections

def run_onnx(frame_bgr: np.ndarray, session, conf: float = None) -> list[Detection]:
    """Run YOLO ONNX model."""
    h, w = frame_bgr.shape[:2]

    # Preprocess: Resize to 640x640, BGR to RGB, normalize, HWC to CHW
    img = cv2.resize(frame_bgr, (640, 640))
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = img.astype(np.float32) / 255.0
    img = img.transpose(2, 0, 1) # CHW
    img = np.expand_dims(img, axis=0) # NCHW

    input_name = session.get_inputs()[0].name
    outputs = session.run(None, {input_name: img})

    # YOLOv10 output is [1, 300, 6] -> [x1, y1, x2, y2, score, class]
    predictions = outputs[0][0]

    detections = []
    for pred in predictions:
        score = pred[4]
        if score < (conf or 0.25):
            continue

        class_idx = int(pred[5])
        COCO_80 = [
            "person", "bicycle", "car", "motorcycle", "airplane", "bus", "train", "truck", "boat",
            "traffic light", "fire hydrant", "stop sign", "parking meter", "bench", "bird", "cat",
            "dog", "horse", "sheep", "cow", "elephant", "bear", "zebra", "giraffe", "backpack",
            "umbrella", "handbag", "tie", "suitcase", "frisbee", "skis", "snowboard", "sports ball",
            "kite", "baseball bat", "baseball glove", "skateboard", "surfboard", "tennis racket",
            "bottle", "wine glass", "cup", "fork", "knife", "spoon", "bowl", "banana", "apple",
            "sandwich", "orange", "broccoli", "carrot", "hot dog", "pizza", "donut", "cake",
            "chair", "couch", "potted plant", "bed", "dining table", "toilet", "tv", "laptop",
            "mouse", "remote", "keyboard", "cell phone", "microwave", "oven", "toaster", "sink",
            "refrigerator", "book", "clock", "vase", "scissors", "teddy bear", "hair drier", "toothbrush"
        ]

        if class_idx >= len(COCO_80):
            continue

        label = COCO_80[class_idx]
        if label not in VEHICLE_LABELS:
            continue

        detections.append(Detection(
            label=label,
            confidence=float(score),
            x_min=pred[0],
            y_min=pred[1],
            x_max=pred[2],
            y_max=pred[3],
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

def load_rfdetr_model(model_name_or_path: str):
    """Dynamically load RF-DETR model based on name or path."""
    if not RFDETR_AVAILABLE:
        raise ImportError("rfdetr is not installed. Please install it with 'pip install rfdetr rfdetr-plus'.")

    name = model_name_or_path.lower()
    if "rfdetr_nano" in name:
        from rfdetr import RFDETRNano
        return RFDETRNano()
    elif "rfdetr_small" in name:
        from rfdetr import RFDETRSmall
        return RFDETRSmall()
    elif "rfdetr_medium" in name:
        from rfdetr import RFDETRMedium
        return RFDETRMedium()
    elif "rfdetr_large" in name:
        from rfdetr import RFDETRLarge
        return RFDETRLarge()
    elif "rfdetr_xl" in name:
        from rfdetr_plus import RFDETRXLarge
        return RFDETRXLarge()
    elif "rfdetr_2xl" in name:
        from rfdetr_plus import RFDETR2XLarge
        return RFDETR2XLarge()
    else:
        # Fallback to generic loader if it's a path or unknown name
        # RF-DETR models often use .pth or .pt, but we'll try RFDETRMedium as default if it's a generic .pt/pth
        from rfdetr import RFDETRMedium
        if model_name_or_path.endswith('.pth') or model_name_or_path.endswith('.pt'):
             return RFDETRMedium(pretrain_weights=model_name_or_path)
        return RFDETRMedium()

def main():
    parser = argparse.ArgumentParser(description="Offline detector benchmark")
    parser.add_argument("--video", required=True, help="Path to test video")
    parser.add_argument("--model", default="efficientdet-lite0.tflite",
                        help="Path to the TFLite model used by the app")
    parser.add_argument("--ref", default="yolov10n.pt", help="Reference (GT) model")
    parser.add_argument("--iou", type=float, default=0.3, help="IoU threshold")
    parser.add_argument("--score", type=float, default=0.45,
                        help="Score threshold for the online detector")
    parser.add_argument("--skip", type=int, default=1,
                        help="Process every Nth frame (1=all)")
    args = parser.parse_args()

    args.model = resolve_model_path(args.model)
    args.ref = resolve_model_path(args.ref)

    # ── Initialise detectors ──────────────────────────────────────────
    is_yolo_online = False
    is_rfdetr_online = False
    is_onnx_online = False
    online_detector = None

    if "rfdetr" in args.model.lower():
        is_rfdetr_online = True
        print(f"Loading '{args.model}' as RF-DETR online model...")
        online_detector = load_rfdetr_model(args.model)
    elif args.model.lower().endswith(".onnx"):
        if not ONNX_AVAILABLE:
            raise ImportError("onnxruntime is not installed.")
        is_onnx_online = True
        print(f"Loading '{args.model}' as ONNX online model...")
        online_detector = ort.InferenceSession(args.model)
    elif "yolo" in args.model.lower() or args.model.lower().endswith(".pt"):
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

    is_rfdetr_ref = "rfdetr" in args.ref.lower()
    is_onnx_ref = args.ref.lower().endswith(".onnx")

    if is_rfdetr_ref:
        print(f"Loading '{args.ref}' as RF-DETR reference model...")
        ref_model = load_rfdetr_model(args.ref)
    elif is_onnx_ref:
        print(f"Loading '{args.ref}' as ONNX reference model...")
        ref_model = ort.InferenceSession(args.ref)
    else:
        ref_model = YOLO(args.ref)

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
        if is_rfdetr_online:
            online_dets = run_rfdetr(frame_bgr, online_detector, conf=args.score)
        elif is_onnx_online:
            online_dets = run_onnx(frame_bgr, online_detector, conf=args.score)
        elif is_yolo_online:
            online_dets = run_yolo(frame_bgr, online_detector, conf=args.score)
        else:
            frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
            online_dets = run_mediapipe(frame_rgb, online_detector, timestamp_ms)
        inference_times.append(time.perf_counter() - t0)

        if is_rfdetr_ref:
            gt_dets = run_rfdetr(frame_bgr, ref_model, conf=args.score)
        elif is_onnx_ref:
            gt_dets = run_onnx(frame_bgr, ref_model, conf=args.score)
        else:
            gt_dets = run_yolo(frame_bgr, ref_model, conf=args.score)

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
    print(f"  GT model:         {args.ref}")
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
