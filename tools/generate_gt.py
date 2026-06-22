import cv2
import json
import argparse
from ultralytics import YOLO
import requests

def generate_gt(video_path, model_path="yolov10n.pt", upload_url=None):
    model = YOLO(model_path)
    cap = cv2.VideoCapture(video_path)
    
    fps = cap.get(cv2.CAP_PROP_FPS)
    frame_count = cap.get(cv2.CAP_PROP_FRAME_COUNT)
    width = cap.get(cv2.CAP_PROP_FRAME_WIDTH)
    height = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
    
    detections = []
    
    frame_idx = 0
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
            
        timestamp_ms = int((frame_idx / fps) * 1000)
        
        results = model(frame, verbose=False)[0]
        
        for box in results.boxes:
            # box.xyxy is [x1, y1, x2, y2]
            coords = box.xyxy[0].tolist()
            label = model.names[int(box.cls[0])]
            conf = float(box.conf[0])
            
            # Filter for vehicles
            if label in ["car", "bus", "truck", "motorcycle"]:
                detections.append({
                    "timestamp_ms": timestamp_ms,
                    "label": label,
                    "confidence": conf,
                    "x_min": coords[0] / width,
                    "y_min": coords[1] / height,
                    "x_max": coords[2] / width,
                    "y_max": coords[3] / height,
                    "source": "gt"
                })
        
        frame_idx += 1
        if frame_idx % 100 == 0:
            print(f"Processed {frame_idx}/{int(frame_count)} frames")

    cap.release()
    
    if upload_url:
        print(f"Uploading {len(detections)} GT detections to {upload_url}...")
        # Upload in chunks
        chunk_size = 100
        for i in range(0, len(detections), chunk_size):
            chunk = detections[i:i + chunk_size]
            requests.post(upload_url, json=chunk)
    
    return detections

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--video", type=str, required=True)
    parser.add_argument("--model", type=str, default="yolov10n.pt")
    parser.add_argument("--upload", type=str, help="Backend URL to upload detections, e.g. http://localhost:8000/detections")
    args = parser.parse_args()
    
    generate_gt(args.video, args.model, args.upload)
