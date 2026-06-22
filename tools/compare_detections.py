import requests
import argparse
import pandas as pd
import numpy as np
from tqdm import tqdm

def calculate_iou(boxA, boxB):
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[2], boxB[2])
    yB = min(boxA[3], boxB[3])
    
    interWidth = max(0, xB - xA)
    interHeight = max(0, yB - yA)
    interArea = interWidth * interHeight
    
    boxAArea = (boxA[2] - boxA[0]) * (boxA[3] - boxA[1])
    boxBArea = (boxB[2] - boxB[0]) * (boxB[3] - boxB[1])
    
    iou = interArea / float(boxAArea + boxBArea - interArea + 1e-6)
    return iou

def evaluate_offset(online_detections, gt_detections, offset_ms, iou_threshold, time_window_ms):
    tp = 0
    matched_online_ids = set()
    
    # Sort online detections by timestamp for faster matching if needed, 
    # but for small sets simple loop is fine.
    
    for gt in gt_detections:
        gt_box = [gt['x_min'], gt['y_min'], gt['x_max'], gt['y_max']]
        gt_ts = gt['timestamp_ms'] + offset_ms
        
        best_iou = 0
        best_match_id = None
        
        for online in online_detections:
            if online['id'] in matched_online_ids:
                continue
            
            if abs(online['timestamp_ms'] - gt_ts) <= time_window_ms:
                online_box = [online['x_min'], online['y_min'], online['x_max'], online['y_max']]
                iou = calculate_iou(gt_box, online_box)
                if iou > best_iou:
                    best_iou = iou
                    best_match_id = online['id']
        
        if best_iou >= iou_threshold:
            tp += 1
            matched_online_ids.add(best_match_id)
            
    return tp

def compare_detections(backend_url, iou_threshold=0.5, time_window_ms=100, auto_sync=True, max_offset_s=30):
    print(f"Fetching detections from {backend_url}...")
    online_res = requests.get(f"{backend_url}/detections", params={"source": "online"})
    gt_res = requests.get(f"{backend_url}/detections", params={"source": "gt"})
    
    if online_res.status_code != 200 or gt_res.status_code != 200:
        print("Error fetching detections")
        return
        
    online_detections = online_res.json()
    gt_detections = gt_res.json()
    
    if not online_detections or not gt_detections:
        print("No detections found to compare.")
        return

    best_offset = 0
    if auto_sync:
        print(f"Finding optimal temporal offset (range: +/-{max_offset_s}s)...")
        # Optimization: sample offsets every 100ms
        offsets = range(-max_offset_s * 1000, max_offset_s * 1000, 100)
        best_tp = -1
        
        # To speed up, we just take a subset of GT detections for sync
        sync_gt = gt_detections[::5] 
        
        for offset in tqdm(offsets):
            tp = evaluate_offset(online_detections, sync_gt, offset, iou_threshold, time_window_ms)
            if tp > best_tp:
                best_tp = tp
                best_offset = offset
        
        print(f"Optimal offset found: {best_offset} ms")

    # Final evaluation with best offset
    tp = 0
    matched_online_ids = set()
    matched_gt_ids = set()
    
    for gt in gt_detections:
        gt_box = [gt['x_min'], gt['y_min'], gt['x_max'], gt['y_max']]
        gt_ts = gt['timestamp_ms'] + best_offset
        
        best_iou = 0
        best_match_id = None
        
        for online in online_detections:
            if online['id'] in matched_online_ids:
                continue
            
            if abs(online['timestamp_ms'] - gt_ts) <= time_window_ms:
                online_box = [online['x_min'], online['y_min'], online['x_max'], online['y_max']]
                iou = calculate_iou(gt_box, online_box)
                if iou > best_iou:
                    best_iou = iou
                    best_match_id = online['id']
        
        if best_iou >= iou_threshold:
            tp += 1
            matched_online_ids.add(best_match_id)
            matched_gt_ids.add(gt['id'])
            
    fp = len(online_detections) - len(matched_online_ids)
    fn = len(gt_detections) - len(matched_gt_ids)
    
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
    
    print("\n--- Detection Performance Report ---")
    print(f"Total GT Detections:     {len(gt_detections)}")
    print(f"Total Online Detections: {len(online_detections)}")
    print(f"Temporal Offset:         {best_offset} ms")
    print(f"True Positives:          {tp}")
    print(f"False Positives:         {fp}")
    print(f"False Negatives:         {fn}")
    print(f"Precision:               {precision:.4f}")
    print(f"Recall:                  {recall:.4f}")
    print(f"F1-Score:                {f1:.4f}")
    print("------------------------------------\n")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--url", type=str, default="http://localhost:8000")
    parser.add_argument("--iou", type=float, default=0.25, help="IoU threshold (0.25 for rough matching)")
    parser.add_argument("--window", type=int, default=200, help="Time matching window in ms")
    parser.add_argument("--no-sync", action="store_true", help="Disable automatic temporal sync")
    parser.add_argument("--max-offset", type=int, default=60, help="Max offset range in seconds")
    args = parser.parse_args()
    
    compare_detections(args.url, args.iou, args.window, not args.no_sync, args.max_offset)
