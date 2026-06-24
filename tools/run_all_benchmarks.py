#!/usr/bin/env python3
import os
import re
import subprocess
import glob
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

def run_benchmark(video_path, model_path, gt_model_path, skip=1):
    print(f"\n==================================================")
    print(f"Running benchmark for online model: {os.path.basename(model_path)}")
    print(f"==================================================")
    
    cmd = [
        "uv", "run", "python", "tools/benchmark.py",
        "--video", video_path,
        "--model", model_path,
        "--ref", gt_model_path,
        "--skip", str(skip)
    ]
    
    # Run the command and capture output
    result = subprocess.run(cmd, capture_output=True, text=True)
    
    if result.returncode != 0:
        print(f"Error running benchmark for {model_path}:")
        print(result.stderr)
        return None
        
    stdout = result.stdout
    print(stdout)
    
    # Parse metrics
    tp = int(re.search(r"True Positives:\s+(\d+)", stdout).group(1))
    fp = int(re.search(r"False Positives:\s+(\d+)", stdout).group(1))
    fn = int(re.search(r"False Negatives:\s+(\d+)", stdout).group(1))
    precision = float(re.search(r"Precision:\s+([\d.]+)", stdout).group(1))
    recall = float(re.search(r"Recall:\s+([\d.]+)", stdout).group(1))
    f1 = float(re.search(r"F1-Score:\s+([\d.]+)", stdout).group(1))
    latency = float(re.search(r"Avg Latency:\s+([\d.]+)\s+ms", stdout).group(1))
    ref_latency = float(re.search(r"Ref Latency:\s+([\d.]+)\s+ms", stdout).group(1))
    fps = float(re.search(r"FPS:\s+([\d.]+)", stdout).group(1))
    
    return {
        "Model": os.path.basename(model_path),
        "TP": tp,
        "FP": fp,
        "FN": fn,
        "Precision": precision,
        "Recall": recall,
        "F1-Score": f1,
        "Latency (ms)": latency,
        "Ref Latency (ms)": ref_latency,
        "Speed (FPS)": fps
    }

def main():
    video_path = "/home/yuval/Downloads/test1.mp4"
    assets_dir = "/home/yuval/code/PedalRevolution/app/src/main/assets"
    gt_model_path = "rfdetr_2xl"
    skip = 20
    
    # Find all models in assets directory
    model_patterns = ["*.tflite", "*.pt", "*.onnx"]
    model_paths = []
    for pattern in model_patterns:
        model_paths.extend(glob.glob(os.path.join(assets_dir, pattern)))
        
    # Remove duplicates and sort
    model_paths = sorted(list(set(model_paths)))
    
    print(f"Found {len(model_paths)} models to benchmark: {[os.path.basename(p) for p in model_paths]}")
    
    results = []
    for model_path in model_paths:
        res = run_benchmark(video_path, model_path, gt_model_path, skip)
        if res:
            results.append(res)
            
    if not results:
        print("No benchmarks ran successfully.")
        return
        
    df = pd.DataFrame(results)
    print("\nBenchmark Results Summary:")
    print(df.to_string(index=False))
    
    # Save results to a CSV for record keeping
    os.makedirs("docs", exist_ok=True)
    df.to_csv("docs/benchmark_results.csv", index=False)
    
    # --- Plot Generation ---
    plt.style.use('seaborn-v0_8-whitegrid')
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 10), sharex=False)
    
    # Sleek colors
    colors = {
        'Precision': '#4ea8de',
        'Recall': '#560bad',
        'F1-Score': '#4cc9f0',
        'Latency': '#ff477e',
        'FPS': '#06d6a0'
    }
    
    x = np.arange(len(df['Model']))
    width = 0.25
    
    # Plot Accuracy Metrics (Precision, Recall, F1)
    ax1.bar(x - width, df['Precision'], width, label='Precision', color=colors['Precision'])
    ax1.bar(x, df['Recall'], width, label='Recall', color=colors['Recall'])
    ax1.bar(x + width, df['F1-Score'], width, label='F1-Score', color=colors['F1-Score'])
    
    ax1.set_ylabel('Score (0.0 - 1.0)', fontsize=12, fontweight='bold')
    ax1.set_title('Detection Accuracy Comparison (GT: rfdetr_2xl)', fontsize=14, fontweight='bold', pad=15)
    ax1.set_xticks(x)
    ax1.set_xticklabels(df['Model'], rotation=15, ha='right', fontsize=10)
    ax1.set_ylim(0, 1.05)
    ax1.legend(loc='lower left', frameon=True, facecolor='white', edgecolor='none')
    
    # Plot Speed / Latency Metric
    # Left axis for Latency, Right axis for FPS
    color_lat = colors['Latency']
    ax2.bar(x - 0.2, df['Latency (ms)'], 0.4, label='Latency (ms)', color=color_lat, alpha=0.85)
    ax2.set_ylabel('Inference Latency (ms)', color=color_lat, fontsize=12, fontweight='bold')
    ax2.tick_params(axis='y', labelcolor=color_lat)
    ax2.set_title('Inference Speed & Latency Comparison (CPU)', fontsize=14, fontweight='bold', pad=15)
    ax2.set_xticks(x)
    ax2.set_xticklabels(df['Model'], rotation=15, ha='right', fontsize=10)
    
    ax2_fps = ax2.twinx()
    color_fps = colors['FPS']
    ax2_fps.plot(x, df['Speed (FPS)'], color=color_fps, marker='o', linewidth=2.5, markersize=8, label='Speed (FPS)')
    ax2_fps.set_ylabel('Inference Speed (FPS)', color=color_fps, fontsize=12, fontweight='bold')
    ax2_fps.tick_params(axis='y', labelcolor=color_fps)
    
    # Combine legends for second subplot
    lines, labels = ax2.get_legend_handles_labels()
    lines2, labels2 = ax2_fps.get_legend_handles_labels()
    ax2.legend(lines + lines2, labels + labels2, loc='upper right', frameon=True, facecolor='white', edgecolor='none')
    
    plt.tight_layout()
    plot_path = "docs/benchmark_plot.png"
    plt.savefig(plot_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"\nPlot saved to: {plot_path}")
    
    # --- Update README.md ---
    readme_path = "README.md"
    with open(readme_path, "r") as f:
        readme_content = f.read()
        
    # Locate or create ## Benchmarks section
    benchmark_header = "## Benchmarks"
    if benchmark_header in readme_content:
        # Keep everything before ## Benchmarks
        readme_base = readme_content.split(benchmark_header)[0]
    else:
        readme_base = readme_content + "\n"
        
    # Generate Markdown Table
    md_table = df.to_markdown(index=False)
    
    benchmark_section = f"""{benchmark_header}

Below is the benchmarking result of the object detection models available in the app's assets folder (`app/src/main/assets`). 

The models were evaluated frame-by-frame on a test video against the largest ground truth model (`rfdetr_2xl`) with an IoU threshold of 0.3 and a score confidence threshold of 0.45.

### Results Table

{md_table}

### Performance Visualization

![Detection Model Benchmark Visualization](docs/benchmark_plot.png)
"""
    
    new_readme_content = readme_base + benchmark_section
    with open(readme_path, "w") as f:
        f.write(new_readme_content)
        
    print("README.md updated with benchmark results and plot.")

if __name__ == "__main__":
    main()
