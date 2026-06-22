# Pedal Revolution

Pedal Revolution is a gamified bicycle side-camera app.

When riding a bike in traffic, the app should help show when you pass cars and when cars pass you.
Over time, those events can turn into points, streaks, and friendly competition.

## Goal

- Detect passing events while the phone is mounted on a bike and filming traffic.
- Track when the rider overtakes cars and when cars overtake the rider.
- Turn those events into a game-like score and visual feedback.

## Method

- Native Android app written in Kotlin.
- Jetpack Compose for the UI.
- CameraX for live camera preview and frame analysis.
- Computer vision for car detection and later tracking.
- Prefer on-device processing for low latency, cost and future-proof.

## Roadmap

- [ ] 0. Pick a good name
- [x] 1. Create a minimal Android/Kotlin CameraX app.
- [x] 2. Add a live camera preview.
- [x] 3. Add frame analysis.
- [x] 4. Add a car detector, probably TensorFlow Lite or ML Kit object detection.
- [ ] 5. Add simple tracking and counters.
- [ ] 6. Add an arcade-style overlay with effects like "combo" when passing multiple cars in a short time.
- [ ] 7. Add a social feature to compare with friends.
- [ ] 8. Collect data to help Ido.
- [ ] 9. Sell said data to data-brokers and retire early.

## Benchmarks

Below is the benchmarking result of the object detection models available in the app's assets folder (`app/src/main/assets`). 

The models were evaluated frame-by-frame on a test video against the largest ground truth model (`yolo26x.pt`) with an IoU threshold of 0.3 and a score confidence threshold of 0.45.

### Results Table

| Model                     |   TP |   FP |   FN |   Precision |   Recall |   F1-Score |   Latency (ms) |   Speed (FPS) |
|:--------------------------|-----:|-----:|-----:|------------:|---------:|-----------:|---------------:|--------------:|
| efficientdet-lite0.tflite |   20 |   29 |   92 |      0.4082 |   0.1786 |     0.2484 |          37.94 |         26.36 |
| yolo26n.pt                |   24 |    6 |   88 |      0.8    |   0.2143 |     0.338  |          15.22 |         65.69 |
| yolo26n_float16.tflite    |   22 |    6 |   90 |      0.7857 |   0.1964 |     0.3143 |          76.11 |         13.14 |
| yolo26n_float32.tflite    |   22 |    6 |   90 |      0.7857 |   0.1964 |     0.3143 |          75.95 |         13.17 |
| yolo26x.pt                |   75 |    0 |   37 |      1      |   0.6696 |     0.8021 |          36.34 |         27.52 |
| yolo26x_float16.tflite    |   68 |    1 |   44 |      0.9855 |   0.6071 |     0.7514 |        2047.87 |          0.49 |
| yolo26x_float32.tflite    |   68 |    1 |   44 |      0.9855 |   0.6071 |     0.7514 |        2132.53 |          0.47 |

### Performance Visualization

![Detection Model Benchmark Visualization](docs/benchmark_plot.png)
