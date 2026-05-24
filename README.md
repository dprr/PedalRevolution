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
- [ ] 4. Add a car detector, probably TensorFlow Lite or ML Kit object detection.
- [ ] 5. Add simple tracking and counters.
- [ ] 6. Add an arcade-style overlay with effects like "combo" when passing multiple cars in a short time.
- [ ] 7. Add a social feature to compare with friends.
- [ ] 8. Collect data to help Ido.
- [ ] 9. Sell said data to data-brokers and retire early.
