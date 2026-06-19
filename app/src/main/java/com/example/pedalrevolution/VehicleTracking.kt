package com.example.pedalrevolution

import androidx.compose.ui.geometry.Rect
import kotlin.math.max

data class TrackedVehicle(
    val id: Int,
    val detection: VehicleDetection,
    val area: Float,
    val lastTimestamp: Long,
    val areaHistory: List<Float>,
)

data class TrackedVehicleFrameResult(
    val imageWidth: Int,
    val imageHeight: Int,
    val rotationDegrees: Int,
    val timestamp: Long,
    val trackedVehicles: List<TrackedVehicle>,
)

class VehicleTracker(
    private val minIouToMatch: Float = 0.25f,
    private val maxMissedFrames: Int = 8,
    private val historySize: Int = 12,
) {
    private var nextTrackId = 1
    private val tracks = mutableListOf<TrackState>()

    fun update(frameResult: VehicleFrameResult): TrackedVehicleFrameResult {
        val detections = frameResult.detections
        val matchedDetections = BooleanArray(detections.size)
        val detectionByTrackIndex = IntArray(tracks.size) { -1 }

        while (true) {
            var bestTrackIndex = -1
            var bestDetectionIndex = -1
            var bestIou = minIouToMatch

            for (trackIndex in tracks.indices) {
                if (detectionByTrackIndex[trackIndex] != -1) continue

                val track = tracks[trackIndex]
                for (detectionIndex in detections.indices) {
                    if (matchedDetections[detectionIndex]) continue

                    val iou = intersectionOverUnion(
                        track.detection.bounds,
                        detections[detectionIndex].bounds
                    )
                    if (iou > bestIou) {
                        bestIou = iou
                        bestTrackIndex = trackIndex
                        bestDetectionIndex = detectionIndex
                    }
                }
            }

            if (bestTrackIndex == -1 || bestDetectionIndex == -1) {
                break
            }

            matchedDetections[bestDetectionIndex] = true
            detectionByTrackIndex[bestTrackIndex] = bestDetectionIndex
        }

        val updatedTracks = mutableListOf<TrackState>()

        for (trackIndex in tracks.indices) {
            val detectionIndex = detectionByTrackIndex[trackIndex]
            if (detectionIndex == -1) continue

            val track = tracks[trackIndex]
            track.update(detections[detectionIndex], frameResult.timestamp, historySize)
            updatedTracks.add(track)
        }

        for (trackIndex in tracks.indices) {
            if (detectionByTrackIndex[trackIndex] != -1) continue

            val track = tracks[trackIndex]
            track.missedFrames += 1
            if (track.missedFrames <= maxMissedFrames) {
                updatedTracks.add(track)
            }
        }

        for (detectionIndex in detections.indices) {
            if (matchedDetections[detectionIndex]) continue

            updatedTracks.add(
                TrackState.create(
                    id = nextTrackId++,
                    detection = detections[detectionIndex],
                    timestamp = frameResult.timestamp,
                    historySize = historySize,
                )
            )
        }

        tracks.clear()
        tracks.addAll(updatedTracks.sortedBy { it.id })

        return TrackedVehicleFrameResult(
            imageWidth = frameResult.imageWidth,
            imageHeight = frameResult.imageHeight,
            rotationDegrees = frameResult.rotationDegrees,
            timestamp = frameResult.timestamp,
            trackedVehicles = tracks.map { it.toTrackedVehicle() },
        )
    }
}

private class TrackState(
    val id: Int,
    var detection: VehicleDetection,
    var missedFrames: Int = 0,
    var lastTimestamp: Long,
    private val areaHistory: ArrayDeque<Float> = ArrayDeque(),
) {
    companion object {
        fun create(
            id: Int,
            detection: VehicleDetection,
            timestamp: Long,
            historySize: Int,
        ): TrackState {
            return TrackState(
                id = id,
                detection = detection,
                lastTimestamp = timestamp,
            ).apply {
                appendArea(area(), historySize)
            }
        }
    }

    fun update(detection: VehicleDetection, timestamp: Long, historySize: Int) {
        this.detection = detection
        missedFrames = 0
        lastTimestamp = timestamp
        appendArea(area(), historySize)
    }

    fun toTrackedVehicle(): TrackedVehicle {
        return TrackedVehicle(
            id = id,
            detection = detection,
            area = area(),
            lastTimestamp = lastTimestamp,
            areaHistory = areaHistory.toList(),
        )
    }

    private fun appendArea(value: Float, historySize: Int) {
        areaHistory.addLast(value)
        while (areaHistory.size > historySize) {
            areaHistory.removeFirst()
        }
    }

    private fun area(): Float {
        return max(
            0f,
            detection.bounds.width) * max(0f, detection.bounds.height
            )
    }
}

private fun intersectionOverUnion(first: Rect, second: Rect): Float {
    val left = maxOf(first.left, second.left)
    val top = maxOf(first.top, second.top)
    val right = minOf(first.right, second.right)
    val bottom = minOf(first.bottom, second.bottom)

    if (right <= left || bottom <= top) {
        return 0f
    }

    val intersection = (right - left) * (bottom - top)
    val union = rectArea(first) + rectArea(second) - intersection
    return if (union <= 0f) 0f else intersection / union
}

private fun rectArea(rect: Rect): Float {
    return max(0f, rect.width) * max(0f, rect.height)
}
