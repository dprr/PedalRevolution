package com.example.pedalrevolution

import android.content.Context
import android.graphics.Bitmap
import android.graphics.Matrix
import androidx.compose.ui.geometry.Rect
import android.util.Log
import androidx.camera.core.ImageAnalysis
import androidx.camera.core.ImageProxy
import com.google.mediapipe.framework.image.BitmapImageBuilder
import com.google.mediapipe.tasks.components.containers.Detection
import com.google.mediapipe.tasks.core.BaseOptions
import com.google.mediapipe.tasks.vision.core.RunningMode
import com.google.mediapipe.tasks.vision.objectdetector.ObjectDetector
import java.util.Locale
import kotlin.math.max

private const val TAG = "VehicleDetect"
private const val MODEL_FILE = "efficientdet-lite0.tflite"
private const val SCORE_THRESHOLD = 0.45f
private const val MAX_RESULTS = 10

private val VEHICLE_LABELS = setOf("car", "bus", "motorcycle", "motorbike", "truck")

data class VehicleDetection(
    val bounds: Rect,
    val label: String,
    val confidence: Float,
)

data class VehicleFrameResult(
    val imageWidth: Int,
    val imageHeight: Int,
    val rotationDegrees: Int,
    val timestamp: Long,
    val detections: List<VehicleDetection>,
)

interface VehicleDetector : AutoCloseable {
    fun detect(imageProxy: ImageProxy): VehicleFrameResult
}

class MediaPipeVehicleDetector(context: Context) : VehicleDetector {
    private val objectDetector = ObjectDetector.createFromOptions(
        context,
        ObjectDetector.ObjectDetectorOptions.builder()
            .setBaseOptions(
                BaseOptions.builder()
                    .setModelAssetPath(MODEL_FILE)
                    .build()
            )
            .setRunningMode(RunningMode.VIDEO)
            .setScoreThreshold(SCORE_THRESHOLD)
            .setMaxResults(MAX_RESULTS)
            .build()
    )
    private var lastTimestampMs = 0L

    override fun detect(imageProxy: ImageProxy): VehicleFrameResult {
        val sourceBitmap = imageProxy.toBitmap()
        val rotationDegrees = imageProxy.imageInfo.rotationDegrees
        val rotatedBitmap = sourceBitmap.rotate(rotationDegrees)
        if (rotatedBitmap !== sourceBitmap) {
            sourceBitmap.recycle()
        }

        return try {
            val timestampMs = max(lastTimestampMs + 1, imageProxy.imageInfo.timestamp / 1_000_000L)
            lastTimestampMs = timestampMs

            BitmapImageBuilder(rotatedBitmap).build().use { mpImage ->
                val detectionResult = objectDetector.detectForVideo(mpImage, timestampMs)
                val detections = detectionResult.detections().mapNotNull { it.toVehicleDetection() }

                VehicleFrameResult(
                    imageWidth = rotatedBitmap.width,
                    imageHeight = rotatedBitmap.height,
                    rotationDegrees = rotationDegrees,
                    timestamp = detectionResult.timestampMs(),
                    detections = detections,
                )
            }
        } finally {
            if (!rotatedBitmap.isRecycled) {
                rotatedBitmap.recycle()
            }
        }
    }

    override fun close() {
        objectDetector.close()
    }
}

class VehicleFrameAnalyzer(
    private val detector: VehicleDetector,
    private val tracker: VehicleTracker,
    private val onFrameResult: (TrackedVehicleFrameResult) -> Unit,
) : ImageAnalysis.Analyzer {
    override fun analyze(imageProxy: ImageProxy) {
        try {
            val frameResult = detector.detect(imageProxy)
            val trackedFrameResult = tracker.update(frameResult)
            Log.d(
                TAG,
                "frame width=${trackedFrameResult.imageWidth} " +
                    "height=${trackedFrameResult.imageHeight} " +
                    "rotation=${trackedFrameResult.rotationDegrees} " +
                    "timestamp=${trackedFrameResult.timestamp} " +
                    "vehicles=${trackedFrameResult.trackedVehicles.size}"
            )
            onFrameResult(trackedFrameResult)
        } catch (throwable: Throwable) {
            Log.e(TAG, "Vehicle detection failed", throwable)
            onFrameResult(
                TrackedVehicleFrameResult(
                    imageWidth = max(imageProxy.width, 1),
                    imageHeight = max(imageProxy.height, 1),
                    rotationDegrees = imageProxy.imageInfo.rotationDegrees,
                    timestamp = imageProxy.imageInfo.timestamp / 1_000_000L,
                    trackedVehicles = emptyList(),
                )
            )
        } finally {
            imageProxy.close()
        }
    }
}

private fun Bitmap.rotate(rotationDegrees: Int): Bitmap {
    if (rotationDegrees == 0) {
        return this
    }

    val matrix = Matrix().apply {
        postRotate(rotationDegrees.toFloat())
    }
    return Bitmap.createBitmap(this, 0, 0, width, height, matrix, true)
}

private fun Detection.toVehicleDetection(): VehicleDetection? {
    val category = categories().maxByOrNull { it.score() } ?: return null
    val label = (category.displayName().ifBlank { category.categoryName() }).lowercase(Locale.US)

    if (label !in VEHICLE_LABELS) {
        return null
    }

    val box = boundingBox()
    return VehicleDetection(
        bounds = Rect(box.left, box.top, box.right, box.bottom),
        label = label,
        confidence = category.score(),
    )
}
