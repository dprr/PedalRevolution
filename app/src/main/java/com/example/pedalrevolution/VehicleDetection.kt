package com.example.pedalrevolution

import android.content.Context
import android.graphics.Bitmap
import android.graphics.Matrix
import android.graphics.RectF
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
import kotlinx.coroutines.CoroutineScope
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.launch
import okhttp3.OkHttpClient
import okhttp3.logging.HttpLoggingInterceptor
import retrofit2.Retrofit
import retrofit2.converter.kotlinx.serialization.asConverterFactory
import kotlinx.serialization.json.Json
import okhttp3.MediaType.Companion.toMediaType

private const val TAG = "VehicleDetect"
private const val MODEL_FILE = "efficientdet-lite0.tflite"
private const val SCORE_THRESHOLD = 0.45f
private const val MAX_RESULTS = 10

private val VEHICLE_LABELS = setOf("car", "bus", "motorcycle", "motorbike", "truck")
private val IGNORE_LABELS = setOf("bicycle")

data class VehicleDetection(
    val bounds: RectF,
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
    private val apiService: DetectionApiService,
    private val locationProvider: () -> android.location.Location?,
    private val onFrameResult: (TrackedVehicleFrameResult) -> Unit,
) : ImageAnalysis.Analyzer {
    private val scope = CoroutineScope(Dispatchers.IO)

    override fun analyze(imageProxy: ImageProxy) {
        try {
            val frameResult = detector.detect(imageProxy)
            
            // Send detections to cloud
            if (frameResult.detections.isNotEmpty()) {
                val loc = locationProvider()
                val requests = frameResult.detections.map { 
                    DetectionRequest(
                        timestamp_ms = frameResult.timestamp,
                        label = it.label,
                        confidence = it.confidence,
                        x_min = it.bounds.left / frameResult.imageWidth,
                        y_min = it.bounds.top / frameResult.imageHeight,
                        x_max = it.bounds.right / frameResult.imageWidth,
                        y_max = it.bounds.bottom / frameResult.imageHeight,
                        latitude = loc?.latitude,
                        longitude = loc?.longitude,
                        altitude = if (loc?.hasAltitude() == true) loc.altitude else null
                    )
                }
                scope.launch {
                    try {
                        apiService.sendDetections(requests)
                    } catch (e: Exception) {
                        Log.e(TAG, "Failed to send detections", e)
                    }
                }
            }

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

    if (label in IGNORE_LABELS) {
        return null
    }
    if (label !in VEHICLE_LABELS) {
        return null
    }

    return VehicleDetection(
        bounds = RectF(boundingBox()),
        label = label,
        confidence = category.score(),
    )
}
