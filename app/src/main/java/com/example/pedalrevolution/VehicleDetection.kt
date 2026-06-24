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
import java.io.FileInputStream
import java.nio.ByteBuffer
import java.nio.ByteOrder
import java.nio.channels.FileChannel
import java.util.Locale
import kotlin.math.max
import kotlinx.coroutines.CoroutineScope
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.launch
import org.tensorflow.lite.Interpreter
import retrofit2.Retrofit
import retrofit2.converter.kotlinx.serialization.asConverterFactory
import kotlinx.serialization.json.Json
import okhttp3.MediaType.Companion.toMediaType
import ai.onnxruntime.OnnxTensor
import ai.onnxruntime.OrtEnvironment
import ai.onnxruntime.OrtSession
import java.nio.FloatBuffer

private const val TAG = "VehicleDetect"
private const val YOLO_TFLITE_MODEL_FILE = "yolo26n_float32.tflite"
private const val YOLO_ONNX_MODEL_FILE = "yolo26n.onnx"
private const val MEDIAPIPE_MODEL_FILE = "efficientdet-lite0.tflite"
private const val SCORE_THRESHOLD = 0.45f
private const val MAX_RESULTS = 10

private val VEHICLE_LABELS = setOf("car", "bus", "motorcycle", "motorbike", "truck", "person", "bicycle")
private val IGNORE_LABELS = setOf("book")

private val COCO_LABELS = listOf(
    "person", "bicycle", "car", "motorcycle", "airplane", "bus", "train", "truck", "boat",
    "traffic light", "fire hydrant", "stop sign", "parking meter", "bench", "bird", "cat",
    "dog", "horse", "sheep", "cow", "elephant", "bear", "zebra", "giraffe", "backpack",
    "umbrella", "handbag", "tie", "suitcase", "frisbee", "skis", "snowboard", "sports ball",
    "kite", "baseball bat", "baseball glove", "skateboard", "surfboard", "tennis racket",
    "bottle", "wine glass", "cup", "fork", "knife", "spoon", "bowl", "banana", "apple",
    "sandwich", "orange", "broccoli", "carrot", "hot dog", "pizza", "donut", "cake",
    "chair", "couch", "potted plant", "bed", "dining table", "toilet", "tv", "laptop",
    "mouse", "remote", "keyboard", "cell phone", "microwave", "oven", "toaster", "sink",
    "refrigerator", "book", "clock", "vase", "scissors", "teddy bear", "hair drier", "toothbrush",
)

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

class YoloVehicleDetector(context: Context) : VehicleDetector {
    private val interpreter: Interpreter
    private val inputBuffer: ByteBuffer
    private val outputBuffer: Array<Array<FloatArray>>
    private val intValues = IntArray(640 * 640)

    init {
        val model = context.assets.openFd(YOLO_TFLITE_MODEL_FILE).use { fd ->
            FileInputStream(fd.fileDescriptor).channel.map(
                FileChannel.MapMode.READ_ONLY,
                fd.startOffset,
                fd.declaredLength
            )
        }
        interpreter = Interpreter(model)
        inputBuffer = ByteBuffer.allocateDirect(1 * 640 * 640 * 3 * 4).apply {
            order(ByteOrder.nativeOrder())
        }
        // YOLOv10 output is [1, 300, 6]
        outputBuffer = Array(1) { Array(300) { FloatArray(6) } }
    }

    override fun detect(imageProxy: ImageProxy): VehicleFrameResult {
        val rotationDegrees = imageProxy.imageInfo.rotationDegrees
        val bitmap = imageProxy.toBitmap()
        val rotatedBitmap = bitmap.rotate(rotationDegrees)
        if (rotatedBitmap !== bitmap) {
            bitmap.recycle()
        }

        val scaledBitmap = Bitmap.createScaledBitmap(rotatedBitmap, 640, 640, true)

        // Preprocess
        scaledBitmap.getPixels(intValues, 0, 640, 0, 0, 640, 640)
        inputBuffer.rewind()
        for (pixel in intValues) {
            inputBuffer.putFloat(((pixel shr 16) and 0xFF) / 255.0f)
            inputBuffer.putFloat(((pixel shr 8) and 0xFF) / 255.0f)
            inputBuffer.putFloat((pixel and 0xFF) / 255.0f)
        }

        // Run inference
        interpreter.run(inputBuffer, outputBuffer)

        // Postprocess
        val detections = mutableListOf<VehicleDetection>()
        val results = outputBuffer[0]
        for (i in 0 until 300) {
            val score = results[i][4]
            if (score < SCORE_THRESHOLD) continue

            val classIdx = results[i][5].toInt()
            if (classIdx !in COCO_LABELS.indices) continue

            val label = COCO_LABELS[classIdx].lowercase(Locale.US)
            if (label !in VEHICLE_LABELS || (label in IGNORE_LABELS)) continue

            // YOLOv10 output is normalized 0-640 (usually) or 0-1 depending on export
            // Based on common Ultralytics TFLite exports, it's often 0-1 for boxes
            val x1 = results[i][0] * rotatedBitmap.width
            val y1 = results[i][1] * rotatedBitmap.height
            val x2 = results[i][2] * rotatedBitmap.width
            val y2 = results[i][3] * rotatedBitmap.height

            detections.add(
                VehicleDetection(
                    bounds = RectF(x1, y1, x2, y2),
                    label = label,
                    confidence = score
                )
            )
        }

        if (scaledBitmap !== rotatedBitmap) {
            scaledBitmap.recycle()
        }

        val timestamp = imageProxy.imageInfo.timestamp / 1_000_000L

        val result = VehicleFrameResult(
            imageWidth = rotatedBitmap.width,
            imageHeight = rotatedBitmap.height,
            rotationDegrees = rotationDegrees,
            timestamp = timestamp,
            detections = detections
        )

        rotatedBitmap.recycle()
        return result
    }

    override fun close() {
        interpreter.close()
    }
}

class OnnxYoloVehicleDetector(context: Context) : VehicleDetector {
    private val ortEnv: OrtEnvironment = OrtEnvironment.getEnvironment()
    private val ortSession: OrtSession
    private val intValues = IntArray(640 * 640)
    private val floatValues = FloatArray(640 * 640 * 3)

    init {
        val modelBytes = context.assets.open(YOLO_ONNX_MODEL_FILE).use { it.readBytes() }
        ortSession = ortEnv.createSession(modelBytes)
    }

    override fun detect(imageProxy: ImageProxy): VehicleFrameResult {
        val rotationDegrees = imageProxy.imageInfo.rotationDegrees
        val bitmap = imageProxy.toBitmap()
        val rotatedBitmap = bitmap.rotate(rotationDegrees)
        if (rotatedBitmap !== bitmap) {
            bitmap.recycle()
        }

        val scaledBitmap = Bitmap.createScaledBitmap(rotatedBitmap, 640, 640, true)
        scaledBitmap.getPixels(intValues, 0, 640, 0, 0, 640, 640)

        // Preprocess to CHW FloatBuffer
        for (i in intValues.indices) {
            val pixel = intValues[i]
            floatValues[i] = ((pixel shr 16) and 0xFF) / 255.0f // R
            floatValues[i + 640 * 640] = ((pixel shr 8) and 0xFF) / 255.0f // G
            floatValues[i + 2 * 640 * 640] = (pixel and 0xFF) / 255.0f // B
        }

        val inputTensor = OnnxTensor.createTensor(
            ortEnv,
            FloatBuffer.wrap(floatValues),
            longArrayOf(1, 3, 640, 640)
        )

        val detections = mutableListOf<VehicleDetection>()
        inputTensor.use {
            val results = ortSession.run(mapOf(ortSession.inputNames.first() to inputTensor))
            results.use {
                val output = results[0].value as Array<Array<FloatArray>>
                val predictions = output[0] // [300, 6] for YOLOv10

                for (i in predictions.indices) {
                    val score = predictions[i][4]
                    if (score < SCORE_THRESHOLD) continue

                    val classIdx = predictions[i][5].toInt()
                    if (classIdx !in COCO_LABELS.indices) continue

                    val label = COCO_LABELS[classIdx].lowercase(Locale.US)
                    if (label !in VEHICLE_LABELS || (label in IGNORE_LABELS)) continue

                    val x1 = predictions[i][0] * rotatedBitmap.width
                    val y1 = predictions[i][1] * rotatedBitmap.height
                    val x2 = predictions[i][2] * rotatedBitmap.width
                    val y2 = predictions[i][3] * rotatedBitmap.height

                    detections.add(
                        VehicleDetection(
                            bounds = RectF(x1, y1, x2, y2),
                            label = label,
                            confidence = score
                        )
                    )
                }
            }
        }

        if (scaledBitmap !== rotatedBitmap) {
            scaledBitmap.recycle()
        }

        val timestamp = imageProxy.imageInfo.timestamp / 1_000_000L
        val result = VehicleFrameResult(
            imageWidth = rotatedBitmap.width,
            imageHeight = rotatedBitmap.height,
            rotationDegrees = rotationDegrees,
            timestamp = timestamp,
            detections = detections
        )

        rotatedBitmap.recycle()
        return result
    }

    override fun close() {
        ortSession.close()
        ortEnv.close()
    }
}

class MediaPipeVehicleDetector(context: Context) : VehicleDetector {
    private val objectDetector = ObjectDetector.createFromOptions(
        context,
        ObjectDetector.ObjectDetectorOptions.builder()
            .setBaseOptions(
                BaseOptions.builder()
                    .setModelAssetPath(MEDIAPIPE_MODEL_FILE)
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
