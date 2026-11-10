package com.example.pedalrevolution

import androidx.camera.core.CameraSelector
import androidx.camera.core.ImageAnalysis
import androidx.camera.lifecycle.ProcessCameraProvider
import androidx.camera.view.PreviewView
import androidx.compose.foundation.layout.Box
import androidx.compose.foundation.layout.fillMaxSize
import androidx.compose.material3.MaterialTheme
import androidx.compose.material3.Surface
import androidx.compose.material3.Text
import androidx.compose.runtime.Composable
import androidx.compose.runtime.DisposableEffect
import androidx.compose.runtime.getValue
import androidx.compose.runtime.mutableStateOf
import androidx.compose.runtime.remember
import androidx.compose.runtime.setValue
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.layout.onSizeChanged
import androidx.compose.ui.platform.LocalContext
import androidx.compose.ui.unit.IntSize
import androidx.compose.ui.viewinterop.AndroidView
import androidx.core.content.ContextCompat
import androidx.lifecycle.compose.LocalLifecycleOwner
import java.util.concurrent.Executors
import java.util.concurrent.atomic.AtomicBoolean

@Composable
fun VehicleCameraScreen() {
    val context = LocalContext.current
    val lifecycleOwner = LocalLifecycleOwner.current
    val cameraProviderFuture = remember(context) { ProcessCameraProvider.getInstance(context) }
    val mainExecutor = remember(context) { ContextCompat.getMainExecutor(context) }
    val detectorResult = remember(context) { runCatching { MediaPipeVehicleDetector(context) } }

    if (detectorResult.isFailure) {
        Surface(modifier = Modifier.fillMaxSize()) {
            Box(
                modifier = Modifier.fillMaxSize(),
                contentAlignment = Alignment.Center,
            ) {
                Text(
                    text = "Vehicle detector failed to load: ${detectorResult.exceptionOrNull()?.message}",
                    style = MaterialTheme.typography.bodyLarge,
                )
            }
        }
        return
    }

    val detector = detectorResult.getOrThrow()
    val cameraExecutor = remember { Executors.newSingleThreadExecutor() }
    val tracker = remember { VehicleTracker() }
    var previewView by remember { mutableStateOf<PreviewView?>(null) }
    var previewSize by remember { mutableStateOf(IntSize.Zero) }
    var cameraProvider by remember { mutableStateOf<ProcessCameraProvider?>(null) }
    var cameraAnalysis by remember { mutableStateOf<ImageAnalysis?>(null) }
    val detectorDisposed = remember(detector) { AtomicBoolean(false) }
    var frameResult by remember { mutableStateOf<TrackedVehicleFrameResult?>(null) }

    Surface(modifier = Modifier.fillMaxSize()) {
        Box(modifier = Modifier.fillMaxSize()) {
            AndroidView(
                factory = { viewContext ->
                    PreviewView(viewContext).apply {
                        scaleType = PreviewView.ScaleType.FIT_CENTER
                        setBackgroundColor(android.graphics.Color.BLACK)
                        previewView = this
                    }
                },
                modifier = Modifier.fillMaxSize().onSizeChanged { previewSize = it },
            )

            VehicleOverlay(
                frameResult = frameResult,
                modifier = Modifier.fillMaxSize(),
            )
        }
    }

    // Owns the detector and its worker for this screen.
    DisposableEffect(detector, cameraExecutor) {
        onDispose {
            detectorDisposed.set(true)

            // Stop new frames before scheduling detector closure.
            cameraAnalysis?.clearAnalyzer()
            cameraProvider?.unbindAll()

            // The current frame finishes before this task runs.
            cameraExecutor.execute {
                detector.close()
            }

            // Allow queued tasks, including close(), to finish.
            cameraExecutor.shutdown()
        }
    }

    // Owns the camera binding. This can restart when the preview size changes.
    DisposableEffect(previewView, previewSize, lifecycleOwner, detector, tracker) {
        val view = previewView

        if (view == null || previewSize.width == 0 || previewSize.height == 0) {
            onDispose { }
        } else {
            val bindingDisposed = AtomicBoolean(false)
            var boundProvider: ProcessCameraProvider? = null
            var boundAnalysis: ImageAnalysis? = null

            val listener = Runnable {
                if (bindingDisposed.get() || detectorDisposed.get()) {
                    return@Runnable
                }

                val provider = runCatching {
                    cameraProviderFuture.get()
                }.getOrNull() ?: return@Runnable

                boundProvider = provider
                cameraProvider = provider

                val targetRotation =
                    view.display?.rotation ?: android.view.Surface.ROTATION_0

                val preview = androidx.camera.core.Preview.Builder()
                    .setTargetRotation(targetRotation)
                    .build()
                    .also { it.surfaceProvider = view.surfaceProvider }

                val analysis = ImageAnalysis.Builder()
                    .setTargetRotation(targetRotation)
                    .setBackpressureStrategy(
                        ImageAnalysis.STRATEGY_KEEP_ONLY_LATEST
                    )
                    .setOutputImageFormat(
                        ImageAnalysis.OUTPUT_IMAGE_FORMAT_RGBA_8888
                    )
                    .build()

                boundAnalysis = analysis
                cameraAnalysis = analysis

                val analyzer = VehicleFrameAnalyzer(detector, tracker) { result ->
                    mainExecutor.execute {
                        if (!bindingDisposed.get() && !detectorDisposed.get()) {
                            frameResult = result
                        }
                    }
                }

                analysis.setAnalyzer(cameraExecutor) { image ->
                    if (bindingDisposed.get() || detectorDisposed.get()) {
                        image.close()
                    } else {
                        analyzer.analyze(image)
                    }
                }

                provider.unbindAll()
                provider.bindToLifecycle(
                    lifecycleOwner,
                    CameraSelector.DEFAULT_BACK_CAMERA,
                    preview,
                    analysis,
                )
            }

            cameraProviderFuture.addListener(listener, mainExecutor)

            onDispose {
                bindingDisposed.set(true)
                boundAnalysis?.clearAnalyzer()
                boundProvider?.unbindAll()
            }
        }
    }
}
