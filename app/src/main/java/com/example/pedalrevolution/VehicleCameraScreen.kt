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
import androidx.compose.ui.platform.LocalContext
import androidx.compose.ui.viewinterop.AndroidView
import androidx.core.content.ContextCompat
import androidx.lifecycle.compose.LocalLifecycleOwner
import java.util.concurrent.Executors
import android.annotation.SuppressLint
import com.google.android.gms.location.LocationServices
import com.google.android.gms.location.LocationRequest
import com.google.android.gms.location.Priority
import com.google.android.gms.location.LocationCallback
import com.google.android.gms.location.LocationResult

@SuppressLint("MissingPermission")
@Composable
fun VehicleCameraScreen() {
    val context = LocalContext.current
    val lifecycleOwner = LocalLifecycleOwner.current
    val cameraProviderFuture = remember(context) { ProcessCameraProvider.getInstance(context) }
    val mainExecutor = remember(context) { ContextCompat.getMainExecutor(context) }
    val detectorResult = remember(context) { runCatching { YoloVehicleDetector(context) } }
//    val detectorResult = remember(context) { runCatching { MediaPipeVehicleDetector(context) } }

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
    var cameraProvider by remember { mutableStateOf<ProcessCameraProvider?>(null) }
    var frameResult by remember { mutableStateOf<TrackedVehicleFrameResult?>(null) }
    var currentLocation by remember { mutableStateOf<android.location.Location?>(null) }

    val apiService = remember { DetectionApiService.create() }
    val fusedLocationClient = remember(context) { LocationServices.getFusedLocationProviderClient(context) }

    DisposableEffect(fusedLocationClient) {
        val locationRequest = LocationRequest.Builder(Priority.PRIORITY_HIGH_ACCURACY, 1000L).build()
        val callback = object : LocationCallback() {
            override fun onLocationResult(result: LocationResult) {
                result.lastLocation?.let { currentLocation = it }
            }
        }
        try {
            fusedLocationClient.requestLocationUpdates(locationRequest, mainExecutor, callback)
        } catch (e: SecurityException) {
            // Ignore if missing permissions
        }
        onDispose {
            fusedLocationClient.removeLocationUpdates(callback)
        }
    }

    Surface(modifier = Modifier.fillMaxSize()) {
        Box(modifier = Modifier.fillMaxSize()) {
            AndroidView(
                factory = { viewContext ->
                    PreviewView(viewContext).apply {
                        scaleType = PreviewView.ScaleType.FILL_CENTER
                        previewView = this
                    }
                },
                modifier = Modifier.fillMaxSize(),
            )

            VehicleOverlay(
                frameResult = frameResult,
                modifier = Modifier.fillMaxSize(),
            )
        }
    }

    DisposableEffect(detector) {
        onDispose {
            detector.close()
        }
    }

    DisposableEffect(previewView, lifecycleOwner, detector, tracker) {
        val view = previewView
        if (view == null) {
            onDispose { }
        } else {
            val listener = Runnable {
                val provider = runCatching { cameraProviderFuture.get() }.getOrNull() ?: return@Runnable
                cameraProvider = provider

                val targetRotation = view.display?.rotation ?: android.view.Surface.ROTATION_0
                val preview = androidx.camera.core.Preview.Builder()
                    .setTargetRotation(targetRotation)
                    .build()
                    .also { it.surfaceProvider = view.surfaceProvider }

                val analysis = ImageAnalysis.Builder()
                    .setTargetRotation(targetRotation)
                    .setBackpressureStrategy(ImageAnalysis.STRATEGY_KEEP_ONLY_LATEST)
                    .setOutputImageFormat(ImageAnalysis.OUTPUT_IMAGE_FORMAT_RGBA_8888)
                    .build()
                    .also { useCase ->
                        useCase.setAnalyzer(
                            cameraExecutor,
                            VehicleFrameAnalyzer(detector, tracker, apiService, { currentLocation }) { result ->
                                mainExecutor.execute {
                                    frameResult = result
                                }
                            }
                        )
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
                cameraProvider?.unbindAll()
            }
        }
    }

    DisposableEffect(Unit) {
        onDispose {
            cameraExecutor.shutdown()
        }
    }
}
