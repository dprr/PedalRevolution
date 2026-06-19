package com.example.pedalrevolution

import android.graphics.Paint
import android.graphics.Typeface
import androidx.compose.foundation.Canvas
import androidx.compose.foundation.layout.fillMaxSize
import androidx.compose.runtime.Composable
import androidx.compose.runtime.remember
import androidx.compose.ui.Modifier
import androidx.compose.ui.geometry.Offset
import androidx.compose.ui.geometry.Size
import androidx.compose.ui.graphics.Color
import androidx.compose.ui.graphics.drawscope.Stroke
import androidx.compose.ui.graphics.nativeCanvas
import androidx.compose.ui.platform.LocalDensity
import androidx.compose.ui.unit.dp
import androidx.compose.ui.unit.sp
import kotlin.math.roundToInt

@Composable
fun VehicleOverlay(
    frameResult: TrackedVehicleFrameResult?,
    modifier: Modifier = Modifier,
) {
    val density = LocalDensity.current
    val textSize = with(density) { 14.sp.toPx() }
    val strokeWidth = with(density) { 3.dp.toPx() }

    val textPaint = remember(textSize) {
        Paint().apply {
            isAntiAlias = true
            typeface = Typeface.create(Typeface.DEFAULT, Typeface.BOLD)
            this.textSize = textSize
            color = android.graphics.Color.WHITE
        }
    }

    val labelBackgroundPaint = remember {
        Paint().apply {
            isAntiAlias = true
            color = android.graphics.Color.argb(180, 0, 0, 0)
        }
    }

    Canvas(modifier = modifier.fillMaxSize()) {
        val frame = frameResult ?: return@Canvas
        if (frame.imageWidth <= 0 || frame.imageHeight <= 0) return@Canvas

        val scaleX = size.width / frame.imageWidth.toFloat()
        val scaleY = size.height / frame.imageHeight.toFloat()

        frame.trackedVehicles.forEach { vehicle ->
            val highlight = trackColor(vehicle.id)
            val left = vehicle.detection.bounds.left * scaleX
            val top = vehicle.detection.bounds.top * scaleY
            val right = vehicle.detection.bounds.right * scaleX
            val bottom = vehicle.detection.bounds.bottom * scaleY
            val label = "${vehicle.detection.label} #${vehicle.id} ${(vehicle.detection.confidence * 100).roundToInt()}%"

            drawRect(
                color = highlight,
                topLeft = Offset(left, top),
                size = Size(right - left, bottom - top),
                style = Stroke(width = strokeWidth),
            )

            val textWidth = textPaint.measureText(label)
            val textHeight = textPaint.textSize
            val backgroundLeft = left
            val backgroundTop = (top - textHeight - 12.dp.toPx()).coerceAtLeast(0f)
            val backgroundRight = left + textWidth + 24.dp.toPx()
            val backgroundBottom = backgroundTop + textHeight + 16.dp.toPx()

            drawContext.canvas.nativeCanvas.drawRoundRect(
                backgroundLeft,
                backgroundTop,
                backgroundRight,
                backgroundBottom,
                12.dp.toPx(),
                12.dp.toPx(),
                labelBackgroundPaint,
            )

            drawContext.canvas.nativeCanvas.drawText(
                label,
                backgroundLeft + 12.dp.toPx(),
                backgroundBottom - 10.dp.toPx(),
                textPaint,
            )
        }
    }
}

private fun trackColor(trackId: Int): Color {
    val palette = listOf(
        Color(0xFF3DDC84),
        Color(0xFF42A5F5),
        Color(0xFFFFC107),
        Color(0xFFE53935),
        Color(0xFFAB47BC),
        Color(0xFF26C6DA),
    )
    return palette[(trackId - 1) % palette.size]
}
