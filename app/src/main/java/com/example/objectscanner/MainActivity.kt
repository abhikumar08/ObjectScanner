package com.example.objectscanner

import android.Manifest
import android.content.Intent
import android.content.pm.PackageManager
import android.net.Uri
import android.os.Bundle
import android.provider.Settings
import androidx.activity.ComponentActivity
import androidx.activity.SystemBarStyle
import androidx.activity.compose.rememberLauncherForActivityResult
import androidx.activity.compose.setContent
import androidx.activity.enableEdgeToEdge
import androidx.activity.result.contract.ActivityResultContracts
import androidx.compose.foundation.Canvas
import androidx.compose.foundation.background
import androidx.compose.foundation.layout.Arrangement
import androidx.compose.foundation.layout.Box
import androidx.compose.foundation.layout.Column
import androidx.compose.foundation.layout.Spacer
import androidx.compose.foundation.layout.fillMaxSize
import androidx.compose.foundation.layout.height
import androidx.compose.foundation.layout.padding
import androidx.compose.foundation.layout.safeDrawingPadding
import androidx.compose.foundation.shape.RoundedCornerShape
import androidx.compose.material3.Button
import androidx.compose.material3.MaterialTheme
import androidx.compose.material3.Text
import androidx.compose.material3.TextButton
import androidx.compose.material3.darkColorScheme
import androidx.compose.runtime.Composable
import androidx.compose.runtime.LaunchedEffect
import androidx.compose.runtime.getValue
import androidx.compose.runtime.mutableStateOf
import androidx.compose.runtime.remember
import androidx.compose.runtime.saveable.rememberSaveable
import androidx.compose.runtime.setValue
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.geometry.Offset
import androidx.compose.ui.geometry.Size
import androidx.compose.ui.graphics.Color
import androidx.compose.ui.graphics.drawscope.Stroke
import androidx.compose.ui.platform.LocalContext
import androidx.compose.ui.text.TextStyle
import androidx.compose.ui.text.drawText
import androidx.compose.ui.text.font.FontWeight
import androidx.compose.ui.text.rememberTextMeasurer
import androidx.compose.ui.text.style.TextAlign
import androidx.compose.ui.unit.dp
import androidx.compose.ui.unit.sp
import androidx.core.content.ContextCompat
import androidx.lifecycle.compose.LifecycleResumeEffect
import androidx.lifecycle.compose.collectAsStateWithLifecycle
import androidx.lifecycle.viewmodel.compose.viewModel
import kotlin.math.max
import kotlin.math.roundToInt

class MainActivity : ComponentActivity() {
    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        // Light system bar icons: the camera screen and permission screen are both dark.
        enableEdgeToEdge(
            statusBarStyle = SystemBarStyle.dark(android.graphics.Color.TRANSPARENT),
            navigationBarStyle = SystemBarStyle.dark(android.graphics.Color.TRANSPARENT),
        )
        setContent {
            MaterialTheme(colorScheme = darkColorScheme()) {
                CameraPermissionGate { ScannerScreen() }
            }
        }
    }
}

@Composable
private fun CameraPermissionGate(content: @Composable () -> Unit) {
    val context = LocalContext.current
    fun isGranted() = ContextCompat.checkSelfPermission(context, Manifest.permission.CAMERA) ==
        PackageManager.PERMISSION_GRANTED

    var granted by remember { mutableStateOf(isGranted()) }
    var asked by rememberSaveable { mutableStateOf(false) }
    val launcher = rememberLauncherForActivityResult(ActivityResultContracts.RequestPermission()) {
        granted = it
    }

    // Re-check on resume so a grant from system Settings is picked up.
    LifecycleResumeEffect(Unit) {
        granted = isGranted()
        onPauseOrDispose {}
    }
    LaunchedEffect(Unit) {
        if (!granted && !asked) {
            asked = true
            launcher.launch(Manifest.permission.CAMERA)
        }
    }

    if (granted) {
        content()
        return
    }
    Column(
        modifier = Modifier.fillMaxSize().background(Color.Black).padding(24.dp),
        verticalArrangement = Arrangement.Center,
        horizontalAlignment = Alignment.CenterHorizontally,
    ) {
        Text("Camera access is needed to detect objects.", color = Color.White, textAlign = TextAlign.Center)
        Spacer(Modifier.height(16.dp))
        Button(onClick = { launcher.launch(Manifest.permission.CAMERA) }) { Text("Grant access") }
        TextButton(onClick = {
            context.startActivity(
                Intent(Settings.ACTION_APPLICATION_DETAILS_SETTINGS, Uri.fromParts("package", context.packageName, null))
            )
        }) { Text("Open settings") }
    }
}

@Composable
private fun ScannerScreen(viewModel: ScannerViewModel = viewModel()) {
    val state by viewModel.uiState.collectAsStateWithLifecycle()

    Box(Modifier.fillMaxSize().background(Color.Black)) {
        CameraPreviewScanner(
            canAcceptFrame = { viewModel.canAcceptFrame },
            onFrame = viewModel::onFrame,
            modifier = Modifier.fillMaxSize(),
        )
        DetectionOverlay(state, Modifier.fillMaxSize())
        Text(
            text = state.error
                ?: if (state.backend.isEmpty()) "Loading model..."
                else "YOLOv8 | ${state.backend} | ${state.inferenceTimeMs} ms | ${state.detections.size} objects",
            color = Color.White,
            fontSize = 14.sp,
            modifier = Modifier
                .align(Alignment.TopStart)
                .safeDrawingPadding()
                .padding(12.dp)
                .background(Color.Black.copy(alpha = 0.6f), RoundedCornerShape(8.dp))
                .padding(horizontal = 10.dp, vertical = 6.dp),
        )
    }
}

/**
 * Draws boxes in the PreviewView's coordinate space. FILL_CENTER scales the frame by
 * max(viewW / frameW, viewH / frameH) and crops the overflow equally on both sides,
 * so: screen = fraction * frameSize * scale + offset.
 */
@Composable
private fun DetectionOverlay(state: ScannerUiState, modifier: Modifier = Modifier) {
    val textMeasurer = rememberTextMeasurer()
    val labelStyle = TextStyle(color = Color.White, fontSize = 13.sp, fontWeight = FontWeight.SemiBold)

    Canvas(modifier) {
        if (state.frameWidth == 0 || state.frameHeight == 0) return@Canvas
        val frameW = state.frameWidth.toFloat()
        val frameH = state.frameHeight.toFloat()
        val scale = max(size.width / frameW, size.height / frameH)
        val offsetX = (size.width - frameW * scale) / 2f
        val offsetY = (size.height - frameH * scale) / 2f
        val stroke = Stroke(width = 3.dp.toPx())
        val labelPadding = 4.dp.toPx()

        for (d in state.detections) {
            val color = BoxColors[d.classId % BoxColors.size]
            val topLeft = Offset(d.box.left * frameW * scale + offsetX, d.box.top * frameH * scale + offsetY)
            drawRect(color, topLeft, Size(d.box.width() * frameW * scale, d.box.height() * frameH * scale), style = stroke)

            // Keep the label on screen even when FILL_CENTER crops part of the box.
            val label = textMeasurer.measure("${d.label} ${(d.score * 100).roundToInt()}%", labelStyle)
            val labelWidth = label.size.width + labelPadding * 2
            val labelLeft = topLeft.x.coerceIn(0f, (size.width - labelWidth).coerceAtLeast(0f))
            val labelTop = (topLeft.y - label.size.height).coerceAtLeast(0f)
            drawRect(color, Offset(labelLeft, labelTop), Size(labelWidth, label.size.height.toFloat()))
            drawText(label, topLeft = Offset(labelLeft + labelPadding, labelTop))
        }
    }
}

private val BoxColors = listOf(
    Color(0xFF00C853), Color(0xFF2962FF), Color(0xFFFF6D00),
    Color(0xFFD50000), Color(0xFFAA00FF), Color(0xFF00B8D4),
)
