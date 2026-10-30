package com.example.objectscanner

import android.graphics.Bitmap
import android.util.Log
import android.util.Range
import android.util.Size
import androidx.camera.core.CameraSelector
import androidx.camera.core.ImageAnalysis
import androidx.camera.core.Preview
import androidx.camera.core.resolutionselector.AspectRatioStrategy
import androidx.camera.core.resolutionselector.ResolutionSelector
import androidx.camera.core.resolutionselector.ResolutionStrategy
import androidx.camera.lifecycle.ProcessCameraProvider
import androidx.camera.view.PreviewView
import androidx.compose.runtime.Composable
import androidx.compose.runtime.DisposableEffect
import androidx.compose.runtime.getValue
import androidx.compose.runtime.remember
import androidx.compose.runtime.rememberUpdatedState
import androidx.compose.ui.Modifier
import androidx.compose.ui.platform.LocalContext
import androidx.compose.ui.viewinterop.AndroidView
import androidx.core.content.ContextCompat
import androidx.lifecycle.compose.LocalLifecycleOwner
import java.util.concurrent.Executors

/**
 * Back camera viewfinder that hands 30 FPS frames to [onFrame] on a background thread.
 *
 * Preview and analysis both run at 4:3, so they share one field of view. That lets the overlay
 * map a frame onto this FILL_CENTER [PreviewView] with a single scale plus a centered crop.
 */
@Composable
fun CameraPreviewScanner(
    canAcceptFrame: () -> Boolean,
    onFrame: (bitmap: Bitmap, rotationDegrees: Int) -> Unit,
    modifier: Modifier = Modifier,
) {
    val context = LocalContext.current
    val lifecycleOwner = LocalLifecycleOwner.current
    val currentCanAcceptFrame by rememberUpdatedState(canAcceptFrame)
    val currentOnFrame by rememberUpdatedState(onFrame)
    val previewView = remember {
        PreviewView(context).apply { scaleType = PreviewView.ScaleType.FILL_CENTER }
    }

    DisposableEffect(lifecycleOwner) {
        val analysisExecutor = Executors.newSingleThreadExecutor()
        val providerFuture = ProcessCameraProvider.getInstance(context)
        var disposed = false

        providerFuture.addListener({
            if (disposed) return@addListener
            val preview = Preview.Builder()
                .setResolutionSelector(resolutionSelector())
                .setTargetFrameRate(Range(30, 30))
                .build()
                .also { it.setSurfaceProvider(previewView.surfaceProvider) }

            // 640x480 RGBA is close to the 640x640 model input and makes toBitmap() a plain copy.
            val analysis = ImageAnalysis.Builder()
                .setResolutionSelector(resolutionSelector(Size(640, 480)))
                .setBackpressureStrategy(ImageAnalysis.STRATEGY_KEEP_ONLY_LATEST)
                .setOutputImageFormat(ImageAnalysis.OUTPUT_IMAGE_FORMAT_RGBA_8888)
                .build()
            analysis.setAnalyzer(analysisExecutor) { imageProxy ->
                try {
                    // Skip the bitmap copy entirely while the detector is still on the last frame.
                    if (currentCanAcceptFrame()) {
                        currentOnFrame(imageProxy.toBitmap(), imageProxy.imageInfo.rotationDegrees)
                    }
                } finally {
                    imageProxy.close()
                }
            }

            try {
                providerFuture.get().apply {
                    unbindAll()
                    bindToLifecycle(lifecycleOwner, CameraSelector.DEFAULT_BACK_CAMERA, preview, analysis)
                }
            } catch (e: Exception) {
                Log.e("CameraPreviewScanner", "Camera bind failed", e)
            }
        }, ContextCompat.getMainExecutor(context))

        onDispose {
            disposed = true
            if (providerFuture.isDone) providerFuture.get().unbindAll()
            analysisExecutor.shutdown()
        }
    }

    AndroidView(factory = { previewView }, modifier = modifier)
}

private fun resolutionSelector(target: Size? = null): ResolutionSelector =
    ResolutionSelector.Builder()
        .setAspectRatioStrategy(AspectRatioStrategy.RATIO_4_3_FALLBACK_AUTO_STRATEGY)
        .apply {
            if (target != null) {
                setResolutionStrategy(
                    ResolutionStrategy(target, ResolutionStrategy.FALLBACK_RULE_CLOSEST_HIGHER_THEN_LOWER)
                )
            }
        }
        .build()
