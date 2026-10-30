package com.example.objectscanner

import android.app.Application
import android.graphics.Bitmap
import androidx.lifecycle.AndroidViewModel
import kotlinx.coroutines.flow.MutableStateFlow
import kotlinx.coroutines.flow.StateFlow
import kotlinx.coroutines.flow.asStateFlow
import kotlinx.coroutines.flow.update

data class ScannerUiState(
    val detections: List<Detection> = emptyList(),
    val inferenceTimeMs: Long = 0L,
    /** Upright analysis frame size. The overlay needs it to map boxes onto the screen. */
    val frameWidth: Int = 0,
    val frameHeight: Int = 0,
    val backend: String = "",
    val error: String? = null,
)

class ScannerViewModel(application: Application) :
    AndroidViewModel(application), ObjectDetectorHelper.DetectorListener {

    private val _uiState = MutableStateFlow(ScannerUiState())
    val uiState: StateFlow<ScannerUiState> = _uiState.asStateFlow()

    private val detector = ObjectDetectorHelper(application, listener = this)

    val canAcceptFrame: Boolean get() = detector.canAcceptFrame

    fun onFrame(bitmap: Bitmap, rotationDegrees: Int) {
        detector.detect(bitmap, rotationDegrees)
    }

    // Called on the detector thread. StateFlow is thread safe, Compose picks it up on main.
    override fun onResults(result: DetectionResult) = _uiState.update {
        it.copy(
            detections = result.detections,
            inferenceTimeMs = result.inferenceTimeMs,
            frameWidth = result.frameWidth,
            frameHeight = result.frameHeight,
            backend = result.backend,
            error = null,
        )
    }

    override fun onError(message: String) = _uiState.update { it.copy(error = message) }

    override fun onCleared() = detector.close()
}
