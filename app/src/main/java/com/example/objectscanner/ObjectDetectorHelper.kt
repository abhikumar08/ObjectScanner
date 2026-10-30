package com.example.objectscanner

import android.content.Context
import android.graphics.Bitmap
import android.graphics.Canvas
import android.graphics.Color
import android.graphics.Matrix
import android.graphics.Paint
import android.graphics.RectF
import android.os.Build
import android.os.SystemClock
import android.util.Log
import androidx.core.content.edit
import androidx.core.graphics.createBitmap
import com.qualcomm.qti.QnnDelegate
import com.qualcomm.qti.QnnDelegate.Options.BackendType
import org.tensorflow.lite.DataType
import org.tensorflow.lite.Delegate
import org.tensorflow.lite.Interpreter
import org.tensorflow.lite.Tensor
import org.tensorflow.lite.gpu.CompatibilityList
import org.tensorflow.lite.gpu.GpuDelegate
import org.tensorflow.lite.gpu.GpuDelegateFactory
import org.tensorflow.lite.nnapi.NnApiDelegate
import java.io.File
import java.io.FileInputStream
import java.nio.ByteBuffer
import java.nio.ByteOrder
import java.nio.channels.FileChannel
import java.util.concurrent.Executors
import java.util.concurrent.atomic.AtomicBoolean
import kotlin.math.max
import kotlin.math.min
import kotlin.math.roundToInt

data class ObjectDetectorOptions(
    val modelPath: String = "object_detector.tflite",
    val labelsPath: String = "coco_labels.txt",
    val scoreThreshold: Float = 0.5f,
    val iouThreshold: Float = 0.45f,
    val maxResults: Int = 25,
    val numThreads: Int = 4,
    /** Try the NPU first: Qualcomm HTP, then Qualcomm DSP, then NNAPI. Falls back to GPU and CPU. */
    val useNpu: Boolean = true,
    /** Try the GPU delegate before the CPU. Falls back to CPU when the device or model can't use it. */
    val useGpu: Boolean = true,
)

/** [box] edges are fractions (0..1) of the upright camera frame. */
data class Detection(val classId: Int, val label: String, val score: Float, val box: RectF)

data class DetectionResult(
    val detections: List<Detection>,
    val inferenceTimeMs: Long,
    val frameWidth: Int,
    val frameHeight: Int,
    /** "NPU", "DSP", "NNAPI", "GPU" or "CPU": where inference actually ran. */
    val backend: String,
)

/**
 * Runs the Qualcomm AI Hub YOLOv8 export (`qai-hub-models export yolov8_det`) with the LiteRT
 * Interpreter. The Task Library ObjectDetector can't load this model: it needs TFLite metadata
 * and SSD-style outputs. This export has no metadata and emits raw candidates instead:
 * boxes [1, N, 4] as xyxy input pixels, scores [1, N], class_idx [1, N] uint8, and no NMS.
 *
 * Everything model related runs on one background thread. The interpreter never leaves it,
 * which the GPU delegate requires: it must be created and run on the same thread.
 */
class ObjectDetectorHelper(
    context: Context,
    private val listener: DetectorListener,
    private val options: ObjectDetectorOptions = ObjectDetectorOptions(),
) {
    interface DetectorListener {
        fun onResults(result: DetectionResult)
        fun onError(message: String)
    }

    private val executor = Executors.newSingleThreadExecutor()
    private val busy = AtomicBoolean(false)
    @Volatile private var ready = false

    // Owned by the executor thread.
    private var backend: Backend? = null
    private lateinit var labels: List<String>
    private lateinit var input: ModelInput
    private lateinit var boxes: ModelOutput
    private lateinit var scores: ModelOutput
    private lateinit var classes: ModelOutput
    private lateinit var outputs: Map<Int, Any>
    private val transform = Matrix()

    /** False while a frame is in flight or the model is not loaded. Check it before copying a frame. */
    val canAcceptFrame: Boolean get() = ready && !busy.get()

    init {
        val appContext = context.applicationContext
        executor.execute { setup(appContext) }
    }

    /**
     * Queues [bitmap] for detection and takes ownership of it. [rotation] is the clockwise
     * rotation that makes the frame upright. Returns false and drops the frame when busy.
     */
    fun detect(bitmap: Bitmap, rotation: Int): Boolean {
        if (!ready || !busy.compareAndSet(false, true)) return false
        val upright = rotation % 180 == 0
        val frameWidth = if (upright) bitmap.width else bitmap.height
        val frameHeight = if (upright) bitmap.height else bitmap.width

        val task = Runnable {
            try {
                val start = SystemClock.uptimeMillis()
                val detections = runDetection(bitmap, rotation, frameWidth, frameHeight)
                val elapsed = SystemClock.uptimeMillis() - start
                listener.onResults(DetectionResult(detections, elapsed, frameWidth, frameHeight, backend!!.name))
            } catch (e: Exception) {
                Log.e(TAG, "Detection failed", e)
                listener.onError("Detection failed: ${e.message}")
            } finally {
                bitmap.recycle()
                busy.set(false)
            }
        }
        return runCatching { executor.execute(task) }.onFailure { busy.set(false) }.isSuccess
    }

    fun close() {
        ready = false
        executor.execute {
            backend?.close()
            backend = null
        }
        executor.shutdown()
    }

    private fun setup(context: Context) {
        try {
            val model = context.assets.openFd(options.modelPath).use { fd ->
                FileInputStream(fd.fileDescriptor).use {
                    it.channel.map(FileChannel.MapMode.READ_ONLY, fd.startOffset, fd.declaredLength)
                }
            }
            val backend = createBackend(context, model)
            val interpreter = backend.interpreter
            Log.i(TAG, "Running on ${backend.name}")
            labels = context.assets.open(options.labelsPath).bufferedReader().readLines()
            input = ModelInput(interpreter.getInputTensor(0))

            // Find outputs by shape, name, then type, so tensor order in the export doesn't matter.
            val tensors = (0 until interpreter.outputTensorCount).map { interpreter.getOutputTensor(it) }
            val boxesIndex = tensors.indexOfFirst { it.numDimensions() == 3 && it.shape()[2] == 4 }
            require(tensors.size == 3 && boxesIndex >= 0) {
                "Expected YOLOv8 outputs boxes, scores, class_idx. Got ${tensors.map { it.shape().contentToString() }}"
            }
            val rest = tensors.indices - boxesIndex
            val classIndex = rest.firstOrNull { tensors[it].name().contains("class", ignoreCase = true) }
                ?: rest.firstOrNull { tensors[it].dataType() != DataType.FLOAT32 && tensors[it].quantizationParams().scale == 0f }
                ?: rest.last()
            val scoresIndex = rest.first { it != classIndex }
            boxes = ModelOutput(tensors[boxesIndex])
            scores = ModelOutput(tensors[scoresIndex])
            classes = ModelOutput(tensors[classIndex])
            outputs = mapOf(boxesIndex to boxes.buffer, scoresIndex to scores.buffer, classIndex to classes.buffer)

            this.backend = backend
            ready = true
        } catch (e: Exception) {
            Log.e(TAG, "Model setup failed", e)
            listener.onError("Could not load assets/${options.modelPath}: ${e.message}")
        }
    }

    /**
     * Picks the first backend that loads, in order: Qualcomm NPU (HTP), Qualcomm DSP, NNAPI, GPU,
     * then CPU (XNNPACK). QNN only loads on Snapdragons it supports, so it is used whenever it
     * loads. NNAPI can't report which chip it runs on, so it must also beat the GPU (or CPU).
     *
     * Compiled NPU, DSP and GPU programs are cached in codeCacheDir, which the system clears on
     * app updates, so only the first launch after an install pays the compile cost.
     */
    private fun createBackend(context: Context, model: ByteBuffer): Backend {
        val cacheToken = "${options.modelPath.substringAfterLast('/')}-${model.capacity()}"
        if (options.useNpu) {
            createQnn(context, model, cacheToken, BackendType.HTP_BACKEND)?.let { return it }
            createQnn(context, model, cacheToken, BackendType.DSP_BACKEND)?.let { return it }
        }
        val fallback = (if (options.useGpu) createGpu(context, model, cacheToken) else null) ?: createCpu(model)
        return if (options.useNpu) nnapiIfFaster(context, model, cacheToken, fallback) else fallback
    }

    /** Qualcomm AI Engine Direct on the HTP (NPU) or the older Hexagon DSP. Works on any API level. */
    private fun createQnn(context: Context, model: ByteBuffer, cacheToken: String, type: BackendType): Backend? {
        val htp = type == BackendType.HTP_BACKEND
        val name = if (htp) "NPU" else "DSP"
        return attempt(name) {
            // The bundled model is int8. A float model fails here and falls through to GPU.
            val capability = if (htp) QnnDelegate.Capability.HTP_RUNTIME_QUANTIZED else QnnDelegate.Capability.DSP_RUNTIME
            if (!QnnDelegate.checkCapability(capability)) {
                Log.i(TAG, "$name: not supported on this device")
                return@attempt null
            }
            val qnnOptions = QnnDelegate.Options().apply {
                setBackendType(type)
                setSkelLibraryDir(context.applicationInfo.nativeLibraryDir)
                setCacheDir(cacheDir(context, "qnn"))
                setModelToken(cacheToken)
                if (htp) {
                    setHtpPrecision(QnnDelegate.Options.HtpPrecision.HTP_PRECISION_QUANTIZED)
                    setHtpPerformanceMode(QnnDelegate.Options.HtpPerformanceMode.HTP_PERFORMANCE_SUSTAINED_HIGH_PERFORMANCE)
                } else {
                    setDspPerformanceMode(QnnDelegate.Options.DspPerformanceMode.DSP_PERFORMANCE_SUSTAINED_HIGH_PERFORMANCE)
                }
            }
            withDelegate(name, QnnDelegate(qnnOptions), model)
        }
    }

    private fun createGpu(context: Context, model: ByteBuffer, cacheToken: String): Backend? = attempt("GPU") {
        CompatibilityList().use { compat ->
            if (!compat.isDelegateSupportedOnThisDevice) return@attempt null
            val delegate = GpuDelegate(
                compat.bestOptionsForThisDevice
                    .setInferencePreference(GpuDelegateFactory.Options.INFERENCE_PREFERENCE_SUSTAINED_SPEED)
                    .setSerializationParams(context.codeCacheDir.absolutePath, cacheToken)
            )
            withDelegate("GPU", delegate, model)
        }
    }

    private fun createNnapi(context: Context, model: ByteBuffer, cacheToken: String): Backend? = attempt("NNAPI") {
        val nnapiOptions = NnApiDelegate.Options()
            .setUseNnapiCpu(false) // Never NNAPI's slow reference CPU implementation.
            .setExecutionPreference(NnApiDelegate.Options.EXECUTION_PREFERENCE_SUSTAINED_SPEED)
            .setCacheDir(cacheDir(context, "nnapi"))
            .setModelToken(cacheToken)
        withDelegate("NNAPI", NnApiDelegate(nnapiOptions), model)
    }

    private fun createCpu(model: ByteBuffer) =
        Backend("CPU", Interpreter(model, Interpreter.Options().setNumThreads(options.numThreads)), null)

    /**
     * Returns NNAPI instead of [fallback] when it is clearly faster on this device. The probe runs
     * once per OS build and model, and the verdict is kept in SharedPreferences.
     */
    private fun nnapiIfFaster(context: Context, model: ByteBuffer, cacheToken: String, fallback: Backend): Backend {
        if (Build.VERSION.SDK_INT < Build.VERSION_CODES.O_MR1) return fallback
        val prefs = context.getSharedPreferences(PREFS_NAME, Context.MODE_PRIVATE)
        val key = "nnapi_wins:${Build.FINGERPRINT}:$cacheToken"
        val verdict = if (prefs.contains(key)) prefs.getBoolean(key, false) else null
        if (verdict == false) return fallback
        // Some NNAPI drivers crash the process while compiling. Record a loss first so a crash
        // isn't retried on every launch. commit, not apply: it must reach disk before we try.
        if (verdict == null) prefs.edit(commit = true) { putBoolean(key, false) }

        val nnapi = createNnapi(context, model, cacheToken)
        val wins = nnapi != null && (verdict == true || isFaster(nnapi, fallback))
        prefs.edit { putBoolean(key, wins) }
        if (nnapi != null && wins) {
            fallback.close()
            return nnapi
        }
        nnapi?.close()
        return fallback
    }

    /** True if [candidate] beats [baseline] by at least 10%, so noise can't flip the choice. */
    private fun isFaster(candidate: Backend, baseline: Backend): Boolean = attempt("${candidate.name} probe") {
        val candidateMs = medianLatencyMs(candidate.interpreter)
        val baselineMs = medianLatencyMs(baseline.interpreter)
        Log.i(TAG, "Probe: ${candidate.name} %.1f ms, ${baseline.name} %.1f ms".format(candidateMs, baselineMs))
        candidateMs < baselineMs * PROBE_MARGIN
    } ?: false

    /** Median of a few runs on a blank input, after one warm-up run. */
    private fun medianLatencyMs(interpreter: Interpreter): Double {
        val probeInput = ByteBuffer.allocateDirect(interpreter.getInputTensor(0).numBytes()).order(ByteOrder.nativeOrder())
        val probeOutputs = (0 until interpreter.outputTensorCount).associateWith {
            ByteBuffer.allocateDirect(interpreter.getOutputTensor(it).numBytes()).order(ByteOrder.nativeOrder())
        }
        val times = List(PROBE_RUNS + 1) {
            probeInput.rewind()
            probeOutputs.values.forEach { it.rewind() }
            val start = SystemClock.elapsedRealtimeNanos()
            interpreter.runForMultipleInputsOutputs(arrayOf(probeInput), probeOutputs)
            (SystemClock.elapsedRealtimeNanos() - start) / 1e6
        }
        return times.drop(1).sorted()[PROBE_RUNS / 2]
    }

    /** Builds an interpreter on [delegate], closing the delegate if the model can't run on it. */
    private fun withDelegate(name: String, delegate: Delegate, model: ByteBuffer): Backend =
        try {
            val interpreterOptions = Interpreter.Options().setNumThreads(options.numThreads).addDelegate(delegate)
            Backend(name, Interpreter(model, interpreterOptions), delegate)
        } catch (e: Exception) {
            delegate.close()
            throw e
        }

    /** Runs [block], returning null if it throws. A missing native library throws LinkageError. */
    private inline fun <T> attempt(what: String, block: () -> T?): T? =
        try {
            block()
        } catch (e: Exception) {
            Log.w(TAG, "$what unavailable", e)
            null
        } catch (e: LinkageError) {
            Log.w(TAG, "$what unavailable", e)
            null
        }

    private fun cacheDir(context: Context, name: String): String =
        File(context.codeCacheDir, name).apply { mkdirs() }.absolutePath

    private fun runDetection(bitmap: Bitmap, rotation: Int, frameWidth: Int, frameHeight: Int): List<Detection> {
        // Rotate upright and letterbox into the model input in a single draw.
        val scale = min(input.width.toFloat() / frameWidth, input.height.toFloat() / frameHeight)
        val padX = (input.width - frameWidth * scale) / 2f
        val padY = (input.height - frameHeight * scale) / 2f
        transform.reset()
        transform.postTranslate(-bitmap.width / 2f, -bitmap.height / 2f)
        transform.postRotate(rotation.toFloat())
        transform.postScale(scale, scale)
        transform.postTranslate(input.width / 2f, input.height / 2f)
        input.write(bitmap, transform)

        boxes.buffer.rewind()
        scores.buffer.rewind()
        classes.buffer.rewind()
        backend!!.interpreter.runForMultipleInputsOutputs(arrayOf(input.buffer), outputs)

        // Boxes come back in input pixels. Undo the letterbox and normalize to the upright frame.
        val contentWidth = frameWidth * scale
        val contentHeight = frameHeight * scale
        val candidates = ArrayList<Detection>()
        for (i in 0 until scores.size) {
            val score = scores[i]
            if (score < options.scoreThreshold) continue
            val b = i * 4
            val box = RectF(
                ((boxes[b] - padX) / contentWidth).coerceIn(0f, 1f),
                ((boxes[b + 1] - padY) / contentHeight).coerceIn(0f, 1f),
                ((boxes[b + 2] - padX) / contentWidth).coerceIn(0f, 1f),
                ((boxes[b + 3] - padY) / contentHeight).coerceIn(0f, 1f),
            )
            val classId = classes.raw(i)
            candidates += Detection(classId, labels.getOrElse(classId) { "class $classId" }, score, box)
        }
        return nonMaxSuppression(candidates)
    }

    /** Greedy per-class NMS. Few candidates survive the score filter, so O(n²) is fine. */
    private fun nonMaxSuppression(candidates: List<Detection>): List<Detection> {
        val kept = ArrayList<Detection>()
        for (d in candidates.sortedByDescending { it.score }) {
            if (kept.size == options.maxResults) break
            if (kept.none { it.classId == d.classId && iou(it.box, d.box) > options.iouThreshold }) kept += d
        }
        return kept
    }

    private fun iou(a: RectF, b: RectF): Float {
        val w = min(a.right, b.right) - max(a.left, b.left)
        val h = min(a.bottom, b.bottom) - max(a.top, b.top)
        if (w <= 0f || h <= 0f) return 0f
        val intersection = w * h
        return intersection / (a.width() * a.height() + b.width() * b.height() - intersection)
    }

    /** An interpreter and the delegate it runs on, if any. [name] is what the UI shows. */
    private class Backend(val name: String, val interpreter: Interpreter, private val delegate: Delegate?) {
        fun close() {
            interpreter.close()
            delegate?.close()
        }
    }

    /** RGB input in 0..1. Handles NHWC or NCHW layout and float or 8-bit quantized types. */
    private class ModelInput(tensor: Tensor) {
        private val shape = tensor.shape()
        private val channelsFirst = shape[1] == 3
        val height = if (channelsFirst) shape[2] else shape[1]
        val width = if (channelsFirst) shape[3] else shape[2]
        val buffer: ByteBuffer = ByteBuffer.allocateDirect(tensor.numBytes()).order(ByteOrder.nativeOrder())

        private val type = tensor.dataType()
        private val scale = tensor.quantizationParams().scale
        private val zeroPoint = tensor.quantizationParams().zeroPoint
        private val bitmap = createBitmap(width, height)
        private val canvas = Canvas(bitmap)
        private val paint = Paint(Paint.FILTER_BITMAP_FLAG)
        private val pixels = IntArray(width * height)
        private val floats = FloatArray(width * height * 3)

        init {
            require(type == DataType.FLOAT32 || type == DataType.UINT8 || type == DataType.INT8) {
                "Unsupported input type $type"
            }
        }

        fun write(source: Bitmap, transform: Matrix) {
            canvas.drawColor(LETTERBOX_COLOR)
            canvas.drawBitmap(source, transform, paint)
            bitmap.getPixels(pixels, 0, width, 0, 0, width, height)

            val plane = pixels.size
            for (i in pixels.indices) {
                val p = pixels[i]
                for (c in 0..2) {
                    floats[if (channelsFirst) c * plane + i else i * 3 + c] = ((p shr (16 - 8 * c)) and 0xFF) / 255f
                }
            }
            buffer.rewind()
            if (type == DataType.FLOAT32) {
                buffer.asFloatBuffer().put(floats)
            } else {
                for (v in floats) buffer.put(quantize(v))
                buffer.rewind()
            }
        }

        private fun quantize(v: Float): Byte {
            val q = (if (scale == 0f) v * 255f else v / scale + zeroPoint).roundToInt()
            return if (type == DataType.INT8) q.coerceIn(-128, 127).toByte() else q.coerceIn(0, 255).toByte()
        }
    }

    private class ModelOutput(tensor: Tensor) {
        val buffer: ByteBuffer = ByteBuffer.allocateDirect(tensor.numBytes()).order(ByteOrder.nativeOrder())
        val size = tensor.numElements()
        private val type = tensor.dataType()
        private val scale = tensor.quantizationParams().scale
        private val zeroPoint = tensor.quantizationParams().zeroPoint

        /** Stored value without dequantization. Used for class ids. */
        fun raw(i: Int): Int = when (type) {
            DataType.UINT8 -> buffer.get(i).toInt() and 0xFF
            DataType.INT8 -> buffer.get(i).toInt()
            DataType.INT32 -> buffer.getInt(i * 4)
            DataType.INT64 -> buffer.getLong(i * 8).toInt()
            DataType.FLOAT32 -> buffer.getFloat(i * 4).toInt()
            else -> error("Unsupported output type $type")
        }

        /** Real value, dequantized when the tensor is quantized. */
        operator fun get(i: Int): Float = when {
            type == DataType.FLOAT32 -> buffer.getFloat(i * 4)
            scale == 0f -> raw(i).toFloat()
            else -> (raw(i) - zeroPoint) * scale
        }
    }

    private companion object {
        const val TAG = "ObjectDetectorHelper"
        val LETTERBOX_COLOR = Color.rgb(114, 114, 114) // Ultralytics padding gray.
        const val PREFS_NAME = "object_detector"
        const val PROBE_RUNS = 5
        const val PROBE_MARGIN = 0.9
    }
}
