package org.techtown.writecall.classifier

import android.content.Context
import android.graphics.Bitmap
import android.os.SystemClock
import android.util.Log
import android.util.Size
import org.tensorflow.lite.Delegate
import org.tensorflow.lite.Interpreter
import org.tensorflow.lite.Tensor
import org.tensorflow.lite.gpu.GpuDelegate
import org.tensorflow.lite.nnapi.NnApiDelegate
import org.tensorflow.lite.support.common.FileUtil
import org.tensorflow.lite.support.tensorbuffer.TensorBuffer
import java.io.Closeable
import java.nio.ByteBuffer
import java.nio.ByteOrder

class Classifier(
    context: Context,
    device: Device = Device.CPU,
    numThreads: Int = 4
) {
    // Delegate는 모델의 그래프 노드를 줄여, 시스템이 모델을 추론량을 줄여줌으로서 자원을 아낄 수 있습니다.
    private val delegate: Delegate? = when(device) {
        Device.CPU -> null // CPU를 사용하는 경우, Delegate에 null 할당
        Device.NNAPI -> NnApiDelegate() // NPU용 NNAPI Delegate 사용
        Device.GPU -> GpuDelegate() // GPU용 Delegate 사용
    }

    // 모델을 추론해주는 역할
    private val interpreter: Interpreter = Interpreter(
        FileUtil.loadMappedFile(context, MODEL_FILE_NAME), // tensorflow lite (tflite) 모델이 넣어져있다.
        Interpreter.Options().apply {
            setNumThreads(numThreads)
            delegate?.let { addDelegate(it) }
        }
    )

    private val inputTensor: Tensor = interpreter.getInputTensor(0)

    private val outputTensor: Tensor = interpreter.getOutputTensor(0)

    val inputShape: Size = with(inputTensor.shape()) { Size(this[2], this[1]) }

    private val imagePixels = IntArray(inputShape.height * inputShape.width)

    private val imageBuffer: ByteBuffer =
        ByteBuffer.allocateDirect(4 * inputShape.height * inputShape.width).apply {
            order(ByteOrder.nativeOrder())
        }

    private val outputBuffer: TensorBuffer =
        TensorBuffer.createFixedSize(outputTensor.shape(), outputTensor.dataType())

    init {
        Log.v(LOG_TAG, "[Input] shape = ${inputTensor.shape()?.contentToString()}, " +
                "dataType = ${inputTensor.dataType()}")
        Log.v(LOG_TAG, "[Output] shape = ${outputTensor.shape()?.contentToString()}, " +
                "dataType = ${outputTensor.dataType()}")
    }

    // 실질적인 분류기의 주기능: 분류하기. bitmap영상을 입력값으로 받는다
    fun classify(image: Bitmap): Recognition {
        // 영상을 byte형으로 변환
        convertBitmapToByteBuffer(image)

        outputBuffer
        // 초시계 start!
        val start = SystemClock.uptimeMillis()
        // 인터프리터(모델 + 가중치)를 이용하여 imagebuffer을 입력받아 ouputBuffer(TensorBuffer형)에 저장
        interpreter.run(imageBuffer, outputBuffer.buffer.rewind())
        val end = SystemClock.uptimeMillis()
        val timeCost = end - start // 실행이 얼마나 걸렸는지 반환

        val probs = outputBuffer.floatArray // 레이블별 확률값 배열 probs에 반환
        val top = probs.argMax() // 확률이 가장 높은 레이블을 top에 반환
        Log.v(LOG_TAG, "classify(): timeCost = $timeCost, top = $top, probs = ${probs.contentToString()}")
        return Recognition(top, probs[top], timeCost) // Recog 데이터 클래스에 레이블과 해당 확률값, 걸린 시간을 넣는다.
    }

    fun close() {
        interpreter.close()
        if (delegate is Closeable) {
            delegate.close()
        }
    }

    // Bitmap 형식은 android.graphic 패키지를 참조
    private fun convertBitmapToByteBuffer(bitmap: Bitmap) {
        imageBuffer.rewind()
        // bitmap을 픽셀로 전달 받아 imagePixels(arrayInt 형)에 저장
        bitmap.getPixels(imagePixels, 0, bitmap.width, 0, 0, bitmap.width, bitmap.height)
        // pixel값들을 모두 imageBuffer(byte형 buffer)에 저장
        for (i in 0 until inputShape.width * inputShape.height) {
            val pixel: Int = imagePixels[i]
            imageBuffer.putFloat(convertPixel(pixel))
        }
    }

    private fun convertPixel(color: Int): Float {
        return (255 - ((color shr 16 and 0xFF) * 0.299f
                + (color shr 8 and 0xFF) * 0.587f
                + (color and 0xFF) * 0.114f)) / 255.0f
    }

    companion object {
        private val LOG_TAG: String = Classifier::class.java.simpleName
        private const val MODEL_FILE_NAME: String = "mnist.tflite"
    }
}

fun FloatArray.argMax(): Int {
    return this.withIndex().maxBy { it.value }?.index
        ?: throw IllegalArgumentException("Cannot find arg max in empty list")
}
