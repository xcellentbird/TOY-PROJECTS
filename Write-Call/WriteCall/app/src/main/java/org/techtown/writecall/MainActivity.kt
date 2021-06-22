package org.techtown.writecall

import android.content.Intent
import android.graphics.Bitmap
import android.net.Uri
import android.os.Bundle
import android.util.Log
import android.widget.Toast
import androidx.appcompat.app.AppCompatActivity
import org.techtown.writecall.classifier.Classifier

import org.techtown.writecall.classifier.Recognition
import kotlinx.android.synthetic.main.activity_main.*
import java.io.IOException

class MainActivity : AppCompatActivity() {

    private lateinit var classifier: Classifier

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        setContentView(R.layout.activity_main)
        init()
    }

    private fun init() {
        initClassifier()
        initView()
    }

    private fun initClassifier() {
        try {
            classifier = Classifier(this)
            Log.v(LOG_TAG, "Classifier initialized")
        } catch (e: IOException) {
            Toast.makeText(this, R.string.failed_to_create_classifier, Toast.LENGTH_LONG).show()
            Log.e(LOG_TAG, "init(): Failed to create Classifier", e)
        }
    }

    private fun initView() {
        btn_detect.setOnClickListener { onDetectClick() }
        btn_clear.setOnClickListener { clearResult() }
        btn_add.setOnClickListener { addResult() }
        btn_call.setOnClickListener { callResult() }
    }

    private fun addResult() {
        if (tv_prediction.text.toString() == "--"){
            Toast.makeText(this,"글자를 입력해주세요", Toast.LENGTH_LONG).show()
            return
        } else {
            NumbViewer.text = NumbViewer.text.toString() + tv_prediction.text.toString()
            clearResult()
        }
    }

    private fun callResult() {
        val intent = Intent(Intent.ACTION_VIEW, Uri.parse("tel:${NumbViewer.text}"))
        startActivity(intent)
        NumbViewer.setText("")
    }

    // Detect 버튼의 클릭리스너
    private fun onDetectClick() {
        // classifier가 초기화되지 않았거나 fpv에 아무것도 없으면 그냥 끝낸다.
        if (!this::classifier.isInitialized) {
            Log.e(LOG_TAG, "onDetectClick(): Classifier is not initialized")
            return
        } else if (fpv_paint.isEmpty) {
            Toast.makeText(this, "숫자를 써주세요", Toast.LENGTH_SHORT).show()
            return
        }

        // 분류기에 넣을 크기의 Bitmap(영상)으로 변환시킨다.
        val image: Bitmap = fpv_paint.exportToBitmap(
            classifier.inputShape.width, classifier.inputShape.height
        )

        // 분류기가 영상을 분류하도록 시킨다. Recognition(label, confidence, timeCost) 반환
        val result = classifier.classify(image)
        renderResult(result)
    }

    private fun renderResult(result: Recognition) {
        tv_prediction.text = java.lang.String.valueOf(result.label)
        tv_probability.text = java.lang.String.valueOf(result.confidence)
        tv_timecost.text = java.lang.String.format(
            getString(R.string.timecost_value),
            result.timeCost
        )
    }

    private fun clearResult() {
        fpv_paint.clear()
        tv_prediction.setText("--")
        tv_probability.setText("--")
        tv_timecost.setText("--")
    }

    override fun onDestroy() {
        super.onDestroy()
//        classifier.close()
    }

    companion object {
        private val LOG_TAG: String = MainActivity::class.java.simpleName
    }
}