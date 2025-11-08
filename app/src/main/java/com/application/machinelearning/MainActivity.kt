package com.application.machinelearning

import android.os.Bundle
import android.util.Log
import androidx.activity.enableEdgeToEdge
import androidx.appcompat.app.AppCompatActivity
import androidx.core.view.ViewCompat
import androidx.core.view.WindowInsetsCompat
import org.pytorch.IValue
import org.pytorch.LiteModuleLoader
import org.pytorch.Tensor


class MainActivity : AppCompatActivity() {

    val FEATURE_INPUT_SIZE = 8

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        enableEdgeToEdge()
        setContentView(R.layout.activity_main)
        ViewCompat.setOnApplyWindowInsetsListener(findViewById(R.id.main)) { v, insets ->
            val systemBars = insets.getInsets(WindowInsetsCompat.Type.systemBars())
            v.setPadding(systemBars.left, systemBars.top, systemBars.right, systemBars.bottom)
            insets
        }

        val wordIndexMap = hashMapOf(
            "your" to 0,
            "is" to 1,
            "cash" to 2,
            "account" to 3,
            "otp" to 4,
            "credited" to 5,
            "won" to 6,
            "prize" to 7
        )


        val tfidfWeights = floatArrayOf(
            0.8f,
            0.7f,
            1.2f,
            1.1f,
            1.5f,
            1.3f,
            1.4f,
            1.6f
        )

        val classMap = HashMap<Int, String>()

        classMap.put(0, "OTP")
        classMap.put(1, "Transaction")
        classMap.put(2, "Spam")

        val module = LiteModuleLoader.load(assetFilePath(this, "msg_model.ptl"))
        val inputData = getTfidfVector("Your otp is 12345", wordIndexMap, tfidfWeights)
        val inputTensor = Tensor.fromBlob(inputData, longArrayOf(1, FEATURE_INPUT_SIZE.toLong()))

        val output = module.forward(IValue.from(inputTensor)).toTensor()
        val outputData = output.dataAsFloatArray

        // Convert logits → probabilities using Softmax
        val expValues = outputData.map { Math.exp(it.toDouble()) }
        val sumExp = expValues.sum()
        val probabilities = expValues.map { it / sumExp }
        val predictedIndex = probabilities.indices.maxByOrNull { probabilities[it] } ?: -1

        Log.d("Jatin", "Probabilities: $probabilities")
        Log.d("Jatin", "Predicted class index: $predictedIndex")
        Log.d("Jatin", "Output: ${outputData.joinToString()}")
        Log.d("Jatin", "Output: Category ${classMap[predictedIndex]}")
    }

    fun getTfidfVector(
        message: String,
        wordIndexMap: Map<String, Int>,
        tfidfWeights: FloatArray
    ): FloatArray {
        val words = message.lowercase().split(" ")
        val vector = FloatArray(FEATURE_INPUT_SIZE) { 0f }

        for (word in words) {
            val index = wordIndexMap[word]
            if (index != null) {
                vector[index] = tfidfWeights[index]
            }
        }
        return vector
    }
}