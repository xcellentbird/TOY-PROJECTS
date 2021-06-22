package org.techtown.writecall.classifier

data class Recognition(
    val label: Int,
    val confidence: Float,
    val timeCost: Long
)
