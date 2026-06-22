package com.example.pedalrevolution

import kotlinx.serialization.Serializable
import kotlinx.serialization.json.Json
import okhttp3.MediaType.Companion.toMediaType
import okhttp3.OkHttpClient
import okhttp3.logging.HttpLoggingInterceptor
import retrofit2.Retrofit
import retrofit2.converter.kotlinx.serialization.asConverterFactory
import retrofit2.http.Body
import retrofit2.http.POST

@Serializable
data class DetectionRequest(
    val timestamp_ms: Long,
    val label: String,
    val confidence: Float,
    val x_min: Float,
    val y_min: Float,
    val x_max: Float,
    val y_max: Float,
    val source: String = "online",
    val latitude: Double? = null,
    val longitude: Double? = null,
    val altitude: Double? = null
)

interface DetectionApiService {
    @POST("detections")
    suspend fun sendDetections(@Body detections: List<DetectionRequest>): Map<String, String>

    companion object {
        private const val BASE_URL = "http://10.0.2.2:8000/"

        fun create(): DetectionApiService {
            val logger = HttpLoggingInterceptor().apply { level = HttpLoggingInterceptor.Level.BODY }
            val client = OkHttpClient.Builder()
                .addInterceptor(logger)
                .build()

            val json = Json { ignoreUnknownKeys = true }

            return Retrofit.Builder()
                .baseUrl(BASE_URL)
                .client(client)
                .addConverterFactory(json.asConverterFactory("application/json".toMediaType()))
                .build()
                .create(DetectionApiService::class.java)
        }
    }
}
