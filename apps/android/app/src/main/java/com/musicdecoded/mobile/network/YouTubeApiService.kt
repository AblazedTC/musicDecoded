package com.musicdecoded.mobile.network

import io.ktor.client.HttpClient
import io.ktor.client.request.bearerAuth
import io.ktor.client.request.get
import io.ktor.client.request.parameter
import kotlinx.serialization.Serializable

/** `VideoSummary` in contracts/openapi.yaml. */
@Serializable
data class VideoSummary(
    val videoId: String,
    val title: String,
    val channelTitle: String,
    val thumbnailUrl: String,
    val durationSec: Int? = null,
)

/** 200 body of `GET /youtube/search`. */
@Serializable
data class VideoSearchPage(val items: List<VideoSummary>, val nextPageToken: String? = null)

/**
 * The YouTube sandbox search (PLAN.md D-4): the server proxies the Data API, so the key never
 * reaches the app. 403 `feature_disabled` = search unavailable; 429 `rate_limited` = the server's
 * YouTube quota (not a plan limit, so no upsell).
 */
class YouTubeApiService(private val http: HttpClient, private val baseUrl: String) {
    suspend fun search(token: String, query: String, pageToken: String? = null): VideoSearchPage =
        http.get("$baseUrl/youtube/search") {
            bearerAuth(token)
            parameter("q", query)
            pageToken?.let { parameter("pageToken", it) }
        }.bodyOrThrow()
}
