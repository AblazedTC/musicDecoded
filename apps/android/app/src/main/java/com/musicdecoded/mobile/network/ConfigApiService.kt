package com.musicdecoded.mobile.network

import io.ktor.client.HttpClient
import io.ktor.client.request.bearerAuth
import io.ktor.client.request.get
import kotlinx.serialization.Serializable

/** `ClientConfig` in contracts/openapi.yaml: server-controlled flags, read at launch and cached. */
@Serializable
data class ClientConfig(
    /** R-1 flag. false in store builds. */
    val offlineStreamAudio: Boolean,
    val maxUploadBytes: Long,
    val soundcloudEnabled: Boolean = false,
    val plan: Plan,
    val limits: Limits,
) {
    /** For display and upsell only; the server enforces them. */
    @Serializable
    data class Limits(
        /** `null` = unlimited. */
        val dailyJobs: Int?,
        val jobsUsedToday: Int,
    )
}

class ConfigApiService(private val http: HttpClient, private val baseUrl: String) {
    suspend fun config(token: String): ClientConfig =
        http.get("$baseUrl/config") { bearerAuth(token) }.bodyOrThrow()
}
