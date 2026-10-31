package com.musicdecoded.mobile.network

import io.ktor.client.HttpClient
import io.ktor.client.call.body
import io.ktor.client.engine.HttpClientEngine
import io.ktor.client.plugins.contentnegotiation.ContentNegotiation
import io.ktor.client.plugins.logging.LogLevel
import io.ktor.client.plugins.logging.Logging
import io.ktor.client.statement.HttpResponse
import io.ktor.http.HttpHeaders
import io.ktor.http.isSuccess
import io.ktor.serialization.kotlinx.json.json
import kotlinx.serialization.json.Json

/** JSON for API bodies: unknown fields are ignored so older apps keep working as the API grows. */
val ApiJson = Json { ignoreUnknownKeys = true }

/**
 * The one HTTP client for the API. Non-2xx responses are not thrown by Ktor; services call
 * [bodyOrThrow] so callers get an [ApiException] carrying the contract `Error` body.
 *
 * @param debugLogging logs full bodies (BODY level) — pass `BuildConfig.DEBUG`, never true in release.
 */
fun createHttpClient(engine: HttpClientEngine, debugLogging: Boolean): HttpClient = HttpClient(engine) {
    expectSuccess = false
    install(ContentNegotiation) { json(ApiJson) }
    if (debugLogging) {
        install(Logging) {
            level = LogLevel.BODY
            sanitizeHeader { it == HttpHeaders.Authorization }
        }
    }
}

/** Decodes a 2xx body as [T], or throws [ApiException] with the server's `Error`. */
suspend inline fun <reified T> HttpResponse.bodyOrThrow(): T {
    if (status.isSuccess()) return body()
    val error = runCatching { body<ApiError>() }
        .getOrElse { ApiError("internal", "Something went wrong (${status.value})") }
    throw ApiException(error, status.value)
}
