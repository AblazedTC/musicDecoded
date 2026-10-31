package com.musicdecoded.mobile.network

import com.musicdecoded.mobile.core.Source
import io.ktor.client.HttpClient
import io.ktor.client.request.bearerAuth
import io.ktor.client.request.get
import io.ktor.client.request.parameter
import io.ktor.client.request.prepareGet
import io.ktor.client.request.put
import io.ktor.client.request.setBody
import io.ktor.client.statement.bodyAsChannel
import io.ktor.http.ContentType
import io.ktor.http.contentType
import io.ktor.http.isSuccess
import io.ktor.utils.io.jvm.javaio.toInputStream
import kotlinx.serialization.EncodeDefault
import kotlinx.serialization.ExperimentalSerializationApi
import kotlinx.serialization.Serializable
import java.io.File

/** `LibraryItem` in contracts/openapi.yaml (times are ISO-8601 date-time strings). */
@OptIn(ExperimentalSerializationApi::class)
@Serializable
data class SyncItem(
    val trackId: String,
    val title: String,
    val artist: String? = null,
    val source: Source,
    val sourceRef: String? = null,
    val durationSec: Double? = null,
    val thumbnailUrl: String? = null,
    val audioUploadId: String? = null,
    @EncodeDefault val favorite: Boolean = false,
    val lastPlayedAt: String? = null,
    val createdAt: String,
    val updatedAt: String,
    val deleted: Boolean,
)

/** 200 body of `GET /sync/library`. */
@Serializable
data class SyncPage(val items: List<SyncItem>, val cursor: String)

/** Body of `PUT /sync/library` and its 200 response. */
@Serializable
data class SyncItems(val items: List<SyncItem> = emptyList())

/** 200 body of `GET /uploads/{uploadId}`. */
@Serializable
data class UploadLocation(val url: String, val contentType: String, val expiresAt: String)

/** `/sync/library` (account tokens only) and `GET /uploads/{id}` (T-404a, PLAN.md §5.2). */
class SyncApiService(private val http: HttpClient, private val baseUrl: String) {
    /** One page (up to 500 items); [since] `null` = full pull. */
    suspend fun pull(token: String, since: String?): SyncPage =
        http.get("$baseUrl/sync/library") {
            bearerAuth(token)
            since?.let { parameter("since", it) }
        }.bodyOrThrow()

    /** Up to 500 items; returns the server state of the pushed ids after the merge. */
    suspend fun push(token: String, items: List<SyncItem>): SyncItems =
        http.put("$baseUrl/sync/library") {
            bearerAuth(token)
            contentType(ContentType.Application.Json)
            setBody(SyncItems(items))
        }.bodyOrThrow()

    suspend fun uploadLocation(token: String, uploadId: String): UploadLocation =
        http.get("$baseUrl/uploads/$uploadId") { bearerAuth(token) }.bodyOrThrow()

    /** Streams a presigned GET (no bearer) into [target]. */
    suspend fun downloadTo(url: String, target: File) {
        http.prepareGet(url).execute { response ->
            if (!response.status.isSuccess()) throw StorageException(response.status.value)
            response.bodyAsChannel().toInputStream().use { input -> target.outputStream().use { input.copyTo(it) } }
        }
    }
}
