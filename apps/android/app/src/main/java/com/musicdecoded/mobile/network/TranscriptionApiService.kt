package com.musicdecoded.mobile.network

import com.musicdecoded.mobile.core.Source
import io.ktor.client.HttpClient
import io.ktor.client.request.bearerAuth
import io.ktor.client.request.get
import io.ktor.client.request.post
import io.ktor.client.request.put
import io.ktor.client.request.setBody
import io.ktor.client.statement.bodyAsText
import io.ktor.http.ContentType
import io.ktor.http.content.OutgoingContent
import io.ktor.http.contentType
import io.ktor.http.isSuccess
import io.ktor.util.cio.readChannel
import io.ktor.utils.io.ByteReadChannel
import kotlinx.serialization.SerialName
import kotlinx.serialization.Serializable
import java.io.File

/** Body of `POST /uploads`. */
@Serializable
data class UploadRequest(val contentType: String, val sizeBytes: Long, val sha256: String)

/** 201 body of `POST /uploads`. */
@Serializable
data class UploadTarget(val uploadId: String, val putUrl: String, val expiresAt: String)

/** `JobRequest` in contracts/openapi.yaml: `uploadId` for file/recording, `videoId` for youtube. */
@Serializable
data class JobRequest(
    val source: Source,
    val uploadId: String? = null,
    val videoId: String? = null,
    val soundcloudUrl: String? = null,
    val title: String? = null,
)

/** `Job` in contracts/openapi.yaml. */
@Serializable
data class Job(
    val jobId: String,
    val trackId: String,
    val status: JobStatus,
    /** `Job.stage`, kept as text so a new stage never breaks polling (DESIGN.md §3.5 maps it). */
    val stage: String? = null,
    val progress: Int,
    val error: String? = null,
)

@Serializable
enum class JobStatus {
    @SerialName("queued") QUEUED,
    @SerialName("processing") PROCESSING,
    @SerialName("complete") COMPLETE,
    @SerialName("failed") FAILED,
}

/** 200 body of `GET /tracks/{trackId}/analysis`. */
@Serializable
data class AnalysisLocation(val url: String, val schemaVersion: Int)

/** A presigned PUT/GET to object storage failed (not an API `Error`). */
class StorageException(val status: Int) : Exception("Storage request failed ($status)")

/** The uploads, jobs and tracks endpoints used by the job runner (T-204). */
class TranscriptionApiService(private val http: HttpClient, private val baseUrl: String) {
    suspend fun createUpload(token: String, request: UploadRequest): UploadTarget =
        http.post("$baseUrl/uploads") {
            bearerAuth(token)
            contentType(ContentType.Application.Json)
            setBody(request)
        }.bodyOrThrow()

    /** PUTs the raw bytes to the presigned URL: no bearer, same Content-Type as the upload request. */
    suspend fun putBytes(putUrl: String, file: File, contentType: String) {
        val response = http.put(putUrl) { setBody(FileContent(file, ContentType.parse(contentType))) }
        if (!response.status.isSuccess()) throw StorageException(response.status.value)
    }

    suspend fun createJob(token: String, request: JobRequest): Job =
        http.post("$baseUrl/jobs") {
            bearerAuth(token)
            contentType(ContentType.Application.Json)
            setBody(request)
        }.bodyOrThrow()

    suspend fun job(token: String, jobId: String): Job =
        http.get("$baseUrl/jobs/$jobId") { bearerAuth(token) }.bodyOrThrow()

    suspend fun analysisLocation(token: String, trackId: String): AnalysisLocation =
        http.get("$baseUrl/tracks/$trackId/analysis") { bearerAuth(token) }.bodyOrThrow()

    /** GETs the analysis file from its presigned URL (no bearer). */
    suspend fun download(url: String): String {
        val response = http.get(url)
        if (!response.status.isSuccess()) throw StorageException(response.status.value)
        return response.bodyAsText()
    }
}

/** Streams a file as a request body with a known length (presigned PUTs need Content-Length). */
private class FileContent(private val file: File, override val contentType: ContentType) :
    OutgoingContent.ReadChannelContent() {
    override val contentLength: Long = file.length()
    override fun readFrom(): ByteReadChannel = file.readChannel()
}
