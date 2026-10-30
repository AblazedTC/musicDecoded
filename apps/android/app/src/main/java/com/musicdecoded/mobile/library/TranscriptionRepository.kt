package com.musicdecoded.mobile.library

import com.musicdecoded.mobile.auth.AuthRepository
import com.musicdecoded.mobile.core.Analysis
import com.musicdecoded.mobile.core.Source
import com.musicdecoded.mobile.network.ApiException
import com.musicdecoded.mobile.network.Job
import com.musicdecoded.mobile.network.JobRequest
import com.musicdecoded.mobile.network.JobStatus
import com.musicdecoded.mobile.network.TranscriptionApiService
import com.musicdecoded.mobile.network.UploadRequest
import kotlinx.coroutines.CancellationException
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.delay
import kotlinx.coroutines.flow.MutableStateFlow
import kotlinx.coroutines.flow.StateFlow
import kotlinx.coroutines.flow.asStateFlow
import kotlinx.coroutines.flow.update
import kotlinx.coroutines.withContext
import java.io.File
import java.io.IOException
import java.security.MessageDigest
import java.time.Instant
import java.util.UUID

/** Audio on this device, ready to be decoded (a picked file or a finished recording). */
data class LocalAudio(
    /** A file made by [TranscriptionRepository.newPendingFile]; it's moved to `audio/{trackId}.{ext}`. */
    val file: File,
    val title: String,
    /** [Source.FILE] or [Source.RECORDING]. */
    val source: Source,
    val durationSec: Double? = null,
)

/** A failure with a message that can be shown to the user as is. */
class TranscriptionException(message: String, cause: Throwable? = null) : Exception(message, cause)

/** Runs the polling for a track in the background (WorkManager in the app, a fake in tests). */
fun interface PollScheduler {
    /** [replace] restarts polling (new job); otherwise an already-scheduled poll is kept. */
    fun schedule(trackId: String, replace: Boolean)
}

/**
 * The job runner every source uses (T-204, PLAN.md §3.2):
 * SHA-256 → `POST /uploads` → PUT bytes → `POST /jobs` → LibraryItem `analyzing` →
 * poll `GET /jobs/{id}` (in the background, see [TranscriptionWorker]) → download the analysis →
 * `AnalysisStore.save` → item `ready`. A failed job sets `failed`; [retry] reuses the upload.
 */
class TranscriptionRepository(
    private val api: TranscriptionApiService,
    private val auth: AuthRepository,
    private val dao: LibraryDao,
    private val analysisStore: AnalysisStore,
    /** The app's storage root; audio lives in `audio/` under it (PLAN.md §5.1). */
    private val filesDir: File,
    private val scheduler: PollScheduler,
    private val clock: () -> Instant = Instant::now,
    /** Called when a job finished and its analysis is saved (triggers a sync, PLAN.md §5.2). */
    private val onJobComplete: () -> Unit = {},
) {
    private val audioDir = File(filesDir, "audio")
    private val _jobs = MutableStateFlow<Map<String, Job>>(emptyMap())

    /** The last known `Job` per trackId while it runs (drives Processing screen progress). */
    val jobs: StateFlow<Map<String, Job>> = _jobs.asStateFlow()

    /** A fresh temp file in `audio/` for a source to write into before the trackId is known. */
    fun newPendingFile(extension: String): File {
        audioDir.mkdirs()
        return File(audioDir, "pending-${UUID.randomUUID()}.${extension.lowercase()}")
    }

    /**
     * Uploads [audio], requests the analysis and schedules polling. Returns the trackId as soon as
     * the job exists (the LibraryItem is then `analyzing`). Throws [TranscriptionException].
     */
    suspend fun start(audio: LocalAudio): String = userFacing {
        val uploadId = upload(audio.file)
        val job = auth.authorized { api.createJob(it, JobRequest(audio.source, uploadId = uploadId, title = audio.title)) }
        val audioFile = moveToTrack(audio.file, job.trackId)
        val now = clock()
        val existing = dao.get(job.trackId)
        val item = (existing ?: LibraryItem(
            trackId = job.trackId,
            title = audio.title,
            source = audio.source,
            createdAt = now,
            updatedAt = now,
            status = LibraryStatus.ANALYZING,
        )).copy(
            title = audio.title,
            durationSec = audio.durationSec ?: existing?.durationSec,
            audioUploadId = uploadId,
            localAudioPath = audioFile.relativeTo(filesDir).invariantSeparatorsPath,
            deleted = false,
        )
        begin(item, job)
        job.trackId
    }

    /**
     * Requests the analysis of a YouTube video (T-304a): no upload, the worker fetches the audio by
     * [videoId]. Same LibraryItem + polling as [start]; no local audio is kept (PLAN.md R-1).
     */
    suspend fun startYouTube(videoId: String, title: String, thumbnailUrl: String?, durationSec: Double?): String = userFacing {
        val job = auth.authorized { api.createJob(it, youTubeJob(videoId)) }
        val now = clock()
        val existing = dao.get(job.trackId)
        val item = (existing ?: LibraryItem(
            trackId = job.trackId,
            title = title,
            source = Source.YOUTUBE,
            createdAt = now,
            updatedAt = now,
            status = LibraryStatus.ANALYZING,
        )).copy(
            sourceRef = videoId,
            thumbnailUrl = thumbnailUrl ?: existing?.thumbnailUrl,
            durationSec = durationSec ?: existing?.durationSec,
            deleted = false,
        )
        begin(item, job)
        job.trackId
    }

    /**
     * Requests the analysis of a SoundCloud track (T-502a): no upload, the worker fetches the audio
     * from [url]. Same LibraryItem + polling as [startYouTube]; no local audio is kept (PLAN.md R-1).
     * A server with SoundCloud off answers 403 `feature_disabled` ([SOUNDCLOUD_DISABLED]).
     */
    suspend fun startSoundCloud(url: String, title: String): String = userFacing {
        val job = auth.authorized { api.createJob(it, soundCloudJob(url)) }
        val now = clock()
        val existing = dao.get(job.trackId)
        val item = (existing ?: LibraryItem(
            trackId = job.trackId,
            title = title,
            source = Source.SOUNDCLOUD,
            createdAt = now,
            updatedAt = now,
            status = LibraryStatus.ANALYZING,
        )).copy(sourceRef = url, deleted = false)
        begin(item, job)
        job.trackId
    }

    /** Requests a new analysis for a failed (or re-analyzed) item, reusing its upload when possible. */
    suspend fun retry(trackId: String): Unit = userFacing {
        val item = checkNotNull(dao.get(trackId)) { "Unknown track" }
        try {
            if (item.source == Source.YOUTUBE) {
                val videoId = item.sourceRef ?: throw TranscriptionException(GENERIC)
                return@userFacing begin(item, auth.authorized { api.createJob(it, youTubeJob(videoId)) })
            }
            if (item.source == Source.SOUNDCLOUD) {
                val url = item.sourceRef ?: throw TranscriptionException(GENERIC)
                return@userFacing begin(item, auth.authorized { api.createJob(it, soundCloudJob(url)) })
            }
            val uploadId = item.audioUploadId ?: upload(localAudio(item))
            val job = try {
                createJob(item, uploadId)
            } catch (e: ApiException) {
                // The upload is gone (expired or never finished): upload the local copy again.
                if (e.status != 404) throw e
                createJob(item, upload(localAudio(item)))
            }
            begin(item, job)
        } catch (e: CancellationException) {
            throw e
        } catch (e: Exception) {
            markFailed(trackId, userMessage(e))
            throw e
        }
    }

    /**
     * Polls the item's job until it ends: 2 s, backing off to 10 s after 60 s (openapi `GET /jobs`).
     * Network/server errors are thrown so the caller can retry later; a terminal answer is saved.
     */
    suspend fun poll(trackId: String) {
        val jobId = dao.get(trackId)?.takeIf { it.status == LibraryStatus.ANALYZING }?.jobId ?: return
        var elapsedMs = 0L
        while (true) {
            val job = try {
                auth.authorized { api.job(it, jobId) }
            } catch (e: ApiException) {
                if (e.status != 404) throw e
                markFailed(trackId, "This analysis expired. Try again.")
                return
            }
            _jobs.update { it + (trackId to job) }
            when (job.status) {
                JobStatus.COMPLETE -> return complete(trackId)
                JobStatus.FAILED -> return markFailed(trackId, job.error?.takeIf { it.isNotBlank() } ?: ANALYSIS_FAILED)
                JobStatus.QUEUED, JobStatus.PROCESSING -> {
                    val interval = if (elapsedMs < BACKOFF_AFTER_MS) FAST_POLL_MS else SLOW_POLL_MS
                    delay(interval)
                    elapsedMs += interval
                }
            }
        }
    }

    /** Re-schedules polling for every job that was still running (app launch). */
    suspend fun resumePending() {
        dao.analyzing().forEach { scheduler.schedule(it.trackId, replace = false) }
    }

    suspend fun markFailed(trackId: String, message: String) {
        val item = dao.get(trackId) ?: return
        dao.upsert(item.copy(status = LibraryStatus.FAILED, jobId = null, error = message, updatedAt = clock()))
        _jobs.update { it - trackId }
    }

    private suspend fun complete(trackId: String) {
        val json = try {
            val location = auth.authorized { api.analysisLocation(it, trackId) }
            api.download(location.url).also { analysisStore.save(trackId, it) }
        } catch (e: ApiException) {
            if (e.status != 404) throw e
            return markFailed(trackId, ANALYSIS_FAILED)
        } catch (e: IllegalArgumentException) {
            return markFailed(trackId, ANALYSIS_FAILED) // not a valid v1 analysis (SerializationException)
        }
        val item = dao.get(trackId) ?: return
        dao.upsert(item.withAnalysis(Analysis.decode(json), clock()))
        _jobs.update { it - trackId }
        onJobComplete()
    }

    private suspend fun begin(item: LibraryItem, job: Job) {
        dao.upsert(item.copy(status = LibraryStatus.ANALYZING, jobId = job.jobId, error = null, updatedAt = clock()))
        _jobs.update { it + (job.trackId to job) }
        scheduler.schedule(job.trackId, replace = true)
    }

    private fun youTubeJob(videoId: String) = JobRequest(Source.YOUTUBE, videoId = videoId)

    private fun soundCloudJob(url: String) = JobRequest(Source.SOUNDCLOUD, soundcloudUrl = url)

    private suspend fun createJob(item: LibraryItem, uploadId: String): Job =
        auth.authorized { api.createJob(it, JobRequest(item.source, uploadId = uploadId, title = item.title)) }

    /** SHA-256 → `POST /uploads` → PUT. Returns the uploadId. */
    private suspend fun upload(file: File): String {
        val contentType = contentTypeFor(file.extension)
            ?: throw TranscriptionException("This audio format isn't supported. Try MP3, M4A, AAC, WAV, FLAC or OGG.")
        val size = file.length()
        if (size <= 0) throw TranscriptionException("This file is empty.")
        val maxBytes = auth.config.value?.maxUploadBytes ?: MAX_UPLOAD_BYTES
        if (size > maxBytes) throw TranscriptionException(TOO_LARGE)
        val sha = withContext(Dispatchers.IO) { sha256(file) }
        val target = auth.authorized { api.createUpload(it, UploadRequest(contentType, size, sha)) }
        api.putBytes(target.putUrl, file, contentType)
        return target.uploadId
    }

    private fun localAudio(item: LibraryItem): File =
        item.localAudioPath?.let { File(filesDir, it) }?.takeIf { it.isFile }
            ?: throw TranscriptionException("The audio for this song is no longer on this device.")

    /** Renames the pending file to `audio/{trackId}.{ext}` (or drops it if that file already exists). */
    private suspend fun moveToTrack(pending: File, trackId: String): File = withContext(Dispatchers.IO) {
        require(TRACK_ID.matches(trackId)) { "Invalid trackId" }
        val target = File(audioDir, "$trackId.${pending.extension.lowercase()}")
        if (pending.absolutePath != target.absolutePath) {
            if (target.isFile) pending.delete() else if (!pending.renameTo(target)) {
                pending.copyTo(target, overwrite = true)
                pending.delete()
            }
        }
        target
    }

    /** Rethrows any failure as a [TranscriptionException] with a user-facing message. */
    private suspend fun <T> userFacing(block: suspend () -> T): T = try {
        block()
    } catch (e: CancellationException) {
        throw e
    } catch (e: TranscriptionException) {
        throw e
    } catch (e: Exception) {
        throw TranscriptionException(userMessage(e), e)
    }

    companion object {
        const val MAX_UPLOAD_BYTES = 104_857_600L
        const val FAST_POLL_MS = 2_000L
        const val SLOW_POLL_MS = 10_000L
        const val BACKOFF_AFTER_MS = 60_000L

        const val RATE_LIMITED =
            "You've used today's free transcriptions. Go Pro for more, or try again tomorrow."
        const val TOO_LARGE = "This file is too large to upload (the limit is 100 MB)."
        const val OFFLINE = "Couldn't reach MusicDecoded. Check your connection and try again."
        const val ANALYSIS_FAILED = "We couldn't decode this audio. Try again."
        const val GENERIC = "Something went wrong. Try again."
        const val SOUNDCLOUD_DISABLED = "SoundCloud links aren't available right now. Try another source."

        private val TRACK_ID = Regex("[A-Za-z0-9_-]+")

        /** The upload content types allowed by `POST /uploads`, by file extension. */
        fun contentTypeFor(extension: String): String? = when (extension.lowercase()) {
            "mp3" -> "audio/mpeg"
            "m4a", "mp4" -> "audio/mp4"
            "aac" -> "audio/aac"
            "wav" -> "audio/wav"
            "flac" -> "audio/flac"
            "ogg", "oga", "opus" -> "audio/ogg"
            else -> null
        }

        /** A message for any failure, with `rate_limited` (429) spelled out for the user. */
        fun userMessage(e: Throwable): String = when (e) {
            is TranscriptionException -> e.message ?: GENERIC
            is ApiException -> when {
                e.error.code == "rate_limited" || e.status == 429 -> RATE_LIMITED
                e.error.code == "too_large" || e.status == 413 -> TOO_LARGE
                // Only `POST /jobs` with source soundcloud answers this (openapi).
                e.error.code == "feature_disabled" -> SOUNDCLOUD_DISABLED
                else -> e.message ?: GENERIC
            }
            is IOException -> OFFLINE
            else -> GENERIC
        }

        internal fun sha256(file: File): String {
            val digest = MessageDigest.getInstance("SHA-256")
            file.inputStream().use { input ->
                val buffer = ByteArray(64 * 1024)
                while (true) {
                    val read = input.read(buffer)
                    if (read < 0) break
                    digest.update(buffer, 0, read)
                }
            }
            return digest.digest().joinToString("") { "%02x".format(it) }
        }
    }
}
