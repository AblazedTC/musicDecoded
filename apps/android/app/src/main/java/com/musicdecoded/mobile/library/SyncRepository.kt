package com.musicdecoded.mobile.library

import android.util.Log
import com.musicdecoded.mobile.auth.AuthRepository
import com.musicdecoded.mobile.core.Analysis
import com.musicdecoded.mobile.core.Source
import com.musicdecoded.mobile.network.ApiException
import com.musicdecoded.mobile.network.Kind
import com.musicdecoded.mobile.network.SyncApiService
import com.musicdecoded.mobile.network.SyncItem
import com.musicdecoded.mobile.network.TranscriptionApiService
import kotlinx.coroutines.CancellationException
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.sync.Mutex
import kotlinx.coroutines.sync.withLock
import kotlinx.coroutines.withContext
import java.io.File
import java.time.Instant
import java.time.OffsetDateTime

/** Which account last synced on this device, and its pull cursor. */
interface SyncStateStore {
    suspend fun userId(): String?
    suspend fun cursor(): String?
    suspend fun save(userId: String, cursor: String?)
}

/**
 * Library sync for accounts (T-404a, PLAN.md §5.2). Guests never sync.
 * 1. Push rows changed since their last sync (incl. tombstones) in batches of [PAGE].
 * 2. Pull from the stored cursor until a page has fewer than [PAGE] items.
 * 3. Merge last-write-wins on `updatedAt` (a tie keeps the local row).
 * 4. Fetch what synced rows are missing here: the analysis file, and the user's own audio
 *    (file/recording with an `audioUploadId`). Streaming audio is never synced (R-1).
 * Own-audio sync is Pro only (PLAN.md §9): free accounts push without `audioUploadId`, and the
 * server returns it as null, so nothing is downloaded for them.
 */
class SyncRepository(
    private val api: SyncApiService,
    private val tracksApi: TranscriptionApiService,
    private val auth: AuthRepository,
    private val dao: LibraryDao,
    private val analysisStore: AnalysisStore,
    private val filesDir: File,
    private val state: SyncStateStore,
) {
    private val mutex = Mutex()

    /** Runs one full sync. Returns `false` (and does nothing) for guests. Network errors are thrown. */
    suspend fun sync(): Boolean = mutex.withLock {
        val session = auth.currentSession()?.takeIf { it.kind == Kind.ACCOUNT } ?: return false
        // First sync for this account here: full pull. Rows another account already synced aren't
        // re-pushed (only never-synced rows, i.e. guest data, and later edits are).
        if (state.userId() != session.userId) state.save(session.userId, null)
        push(session.token, syncsOwnAudio = auth.isPro(session))
        pull(session.token, session.userId)
        fetchMissing(session.token)
        true
    }

    private suspend fun push(token: String, syncsOwnAudio: Boolean) {
        while (true) {
            val batch = dao.pendingPush(PAGE)
            if (batch.isEmpty()) return
            val items = batch.map { it.toSyncItem().let { item -> if (syncsOwnAudio) item else item.copy(audioUploadId = null) } }
            val response = try {
                api.push(token, items)
            } catch (e: ApiException) {
                // The whole push is rejected if one audioUploadId isn't this account's (e.g. rows left
                // by another account that logged out here). The rows still sync, without their audio.
                if (e.status != 400) throw e
                api.push(token, items.map { it.copy(audioUploadId = null) })
            }
            batch.forEach { dao.markSynced(it.trackId, it.updatedAt) }
            response.items.forEach { merge(it) }
            if (batch.size < PAGE) return
        }
    }

    private suspend fun pull(token: String, userId: String) {
        var cursor = state.cursor()
        do {
            val page = api.pull(token, cursor)
            page.items.forEach { merge(it) }
            cursor = page.cursor.ifEmpty { cursor }
            state.save(userId, cursor)
        } while (page.items.size >= PAGE)
    }

    /** Applies [remote] if it's newer than the local row; local-only fields are kept. */
    private suspend fun merge(remote: SyncItem) {
        val updatedAt = parseTime(remote.updatedAt)
        val local = dao.get(remote.trackId)
        if (local != null && updatedAt <= local.updatedAt) return
        val base = local ?: LibraryItem(
            trackId = remote.trackId,
            title = remote.title,
            source = remote.source,
            createdAt = parseTime(remote.createdAt),
            updatedAt = updatedAt,
            status = LibraryStatus.READY,
        )
        dao.upsert(
            base.copy(
                title = remote.title,
                artist = remote.artist,
                source = remote.source,
                sourceRef = remote.sourceRef,
                durationSec = remote.durationSec ?: base.durationSec,
                thumbnailUrl = remote.thumbnailUrl,
                audioUploadId = remote.audioUploadId,
                favorite = remote.favorite,
                lastPlayedAt = remote.lastPlayedAt?.let(::parseTime),
                createdAt = parseTime(remote.createdAt),
                updatedAt = updatedAt,
                deleted = remote.deleted,
                syncedAt = updatedAt,
            ),
        )
    }

    /** Downloads missing analysis files and owned audio. One failure doesn't stop the others. */
    private suspend fun fetchMissing(token: String) {
        for (item in dao.syncedVisible()) {
            if (!analysisStore.exists(item.trackId) && (item.status == LibraryStatus.READY || item.error == DOWNLOAD_FAILED)) {
                fetchAnalysis(token, item)
            }
            if (item.source.ownAudio && item.audioUploadId != null && !hasLocalAudio(item)) {
                fetchAudio(token, item.trackId, item.audioUploadId)
            }
        }
    }

    private suspend fun fetchAnalysis(token: String, item: LibraryItem) {
        val result = runCatching {
            val location = tracksApi.analysisLocation(token, item.trackId)
            val json = tracksApi.download(location.url)
            analysisStore.save(item.trackId, json)
            Analysis.decode(json)
        }
        result.exceptionOrNull()?.let { if (it is CancellationException) throw it }
        val current = dao.get(item.trackId) ?: return
        val updated = result.fold(
            // Local metadata only: keep updatedAt so this doesn't count as a user change.
            onSuccess = { current.withAnalysis(it, current.updatedAt) },
            onFailure = {
                Log.w(TAG, "Analysis download failed for ${item.trackId}", it)
                current.copy(status = LibraryStatus.FAILED, error = DOWNLOAD_FAILED)
            },
        )
        dao.upsert(updated)
    }

    private suspend fun fetchAudio(token: String, trackId: String, uploadId: String) {
        try {
            val location = api.uploadLocation(token, uploadId)
            val audioDir = File(filesDir, "audio").apply { mkdirs() }
            val target = File(audioDir, "$trackId.${extensionFor(location.contentType)}")
            val temp = File(audioDir, "${target.name}.part")
            withContext(Dispatchers.IO) {
                api.downloadTo(location.url, temp)
                target.delete()
                check(temp.renameTo(target)) { "Couldn't write ${target.name}" }
            }
            val current = dao.get(trackId) ?: return
            dao.upsert(current.copy(localAudioPath = target.relativeTo(filesDir).invariantSeparatorsPath))
        } catch (e: CancellationException) {
            throw e
        } catch (e: Exception) {
            Log.w(TAG, "Audio download failed for $trackId", e)
        }
    }

    private fun hasLocalAudio(item: LibraryItem): Boolean =
        item.localAudioPath?.let { File(filesDir, it).isFile } == true

    companion object {
        /** Contract page size for both pull and push. */
        const val PAGE = 500
        const val DOWNLOAD_FAILED = "Couldn't download this transcription. Try again."
        private const val TAG = "SyncRepository"

        private val Source.ownAudio: Boolean get() = this == Source.FILE || this == Source.RECORDING

        /** `date-time` from the API; accepts `Z` or an explicit offset. */
        fun parseTime(value: String): Instant = OffsetDateTime.parse(value).toInstant()

        fun extensionFor(contentType: String): String = when (contentType.substringBefore(';').trim().lowercase()) {
            "audio/mpeg" -> "mp3"
            "audio/mp4" -> "m4a"
            "audio/aac" -> "aac"
            "audio/wav", "audio/x-wav" -> "wav"
            "audio/flac" -> "flac"
            "audio/ogg" -> "ogg"
            else -> "audio"
        }

        fun LibraryItem.toSyncItem() = SyncItem(
            trackId = trackId,
            title = title,
            artist = artist,
            source = source,
            sourceRef = sourceRef,
            durationSec = durationSec,
            thumbnailUrl = thumbnailUrl,
            audioUploadId = audioUploadId,
            favorite = favorite,
            lastPlayedAt = lastPlayedAt?.toString(),
            createdAt = createdAt.toString(),
            updatedAt = updatedAt.toString(),
            deleted = deleted,
        )
    }
}
