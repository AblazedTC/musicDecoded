package com.musicdecoded.mobile.library

import androidx.room.ColumnInfo
import androidx.room.Entity
import androidx.room.Index
import androidx.room.PrimaryKey
import com.musicdecoded.mobile.core.Analysis
import com.musicdecoded.mobile.core.Source
import com.musicdecoded.mobile.sources.soundcloud.SoundCloudUrl
import java.time.Instant

/** Local-only row state (docs/data-model.md). */
enum class LibraryStatus { ANALYZING, READY, FAILED }

/**
 * One entry in this device's history. Mirrors `LibraryItem` in contracts/openapi.yaml plus the
 * local-only fields from docs/data-model.md §1. Deletes are tombstones ([deleted] = true) so
 * they can sync; rows are never removed.
 */
@Entity(tableName = "library_items", indices = [Index("createdAt")])
data class LibraryItem(
    @PrimaryKey val trackId: String,
    val title: String,
    val artist: String? = null,
    val source: Source,
    /** videoId / SoundCloud URL. */
    val sourceRef: String? = null,
    val durationSec: Double? = null,
    val thumbnailUrl: String? = null,
    /** file/recording only; lets another device download the user's own audio. */
    val audioUploadId: String? = null,
    val favorite: Boolean = false,
    /** Copied from the analysis file (e.g. `B minor`) so lists can show and search it without decoding JSON. */
    @ColumnInfo(name = "musicKey") val key: String? = null,
    val bpm: Double? = null,
    val lastPlayedAt: Instant? = null,
    val createdAt: Instant,
    /** Drives last-write-wins sync. */
    val updatedAt: Instant,
    /** Tombstone. */
    val deleted: Boolean = false,
    // Local only.
    val status: LibraryStatus,
    /** Set while analyzing. */
    val jobId: String? = null,
    /** `null` if streaming-only. */
    val localAudioPath: String? = null,
    val syncedAt: Instant? = null,
    /** Local only: user-facing reason when [status] is FAILED. */
    val error: String? = null,
) {
    /** Copies the list metadata (duration, key, BPM) from a finished analysis and marks it ready. */
    fun withAnalysis(analysis: Analysis, now: Instant): LibraryItem = copy(
        durationSec = analysis.meta.durationSec,
        key = analysis.key?.let { "${it.tonic} ${it.mode.name.lowercase()}" },
        bpm = analysis.bpm,
        status = LibraryStatus.READY,
        jobId = null,
        error = null,
        updatedAt = now,
    )

    /** Whether [title] is only a stand-in we made up (a pasted YouTube link, a SoundCloud slug). */
    val hasPlaceholderTitle: Boolean
        get() = when (source) {
            Source.YOUTUBE -> title == YOUTUBE_PLACEHOLDER_TITLE
            Source.SOUNDCLOUD -> title == SOUNDCLOUD_PLACEHOLDER_TITLE ||
                sourceRef?.let { title == SoundCloudUrl.titleFrom(it) } == true
            else -> false
        }

    /**
     * Fills in what a job learned about the track (`Job.title/thumbnailUrl/durationSec`): replaces a
     * placeholder title and fills a missing thumbnail or duration, never overwriting real values.
     * Returns this same instance when nothing changes, so callers can skip the write.
     */
    fun withJobMetadata(title: String?, thumbnailUrl: String?, durationSec: Double?, now: Instant): LibraryItem {
        val newTitle = title?.trim()?.takeIf { it.isNotEmpty() && hasPlaceholderTitle && it != this.title }
        val newThumb = thumbnailUrl?.takeIf { it.isNotBlank() && this.thumbnailUrl == null }
        val newDuration = durationSec?.takeIf { it > 0 && this.durationSec == null }
        if (newTitle == null && newThumb == null && newDuration == null) return this
        return copy(
            title = newTitle ?: this.title,
            thumbnailUrl = newThumb ?: this.thumbnailUrl,
            durationSec = newDuration ?: this.durationSec,
            updatedAt = now,
        )
    }

    companion object {
        /** Title of a pasted YouTube link until the job names the video. */
        const val YOUTUBE_PLACEHOLDER_TITLE = "YouTube video"
        /** [SoundCloudUrl.titleFrom] when the link has no readable slug. */
        const val SOUNDCLOUD_PLACEHOLDER_TITLE = "SoundCloud track"
    }
}
