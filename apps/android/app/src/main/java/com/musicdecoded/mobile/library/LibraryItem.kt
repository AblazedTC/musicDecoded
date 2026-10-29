package com.musicdecoded.mobile.library

import androidx.room.ColumnInfo
import androidx.room.Entity
import androidx.room.Index
import androidx.room.PrimaryKey
import com.musicdecoded.mobile.core.Analysis
import com.musicdecoded.mobile.core.Source
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
) {
    /** Copies the list metadata (duration, key, BPM) from a finished analysis and marks it ready. */
    fun withAnalysis(analysis: Analysis, now: Instant): LibraryItem = copy(
        durationSec = analysis.meta.durationSec,
        key = analysis.key?.let { "${it.tonic} ${it.mode.name.lowercase()}" },
        bpm = analysis.bpm,
        status = LibraryStatus.READY,
        jobId = null,
        updatedAt = now,
    )
}
