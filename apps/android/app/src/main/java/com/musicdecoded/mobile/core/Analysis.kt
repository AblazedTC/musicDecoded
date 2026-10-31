package com.musicdecoded.mobile.core

import kotlinx.serialization.SerialName
import kotlinx.serialization.Serializable
import kotlinx.serialization.json.Json

/**
 * The chord file (schema v1). Mirrors `contracts/analysis.schema.json` field for field.
 * All times are seconds from the start of the audio.
 *
 * `key`, `bpm` and `timeSignature` are required by the schema but may be `null`, so they have no
 * default: decoding fails if the key is missing, and encoding always writes them (as `null`).
 * `spectrogram` / `waveformPeaks` are optional and omitted when `null`.
 */
@Serializable
data class Analysis(
    val schemaVersion: Int,
    val meta: Meta,
    val key: Key?,
    val bpm: Double?,
    val timeSignature: TimeSignature?,
    /** Sorted by `t` ascending. */
    val beats: List<Beat>,
    /** Sorted, contiguous, non-overlapping segments. */
    val chords: List<Chord>,
    val spectrogram: Spectrogram? = null,
    val waveformPeaks: WaveformPeaks? = null,
) {
    @Serializable
    data class Meta(
        val trackId: String,
        val source: Source,
        /** YouTube videoId or SoundCloud URL; `null` for file/recording. */
        val sourceRef: String? = null,
        val durationSec: Double,
        /** ISO 8601 date-time, kept as text so the file round-trips verbatim. */
        val analyzedAt: String,
        val analyzerVersion: String,
    )

    @Serializable
    data class Key(
        /** One of `C C# D D# E F F# G G# A A# B`. */
        val tonic: String,
        val mode: Mode,
        val confidence: Double? = null,
    )

    @Serializable
    enum class Mode {
        @SerialName("major") MAJOR,
        @SerialName("minor") MINOR,
    }

    @Serializable
    data class TimeSignature(
        val numerator: Int,
        /** One of 2, 4, 8, 16. */
        val denominator: Int,
    )

    @Serializable
    data class Beat(val t: Double, val downbeat: Boolean)

    @Serializable
    data class Chord(
        val start: Double,
        val end: Double,
        /** Harte-style label, e.g. `A:maj`, `F#:min7`, `D:maj/3`; `N` = no chord, `X` = unknown. */
        val label: String,
        val confidence: Double? = null,
    )

    /** Downsampled mel spectrogram. Row-major frames: `data[frame * bins + bin]`. */
    @Serializable
    data class Spectrogram(
        val fps: Double,
        val bins: Int,
        val frames: Int,
        /** Always `uint8-base64`: each value 0-255 maps to dB in [-80, 0]. */
        val encoding: String,
        /** Base64 text. */
        val data: String,
    )

    @Serializable
    data class WaveformPeaks(
        val fps: Double,
        /** Values in [0, 1]. */
        val peaks: List<Double>,
    )

    companion object {
        /** Decodes an analysis file, e.g. the contents of `analysis/{trackId}.json`. */
        fun decode(json: String): Analysis = AnalysisJson.decodeFromString(serializer(), json)
    }

    fun encode(): String = AnalysisJson.encodeToString(serializer(), this)
}

/** Where a track comes from. Same values in the analysis file, `JobRequest` and `LibraryItem`. */
@Serializable
enum class Source {
    @SerialName("file") FILE,
    @SerialName("recording") RECORDING,
    @SerialName("youtube") YOUTUBE,
    @SerialName("soundcloud") SOUNDCLOUD,
}

/** Lenient on unknown keys (forward compatible); writes required-nullable keys as explicit `null`. */
val AnalysisJson = Json { ignoreUnknownKeys = true }
