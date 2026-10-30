package com.musicdecoded.mobile.ui.components

import com.musicdecoded.mobile.core.Analysis
import kotlin.math.roundToInt

/** `263.4` → `4:23`. */
fun formatDuration(seconds: Double): String {
    val total = seconds.toInt().coerceAtLeast(0)
    return "${total / 60}:${(total % 60).toString().padStart(2, '0')}"
}

/** `D major`, `F♯ minor`. */
fun keyName(key: Analysis.Key): String {
    val tonic = key.tonic.replace("#", "♯").replace("b", "♭")
    return "$tonic ${if (key.mode == Analysis.Mode.MAJOR) "major" else "minor"}"
}

/** Song-row meta line `duration · key · BPM`, skipping parts that are unknown. */
fun songMetaLine(durationSec: Double?, key: Analysis.Key?, bpm: Double?): String =
    listOfNotNull(
        durationSec?.let(::formatDuration),
        key?.let(::keyName),
        bpm?.let { "${it.roundToInt()} BPM" },
    ).joinToString(" · ")

/**
 * The one file-size format (import sheet, migration, Settings storage): decimal units, one
 * decimal for MB and GB. `0 KB`, `640 KB`, `6.0 MB`, `1.2 GB`.
 */
fun formatBytes(bytes: Long): String = when {
    bytes < 999_500 -> "${(bytes / 1000.0).roundToInt()} KB"
    bytes < 999_950_000 -> "%.1f MB".format(java.util.Locale.ROOT, bytes / 1_000_000.0)
    else -> "%.1f GB".format(java.util.Locale.ROOT, bytes / 1_000_000_000.0)
}
