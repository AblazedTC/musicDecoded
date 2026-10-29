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

/** `0 KB`, `840 KB`, `12.4 MB`, `1.2 GB` (1024-based). */
fun formatBytes(bytes: Long): String {
    val kb = bytes / 1024.0
    return when {
        kb < 1024 -> "${kb.roundToInt()} KB"
        kb < 1024 * 1024 -> "%.1f MB".format(java.util.Locale.ROOT, kb / 1024)
        else -> "%.1f GB".format(java.util.Locale.ROOT, kb / 1024 / 1024)
    }
}
