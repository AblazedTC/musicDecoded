package com.musicdecoded.mobile.core

/**
 * A Harte-style label split into its parts: `root:quality/bass`.
 * `N` (no chord) and `X` (unknown) have no parts, so [parse] returns `null` for them.
 */
internal data class ChordLabel(val root: String, val quality: String, val bass: String?) {
    companion object {
        fun parse(label: String): ChordLabel? {
            val colon = label.indexOf(':')
            if (colon < 0) return null
            val rest = label.substring(colon + 1)
            val slash = rest.indexOf('/')
            return if (slash >= 0) {
                ChordLabel(label.substring(0, colon), rest.substring(0, slash), rest.substring(slash + 1))
            } else {
                ChordLabel(label.substring(0, colon), rest, null)
            }
        }
    }
}

/** `N` and `X` are "no chord": no name, no notes, and skipped by up-next and prev/next. */
internal fun isNoChord(label: String): Boolean = label == "N" || label == "X"

/** Wraps any integer into a pitch class 0..11 (handles negative transposes). */
internal fun wrapPitchClass(value: Int): Int = ((value % 12) + 12) % 12
