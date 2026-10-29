package com.musicdecoded.mobile.core

import kotlinx.serialization.Serializable

/**
 * The shared voicing table from `contracts/chord-voicings.json`.
 * Both apps use it so their piano views show the same notes.
 */
@Serializable
data class ChordVoicings(
    val rootOctaveBase: Int,
    val bassOctaveBase: Int,
    val pitchClass: Map<String, Int>,
    val qualities: Map<String, List<Int>>,
    val degreeSemitones: Map<String, Int>,
    /** Label → expected MIDI notes. Test cases for [notes]. */
    val examples: Map<String, List<Int>>? = null,
) {
    /**
     * MIDI notes for a chord label, following the algorithm in the file's `description`.
     * [transpose] shifts the root's pitch class (wrapping within the octave), so
     * `notes("A:maj", transpose = 3) == notes("C:maj")`.
     */
    fun notes(label: String, transpose: Int = 0): List<Int> {
        val chord = ChordLabel.parse(label) ?: return emptyList()
        val pitch = pitchClass[chord.root] ?: return emptyList()
        val root = wrapPitchClass(pitch + transpose)
        val intervals = qualities[chord.quality]
            ?: qualities[chord.quality.substringBefore('(')]
            ?: qualities["maj"]
            ?: emptyList()
        val notes = intervals.map { rootOctaveBase + root + it }
        val degree = chord.bass?.let { degreeSemitones[it] } ?: return notes
        return listOf(bassOctaveBase + (root + degree) % 12) + notes
    }

    companion object {
        fun decode(json: String): ChordVoicings = AnalysisJson.decodeFromString(serializer(), json)
    }
}
