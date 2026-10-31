package com.musicdecoded.mobile.core

/**
 * What the fixed C3–B5 player keyboard highlights for one chord (DESIGN.md §4 "Piano voicing &
 * inversions"): a right hand of at most 4 MIDI notes in C4–B5 and a left-hand bass note in C3–B3,
 * with finger numbers. [inversions] = number of right-hand tones (one inversion per tone).
 */
data class PianoVoicing(
    val inversions: Int,
    val rightHand: List<Int>,
    val bass: Int,
    val rightHandFingers: List<Int>,
    val bassFinger: Int = BASS_FINGER,
) {
    companion object {
        const val KEYBOARD_LOW = 48 // C3
        const val KEYBOARD_HIGH = 83 // B5
        private const val BASS_FINGER = 5
        private const val RIGHT_HAND_LOW = 60 // C4
        private const val MAX_TONES = 4
        private const val PERFECT_FIFTH = 7
        private const val OCTAVE = 12

        /**
         * The voicing of [label] (`root:quality[/bass]`) at [inversion] (wraps modulo the tone count),
         * with [transpose] applied to the root. [simplify] = Chords: Basic (triad family, no slash bass).
         * `null` for `N`/`X` or an unknown root.
         */
        fun of(voicings: ChordVoicings, label: String, inversion: Int = 0, transpose: Int = 0, simplify: Boolean = false): PianoVoicing? {
            val parsed = ChordLabel.parse(label) ?: return null
            val chord = if (simplify) ChordLabel(parsed.root, triadFamily(parsed.quality), null) else parsed
            val root = wrapPitchClass((voicings.pitchClass[chord.root] ?: return null) + transpose)
            var intervals = voicings.qualities[chord.quality]
                ?: voicings.qualities[chord.quality.substringBefore('(')]
                ?: voicings.qualities["maj"]
                ?: return null
            if (intervals.size > MAX_TONES) intervals = intervals.filter { it != PERFECT_FIFTH }
            val tones = intervals.take(MAX_TONES).map { wrapPitchClass(root + it) }
            val n = tones.size
            if (n == 0) return null
            val k = Math.floorMod(inversion, n)
            val rotated = tones.drop(k) + tones.take(k)

            val placed = ArrayList<Int>(n)
            for (pitchClass in rotated) {
                val previous = placed.lastOrNull()
                placed += if (previous == null) {
                    RIGHT_HAND_LOW + pitchClass
                } else {
                    previous + (Math.floorMod(pitchClass - previous, OCTAVE).takeIf { it > 0 } ?: OCTAVE)
                }
            }
            val rightHand = if (placed.last() > KEYBOARD_HIGH) placed.map { it - OCTAVE } else placed

            val degree = chord.bass?.let { voicings.degreeSemitones[it] } ?: 0
            val bass = KEYBOARD_LOW + wrapPitchClass(root + degree)
            return PianoVoicing(n, rightHand, bass, fingers(n, k))
        }

        /** n=2 → 1,5 · n=3 → 1,3,5 (1st inversion 1,2,5) · n=4 → 1,2,3,5. */
        internal fun fingers(n: Int, inversion: Int): List<Int> = when (n) {
            1 -> listOf(1)
            2 -> listOf(1, 5)
            3 -> if (inversion == 1) listOf(1, 2, 5) else listOf(1, 3, 5)
            else -> listOf(1, 2, 3, 5)
        }
    }
}
