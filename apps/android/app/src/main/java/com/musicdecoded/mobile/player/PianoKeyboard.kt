package com.musicdecoded.mobile.player

import androidx.compose.animation.core.animateFloatAsState
import androidx.compose.foundation.Canvas
import androidx.compose.foundation.layout.BoxWithConstraints
import androidx.compose.foundation.layout.Column
import androidx.compose.foundation.layout.fillMaxWidth
import androidx.compose.foundation.layout.height
import androidx.compose.material3.MaterialTheme
import androidx.compose.runtime.Composable
import androidx.compose.runtime.getValue
import androidx.compose.runtime.remember
import androidx.compose.ui.Modifier
import androidx.compose.ui.draw.clipToBounds

import androidx.compose.ui.geometry.Offset
import androidx.compose.ui.geometry.Size
import androidx.compose.ui.graphics.Color
import androidx.compose.ui.graphics.drawscope.DrawScope
import androidx.compose.ui.graphics.drawscope.Stroke
import androidx.compose.ui.platform.LocalDensity
import androidx.compose.ui.semantics.contentDescription
import androidx.compose.ui.semantics.semantics
import androidx.compose.ui.text.TextMeasurer
import androidx.compose.ui.text.TextStyle
import androidx.compose.ui.text.drawText
import androidx.compose.ui.text.font.FontWeight
import androidx.compose.ui.text.rememberTextMeasurer
import androidx.compose.ui.tooling.preview.Preview
import androidx.compose.ui.unit.Dp
import androidx.compose.ui.unit.dp
import androidx.compose.ui.unit.sp
import com.musicdecoded.mobile.ui.components.PreviewSurface
import com.musicdecoded.mobile.ui.components.block
import com.musicdecoded.mobile.ui.theme.MusicDecodedColors
import com.musicdecoded.mobile.ui.theme.Shadows
import com.musicdecoded.mobile.ui.theme.SpaceGrotesk
import com.musicdecoded.mobile.ui.theme.Sizes
import com.musicdecoded.mobile.ui.theme.Spacing
import kotlin.math.ceil
import kotlin.math.floor
import kotlin.math.max
import kotlin.math.roundToInt

/** Keyboard sizes from DESIGN.md §3.7/§3.8 (component sizes, not spacing). */
object PianoKeyboardDefaults {
    /** 2 octaves in the player's visualizer card. */
    val CompactHeight = 96.dp

    /** Edge to edge in the Piano learning view. */
    val LargeHeight = 190.dp

    /** Key separators: ink, 1.5pt (DESIGN.md §1 piano). */
    internal val Separator = 1.5.dp
    internal const val BLACK_KEY_HEIGHT = 0.62f
    internal const val BLACK_KEY_WIDTH = 0.6f
    internal val NoteNameSize = 11.sp
}

/**
 * A Canvas piano that highlights [notes] (MIDI) in the DESIGN.md §1 piano colors, with the slash
 * [bassNote] in the chord color. It shows [octaves] octaves and slides to keep the active keys
 * in view. [noteName] (when set) labels the white keys and the active keys.
 */
@Composable
fun PianoKeyboard(
    notes: List<Int>,
    modifier: Modifier = Modifier,
    bassNote: Int? = null,
    octaves: Int = 2,
    height: Dp = PianoKeyboardDefaults.CompactHeight,
    noteName: ((Int) -> String)? = null,
    /** What TalkBack reads for the highlighted keys, e.g. "B · D · F♯" (the chord's note names). */
    description: String? = null,
    /** Hard shadow of the keyboard frame (3pt black border); lime on ink (DESIGN.md §1). */
    frameShadow: Dp = Shadows.md,
    frameShadowColor: Color = MusicDecodedColors.Lime,
) {
    // Hold the last position while there's no chord, so the keyboard doesn't jump around.
    val lastStart = remember { intArrayOf(DEFAULT_START) }
    val start = keyboardStart(notes, octaves) ?: lastStart[0]
    lastStart[0] = start
    val firstWhite by animateFloatAsState(whiteKeyIndex(start).toFloat(), label = "keyboardScroll")
    val textMeasurer = rememberTextMeasurer()
    val active = remember(notes) { notes.toSet() }

    Canvas(
        modifier
            .fillMaxWidth()
            .height(height)
            .block(MusicDecodedColors.PianoBlack, shadow = frameShadow, shadowColor = frameShadowColor)
            .clipToBounds()
            .semantics { contentDescription = pianoDescription(notes.size, description) },
    ) {
        drawKeys(firstWhite, size.width / (octaves * WHITE_KEYS_PER_OCTAVE), octaves, active, bassNote, noteName, textMeasurer)
    }
}

/**
 * The player's piano panel keyboard (DESIGN.md §3.7 item 5): fills the size it's given, shows as many
 * octaves as fit the width at natural key proportions (at least 2), centered on the active notes,
 * with the highlighted notes' names under their keys.
 */
@Composable
fun FittedPianoKeyboard(
    notes: List<Int>,
    noteName: (Int) -> String,
    modifier: Modifier = Modifier,
    bassNote: Int? = null,
    description: String? = null,
) {
    BoxWithConstraints(modifier) {
        val density = LocalDensity.current
        val namesHeight = with(density) { max(NAMES_BAND.toPx(), NAMES_BAND.value.sp.toPx()).toDp() }
        val keysHeight = (maxHeight - namesHeight - Shadows.md).coerceAtLeast(PianoKeyboardDefaults.CompactHeight)
        val octaves = fittedOctaves(maxWidth.value, keysHeight.value)
        val lastStart = remember { floatArrayOf(Float.NaN) }
        val target = centeredFirstWhite(notes, octaves)?.toFloat()
            ?: lastStart[0].takeUnless { it.isNaN() }
            ?: (whiteKeyIndex(DEFAULT_START) + WHITE_KEYS_PER_OCTAVE - octaves * WHITE_KEYS_PER_OCTAVE / 2f)
        lastStart[0] = target
        val firstWhite by animateFloatAsState(target, label = "keyboardScroll")
        val textMeasurer = rememberTextMeasurer()
        val active = remember(notes) { notes.toSet() }
        val nameStyle = MaterialTheme.typography.labelSmall
        Column {
            Canvas(
                Modifier
                    .fillMaxWidth()
                    .height(keysHeight)
                    .block(MusicDecodedColors.PianoBlack, shadow = Shadows.md, shadowColor = MusicDecodedColors.Lime)
                    .clipToBounds()
                    .semantics { contentDescription = pianoDescription(notes.size, description) },
            ) {
                drawKeys(firstWhite, size.width / (octaves * WHITE_KEYS_PER_OCTAVE), octaves, active, bassNote, null, textMeasurer)
            }
            Canvas(Modifier.fillMaxWidth().height(namesHeight + Shadows.md).clipToBounds()) {
                val whiteWidth = size.width / (octaves * WHITE_KEYS_PER_OCTAVE)
                var lastRight = Float.NEGATIVE_INFINITY
                for (midi in notes.sorted().distinct()) {
                    val keyCenter = if (isBlackKey(midi)) whiteKeyIndex(midi) + 1f else whiteKeyIndex(midi) + 0.5f
                    val x = (keyCenter - firstWhite) * whiteWidth
                    val color = if (midi == bassNote) MusicDecodedColors.PianoBass else MusicDecodedColors.PianoWhiteActive
                    val text = textMeasurer.measure(noteName(midi), nameStyle.copy(color = color))
                    val left = x - text.size.width / 2f
                    if (left < lastRight || left < 0f || left + text.size.width > size.width) continue
                    drawText(text, topLeft = Offset(left, Shadows.md.toPx()))
                    lastRight = left + text.size.width + Spacing.xxs.toPx()
                }
            }
        }
    }
}

private fun DrawScope.drawKeys(
    firstWhite: Float,
    whiteWidth: Float,
    octaves: Int,
    active: Set<Int>,
    bassNote: Int?,
    noteName: ((Int) -> String)?,
    textMeasurer: TextMeasurer,
) {
    val first = floor(firstWhite).toInt() - 1
    val last = ceil(firstWhite).toInt() + octaves * WHITE_KEYS_PER_OCTAVE + 1
    val separator = PianoKeyboardDefaults.Separator.toPx()
    val nameStyle = TextStyle(fontFamily = SpaceGrotesk, fontSize = PianoKeyboardDefaults.NoteNameSize, fontWeight = FontWeight.Bold)

    for (index in first..last) {
        val midi = midiOfWhiteKey(index)
        if (midi !in PIANO_RANGE) continue
        val left = (index - firstWhite) * whiteWidth
        val color = keyColor(midi, active, bassNote, black = false)
        drawRect(color, Offset(left, 0f), Size(whiteWidth, size.height))
        drawRect(MusicDecodedColors.PianoBlack, Offset(left - separator / 2, 0f), Size(separator, size.height))
        if (noteName != null) {
            val label = if (midi in active) noteName(midi) else whiteLabel(midi)
            drawKeyLabel(textMeasurer, label, nameStyle.copy(color = MusicDecodedColors.PianoBlack), left, whiteWidth, size.height)
        }
    }
    val blackWidth = whiteWidth * PianoKeyboardDefaults.BLACK_KEY_WIDTH
    val blackHeight = size.height * PianoKeyboardDefaults.BLACK_KEY_HEIGHT
    for (index in first..last) {
        val midi = midiOfWhiteKey(index) + 1
        if (!isBlackKey(midi) || midi !in PIANO_RANGE) continue
        val left = (index + 1 - firstWhite) * whiteWidth - blackWidth / 2
        drawRect(keyColor(midi, active, bassNote, black = true), Offset(left, 0f), Size(blackWidth, blackHeight))
        if (noteName != null && midi in active) {
            drawKeyLabel(textMeasurer, noteName(midi), nameStyle.copy(color = MusicDecodedColors.OnPrimary), left, blackWidth, blackHeight)
        }
    }
}

/** Height of the note-name band under the fitted keyboard (grows with font scale). */
private val NAMES_BAND = 16.dp

/** White key height ÷ width used to decide how many octaves fit. */
internal const val WHITE_KEY_ASPECT = 5f

/** Octaves that fit [width] at natural proportions for a keyboard [height] tall; at least 2. */
internal fun fittedOctaves(width: Float, height: Float): Int =
    max(2, floor(width / (WHITE_KEYS_PER_OCTAVE * height / WHITE_KEY_ASPECT)).toInt())

/**
 * The first white key (index from MIDI 0) that centers [notes] on an [octaves]-octave keyboard,
 * kept inside the 88-key range. `null` = no notes.
 */
internal fun centeredFirstWhite(notes: List<Int>, octaves: Int): Int? {
    if (notes.isEmpty()) return null
    val keys = octaves * WHITE_KEYS_PER_OCTAVE
    val center = (whiteKeyIndex(notes.min()) + whiteKeyIndex(notes.max()) + 1) / 2.0
    return (center - keys / 2.0).roundToInt()
        .coerceAtMost(whiteKeyIndex(PIANO_RANGE.last) + 1 - keys)
        .coerceAtLeast(whiteKeyIndex(PIANO_RANGE.first))
}

private fun keyColor(midi: Int, active: Set<Int>, bassNote: Int?, black: Boolean): Color = when {
    midi == bassNote -> MusicDecodedColors.PianoBass
    midi in active -> if (black) MusicDecodedColors.PianoBlackActive else MusicDecodedColors.PianoWhiteActive
    else -> if (black) MusicDecodedColors.PianoBlack else MusicDecodedColors.PianoWhite
}

private fun DrawScope.drawKeyLabel(textMeasurer: TextMeasurer, text: String, style: TextStyle, left: Float, width: Float, bottom: Float) {
    val layout = textMeasurer.measure(text, style)
    if (layout.size.width > width) return
    val padding = Sizes.hairline.toPx() * 4
    drawText(layout, topLeft = Offset(left + (width - layout.size.width) / 2f, bottom - layout.size.height - padding))
}

private const val WHITE_KEYS_PER_OCTAVE = 7
private const val DEFAULT_START = 48
private val PIANO_RANGE = 21..108
private val BLACK_PITCH_CLASSES = setOf(1, 3, 6, 8, 10)
private val WHITE_PITCH_CLASSES = listOf(0, 2, 4, 5, 7, 9, 11)
private val WHITE_LETTERS = listOf("C", "D", "E", "F", "G", "A", "B")

internal fun isBlackKey(midi: Int): Boolean = Math.floorMod(midi, 12) in BLACK_PITCH_CLASSES

/** White keys from MIDI 0 up to [midi] (a black key counts as the white key below it). */
internal fun whiteKeyIndex(midi: Int): Int {
    val octave = Math.floorDiv(midi, 12)
    val pitchClass = Math.floorMod(midi, 12)
    return octave * WHITE_KEYS_PER_OCTAVE + WHITE_PITCH_CLASSES.indexOfLast { it <= pitchClass }
}

/** "Piano, keys B, D, F♯ highlighted" (or a count when the names aren't known). */
internal fun pianoDescription(count: Int, noteNames: String?): String = when {
    count == 0 -> "Piano, no chord"
    !noteNames.isNullOrBlank() -> "Piano, keys ${noteNames.split('·').joinToString(", ") { it.trim() }} highlighted"
    else -> "Piano with $count keys highlighted"
}

internal fun midiOfWhiteKey(index: Int): Int =
    Math.floorDiv(index, WHITE_KEYS_PER_OCTAVE) * 12 + WHITE_PITCH_CLASSES[Math.floorMod(index, WHITE_KEYS_PER_OCTAVE)]

/** `C3` on Cs (middle C = C4), plain letters elsewhere. */
private fun whiteLabel(midi: Int): String {
    val letter = WHITE_LETTERS[WHITE_PITCH_CLASSES.indexOf(Math.floorMod(midi, 12))]
    return if (letter == "C") "C${midi / 12 - 1}" else letter
}

/**
 * The C (MIDI) the keyboard should start at so [notes] fit in [octaves] octaves; when they
 * don't, the lowest notes (the slash bass first) are dropped until the rest fit. `null` = no notes.
 */
internal fun keyboardStart(notes: List<Int>, octaves: Int): Int? {
    var candidates = notes.sorted()
    while (candidates.isNotEmpty()) {
        val start = Math.floorDiv(candidates.first(), 12) * 12
        if (candidates.last() < start + 12 * octaves) return start
        candidates = candidates.drop(1)
    }
    return null
}

@Preview(widthDp = 380)
@Composable
private fun PianoKeyboardPreview() = PreviewSurface {
    // D/F♯: bass F♯2, chord D3 F♯3 A3.
    PianoKeyboard(notes = listOf(42, 50, 54, 57), bassNote = 42)
}

@Preview(widthDp = 380)
@Composable
private fun LargePianoKeyboardPreview() = PreviewSurface {
    PianoKeyboard(
        notes = listOf(50, 54, 57, 61),
        height = PianoKeyboardDefaults.LargeHeight,
        noteName = { listOf("C", "C♯", "D", "E♭", "E", "F", "F♯", "G", "A♭", "A", "B♭", "B")[it % 12] },
    )
}
