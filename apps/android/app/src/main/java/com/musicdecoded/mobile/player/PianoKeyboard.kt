package com.musicdecoded.mobile.player

import androidx.compose.foundation.Canvas
import androidx.compose.foundation.background
import androidx.compose.foundation.gestures.detectHorizontalDragGestures
import androidx.compose.foundation.layout.Arrangement
import androidx.compose.foundation.layout.Box
import androidx.compose.foundation.layout.Column
import androidx.compose.foundation.layout.Row
import androidx.compose.foundation.layout.fillMaxWidth
import androidx.compose.foundation.layout.height
import androidx.compose.foundation.layout.size
import androidx.compose.foundation.shape.CircleShape
import androidx.compose.material3.MaterialTheme
import androidx.compose.runtime.Composable
import androidx.compose.runtime.getValue
import androidx.compose.runtime.rememberUpdatedState
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.draw.clipToBounds
import androidx.compose.ui.geometry.Offset
import androidx.compose.ui.geometry.Size
import androidx.compose.ui.graphics.Color
import androidx.compose.ui.graphics.drawscope.DrawScope
import androidx.compose.ui.hapticfeedback.HapticFeedbackType
import androidx.compose.ui.input.pointer.pointerInput
import androidx.compose.ui.platform.LocalDensity
import androidx.compose.ui.platform.LocalHapticFeedback
import androidx.compose.ui.semantics.CustomAccessibilityAction
import androidx.compose.ui.semantics.clearAndSetSemantics
import androidx.compose.ui.semantics.contentDescription
import androidx.compose.ui.semantics.customActions
import androidx.compose.ui.semantics.stateDescription
import androidx.compose.ui.text.TextMeasurer
import androidx.compose.ui.text.TextStyle
import androidx.compose.ui.text.drawText
import androidx.compose.ui.text.font.FontWeight
import androidx.compose.ui.text.rememberTextMeasurer
import androidx.compose.ui.tooling.preview.Preview
import androidx.compose.ui.unit.Dp
import androidx.compose.ui.unit.dp
import androidx.compose.ui.unit.sp
import com.musicdecoded.mobile.core.PianoVoicing
import com.musicdecoded.mobile.ui.components.PreviewSurface
import com.musicdecoded.mobile.ui.components.block
import com.musicdecoded.mobile.ui.theme.MusicDecodedColors
import com.musicdecoded.mobile.ui.theme.Shadows
import com.musicdecoded.mobile.ui.theme.Sizes
import com.musicdecoded.mobile.ui.theme.SpaceGrotesk
import com.musicdecoded.mobile.ui.theme.Spacing
import kotlin.math.abs
import kotlin.math.max

/** Keyboard heights from DESIGN.md §3.7/§3.7.1/§3.8 (component sizes, not spacing). */
object PianoKeyboardDefaults {
    /** Timeline view piano panel: 110–150, from the width. */
    val PanelMin = 110.dp
    val PanelMax = 150.dp
    const val PANEL_HEIGHT_PER_WIDTH = 0.32f

    /** Beat grid / chord sheet compact piano panel. */
    val CompactHeight = 90.dp

    /** Piano focus. */
    val LargeHeight = 190.dp

    /** Key separators: ink, 1.5pt (DESIGN.md §1 piano). */
    internal val Separator = 1.5.dp
    internal const val BLACK_KEY_HEIGHT = 0.62f
    internal const val BLACK_KEY_WIDTH = 0.6f
    internal val FingerSize = 12.sp
    internal val Dot = 6.dp
}

/**
 * The fixed C3–B5 keyboard (DESIGN.md §3.7 piano panel): always the same 21 white keys across the
 * full width, so a chord change only changes the highlights. Bass (left hand) in `blue`, right hand
 * in `lime`, finger numbers on the highlighted keys, note names under them and one page dot per
 * inversion. Swiping left/right calls [onStepInversion] with +1/−1 (it returns whether anything
 * changed, for the haptic); the same steps are accessibility actions.
 */
@Composable
fun FixedPianoKeyboard(
    voicing: PianoVoicing?,
    noteName: (Int) -> String,
    keysHeight: Dp,
    onStepInversion: (Int) -> Boolean,
    modifier: Modifier = Modifier,
    /** The inversion shown (already clamped to the chord), for the page dots. */
    inversion: Int = 0,
    showNoteNames: Boolean = true,
    description: String? = null,
    frameShadow: Dp = Shadows.sm,
) {
    val haptics = LocalHapticFeedback.current
    val step by rememberUpdatedState { delta: Int ->
        onStepInversion(delta).also { if (it) haptics.performHapticFeedback(HapticFeedbackType.TextHandleMove) }
    }
    val swipeThreshold = with(LocalDensity.current) { SWIPE_THRESHOLD.toPx() }
    val textMeasurer = rememberTextMeasurer()
    val fingerStyle = TextStyle(fontFamily = SpaceGrotesk, fontWeight = FontWeight.Bold, fontSize = PianoKeyboardDefaults.FingerSize, color = MusicDecodedColors.PianoBlack)
    val nameStyle = MaterialTheme.typography.labelSmall.copy(fontWeight = FontWeight.Bold)
    val count = voicing?.inversions ?: 0

    Column(
        modifier
            .fillMaxWidth()
            .pointerInput(Unit) {
                var total = 0f
                detectHorizontalDragGestures(
                    onDragStart = { total = 0f },
                    onDragEnd = { if (abs(total) > swipeThreshold) step(if (total < 0) 1 else -1) },
                ) { change, amount ->
                    change.consume()
                    total += amount
                }
            }
            .clearAndSetSemantics {
                contentDescription = pianoDescription(voicing?.let { it.rightHand.size + 1 } ?: 0, description)
                if (count > 0) {
                    stateDescription = inversionName(inversion)
                    customActions = listOf(
                        CustomAccessibilityAction("Next inversion") { step(1) },
                        CustomAccessibilityAction("Previous inversion") { step(-1) },
                    )
                }
            },
        verticalArrangement = Arrangement.spacedBy(Spacing.xxs),
    ) {
        Canvas(
            Modifier
                .fillMaxWidth()
                .height(keysHeight)
                .block(MusicDecodedColors.PianoBlack, shadow = frameShadow, shadowColor = MusicDecodedColors.Lime)
                .clipToBounds(),
        ) {
            drawFixedKeys(voicing, textMeasurer, fingerStyle)
        }
        if (showNoteNames) {
            val lineHeight = with(LocalDensity.current) { max(NAME_LINE.toPx(), NAME_LINE.value.sp.toPx()).toDp() }
            Canvas(Modifier.fillMaxWidth().height(lineHeight * 2).clipToBounds()) {
                if (voicing != null) drawNoteNames(voicing, noteName, textMeasurer, nameStyle)
            }
        }
        Row(
            Modifier.fillMaxWidth().height(PianoKeyboardDefaults.Dot),
            horizontalArrangement = Arrangement.spacedBy(Spacing.xxs, Alignment.CenterHorizontally),
        ) {
            repeat(count) {
                Box(
                    Modifier
                        .size(PianoKeyboardDefaults.Dot)
                        .background(if (it == inversion) MusicDecodedColors.Primary else MusicDecodedColors.TextMuted, CircleShape),
                )
            }
        }
    }
}

/** Voicing chip / accessibility names: Root, 1st inv, 2nd inv, 3rd inv. */
fun inversionName(inversion: Int): String = when (inversion) {
    0 -> "Root"
    1 -> "1st inv"
    2 -> "2nd inv"
    3 -> "3rd inv"
    else -> "${inversion}th inv"
}

private fun DrawScope.drawFixedKeys(voicing: PianoVoicing?, textMeasurer: TextMeasurer, fingerStyle: TextStyle) {
    val whiteWidth = size.width / WHITE_KEYS
    val separator = PianoKeyboardDefaults.Separator.toPx()
    val fingers = HashMap<Int, Int>()
    voicing?.let { v ->
        v.rightHand.forEachIndexed { i, midi -> fingers[midi] = v.rightHandFingers[i] }
        fingers.putIfAbsent(v.bass, v.bassFinger)
    }
    val inset = Sizes.hairline.toPx() * 4

    for (midi in KEYBOARD_RANGE) {
        if (isBlackKey(midi)) continue
        val left = keyboardWhiteIndex(midi) * whiteWidth
        drawRect(keyColor(midi, voicing, black = false), Offset(left, 0f), Size(whiteWidth, size.height))
        drawRect(MusicDecodedColors.PianoBlack, Offset(left - separator / 2, 0f), Size(separator, size.height))
    }
    val blackWidth = whiteWidth * PianoKeyboardDefaults.BLACK_KEY_WIDTH
    val blackHeight = size.height * PianoKeyboardDefaults.BLACK_KEY_HEIGHT
    for (midi in KEYBOARD_RANGE) {
        if (!isBlackKey(midi)) continue
        val left = (keyboardWhiteIndex(midi) + 1) * whiteWidth - blackWidth / 2
        drawRect(keyColor(midi, voicing, black = true), Offset(left, 0f), Size(blackWidth, blackHeight))
    }
    for ((midi, finger) in fingers) {
        val black = isBlackKey(midi)
        val center = if (black) (keyboardWhiteIndex(midi) + 1) * whiteWidth else (keyboardWhiteIndex(midi) + 0.5f) * whiteWidth
        val bottom = if (black) blackHeight else size.height
        val layout = textMeasurer.measure(finger.toString(), fingerStyle)
        drawText(layout, topLeft = Offset(center - layout.size.width / 2f, bottom - layout.size.height - inset))
    }
}

/** Highlighted notes' names under their keys (bass `blue`, right hand `lime`), staggered onto a second line when they'd touch. */
private fun DrawScope.drawNoteNames(voicing: PianoVoicing, noteName: (Int) -> String, textMeasurer: TextMeasurer, style: TextStyle) {
    val whiteWidth = size.width / WHITE_KEYS
    val gap = Spacing.xxs.toPx()
    val lastRight = floatArrayOf(Float.NEGATIVE_INFINITY, Float.NEGATIVE_INFINITY)
    for (midi in (listOf(voicing.bass) + voicing.rightHand).distinct().sorted()) {
        val center = if (isBlackKey(midi)) (keyboardWhiteIndex(midi) + 1) * whiteWidth else (keyboardWhiteIndex(midi) + 0.5f) * whiteWidth
        val color = if (midi == voicing.bass) MusicDecodedColors.PianoBass else MusicDecodedColors.PianoWhiteActive
        val text = textMeasurer.measure(noteName(midi), style.copy(color = color))
        val left = (center - text.size.width / 2f).coerceIn(0f, size.width - text.size.width)
        val row = if (left >= lastRight[0]) 0 else 1
        drawText(text, topLeft = Offset(left, row * size.height / 2))
        lastRight[row] = left + text.size.width + gap
    }
}

private fun keyColor(midi: Int, voicing: PianoVoicing?, black: Boolean): Color = when {
    voicing != null && midi == voicing.bass -> MusicDecodedColors.PianoBass
    voicing != null && midi in voicing.rightHand -> if (black) MusicDecodedColors.PianoBlackActive else MusicDecodedColors.PianoWhiteActive
    else -> if (black) MusicDecodedColors.PianoBlack else MusicDecodedColors.PianoWhite
}

/** A swipe must travel this far to change the inversion. */
private val SWIPE_THRESHOLD = 48.dp

/** One line of note names under the keys (grows with font scale). */
private val NAME_LINE = 14.dp

/** C3 (48) … B5 (83): 3 octaves, 21 white keys. */
private val KEYBOARD_RANGE = PianoVoicing.KEYBOARD_LOW..PianoVoicing.KEYBOARD_HIGH
private const val WHITE_KEYS = 21

private const val WHITE_KEYS_PER_OCTAVE = 7
private val BLACK_PITCH_CLASSES = setOf(1, 3, 6, 8, 10)
private val WHITE_PITCH_CLASSES = listOf(0, 2, 4, 5, 7, 9, 11)

internal fun isBlackKey(midi: Int): Boolean = Math.floorMod(midi, 12) in BLACK_PITCH_CLASSES

/** White keys from MIDI 0 up to [midi] (a black key counts as the white key below it). */
internal fun whiteKeyIndex(midi: Int): Int {
    val octave = Math.floorDiv(midi, 12)
    val pitchClass = Math.floorMod(midi, 12)
    return octave * WHITE_KEYS_PER_OCTAVE + WHITE_PITCH_CLASSES.indexOfLast { it <= pitchClass }
}

/** White key position on the fixed keyboard (C3 = 0 … B5 = 20). */
internal fun keyboardWhiteIndex(midi: Int): Int = whiteKeyIndex(midi) - whiteKeyIndex(PianoVoicing.KEYBOARD_LOW)

internal fun midiOfWhiteKey(index: Int): Int =
    Math.floorDiv(index, WHITE_KEYS_PER_OCTAVE) * 12 + WHITE_PITCH_CLASSES[Math.floorMod(index, WHITE_KEYS_PER_OCTAVE)]

/** "Piano, keys E, E, G♯, B highlighted" (or a count when the names aren't known). */
internal fun pianoDescription(count: Int, noteNames: String?): String = when {
    count == 0 -> "Piano, no chord"
    !noteNames.isNullOrBlank() -> "Piano, keys ${noteNames.split('·').joinToString(", ") { it.trim() }} highlighted"
    else -> "Piano with $count keys highlighted"
}

@Preview(widthDp = 380)
@Composable
private fun FixedPianoKeyboardPreview() = PreviewSurface {
    FixedPianoKeyboard(
        voicing = PianoVoicing(3, listOf(64, 68, 71), 52, listOf(1, 3, 5)),
        noteName = { listOf("C", "C♯", "D", "E♭", "E", "F", "F♯", "G", "G♯", "A", "B♭", "B")[it % 12] },
        keysHeight = PianoKeyboardDefaults.PanelMin,
        onStepInversion = { true },
    )
}
