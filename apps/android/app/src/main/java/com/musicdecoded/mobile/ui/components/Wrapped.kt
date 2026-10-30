package com.musicdecoded.mobile.ui.components

import androidx.compose.foundation.Canvas
import androidx.compose.foundation.background
import androidx.compose.foundation.border
import androidx.compose.foundation.clickable
import androidx.compose.foundation.interaction.MutableInteractionSource
import androidx.compose.foundation.interaction.collectIsPressedAsState
import androidx.compose.foundation.layout.Arrangement
import androidx.compose.foundation.layout.Column
import androidx.compose.foundation.layout.Row
import androidx.compose.foundation.layout.fillMaxWidth
import androidx.compose.foundation.layout.height
import androidx.compose.foundation.layout.padding
import androidx.compose.foundation.layout.width
import androidx.compose.material3.Text
import androidx.compose.runtime.Composable
import androidx.compose.runtime.getValue
import androidx.compose.runtime.remember
import androidx.compose.ui.Modifier
import androidx.compose.ui.draw.clipToBounds
import androidx.compose.ui.draw.drawBehind
import androidx.compose.ui.draw.drawWithCache
import androidx.compose.ui.draw.drawWithContent
import androidx.compose.ui.geometry.Size
import androidx.compose.ui.geometry.Offset
import androidx.compose.ui.graphics.Color
import androidx.compose.ui.graphics.RectangleShape
import androidx.compose.ui.graphics.Shape
import androidx.compose.ui.graphics.drawOutline
import androidx.compose.ui.graphics.drawscope.translate
import androidx.compose.ui.graphics.graphicsLayer
import androidx.compose.ui.semantics.Role
import androidx.compose.ui.semantics.clearAndSetSemantics
import androidx.compose.ui.text.TextStyle
import androidx.compose.ui.text.style.TextAlign
import androidx.compose.ui.tooling.preview.Preview
import androidx.compose.ui.unit.Dp
import androidx.compose.ui.unit.dp
import androidx.compose.ui.unit.sp
import com.musicdecoded.mobile.ui.theme.Borders
import com.musicdecoded.mobile.ui.theme.LocalReduceMotion
import com.musicdecoded.mobile.ui.theme.MusicDecodedColors
import com.musicdecoded.mobile.ui.theme.Shadows
import com.musicdecoded.mobile.ui.theme.Spacing
import com.musicdecoded.mobile.ui.theme.TextStyles
import com.musicdecoded.mobile.ui.theme.Tilt
import com.musicdecoded.mobile.ui.theme.Typography
import kotlin.math.abs
import kotlin.math.sin

/*
 * DESIGN.md §1 "Wrapped" signature elements: hard offset shadows, 3pt borders, the pushed-sticker
 * pressed state, tilt, stickers, eyebrows, big numerals, waveform bars and textures.
 */

/** A solid shadow offset by [offset] down-right, no blur. Draws outside the bounds (leave room). */
fun Modifier.hardShadow(offset: Dp, color: Color = MusicDecodedColors.Border, shape: Shape = RectangleShape): Modifier =
    if (offset <= 0.dp) this else drawWithCache {
        // The outline only changes with the size; build it once, not on every draw.
        val outline = shape.createOutline(size, layoutDirection, this)
        val px = offset.toPx()
        onDrawBehind { translate(px, px) { drawOutline(outline, color) } }
    }

/** A 3pt rule along the top edge, drawn over the content (tab bar, sheets, transport). */
fun Modifier.topRule(color: Color = MusicDecodedColors.Border): Modifier = drawWithContent {
    drawContent()
    drawRect(color, size = Size(size.width, Borders.thick.toPx()))
}

/** Rotates by [degrees] for decoration, unless the system "Remove animations" setting is on. */
@Composable
fun Modifier.tilt(degrees: Float): Modifier {
    if (LocalReduceMotion.current || degrees == 0f) return this
    return graphicsLayer { rotationZ = degrees }
}

/**
 * A Wrapped color block: hard shadow, fill, 3pt border (null = none) and, when [onClick] is set,
 * the pushed-sticker pressed state: the block moves (+2, +2) and the shadow shrinks by 2, with no
 * ripple tint.
 */
@Composable
fun Modifier.block(
    fill: Color,
    shadow: Dp = 0.dp,
    shadowColor: Color = MusicDecodedColors.Border,
    border: Color? = MusicDecodedColors.Border,
    shape: Shape = RectangleShape,
    onClick: (() -> Unit)? = null,
    onClickLabel: String? = null,
    role: Role = Role.Button,
    enabled: Boolean = true,
): Modifier {
    val interaction = remember { MutableInteractionSource() }
    val pressed by interaction.collectIsPressedAsState()
    val push = if (pressed && onClick != null && shadow > 0.dp) Shadows.press else 0.dp
    val base = graphicsLayer {
        translationX = push.toPx()
        translationY = push.toPx()
    }
        .hardShadow(shadow - push, shadowColor, shape)
        .background(fill, shape)
        .let { if (border != null) it.border(Borders.thick, border, shape) else it }
    return this.then(
        if (onClick == null) base
        else base.clickable(interaction, indication = null, enabled = enabled, onClickLabel = onClickLabel, role = role, onClick = onClick),
    )
}

/**
 * Sticker (portfolio `Sticker.tsx`): accent fill, 3pt black border, `shadowMd`, uppercase heavy
 * label, tilted -3°. For eyebrows, status badges and "PRO".
 */
@Composable
fun Sticker(
    text: String,
    modifier: Modifier = Modifier,
    fill: Color = MusicDecodedColors.Lime,
    contentColor: Color = MusicDecodedColors.OnColor,
    rotate: Float = Tilt.sticker,
) {
    Text(
        text.uppercase(),
        style = TextStyles.sticker,
        color = contentColor,
        modifier = modifier
            .tilt(rotate)
            .block(fill, shadow = Shadows.md)
            .padding(horizontal = Spacing.sm, vertical = Spacing.xxs),
    )
}

/** Eyebrow label: uppercase, 11sp bold, +0.25em. `lime` on ink (default), `white` on violet. */
@Composable
fun Eyebrow(text: String, modifier: Modifier = Modifier, color: Color = MusicDecodedColors.Lime) {
    Text(text.uppercase(), style = TextStyles.overline, color = color, modifier = modifier)
}

/** Oversized decorative number at 8% opacity ("01", "#1"). Hidden from accessibility. */
@Composable
fun BigNumeral(
    text: String,
    modifier: Modifier = Modifier,
    color: Color = MusicDecodedColors.White,
    style: TextStyle = Typography.displayLarge.copy(fontSize = 120.sp),
) {
    Text(
        text,
        style = style,
        color = color.copy(alpha = 0.08f),
        textAlign = TextAlign.End,
        maxLines = 1,
        softWrap = false,
        modifier = modifier.clearAndSetSemantics {},
    )
}

/** Flat equalizer bars (portfolio `Waveform.tsx`) for idle/now-playing indicators. Decorative. */
@Composable
fun WaveformBars(modifier: Modifier = Modifier, bars: Int = 24, color: Color = MusicDecodedColors.Lime) {
    Canvas(modifier.clearAndSetSemantics {}) {
        val gap = 4.dp.toPx()
        val w = (size.width - gap * (bars - 1)) / bars
        repeat(bars) { i ->
            val h = size.height * (0.25f + 0.70f * abs(sin(i * 1.7f + 1f)))
            drawRect(color, topLeft = Offset(i * (w + gap), size.height - h), size = Size(w, h))
        }
    }
}

/** Halftone dot grid texture: 2pt dots on a 12pt pitch at 20% opacity, drawn behind the content. */
fun Modifier.halftone(color: Color, alpha: Float = 0.20f): Modifier = clipToBounds().drawBehind {
    val pitch = 12.dp.toPx()
    val r = 1.dp.toPx()
    val c = color.copy(alpha = alpha)
    var y = pitch / 2
    while (y < size.height) {
        var x = pitch / 2
        while (x < size.width) {
            drawCircle(c, r, Offset(x, y))
            x += pitch
        }
        y += pitch
    }
}

/** Diagonal stripe texture (violet at 50% on lime blocks), drawn behind the content. */
fun Modifier.stripes(color: Color = MusicDecodedColors.Violet, alpha: Float = 0.5f): Modifier = clipToBounds().drawBehind {
    val band = 12.dp.toPx()
    val pitch = 28.dp.toPx()
    val c = color.copy(alpha = alpha)
    var x = -size.height
    while (x < size.width) {
        drawLine(c, Offset(x, size.height), Offset(x + size.height, 0f), strokeWidth = band)
        x += pitch
    }
}

@Preview
@Composable
private fun WrappedPreview() = PreviewSurface {
    Column(verticalArrangement = Arrangement.spacedBy(Spacing.lg)) {
        Row(horizontalArrangement = Arrangement.spacedBy(Spacing.md)) {
            Sticker("Your library")
            Sticker("Pro", fill = MusicDecodedColors.Yellow)
            Sticker("Now playing", fill = MusicDecodedColors.Pink, rotate = Tilt.stickerMax)
        }
        Eyebrow("Now playing")
        WaveformBars(Modifier.fillMaxWidth().height(32.dp))
        Column(Modifier.width(200.dp).height(80.dp).block(MusicDecodedColors.Lime, Shadows.lg).stripes()) {}
    }
}
