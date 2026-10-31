package com.musicdecoded.mobile.ui.theme

import android.provider.Settings
import androidx.compose.material3.MaterialTheme
import androidx.compose.material3.Shapes as MaterialShapes
import androidx.compose.material3.darkColorScheme
import androidx.compose.runtime.Composable
import androidx.compose.runtime.CompositionLocalProvider
import androidx.compose.runtime.remember
import androidx.compose.runtime.staticCompositionLocalOf
import androidx.compose.foundation.shape.RoundedCornerShape
import androidx.compose.ui.unit.dp
import androidx.compose.ui.platform.LocalContext

private val C = MusicDecodedColors

private val ColorScheme = darkColorScheme(
    primary = C.Primary,
    onPrimary = C.OnPrimary,
    primaryContainer = C.Primary,
    onPrimaryContainer = C.OnPrimary,
    secondary = C.Secondary,
    onSecondary = C.OnSecondary,
    tertiary = C.Yellow,
    onTertiary = C.OnColor,
    background = C.Canvas,
    onBackground = C.Text,
    surface = C.Canvas,
    onSurface = C.Text,
    surfaceVariant = C.Surface,
    onSurfaceVariant = C.TextSecondary,
    surfaceContainerLowest = C.Ink,
    surfaceContainerLow = C.Surface,
    surfaceContainer = C.Surface,
    surfaceContainerHigh = C.SurfaceRaised,
    surfaceContainerHighest = C.SurfaceRaised,
    outline = C.Line,
    outlineVariant = C.Line,
    error = C.Record,
    onError = C.OnColor,
)

/** Square corners everywhere (DESIGN.md §1), including Material sheets, menus and dialogs. */
private val Square = RoundedCornerShape(0.dp)

private val AppShapes = MaterialShapes(
    extraSmall = Square,
    small = Square,
    medium = Square,
    large = Square,
    extraLarge = Square,
)

/**
 * True when the system "Remove animations" setting is on (animator duration scale 0). Tilt and
 * other decorative motion is turned off then (DESIGN.md §1).
 */
val LocalReduceMotion = staticCompositionLocalOf { false }

/** Dark only in v1 (DESIGN.md §1); no dynamic color. */
@Composable
fun MusicDecodedTheme(content: @Composable () -> Unit) {
    val resolver = LocalContext.current.contentResolver
    val reduceMotion = remember(resolver) {
        Settings.Global.getFloat(resolver, Settings.Global.ANIMATOR_DURATION_SCALE, 1f) == 0f
    }
    CompositionLocalProvider(LocalReduceMotion provides reduceMotion) {
        MaterialTheme(colorScheme = ColorScheme, typography = Typography, shapes = AppShapes, content = content)
    }
}
