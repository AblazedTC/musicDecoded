package com.musicdecoded.mobile.ui.theme

import androidx.compose.material3.MaterialTheme
import androidx.compose.material3.Shapes as MaterialShapes
import androidx.compose.material3.darkColorScheme
import androidx.compose.runtime.Composable

private val ColorScheme = darkColorScheme(
    primary = MusicDecodedColors.Primary,
    onPrimary = MusicDecodedColors.OnPrimary,
    primaryContainer = MusicDecodedColors.PrimarySoft,
    onPrimaryContainer = MusicDecodedColors.Primary,
    secondary = MusicDecodedColors.Secondary,
    onSecondary = MusicDecodedColors.Ink,
    tertiary = MusicDecodedColors.Chord,
    onTertiary = MusicDecodedColors.OnChord,
    background = MusicDecodedColors.Canvas,
    onBackground = MusicDecodedColors.Text,
    surface = MusicDecodedColors.Canvas,
    onSurface = MusicDecodedColors.Text,
    surfaceVariant = MusicDecodedColors.Surface,
    onSurfaceVariant = MusicDecodedColors.TextSecondary,
    surfaceContainerLowest = MusicDecodedColors.Ink,
    surfaceContainerLow = MusicDecodedColors.Surface,
    surfaceContainer = MusicDecodedColors.Surface,
    surfaceContainerHigh = MusicDecodedColors.SurfaceRaised,
    surfaceContainerHighest = MusicDecodedColors.SurfaceSoft,
    outline = MusicDecodedColors.Line,
    outlineVariant = MusicDecodedColors.Line,
    error = MusicDecodedColors.Record,
    onError = MusicDecodedColors.Ink,
)

private val AppShapes = MaterialShapes(
    small = Shapes.chip,
    medium = Shapes.control,
    large = Shapes.card,
)

/** Dark only in v1 (DESIGN.md §1); no dynamic color. */
@Composable
fun MusicDecodedTheme(content: @Composable () -> Unit) {
    MaterialTheme(colorScheme = ColorScheme, typography = Typography, shapes = AppShapes, content = content)
}
