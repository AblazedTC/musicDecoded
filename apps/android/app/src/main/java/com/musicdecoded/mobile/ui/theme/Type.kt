package com.musicdecoded.mobile.ui.theme

import androidx.compose.material3.Typography
import androidx.compose.ui.text.TextStyle
import androidx.compose.ui.text.font.Font
import androidx.compose.ui.text.font.FontFamily
import androidx.compose.ui.text.font.FontWeight
import androidx.compose.ui.unit.em
import androidx.compose.ui.unit.sp
import com.musicdecoded.mobile.R

/** Display face (DESIGN.md §1): Archivo Black, SIL OFL (assets/licenses). */
val ArchivoBlack = FontFamily(Font(R.font.archivo_black, FontWeight.Normal))

/** Body face: Space Grotesk 400/500/700, SIL OFL (assets/licenses). */
val SpaceGrotesk = FontFamily(
    Font(R.font.space_grotesk_regular, FontWeight.Normal),
    Font(R.font.space_grotesk_medium, FontWeight.Medium),
    Font(R.font.space_grotesk_bold, FontWeight.Bold),
)

private val Base = Typography()

private fun body(style: TextStyle, size: Int, line: Int, weight: FontWeight) =
    style.copy(fontFamily = SpaceGrotesk, fontSize = size.sp, lineHeight = line.sp, fontWeight = weight, letterSpacing = 0.sp)

private fun display(style: TextStyle, size: Int) =
    style.copy(fontFamily = ArchivoBlack, fontSize = size.sp, lineHeight = 0.95.em, fontWeight = FontWeight.Normal, letterSpacing = (-0.01).em)

/**
 * DESIGN.md §1 type roles on Material 3 styles. Everything is sp so font scaling works; nothing is
 * below 11sp. Display roles are uppercase: uppercase the text at the call site.
 * page title → headlineMedium (44) · section title → titleMedium (28) · current chord → displayMedium (56) ·
 * row title → titleSmall (17 bold) · body → bodyMedium/bodyLarge (15) · secondary → bodySmall (13 medium) ·
 * eyebrow/caption → labelSmall (11 bold) · buttons/chips → labelLarge (14 bold).
 */
val Typography = Typography(
    displayLarge = display(Base.displayLarge, 56),
    displayMedium = display(Base.displayMedium, 56),
    displaySmall = display(Base.displaySmall, 36),
    headlineLarge = display(Base.headlineLarge, 44),
    headlineMedium = display(Base.headlineMedium, 44),
    headlineSmall = display(Base.headlineSmall, 28),
    titleLarge = display(Base.titleLarge, 28),
    titleMedium = display(Base.titleMedium, 28),
    titleSmall = body(Base.titleSmall, 17, 22, FontWeight.Bold),
    bodyLarge = body(Base.bodyLarge, 15, 21, FontWeight.Normal),
    bodyMedium = body(Base.bodyMedium, 15, 21, FontWeight.Normal),
    bodySmall = body(Base.bodySmall, 13, 18, FontWeight.Medium),
    labelLarge = body(Base.labelLarge, 14, 18, FontWeight.Bold),
    labelMedium = body(Base.labelMedium, 13, 18, FontWeight.Medium),
    labelSmall = body(Base.labelSmall, 11, 14, FontWeight.Bold),
)

object TextStyles {
    /** Current chord in the player: Archivo Black 56sp, lime on ink. */
    val currentChord = Typography.displayMedium.copy(color = MusicDecodedColors.Chord)

    /** Eyebrow / caption: uppercase, 11sp bold, +0.25em tracking. Uppercase the text at the call site. */
    val overline = Typography.labelSmall.copy(letterSpacing = 0.25.em, color = MusicDecodedColors.TextMuted)

    /** Eyebrow in the accent color: lime on ink (use `White` on violet). */
    val eyebrow = overline.copy(color = MusicDecodedColors.Lime)

    /**
     * Caption inside narrow chips (Key · Transpose…): the overline at +0.1em, because +0.25em
     * breaks "TRANSPOSE" mid-word in a third-width chip.
     */
    val chipLabel = overline.copy(letterSpacing = 0.1.em)

    /** Sticker label: uppercase heavy body font, +0.04em tracking. */
    val sticker = Typography.labelLarge.copy(letterSpacing = 0.04.em)

    /** Times, counters, BPM: Space Grotesk Medium 13 with tabular figures. */
    val counter = Typography.labelMedium.copy(fontFeatureSettings = "tnum")
}

/** Times and counters use tabular numbers. */
fun TextStyle.tabular(): TextStyle = copy(fontFeatureSettings = "tnum")
