package com.musicdecoded.mobile.ui.theme

import androidx.compose.material3.Typography
import androidx.compose.ui.text.TextStyle
import androidx.compose.ui.text.font.FontWeight
import androidx.compose.ui.unit.em
import androidx.compose.ui.unit.sp

private val Base = Typography()

/**
 * DESIGN.md §1 type roles mapped onto Material 3 styles (system font, all sizes in sp so font
 * scaling works; nothing below 11sp):
 * page title → headlineMedium · section header → titleMedium · row title → titleSmall ·
 * body/secondary → bodySmall · caption/overline → labelSmall.
 */
val Typography = Typography(
    headlineMedium = Base.headlineMedium.copy(fontWeight = FontWeight.Bold, letterSpacing = (-0.02).em),
    titleMedium = Base.titleMedium.copy(fontSize = 18.sp, lineHeight = 24.sp, fontWeight = FontWeight.Bold),
    titleSmall = Base.titleSmall.copy(fontSize = 14.sp, fontWeight = FontWeight.SemiBold),
    bodySmall = Base.bodySmall.copy(fontSize = 12.sp, lineHeight = 16.sp),
    labelSmall = Base.labelSmall.copy(fontSize = 11.sp, fontWeight = FontWeight.Medium),
)

object TextStyles {
    /** Current chord in the player: 34sp bold, chord color. */
    val currentChord = TextStyle(fontSize = 34.sp, lineHeight = 40.sp, fontWeight = FontWeight.Bold, color = MusicDecodedColors.Chord)

    /** Uppercase overline caption with +0.1em tracking. Uppercase the text at the call site. */
    val overline = Typography.labelSmall.copy(letterSpacing = 0.1.em, color = MusicDecodedColors.TextMuted)
}

/** Times and counters use tabular numbers. */
fun TextStyle.tabular(): TextStyle = copy(fontFeatureSettings = "tnum")
