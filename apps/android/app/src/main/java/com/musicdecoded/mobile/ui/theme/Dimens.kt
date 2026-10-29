package com.musicdecoded.mobile.ui.theme

import androidx.compose.foundation.shape.RoundedCornerShape
import androidx.compose.ui.unit.dp

/**
 * DESIGN.md §1 spacing: a 4-point scale. Every margin, padding and gap in UI code uses one of
 * these values and nothing else.
 */
object Spacing {
    val xxs = 4.dp
    val xs = 8.dp
    val sm = 12.dp
    val md = 16.dp
    val lg = 20.dp
    val xl = 24.dp
    val xxl = 32.dp
    val xxxl = 40.dp

    /** Screen side gutter. */
    val gutter = md
}

/** DESIGN.md §1 corner radii. */
object Radius {
    val card = 18.dp
    val row = 15.dp
    val control = 12.dp
    val button = 13.dp
    val chip = 10.dp
    val art = 10.dp
    val pill = 20.dp
    val sheet = 22.dp
}

object Shapes {
    val card = RoundedCornerShape(Radius.card)
    val row = RoundedCornerShape(Radius.row)
    val control = RoundedCornerShape(Radius.control)
    val button = RoundedCornerShape(Radius.button)
    val chip = RoundedCornerShape(Radius.chip)
    val art = RoundedCornerShape(Radius.art)
    val pill = RoundedCornerShape(Radius.pill)
    val sheet = RoundedCornerShape(topStart = Radius.sheet, topEnd = Radius.sheet)
}

/** Fixed component sizes (not spacing), from DESIGN.md §1 "component sizes". */
object Sizes {
    val hairline = 1.dp
    val minTouchTarget = 48.dp
    val iconButton = 40.dp
    val smallIconButton = 32.dp
    val playButton = 56.dp
    val recordButton = 84.dp
    val songArt = 56.dp
    val sourceTile = 44.dp
    val buttonHeight = 48.dp
    val searchField = 44.dp
    val icon = 24.dp
    val smallIcon = 18.dp
    val progressTrack = 4.dp
}
