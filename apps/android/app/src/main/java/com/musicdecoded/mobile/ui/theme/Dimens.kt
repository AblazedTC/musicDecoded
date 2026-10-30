package com.musicdecoded.mobile.ui.theme

import androidx.compose.ui.graphics.RectangleShape
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

/**
 * DESIGN.md §1: corners are square everywhere. Only circular play/record buttons and avatars are
 * round (they use `CircleShape` directly). The names stay so call sites read by role.
 */
object Shapes {
    val card = RectangleShape
    val row = RectangleShape
    val control = RectangleShape
    val button = RectangleShape
    val chip = RectangleShape
    val art = RectangleShape
    val pill = RectangleShape
    val sheet = RectangleShape
}

/** DESIGN.md §1 borders: 3pt on blocks/cards/art, 2pt separators inside lists. */
object Borders {
    val thick = 3.dp
    val separator = 2.dp
}

/** DESIGN.md §1 hard offset shadows (no blur, x = y). */
object Shadows {
    val sm = 3.dp
    val md = 5.dp
    val lg = 8.dp
    val xl = 10.dp
    /** Pressed state moves the block by this and shrinks the shadow by the same amount. */
    val press = 2.dp
}

/** DESIGN.md §1 tilt in degrees. Decoration only; zero when the system disables animations. */
object Tilt {
    const val hero = -1.5f
    const val listCard = 1f
    const val sticker = -3f
    const val stickerMax = -6f
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
    /** Paywall plan card (and its loading skeleton). */
    val planCard = 72.dp
}
