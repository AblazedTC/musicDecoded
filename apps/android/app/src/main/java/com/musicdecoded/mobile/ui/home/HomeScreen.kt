package com.musicdecoded.mobile.ui.home

import androidx.compose.foundation.layout.Arrangement
import androidx.compose.foundation.layout.Column
import androidx.compose.foundation.layout.fillMaxSize
import androidx.compose.foundation.layout.padding
import androidx.compose.material3.MaterialTheme
import androidx.compose.material3.Text
import androidx.compose.runtime.Composable
import androidx.compose.ui.Modifier
import com.musicdecoded.mobile.ui.components.PageHeader
import com.musicdecoded.mobile.ui.components.SectionHeader
import com.musicdecoded.mobile.ui.theme.MusicDecodedColors
import com.musicdecoded.mobile.ui.theme.Spacing

/** Placeholder until T-203a builds DESIGN.md §3.1. */
@Composable
fun HomeScreen(onSeeAll: () -> Unit) {
    Column(
        Modifier.fillMaxSize().padding(horizontal = Spacing.gutter, vertical = Spacing.lg),
        verticalArrangement = Arrangement.spacedBy(Spacing.lg),
    ) {
        PageHeader("MusicDecoded", overline = "Your music, decoded.")
        SectionHeader("Recently Decoded", actionLabel = "See all", onAction = onSeeAll)
        Text(
            "Songs you decode will show up here.",
            style = MaterialTheme.typography.bodySmall,
            color = MusicDecodedColors.TextSecondary,
        )
    }
}
