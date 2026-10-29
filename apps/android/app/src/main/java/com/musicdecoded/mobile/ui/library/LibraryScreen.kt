package com.musicdecoded.mobile.ui.library

import androidx.compose.foundation.layout.Arrangement
import androidx.compose.foundation.layout.Column
import androidx.compose.foundation.layout.fillMaxSize
import androidx.compose.foundation.layout.padding
import androidx.compose.material3.MaterialTheme
import androidx.compose.material3.Text
import androidx.compose.runtime.Composable
import androidx.compose.ui.Modifier
import com.musicdecoded.mobile.ui.components.PageHeader
import com.musicdecoded.mobile.ui.theme.MusicDecodedColors
import com.musicdecoded.mobile.ui.theme.Spacing

/** Placeholder until T-210a builds DESIGN.md §3.2. */
@Composable
fun LibraryScreen() {
    Column(
        Modifier.fillMaxSize().padding(horizontal = Spacing.gutter, vertical = Spacing.lg),
        verticalArrangement = Arrangement.spacedBy(Spacing.lg),
    ) {
        PageHeader("Library", overline = "Your collection")
        Text(
            "Your library is empty. Decode a song from Home to get started.",
            style = MaterialTheme.typography.bodySmall,
            color = MusicDecodedColors.TextSecondary,
        )
    }
}
