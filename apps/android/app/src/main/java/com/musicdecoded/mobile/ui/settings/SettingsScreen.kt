package com.musicdecoded.mobile.ui.settings

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

/** Placeholder until the Settings task builds DESIGN.md §3.9. */
@Composable
fun SettingsScreen() {
    Column(
        Modifier.fillMaxSize().padding(horizontal = Spacing.gutter, vertical = Spacing.lg),
        verticalArrangement = Arrangement.spacedBy(Spacing.lg),
    ) {
        PageHeader("Settings", overline = "Preferences & account")
        Text(
            "Guest account · Music is stored on this device",
            style = MaterialTheme.typography.bodySmall,
            color = MusicDecodedColors.TextSecondary,
        )
    }
}
