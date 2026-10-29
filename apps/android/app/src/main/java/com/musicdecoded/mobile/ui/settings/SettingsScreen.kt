package com.musicdecoded.mobile.ui.settings

import androidx.compose.foundation.layout.Arrangement
import androidx.compose.foundation.layout.Column
import androidx.compose.foundation.layout.fillMaxSize
import androidx.compose.foundation.layout.padding
import androidx.compose.runtime.Composable
import androidx.compose.ui.Modifier
import com.musicdecoded.mobile.auth.AuthViewModel
import com.musicdecoded.mobile.ui.components.PageHeader
import com.musicdecoded.mobile.ui.components.SecondaryButton
import com.musicdecoded.mobile.ui.theme.Spacing

/** Placeholder until the Settings task builds DESIGN.md §3.9. */
@Composable
fun SettingsScreen(viewModel: AuthViewModel) {
    Column(
        Modifier.fillMaxSize().padding(horizontal = Spacing.gutter, vertical = Spacing.lg),
        verticalArrangement = Arrangement.spacedBy(Spacing.lg),
    ) {
        PageHeader("Settings", overline = "Preferences & account")
        SecondaryButton("Log out", onClick = { viewModel.logout() })
    }
}
