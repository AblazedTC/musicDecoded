package com.musicdecoded.mobile.ui.settings

import androidx.compose.foundation.background
import androidx.compose.foundation.border
import androidx.compose.foundation.clickable
import androidx.compose.foundation.layout.Arrangement
import androidx.compose.foundation.layout.Box
import androidx.compose.foundation.layout.Column
import androidx.compose.foundation.layout.Row
import androidx.compose.foundation.layout.defaultMinSize
import androidx.compose.foundation.layout.fillMaxSize
import androidx.compose.foundation.layout.fillMaxWidth
import androidx.compose.foundation.layout.padding
import androidx.compose.foundation.layout.size
import androidx.compose.foundation.rememberScrollState
import androidx.compose.foundation.selection.selectable
import androidx.compose.foundation.selection.selectableGroup
import androidx.compose.foundation.shape.CircleShape
import androidx.compose.foundation.verticalScroll
import androidx.compose.material.icons.Icons
import androidx.compose.material.icons.automirrored.rounded.ArrowForward
import androidx.compose.material.icons.automirrored.rounded.HelpOutline
import androidx.compose.material.icons.automirrored.rounded.KeyboardArrowRight
import androidx.compose.material.icons.rounded.CheckCircle
import androidx.compose.material.icons.rounded.Cloud
import androidx.compose.material.icons.rounded.Person
import androidx.compose.material.icons.rounded.PrivacyTip
import androidx.compose.material3.AlertDialog
import androidx.compose.material3.Icon
import androidx.compose.material3.MaterialTheme
import androidx.compose.material3.RadioButton
import androidx.compose.material3.RadioButtonDefaults
import androidx.compose.material3.Text
import androidx.compose.material3.TextButton
import androidx.compose.runtime.Composable
import androidx.compose.runtime.getValue
import androidx.compose.runtime.mutableStateOf
import androidx.compose.runtime.saveable.rememberSaveable
import androidx.compose.runtime.setValue
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.platform.LocalContext
import androidx.compose.ui.graphics.vector.ImageVector
import androidx.compose.ui.semantics.Role
import androidx.lifecycle.compose.collectAsStateWithLifecycle
import androidx.lifecycle.viewmodel.compose.viewModel
import com.musicdecoded.mobile.MusicDecodedApplication
import com.musicdecoded.mobile.core.Accidentals
import com.musicdecoded.mobile.player.ChordsMode
import com.musicdecoded.mobile.player.PlayerView
import com.musicdecoded.mobile.player.formatSpeed
import com.musicdecoded.mobile.player.label
import com.musicdecoded.mobile.settings.PlayerPreferences
import com.musicdecoded.mobile.ui.auth.AuthMode
import com.musicdecoded.mobile.ui.components.PageHeader
import com.musicdecoded.mobile.ui.components.PrimaryButton
import com.musicdecoded.mobile.ui.components.SecondaryButton
import com.musicdecoded.mobile.ui.components.formatBytes
import com.musicdecoded.mobile.ui.theme.MusicDecodedColors
import com.musicdecoded.mobile.ui.theme.Shapes
import com.musicdecoded.mobile.ui.theme.Sizes
import com.musicdecoded.mobile.ui.theme.Spacing
import com.musicdecoded.mobile.ui.components.Eyebrow
import com.musicdecoded.mobile.ui.components.Sticker
import com.musicdecoded.mobile.ui.components.block
import com.musicdecoded.mobile.ui.theme.Borders
import com.musicdecoded.mobile.ui.theme.Shadows

private enum class SettingsDialog { SPEED, CHORDS, ACCIDENTALS, VIEW, PRIVACY, HELP }

/** DESIGN.md §3.9 (v1 rows only): profile card, guest sync card, preferences, storage, about. */
@Composable
fun SettingsScreen(onAuth: (AuthMode) -> Unit, onGoPro: (() -> Unit)? = null) {
    val app = LocalContext.current.applicationContext as MusicDecodedApplication
    val viewModel: SettingsViewModel = viewModel(factory = SettingsViewModel.factory(app))
    val state by viewModel.state.collectAsStateWithLifecycle()
    var dialog by rememberSaveable { mutableStateOf<SettingsDialog?>(null) }
    val prefs = state.preferences

    Column(
        Modifier
            .fillMaxSize()
            .verticalScroll(rememberScrollState())
            .padding(horizontal = Spacing.gutter, vertical = Spacing.lg),
        verticalArrangement = Arrangement.spacedBy(Spacing.lg),
    ) {
        PageHeader("Settings", overline = "Preferences & account")
        ProfileCard(state, onAuth = onAuth, onLogout = viewModel::logout)
        if (!state.isAccount) SyncCard(onClick = { onAuth(AuthMode.SIGN_UP) })
        if (state.showsGoPro && onGoPro != null) {
            SettingsGroup("MusicDecoded Pro") { SettingsRow("Go Pro", "Decode more", onClick = onGoPro) }
        }

        SettingsGroup("Playback") {
            SettingsRow("Default speed", formatSpeed(prefs.defaultSpeed)) { dialog = SettingsDialog.SPEED }
        }
        SettingsGroup("Music analysis") {
            SettingsRow("Chord notation", prefs.chordsMode.label) { dialog = SettingsDialog.CHORDS }
            SettingsRow("Accidentals", prefs.accidentals.label) { dialog = SettingsDialog.ACCIDENTALS }
            SettingsRow("Default view", prefs.playerView.label) { dialog = SettingsDialog.VIEW }
        }
        SettingsGroup("Storage") {
            SettingsRow("Local storage", formatBytes(state.storageBytes))
            SettingsRow("Offline files", state.offlineFiles.toString())
        }
        SettingsGroup("Application") {
            SettingsRow("Privacy & data") { dialog = SettingsDialog.PRIVACY }
            SettingsRow("Help & support") { dialog = SettingsDialog.HELP }
        }
    }

    val close = { dialog = null }
    when (dialog) {
        SettingsDialog.SPEED -> ChoiceDialog("Default speed", PlayerPreferences.SPEEDS, prefs.defaultSpeed, ::formatSpeed, close) { v ->
            viewModel.update { it.copy(defaultSpeed = v) }
        }
        SettingsDialog.CHORDS -> ChoiceDialog("Chord notation", ChordsMode.entries, prefs.chordsMode, { it.label }, close) { v ->
            viewModel.update { it.copy(chordsMode = v) }
        }
        SettingsDialog.ACCIDENTALS -> ChoiceDialog("Accidentals", Accidentals.entries, prefs.accidentals, { it.label }, close) { v ->
            viewModel.update { it.copy(accidentals = v) }
        }
        SettingsDialog.VIEW -> ChoiceDialog("Default view", PlayerView.entries, prefs.playerView, { it.label }, close) { v ->
            viewModel.update { it.copy(playerView = v) }
        }
        SettingsDialog.PRIVACY -> InfoDialog(
            "Privacy & data",
            "Your library lives on this device. Files and recordings you decode are uploaded to our servers to be analyzed. " +
                "A copy stays on this device for offline playback. With an account, your library, chord timelines and your own " +
                "audio sync across your devices. YouTube and SoundCloud audio is never uploaded from this device.",
            Icons.Rounded.PrivacyTip,
            close,
        )
        SettingsDialog.HELP -> InfoDialog(
            "Help & support",
            "Decode a song from Home: import a file, record, or pick a YouTube video. Open it to see the chord timeline. " +
                "Tap a chord to jump to it, drag to scrub, and tap loop to repeat the current bar. Speed changes keep the pitch.",
            Icons.AutoMirrored.Rounded.HelpOutline,
            close,
        )
        null -> Unit
    }
}

private val ChordsMode.label get() = if (this == ChordsMode.FULL) "Full" else "Basic"
private val Accidentals.label get() = when (this) {
    Accidentals.AUTO -> "Auto"
    Accidentals.SHARP -> "♯"
    Accidentals.FLAT -> "♭"
}

/** The profile card is a yellow block (DESIGN.md §1 screen color map): ink text, black border, violet shadow. */
@Composable
private fun ProfileCard(state: SettingsUiState, onAuth: (AuthMode) -> Unit, onLogout: () -> Unit) {
    Column(
        Modifier
            .fillMaxWidth()
            .block(MusicDecodedColors.Yellow, shadow = Shadows.lg, shadowColor = MusicDecodedColors.Violet)
            .padding(Spacing.md),
        verticalArrangement = Arrangement.spacedBy(Spacing.md),
    ) {
        Row(horizontalArrangement = Arrangement.spacedBy(Spacing.sm), verticalAlignment = Alignment.CenterVertically) {
            Box(
                Modifier.size(Sizes.songArt).background(MusicDecodedColors.White, CircleShape).border(Borders.thick, MusicDecodedColors.Border, CircleShape),
                contentAlignment = Alignment.Center,
            ) {
                Icon(Icons.Rounded.Person, contentDescription = null, tint = MusicDecodedColors.OnColor)
            }
            Column(Modifier.weight(1f)) {
                if (state.isAccount) {
                    val email = state.session?.email
                    Text(email?.substringBefore('@') ?: "Account", style = MaterialTheme.typography.titleSmall, color = MusicDecodedColors.OnColor)
                    Text(
                        listOfNotNull(email, "Synced").joinToString(" · "),
                        style = MaterialTheme.typography.bodySmall,
                        color = MusicDecodedColors.OnColor,
                    )
                } else {
                    Text("Guest account", style = MaterialTheme.typography.titleSmall, color = MusicDecodedColors.OnColor)
                    Text("Music is stored on this device", style = MaterialTheme.typography.bodySmall, color = MusicDecodedColors.OnColor)
                }
            }
            if (state.isPro) ProChip()
            if (state.isAccount) {
                Icon(Icons.Rounded.CheckCircle, contentDescription = "Synced", tint = MusicDecodedColors.OnColor)
            }
        }
        if (state.isAccount) {
            SecondaryButton("Log out", onClick = onLogout)
        } else {
            PrimaryButton("Create account", onClick = { onAuth(AuthMode.SIGN_UP) }, shadowColor = MusicDecodedColors.Border)
            TextButton(onClick = { onAuth(AuthMode.LOG_IN) }, modifier = Modifier.fillMaxWidth(), shape = Shapes.button) {
                Text("I already have an account", style = MaterialTheme.typography.labelLarge, color = MusicDecodedColors.OnColor)
            }
        }
    }
}

/** "PRO" sticker; pink because the profile card it sits on is already yellow. */
@Composable
private fun ProChip() {
    Sticker("Pro", fill = MusicDecodedColors.Pink)
}

@Composable
private fun SyncCard(onClick: () -> Unit) {
    Row(
        Modifier
            .fillMaxWidth()
            .block(MusicDecodedColors.Surface, shadow = Shadows.md, shadowColor = MusicDecodedColors.Pink, border = MusicDecodedColors.BorderOnInk, onClick = onClick)
            .padding(Spacing.md),
        horizontalArrangement = Arrangement.spacedBy(Spacing.sm),
        verticalAlignment = Alignment.CenterVertically,
    ) {
        Icon(Icons.Rounded.Cloud, contentDescription = null, tint = MusicDecodedColors.Pink)
        Column(Modifier.weight(1f)) {
            Text("Sync your music", style = MaterialTheme.typography.titleSmall, color = MusicDecodedColors.Text)
            Text(
                "Keep eligible chord timelines and files available across your devices.",
                style = MaterialTheme.typography.bodySmall,
                color = MusicDecodedColors.TextSecondary,
            )
        }
        Icon(Icons.AutoMirrored.Rounded.ArrowForward, contentDescription = null, tint = MusicDecodedColors.Primary)
    }
}

@Composable
private fun SettingsGroup(title: String, content: @Composable () -> Unit) {
    Column(verticalArrangement = Arrangement.spacedBy(Spacing.xs)) {
        Eyebrow(title)
        Column(
            Modifier
                .fillMaxWidth()
                .background(MusicDecodedColors.Surface)
                .border(Borders.thick, MusicDecodedColors.BorderOnInk),
        ) { content() }
    }
}

/** Label · value · chevron. Without [onClick] the row is read-only (no chevron). */
@Composable
private fun SettingsRow(label: String, value: String? = null, onClick: (() -> Unit)? = null) {
    Row(
        Modifier
            .fillMaxWidth()
            .defaultMinSize(minHeight = Sizes.minTouchTarget)
            .then(if (onClick != null) Modifier.clickable(role = Role.Button, onClick = onClick) else Modifier)
            .padding(horizontal = Spacing.md, vertical = Spacing.sm),
        horizontalArrangement = Arrangement.spacedBy(Spacing.xs),
        verticalAlignment = Alignment.CenterVertically,
    ) {
        Text(label, style = MaterialTheme.typography.titleSmall, color = MusicDecodedColors.Text, modifier = Modifier.weight(1f))
        if (value != null) Text(value, style = MaterialTheme.typography.bodySmall, color = MusicDecodedColors.TextSecondary)
        if (onClick != null) {
            Icon(Icons.AutoMirrored.Rounded.KeyboardArrowRight, contentDescription = null, tint = MusicDecodedColors.Primary)
        }
    }
}

@Composable
private fun <T> ChoiceDialog(
    title: String,
    options: List<T>,
    selected: T,
    label: (T) -> String,
    onDismiss: () -> Unit,
    onSelect: (T) -> Unit,
) {
    AlertDialog(
        onDismissRequest = onDismiss,
        containerColor = MusicDecodedColors.Surface,
        title = { Text(title, color = MusicDecodedColors.Text) },
        text = {
            Column(Modifier.selectableGroup()) {
                options.forEach { option ->
                    Row(
                        Modifier
                            .fillMaxWidth()
                            .defaultMinSize(minHeight = Sizes.minTouchTarget)
                            .selectable(selected = option == selected, role = Role.RadioButton) {
                                onSelect(option)
                                onDismiss()
                            },
                        verticalAlignment = Alignment.CenterVertically,
                        horizontalArrangement = Arrangement.spacedBy(Spacing.xs),
                    ) {
                        RadioButton(
                            selected = option == selected,
                            onClick = null,
                            colors = RadioButtonDefaults.colors(selectedColor = MusicDecodedColors.Primary),
                        )
                        Text(label(option), style = MaterialTheme.typography.bodyLarge, color = MusicDecodedColors.Text)
                    }
                }
            }
        },
        confirmButton = {
            TextButton(onClick = onDismiss) { Text("Cancel", color = MusicDecodedColors.Primary) }
        },
    )
}

@Composable
private fun InfoDialog(title: String, text: String, icon: ImageVector, onDismiss: () -> Unit) {
    AlertDialog(
        onDismissRequest = onDismiss,
        containerColor = MusicDecodedColors.Surface,
        icon = {
            Icon(icon, contentDescription = null, tint = MusicDecodedColors.Secondary)
        },
        title = { Text(title, color = MusicDecodedColors.Text) },
        text = { Text(text, style = MaterialTheme.typography.bodyMedium, color = MusicDecodedColors.TextSecondary) },
        confirmButton = {
            TextButton(onClick = onDismiss) { Text("OK", color = MusicDecodedColors.Primary) }
        },
    )
}
