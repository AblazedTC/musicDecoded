package com.musicdecoded.mobile.ui.auth

import androidx.compose.foundation.background
import androidx.compose.foundation.border
import androidx.compose.foundation.layout.Arrangement
import androidx.compose.foundation.layout.Box
import androidx.compose.foundation.layout.Column
import androidx.compose.foundation.layout.Row
import androidx.compose.foundation.layout.fillMaxSize
import androidx.compose.foundation.layout.fillMaxWidth
import androidx.compose.foundation.layout.padding
import androidx.compose.foundation.layout.size
import androidx.compose.foundation.rememberScrollState
import androidx.compose.foundation.verticalScroll
import androidx.compose.material.icons.Icons
import androidx.compose.material.icons.rounded.CheckBox
import androidx.compose.material.icons.rounded.CheckBoxOutlineBlank
import androidx.compose.material.icons.rounded.Close
import androidx.compose.material.icons.rounded.CloudUpload
import androidx.compose.material3.Icon
import androidx.compose.material3.MaterialTheme
import androidx.compose.material3.Text
import androidx.compose.runtime.Composable
import androidx.compose.runtime.getValue
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.semantics.contentDescription
import androidx.compose.ui.semantics.semantics
import androidx.lifecycle.ViewModel
import androidx.lifecycle.ViewModelProvider
import androidx.lifecycle.viewModelScope
import androidx.lifecycle.viewmodel.initializer
import androidx.lifecycle.viewmodel.viewModelFactory
import androidx.lifecycle.compose.collectAsStateWithLifecycle
import com.musicdecoded.mobile.MusicDecodedApplication
import com.musicdecoded.mobile.core.Source
import com.musicdecoded.mobile.library.LibraryItem
import com.musicdecoded.mobile.ui.components.MdIconButton
import com.musicdecoded.mobile.ui.components.PolicyNote
import com.musicdecoded.mobile.ui.components.PrimaryButton
import com.musicdecoded.mobile.ui.components.formatBytes
import com.musicdecoded.mobile.ui.components.block
import com.musicdecoded.mobile.ui.components.tilt
import com.musicdecoded.mobile.ui.theme.Borders
import com.musicdecoded.mobile.ui.theme.MusicDecodedColors
import com.musicdecoded.mobile.ui.theme.Shadows
import com.musicdecoded.mobile.ui.theme.Tilt
import com.musicdecoded.mobile.ui.theme.Shapes
import com.musicdecoded.mobile.ui.theme.Sizes
import com.musicdecoded.mobile.ui.theme.Spacing
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.flow.MutableStateFlow
import kotlinx.coroutines.flow.StateFlow
import kotlinx.coroutines.flow.asStateFlow
import kotlinx.coroutines.launch
import kotlinx.coroutines.withContext
import java.io.File

/** What the migration screen lists (DESIGN.md §3.10). */
data class MigrationSummary(
    /** Chord timelines (analysis, tempo and key data): every guest row. */
    val timelines: Int,
    /** file/recording rows: their audio is eligible for private sync. */
    val ownAudio: Int,
    val ownAudioBytes: Long,
    /** youtube/soundcloud rows: the timeline syncs, the audio never does. */
    val restricted: Int,
) {
    companion object {
        fun of(items: List<LibraryItem>, filesDir: File): MigrationSummary {
            val own = items.filter { it.source == Source.FILE || it.source == Source.RECORDING }
            return MigrationSummary(
                timelines = items.size,
                ownAudio = own.size,
                ownAudioBytes = own.sumOf { item -> item.localAudioPath?.let { File(filesDir, it).length() } ?: 0L },
                restricted = items.count { it.source == Source.YOUTUBE || it.source == Source.SOUNDCLOUD },
            )
        }
    }
}

class MigrationViewModel(
    private val load: suspend () -> MigrationSummary,
    private val requestSync: () -> Unit,
) : ViewModel() {
    private val _summary = MutableStateFlow<MigrationSummary?>(null)
    val summary: StateFlow<MigrationSummary?> = _summary.asStateFlow()

    init {
        viewModelScope.launch { _summary.value = load() }
    }

    /** The server already linked the guest data at login; this runs the first sync. */
    fun confirm() = requestSync()

    companion object {
        fun factory(app: MusicDecodedApplication): ViewModelProvider.Factory = viewModelFactory {
            initializer {
                MigrationViewModel(
                    load = {
                        withContext(Dispatchers.IO) {
                            MigrationSummary.of(app.libraryDatabase.libraryDao().unsynced(), app.filesDir)
                        }
                    },
                    requestSync = app::requestSync,
                )
            }
        }
    }
}

/** DESIGN.md §3.10 "Bring your music with you" → Confirm & sync. Close also keeps everything local. */
@Composable
fun MigrationScreen(viewModel: MigrationViewModel, onDone: () -> Unit) {
    val summary by viewModel.summary.collectAsStateWithLifecycle()
    Column(
        Modifier
            .fillMaxSize()
            .background(MusicDecodedColors.Canvas)
            .verticalScroll(rememberScrollState()),
    ) {
        Row(Modifier.padding(horizontal = Spacing.gutter, vertical = Spacing.xs)) {
            MdIconButton(Icons.Rounded.Close, contentDescription = "Not now", onClick = onDone)
        }
        Column(
            Modifier.padding(horizontal = Spacing.gutter, vertical = Spacing.lg),
            verticalArrangement = Arrangement.spacedBy(Spacing.md),
            horizontalAlignment = Alignment.CenterHorizontally,
        ) {
            Box(
                Modifier.size(Sizes.recordButton).tilt(Tilt.sticker).block(MusicDecodedColors.Blue, shadow = Shadows.md, shadowColor = MusicDecodedColors.Violet),
                contentAlignment = Alignment.Center,
            ) {
                Icon(Icons.Rounded.CloudUpload, contentDescription = null, tint = MusicDecodedColors.OnColor)
            }
            Text("BRING YOUR MUSIC WITH YOU", style = MaterialTheme.typography.headlineMedium, color = MusicDecodedColors.Text)
            Text(
                "Move local music to your account.",
                style = MaterialTheme.typography.bodySmall,
                color = MusicDecodedColors.TextSecondary,
            )
            val s = summary
            if (s != null) {
                Column(Modifier.fillMaxWidth(), verticalArrangement = Arrangement.spacedBy(Spacing.xs)) {
                    CheckRow(true, plural(s.timelines, "chord timeline"), "Analysis, tempo and key data")
                    CheckRow(true, plural(s.ownAudio, "local audio file"), "Eligible for private sync · ${formatBytes(s.ownAudioBytes)}")
                    CheckRow(false, plural(s.restricted, "restricted track"), "Timelines sync; source audio stays local")
                }
            }
            PolicyNote("Nothing is removed from this device. Restricted third-party audio never uploads.")
            PrimaryButton("Confirm & sync", onClick = {
                viewModel.confirm()
                onDone()
            }, enabled = s != null)
        }
    }
}

@Composable
private fun CheckRow(checked: Boolean, title: String, subtitle: String) {
    val description = "$title, $subtitle, ${if (checked) "included" else "not included"}"
    Row(
        Modifier
            .fillMaxWidth()
            .background(MusicDecodedColors.Surface)
            .border(Borders.thick, MusicDecodedColors.BorderOnInk)
            .padding(Spacing.md)
            .semantics(mergeDescendants = true) { contentDescription = description },
        horizontalArrangement = Arrangement.spacedBy(Spacing.sm),
        verticalAlignment = Alignment.CenterVertically,
    ) {
        Icon(
            if (checked) Icons.Rounded.CheckBox else Icons.Rounded.CheckBoxOutlineBlank,
            contentDescription = null,
            tint = if (checked) MusicDecodedColors.Primary else MusicDecodedColors.TextMuted,
        )
        Column(Modifier.weight(1f)) {
            Text(
                title,
                style = MaterialTheme.typography.titleSmall,
                color = if (checked) MusicDecodedColors.Text else MusicDecodedColors.TextMuted,
            )
            Text(subtitle, style = MaterialTheme.typography.bodySmall, color = MusicDecodedColors.TextSecondary)
        }
    }
}

private fun plural(count: Int, noun: String) = "$count $noun${if (count == 1) "" else "s"}"
