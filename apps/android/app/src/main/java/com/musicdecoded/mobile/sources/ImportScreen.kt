package com.musicdecoded.mobile.sources

import android.app.Activity
import android.content.ActivityNotFoundException
import android.content.Context
import android.content.Intent
import android.net.Uri
import android.provider.MediaStore
import androidx.activity.compose.rememberLauncherForActivityResult
import androidx.activity.result.contract.ActivityResultContract
import androidx.activity.result.contract.ActivityResultContracts
import androidx.compose.foundation.background
import androidx.compose.foundation.layout.Arrangement
import androidx.compose.foundation.layout.Box
import androidx.compose.foundation.layout.Column
import androidx.compose.foundation.layout.Row
import androidx.compose.foundation.layout.fillMaxSize
import androidx.compose.foundation.layout.fillMaxWidth
import androidx.compose.foundation.layout.navigationBarsPadding
import androidx.compose.foundation.layout.padding
import androidx.compose.foundation.layout.size
import androidx.compose.material.icons.Icons
import androidx.compose.material.icons.rounded.AudioFile
import androidx.compose.material.icons.rounded.CloudUpload
import androidx.compose.material.icons.rounded.GraphicEq
import androidx.compose.material.icons.rounded.PhoneAndroid
import androidx.compose.material.icons.rounded.Schedule
import androidx.compose.material.icons.rounded.SwapHoriz
import androidx.compose.material3.CircularProgressIndicator
import androidx.compose.material3.ExperimentalMaterial3Api
import androidx.compose.material3.Icon
import androidx.compose.material3.MaterialTheme
import androidx.compose.material3.ModalBottomSheet
import androidx.compose.material3.Text
import androidx.compose.material3.rememberModalBottomSheetState
import androidx.compose.runtime.Composable
import androidx.compose.runtime.LaunchedEffect
import androidx.compose.runtime.getValue
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.platform.LocalContext
import androidx.compose.ui.text.style.TextOverflow
import androidx.compose.ui.tooling.preview.Preview
import androidx.lifecycle.compose.collectAsStateWithLifecycle
import androidx.lifecycle.viewmodel.compose.viewModel
import com.musicdecoded.mobile.MusicDecodedApplication
import com.musicdecoded.mobile.ui.components.InfoChip
import com.musicdecoded.mobile.ui.components.PolicyNote
import com.musicdecoded.mobile.ui.components.PreviewFixture
import com.musicdecoded.mobile.ui.components.PreviewSurface
import com.musicdecoded.mobile.ui.components.PrimaryButton
import com.musicdecoded.mobile.ui.components.SecondaryButton
import com.musicdecoded.mobile.ui.components.formatDuration
import com.musicdecoded.mobile.ui.theme.MusicDecodedColors
import com.musicdecoded.mobile.ui.theme.Shapes
import com.musicdecoded.mobile.ui.theme.Sizes
import com.musicdecoded.mobile.ui.theme.Spacing
import java.io.File

/** DESIGN.md §3.3, with the corrected (server-side analysis) copy. */
const val IMPORT_NOTE =
    "MusicDecoded uploads this file to our servers to analyze it and create an interactive chord timeline. " +
        "A copy stays on your device for offline playback."

/**
 * Opens the system picker for [from], then shows the Import sheet. [onFinished] gets the trackId
 * once decoding has started, or `null` if the user backed out.
 *
 * Pickers (minSdk 26, no storage permission needed):
 * - Files: `ACTION_OPEN_DOCUMENT` limited to `audio/` MIME types.
 * - Media Library: `ACTION_PICK` on `MediaStore.Audio.Media.EXTERNAL_CONTENT_URI` (the system's
 *   music picker). Devices without one fall back to `ACTION_GET_CONTENT` for `audio/` files.
 */
@Composable
fun ImportScreen(from: ImportFrom, onFinished: (trackId: String?) -> Unit) {
    val app = LocalContext.current.applicationContext as MusicDecodedApplication
    val viewModel = viewModel {
        ImportViewModel(AudioImporter(app, app.transcriptionRepository), app.transcriptionRepository, app.appScope)
    }
    val state by viewModel.state.collectAsStateWithLifecycle()

    val openDocument = rememberLauncherForActivityResult(ActivityResultContracts.OpenDocument(), viewModel::onPicked)
    val getContent = rememberLauncherForActivityResult(ActivityResultContracts.GetContent(), viewModel::onPicked)
    val pickMedia = rememberLauncherForActivityResult(PickMediaStoreAudio, viewModel::onPicked)

    LaunchedEffect(state.pickerPending) {
        if (!state.pickerPending) return@LaunchedEffect
        viewModel.onPickerLaunched()
        when (from) {
            ImportFrom.FILES -> openDocument.launch(arrayOf(AUDIO_MIME))
            ImportFrom.MEDIA_LIBRARY -> try {
                pickMedia.launch(Unit)
            } catch (_: ActivityNotFoundException) {
                getContent.launch(AUDIO_MIME)
            }
        }
    }
    LaunchedEffect(state.finished) { state.finished?.let { onFinished(it.trackId) } }

    Box(Modifier.fillMaxSize().background(MusicDecodedColors.Canvas))
    if (state.draft != null || state.copying || state.error != null) {
        ImportSheet(
            state = state,
            onImport = viewModel::importAndDecode,
            onChooseAnother = viewModel::chooseAnother,
            onDismiss = viewModel::dismiss,
        )
    }
}

private const val AUDIO_MIME = "audio/*"

/** The system music picker (`ACTION_PICK` on `MediaStore.Audio`). */
private object PickMediaStoreAudio : ActivityResultContract<Unit, Uri?>() {
    override fun createIntent(context: Context, input: Unit) =
        Intent(Intent.ACTION_PICK, MediaStore.Audio.Media.EXTERNAL_CONTENT_URI)

    override fun parseResult(resultCode: Int, intent: Intent?): Uri? =
        intent?.data?.takeIf { resultCode == Activity.RESULT_OK }
}

@OptIn(ExperimentalMaterial3Api::class)
@Composable
private fun ImportSheet(
    state: ImportUiState,
    onImport: () -> Unit,
    onChooseAnother: () -> Unit,
    onDismiss: () -> Unit,
) {
    ModalBottomSheet(
        onDismissRequest = onDismiss,
        sheetState = rememberModalBottomSheetState(skipPartiallyExpanded = true),
        shape = Shapes.sheet,
        containerColor = MusicDecodedColors.TransportBg,
    ) {
        ImportSheetContent(state, onImport, onChooseAnother)
    }
}

@Composable
private fun ImportSheetContent(state: ImportUiState, onImport: () -> Unit, onChooseAnother: () -> Unit) {
    Column(
        Modifier
            .fillMaxWidth()
            .padding(horizontal = Spacing.gutter)
            .padding(bottom = Spacing.lg)
            .navigationBarsPadding(),
        verticalArrangement = Arrangement.spacedBy(Spacing.md),
    ) {
        val draft = state.draft
        when {
            state.copying -> Row(horizontalArrangement = Arrangement.spacedBy(Spacing.sm), verticalAlignment = Alignment.CenterVertically) {
                CircularProgressIndicator(Modifier.size(Sizes.icon), color = MusicDecodedColors.Primary)
                Text("Preparing file…", style = MaterialTheme.typography.bodySmall, color = MusicDecodedColors.TextSecondary)
            }
            draft != null -> {
                FileHeader(draft)
                Row(horizontalArrangement = Arrangement.spacedBy(Spacing.xs)) {
                    InfoChip(Icons.Rounded.Schedule, "Duration", draft.durationSec?.let(::formatDuration) ?: "—", Modifier.weight(1f))
                    InfoChip(Icons.Rounded.GraphicEq, "Quality", draft.bitrate?.let { "${it / 1000} kbps" } ?: draft.typeLabel, Modifier.weight(1f))
                    InfoChip(Icons.Rounded.PhoneAndroid, "Storage", "On device", Modifier.weight(1f))
                }
                PolicyNote(IMPORT_NOTE, icon = Icons.Rounded.CloudUpload)
            }
        }
        state.error?.let { Text(it, style = MaterialTheme.typography.bodySmall, color = MusicDecodedColors.Record) }
        if (draft != null) {
            PrimaryButton(
                text = if (state.starting) "Uploading…" else "Import & decode",
                onClick = onImport,
                enabled = !state.starting,
            )
        }
        SecondaryButton(
            text = "Choose another file",
            onClick = onChooseAnother,
            icon = Icons.Rounded.SwapHoriz,
            enabled = !state.starting && !state.copying,
        )
    }
}

@Composable
private fun FileHeader(draft: ImportDraft) {
    Row(horizontalArrangement = Arrangement.spacedBy(Spacing.sm), verticalAlignment = Alignment.CenterVertically) {
        Box(
            Modifier.size(Sizes.sourceTile).background(MusicDecodedColors.PrimarySoft, Shapes.control),
            contentAlignment = Alignment.Center,
        ) {
            Icon(Icons.Rounded.AudioFile, contentDescription = null, tint = MusicDecodedColors.Primary)
        }
        Column(Modifier.weight(1f)) {
            Text(
                draft.fileName,
                style = MaterialTheme.typography.titleSmall,
                color = MusicDecodedColors.Text,
                maxLines = 2,
                overflow = TextOverflow.Ellipsis,
            )
            Text(
                "${draft.typeLabel} · ${AudioImporter.formatSize(draft.sizeBytes)}",
                style = MaterialTheme.typography.bodySmall,
                color = MusicDecodedColors.TextSecondary,
            )
        }
    }
}

@Preview
@Composable
private fun ImportSheetPreview() = PreviewSurface {
    Box(Modifier.background(MusicDecodedColors.TransportBg, Shapes.sheet)) {
        ImportSheetContent(
            ImportUiState(
                pickerPending = false,
                draft = ImportDraft(
                    file = File("pending.mp3"),
                    fileName = "${PreviewFixture.TITLE}.mp3",
                    title = PreviewFixture.TITLE,
                    sizeBytes = 8_400_000,
                    durationSec = PreviewFixture.DURATION_SEC,
                    bitrate = 256_000,
                ),
            ),
            onImport = {},
            onChooseAnother = {},
        )
    }
}
