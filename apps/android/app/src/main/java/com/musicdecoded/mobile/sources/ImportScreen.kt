package com.musicdecoded.mobile.sources

import android.app.Activity
import android.content.ActivityNotFoundException
import android.content.Context
import android.content.Intent
import android.net.Uri
import android.provider.DocumentsContract
import androidx.activity.compose.rememberLauncherForActivityResult
import androidx.activity.result.contract.ActivityResultContract
import androidx.activity.result.contract.ActivityResultContracts
import androidx.activity.compose.BackHandler
import androidx.compose.animation.AnimatedVisibility
import androidx.compose.animation.slideInVertically
import androidx.compose.animation.slideOutVertically
import androidx.compose.foundation.background
import androidx.compose.foundation.border
import androidx.compose.foundation.clickable
import androidx.compose.foundation.rememberScrollState
import androidx.compose.foundation.shape.CircleShape
import androidx.compose.foundation.verticalScroll
import androidx.compose.foundation.layout.imePadding
import androidx.compose.foundation.layout.navigationBarsPadding
import androidx.compose.ui.draw.clip
import androidx.compose.ui.unit.dp
import androidx.compose.foundation.layout.Arrangement
import androidx.compose.foundation.layout.Box
import androidx.compose.foundation.layout.Column
import androidx.compose.foundation.layout.Row
import androidx.compose.foundation.layout.fillMaxSize
import androidx.compose.foundation.layout.fillMaxWidth
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
import androidx.compose.material3.Icon
import androidx.compose.material3.MaterialTheme
import androidx.compose.material3.Text
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
import com.musicdecoded.mobile.ui.components.ErrorState
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
 * - Media Library: the same documents UI, opened on the media provider's **Audio** root
 *   (`EXTRA_INITIAL_URI`, API 26+) so it starts in the music collection instead of the old
 *   `ACTION_PICK` ringtone-style list. Without a documents UI it falls back to `ACTION_GET_CONTENT`.
 *   An in-app MediaStore browser was rejected: it needs `READ_MEDIA_AUDIO` / `READ_EXTERNAL_STORAGE`
 *   (a runtime prompt to pick one file), and the Photo Picker doesn't offer audio.
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

    BackHandler(onBack = viewModel::dismiss)
    // Shown over Home (this is a dialog destination); a tap on the dimmed area dismisses.
    Box(
        Modifier
            .fillMaxSize()
            .clickable(interactionSource = null, indication = null, onClickLabel = "Close", onClick = viewModel::dismiss),
        contentAlignment = Alignment.BottomCenter,
    ) {
        AnimatedVisibility(
            visible = state.draft != null || state.copying || state.error != null,
            enter = slideInVertically { it },
            exit = slideOutVertically { it },
        ) {
            ImportSheet(state, onImport = viewModel::importAndDecode, onChooseAnother = viewModel::chooseAnother)
        }
    }
}

private const val AUDIO_MIME = "audio/*"

/** The documents UI opened on the media provider's Audio root (no storage permission needed). */
private object PickMediaStoreAudio : ActivityResultContract<Unit, Uri?>() {
    private const val MEDIA_DOCUMENTS = "com.android.providers.media.documents"
    private const val AUDIO_ROOT = "audio_root"

    override fun createIntent(context: Context, input: Unit): Intent =
        Intent(Intent.ACTION_OPEN_DOCUMENT)
            .addCategory(Intent.CATEGORY_OPENABLE)
            .setType(AUDIO_MIME)
            .putExtra(DocumentsContract.EXTRA_INITIAL_URI, DocumentsContract.buildDocumentUri(MEDIA_DOCUMENTS, AUDIO_ROOT))

    override fun parseResult(resultCode: Int, intent: Intent?): Uri? =
        intent?.data?.takeIf { resultCode == Activity.RESULT_OK }
}

/** DESIGN.md §3.3: `transportBg`, rounded top corners, a drag handle, above the nav bar and keyboard. */
@Composable
private fun ImportSheet(state: ImportUiState, onImport: () -> Unit, onChooseAnother: () -> Unit) {
    Column(
        Modifier
            .fillMaxWidth()
            .clip(Shapes.sheet)
            .background(MusicDecodedColors.TransportBg)
            .border(Sizes.hairline, MusicDecodedColors.TransportBorder, Shapes.sheet)
            .clickable(interactionSource = null, indication = null, onClick = {}) // taps inside don't dismiss
            .navigationBarsPadding()
            .imePadding()
            .verticalScroll(rememberScrollState())
            .padding(top = Spacing.sm),
        horizontalAlignment = Alignment.CenterHorizontally,
        verticalArrangement = Arrangement.spacedBy(Spacing.md),
    ) {
        Box(Modifier.size(SheetHandleWidth, Spacing.xxs).background(MusicDecodedColors.Line, CircleShape))
        ImportSheetContent(state, onImport, onChooseAnother)
    }
}

/** The sheet's drag handle (a component size). */
private val SheetHandleWidth = 32.dp

@Composable
private fun ImportSheetContent(state: ImportUiState, onImport: () -> Unit, onChooseAnother: () -> Unit) {
    Column(
        Modifier
            .fillMaxWidth()
            .padding(horizontal = Spacing.gutter)
            .padding(bottom = Spacing.lg),
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
        // Upload/job errors (offline, rate limit…) retry with the same file; copy errors can only pick another.
        state.error?.let { ErrorState(it, onRetry = onImport.takeIf { draft != null && !state.starting }) }
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
