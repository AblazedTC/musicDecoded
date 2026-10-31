package com.musicdecoded.mobile.sources.soundcloud

import android.net.Uri
import androidx.compose.foundation.background
import androidx.compose.foundation.border
import androidx.compose.foundation.layout.Arrangement
import androidx.compose.foundation.layout.Box
import androidx.compose.foundation.layout.Column
import androidx.compose.foundation.layout.Row
import androidx.compose.foundation.layout.fillMaxWidth
import androidx.compose.foundation.layout.padding
import androidx.compose.foundation.layout.size
import androidx.compose.foundation.rememberScrollState
import androidx.compose.foundation.text.KeyboardActions
import androidx.compose.foundation.text.KeyboardOptions
import androidx.compose.foundation.verticalScroll
import androidx.compose.material3.Icon
import androidx.compose.material3.MaterialTheme
import androidx.compose.material3.OutlinedTextField
import androidx.compose.material3.OutlinedTextFieldDefaults
import androidx.compose.material3.Text
import androidx.compose.runtime.Composable
import androidx.compose.runtime.LaunchedEffect
import androidx.compose.runtime.getValue
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.platform.LocalContext
import androidx.compose.ui.semantics.heading
import androidx.compose.ui.semantics.semantics
import androidx.compose.ui.text.input.ImeAction
import androidx.compose.ui.text.input.KeyboardType
import androidx.compose.ui.tooling.preview.Preview
import androidx.lifecycle.compose.collectAsStateWithLifecycle
import androidx.lifecycle.viewmodel.compose.viewModel
import androidx.navigation.NavGraphBuilder
import androidx.navigation.NavHostController
import androidx.navigation.NavType
import androidx.navigation.compose.dialog
import androidx.navigation.navArgument
import com.musicdecoded.mobile.MusicDecodedApplication
import com.musicdecoded.mobile.ui.components.ErrorState
import com.musicdecoded.mobile.ui.components.PreviewSurface
import com.musicdecoded.mobile.ui.components.PrimaryButton
import com.musicdecoded.mobile.ui.components.SecondaryButton
import com.musicdecoded.mobile.ui.components.SourceIconTile
import com.musicdecoded.mobile.ui.components.SourceKind
import com.musicdecoded.mobile.ui.components.block
import com.musicdecoded.mobile.ui.theme.Shadows
import com.musicdecoded.mobile.ui.theme.MusicDecodedColors
import com.musicdecoded.mobile.ui.theme.Shapes
import com.musicdecoded.mobile.ui.theme.Sizes
import com.musicdecoded.mobile.ui.theme.Spacing

object SoundCloudRoutes {
    const val LINK = "soundcloud?text={text}"

    /** The link dialog, optionally prefilled with pasted or shared [text]. */
    fun link(text: String = "") = "soundcloud?text=${Uri.encode(text)}"
}

/** The SoundCloud link dialog over the current screen (Home card or a share-sheet intent). */
fun NavGraphBuilder.soundCloudRoute(navController: NavHostController, onDecodeStarted: (trackId: String) -> Unit) {
    dialog(SoundCloudRoutes.LINK, arguments = listOf(navArgument("text") { type = NavType.StringType; defaultValue = "" })) { entry ->
        val text = entry.arguments?.getString("text").orEmpty()
        val app = LocalContext.current.applicationContext as MusicDecodedApplication
        val viewModel: SoundCloudViewModel = viewModel(factory = SoundCloudViewModel.factory(app, text))
        val state by viewModel.state.collectAsStateWithLifecycle()
        LaunchedEffect(state.startedTrackId) {
            state.startedTrackId?.let {
                viewModel.navigated()
                onDecodeStarted(it)
            }
        }
        SoundCloudContent(
            state,
            onTextChange = viewModel::onTextChange,
            onDecode = viewModel::decode,
            onCancel = { navController.popBackStack() },
        )
    }
}

@Composable
private fun SoundCloudContent(
    state: SoundCloudUiState,
    onTextChange: (String) -> Unit,
    onDecode: () -> Unit,
    onCancel: () -> Unit,
) {
    Column(
        Modifier
            .fillMaxWidth()
            .block(MusicDecodedColors.Surface, shadow = Shadows.lg, shadowColor = MusicDecodedColors.Yellow)
            .verticalScroll(rememberScrollState())
            .padding(Spacing.lg),
        verticalArrangement = Arrangement.spacedBy(Spacing.md),
    ) {
        Row(horizontalArrangement = Arrangement.spacedBy(Spacing.sm), verticalAlignment = Alignment.CenterVertically) {
            SourceIconTile(SourceKind.SOUNDCLOUD)
            Text(
                "DECODE A SOUNDCLOUD TRACK",
                style = MaterialTheme.typography.titleMedium,
                color = MusicDecodedColors.Text,
                modifier = Modifier.weight(1f).semantics { heading() },
            )
        }
        Text(
            "Paste a track link, or share one to MusicDecoded from the SoundCloud app.",
            style = MaterialTheme.typography.bodySmall,
            color = MusicDecodedColors.TextSecondary,
        )
        OutlinedTextField(
            value = state.text,
            onValueChange = onTextChange,
            modifier = Modifier.fillMaxWidth(),
            enabled = state.enabled && !state.starting,
            label = { Text("SoundCloud link") },
            placeholder = { Text("soundcloud.com/artist/track", color = MusicDecodedColors.TextMuted) },
            isError = state.linkError != null,
            supportingText = state.linkError?.let { { Text(it) } },
            keyboardOptions = KeyboardOptions(keyboardType = KeyboardType.Uri, imeAction = ImeAction.Go),
            keyboardActions = KeyboardActions(onGo = { onDecode() }),
            shape = Shapes.control,
            colors = OutlinedTextFieldDefaults.colors(
                focusedBorderColor = MusicDecodedColors.Primary,
                unfocusedBorderColor = MusicDecodedColors.BorderOnInk,
                focusedTextColor = MusicDecodedColors.Text,
                unfocusedTextColor = MusicDecodedColors.Text,
                focusedLabelColor = MusicDecodedColors.Primary,
                unfocusedLabelColor = MusicDecodedColors.TextSecondary,
                cursorColor = MusicDecodedColors.Primary,
            ),
        )
        val error = state.error
        if (error != null) {
            // A disabled server can't be retried from here; network and quota errors can.
            ErrorState(error, onRetry = onDecode.takeIf { state.enabled && !state.starting })
        } else {
            PrimaryButton(
                if (state.starting) "Starting…" else "Decode this song",
                onClick = onDecode,
                enabled = state.enabled && !state.starting && state.text.isNotBlank(),
            )
        }
        SecondaryButton("Cancel", onClick = onCancel)
    }
}

@Preview
@Composable
private fun SoundCloudPreview() = PreviewSurface {
    SoundCloudContent(SoundCloudUiState(text = "https://soundcloud.com/artist/track"), {}, {}, {})
}
