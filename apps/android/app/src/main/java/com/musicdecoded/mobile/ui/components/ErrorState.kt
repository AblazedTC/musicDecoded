package com.musicdecoded.mobile.ui.components

import androidx.compose.foundation.background
import androidx.compose.foundation.border
import androidx.compose.foundation.layout.Arrangement
import androidx.compose.foundation.layout.Column
import androidx.compose.foundation.layout.Row
import androidx.compose.foundation.layout.fillMaxWidth
import androidx.compose.foundation.layout.padding
import androidx.compose.material.icons.Icons
import androidx.compose.material.icons.rounded.CloudOff
import androidx.compose.material.icons.rounded.ErrorOutline
import androidx.compose.material.icons.rounded.Refresh
import androidx.compose.material3.Icon
import androidx.compose.material3.MaterialTheme
import androidx.compose.material3.Text
import androidx.compose.runtime.Composable
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.semantics.LiveRegionMode
import androidx.compose.ui.semantics.liveRegion
import androidx.compose.ui.semantics.semantics
import androidx.compose.ui.tooling.preview.Preview
import com.musicdecoded.mobile.library.TranscriptionRepository
import com.musicdecoded.mobile.ui.theme.MusicDecodedColors
import com.musicdecoded.mobile.ui.theme.Shapes
import com.musicdecoded.mobile.ui.theme.Sizes
import com.musicdecoded.mobile.ui.theme.Spacing

/** The title [ErrorState] uses for a message: "You're offline" for the shared offline copy. */
fun errorTitle(message: String): String =
    if (message == TranscriptionRepository.OFFLINE) "You're offline" else "Something went wrong"

/**
 * The one error state for anything that talks to the server (T-503a): icon, title, message and a
 * Retry button ([onRetry] `null` hides it, e.g. when retrying can't help). Announced politely by
 * screen readers when it appears.
 */
@Composable
fun ErrorState(
    message: String,
    modifier: Modifier = Modifier,
    title: String = errorTitle(message),
    onRetry: (() -> Unit)? = null,
    retryLabel: String = "Try again",
) {
    Column(
        modifier
            .fillMaxWidth()
            .background(MusicDecodedColors.Surface, Shapes.card)
            .border(Sizes.hairline, MusicDecodedColors.Line, Shapes.card)
            .padding(Spacing.md)
            .semantics { liveRegion = LiveRegionMode.Polite },
        verticalArrangement = Arrangement.spacedBy(Spacing.sm),
    ) {
        Row(horizontalArrangement = Arrangement.spacedBy(Spacing.xs), verticalAlignment = Alignment.CenterVertically) {
            Icon(
                if (message == TranscriptionRepository.OFFLINE) Icons.Rounded.CloudOff else Icons.Rounded.ErrorOutline,
                contentDescription = null,
                tint = MusicDecodedColors.Record,
            )
            Text(title, style = MaterialTheme.typography.titleSmall, color = MusicDecodedColors.Text)
        }
        Text(message, style = MaterialTheme.typography.bodySmall, color = MusicDecodedColors.TextSecondary)
        if (onRetry != null) SecondaryButton(retryLabel, onClick = onRetry, icon = Icons.Rounded.Refresh)
    }
}

@Preview
@Composable
private fun ErrorStatePreview() = PreviewSurface {
    ErrorState(TranscriptionRepository.OFFLINE, onRetry = {})
}
