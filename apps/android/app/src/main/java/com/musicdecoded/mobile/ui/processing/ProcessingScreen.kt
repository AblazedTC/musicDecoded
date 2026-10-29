package com.musicdecoded.mobile.ui.processing

import androidx.compose.foundation.background
import androidx.compose.foundation.border
import androidx.compose.foundation.layout.Arrangement
import androidx.compose.foundation.layout.Box
import androidx.compose.foundation.layout.Column
import androidx.compose.foundation.layout.Row
import androidx.compose.foundation.layout.fillMaxSize
import androidx.compose.foundation.layout.fillMaxWidth
import androidx.compose.foundation.layout.height
import androidx.compose.foundation.layout.padding
import androidx.compose.foundation.layout.size
import androidx.compose.foundation.layout.statusBarsPadding
import androidx.compose.foundation.rememberScrollState
import androidx.compose.foundation.shape.CircleShape
import androidx.compose.foundation.verticalScroll
import androidx.compose.material.icons.Icons
import androidx.compose.material.icons.rounded.Check
import androidx.compose.material.icons.rounded.Close
import androidx.compose.material.icons.rounded.GraphicEq
import androidx.compose.material3.Icon
import androidx.compose.material3.LinearProgressIndicator
import androidx.compose.material3.MaterialTheme
import androidx.compose.material3.Text
import androidx.compose.runtime.Composable
import androidx.compose.runtime.getValue
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.draw.clip
import androidx.compose.ui.platform.LocalContext
import androidx.compose.ui.text.style.TextAlign
import androidx.compose.ui.text.style.TextOverflow
import androidx.compose.ui.tooling.preview.Preview
import androidx.compose.ui.unit.dp
import androidx.lifecycle.compose.collectAsStateWithLifecycle
import androidx.lifecycle.viewmodel.compose.viewModel
import androidx.navigation.NavGraphBuilder
import androidx.navigation.NavHostController
import androidx.navigation.compose.composable
import com.musicdecoded.mobile.MusicDecodedApplication
import com.musicdecoded.mobile.player.PlayerRoutes
import com.musicdecoded.mobile.ui.components.MdIconButton
import com.musicdecoded.mobile.ui.components.PreviewSurface
import com.musicdecoded.mobile.ui.components.PrimaryButton
import com.musicdecoded.mobile.ui.components.SecondaryButton
import com.musicdecoded.mobile.ui.theme.MusicDecodedColors
import com.musicdecoded.mobile.ui.theme.Shapes
import com.musicdecoded.mobile.ui.theme.Sizes
import com.musicdecoded.mobile.ui.theme.Spacing
import com.musicdecoded.mobile.ui.theme.tabular

object ProcessingRoutes {
    const val PROCESSING = "processing/{trackId}"
    fun processing(trackId: String) = "processing/$trackId"
}

/** Opens Processing for a job that just started, replacing the source screen (import sheet / recorder). */
fun NavHostController.navigateToProcessing(trackId: String) {
    val source = currentDestination?.route
    navigate(ProcessingRoutes.processing(trackId)) {
        if (source != null) popUpTo(source) { inclusive = true }
    }
}

/** `processing/{trackId}` (pushed, no tab bar). Leaving it never cancels the job. */
fun NavGraphBuilder.processingDestination(navController: NavHostController) {
    composable(ProcessingRoutes.PROCESSING) { entry ->
        val trackId = entry.arguments?.getString("trackId").orEmpty()
        val app = LocalContext.current.applicationContext as MusicDecodedApplication
        val viewModel: ProcessingViewModel = viewModel(factory = ProcessingViewModel.factory(app, trackId))
        ProcessingScreen(
            viewModel,
            onBackground = { navController.popBackStack() },
            onOpen = {
                navController.navigate(PlayerRoutes.player(trackId)) {
                    popUpTo(ProcessingRoutes.PROCESSING) { inclusive = true }
                }
            },
        )
    }
}

/** DESIGN.md §3.5. × and "Continue in background" both just leave; the row keeps showing progress. */
@Composable
fun ProcessingScreen(viewModel: ProcessingViewModel, onBackground: () -> Unit, onOpen: () -> Unit) {
    val state by viewModel.state.collectAsStateWithLifecycle()
    ProcessingContent(state, onBackground = onBackground, onOpen = onOpen, onRetry = viewModel::retry)
}

@Composable
private fun ProcessingContent(state: ProcessingUiState, onBackground: () -> Unit, onOpen: () -> Unit, onRetry: () -> Unit) {
    Column(
        Modifier
            .fillMaxSize()
            .background(MusicDecodedColors.Canvas)
            .statusBarsPadding()
            .verticalScroll(rememberScrollState())
            .padding(horizontal = Spacing.gutter, vertical = Spacing.xs),
        horizontalAlignment = Alignment.CenterHorizontally,
        verticalArrangement = Arrangement.spacedBy(Spacing.lg),
    ) {
        Box(Modifier.fillMaxWidth()) {
            MdIconButton(Icons.Rounded.Close, contentDescription = "Continue in background", onClick = onBackground, modifier = Modifier.align(Alignment.CenterEnd))
        }
        OrbitArt(state.title)
        Column(horizontalAlignment = Alignment.CenterHorizontally, verticalArrangement = Arrangement.spacedBy(Spacing.xxs)) {
            Text(
                when (state.phase) {
                    ProcessingPhase.COMPLETE -> "Your transcription is ready"
                    ProcessingPhase.FAILED -> "We couldn't decode this song"
                    else -> "Decoding your music"
                },
                style = MaterialTheme.typography.headlineMedium,
                color = MusicDecodedColors.Text,
                textAlign = TextAlign.Center,
            )
            val subtitle = listOfNotNull(state.title.takeIf { it.isNotBlank() }, state.artist).joinToString(" · ")
            if (subtitle.isNotEmpty()) {
                Text(subtitle, style = MaterialTheme.typography.bodySmall, color = MusicDecodedColors.TextSecondary, maxLines = 1, overflow = TextOverflow.Ellipsis)
            }
        }

        if (state.phase == ProcessingPhase.FAILED) {
            Text(
                state.error ?: "Something went wrong. Try again.",
                style = MaterialTheme.typography.bodySmall,
                color = MusicDecodedColors.Record,
                textAlign = TextAlign.Center,
            )
        } else {
            OverallProgress(state)
            Column(Modifier.fillMaxWidth(), verticalArrangement = Arrangement.spacedBy(Spacing.xs)) {
                ProcessingStep.entries.forEach { StepRow(it, done = state.isDone(it), active = state.isActive(it)) }
            }
        }

        Column(Modifier.fillMaxWidth(), verticalArrangement = Arrangement.spacedBy(Spacing.sm)) {
            when (state.phase) {
                ProcessingPhase.COMPLETE -> PrimaryButton("Open transcription", onClick = onOpen)
                ProcessingPhase.FAILED -> {
                    PrimaryButton("Try again", onClick = onRetry)
                    SecondaryButton("Back", onClick = onBackground)
                }
                else -> SecondaryButton("Continue in background", onClick = onBackground)
            }
        }
    }
}

private val OrbitOuter = 208.dp
private val OrbitInner = 164.dp
private val ArtSize = 120.dp

/** Art (the title's initial) inside two orbit rings, with a wave badge. */
@Composable
private fun OrbitArt(title: String) {
    Box(Modifier.size(OrbitOuter), contentAlignment = Alignment.Center) {
        Box(Modifier.size(OrbitOuter).border(Sizes.hairline, MusicDecodedColors.Line, CircleShape))
        Box(Modifier.size(OrbitInner).border(Sizes.hairline, MusicDecodedColors.PrimarySoft, CircleShape))
        Box(Modifier.size(ArtSize).background(MusicDecodedColors.SurfaceSoft, Shapes.card), contentAlignment = Alignment.Center) {
            Text(
                title.trim().firstOrNull()?.uppercase() ?: "",
                style = MaterialTheme.typography.headlineMedium,
                color = MusicDecodedColors.Secondary,
            )
        }
        Box(
            Modifier
                .align(Alignment.BottomEnd)
                .padding(Spacing.xl)
                .size(Sizes.iconButton)
                .background(MusicDecodedColors.Primary, CircleShape),
            contentAlignment = Alignment.Center,
        ) {
            Icon(Icons.Rounded.GraphicEq, contentDescription = null, tint = MusicDecodedColors.OnPrimary, modifier = Modifier.size(Sizes.smallIcon))
        }
    }
}

@Composable
private fun OverallProgress(state: ProcessingUiState) {
    Column(Modifier.fillMaxWidth(), verticalArrangement = Arrangement.spacedBy(Spacing.xs)) {
        Row(Modifier.fillMaxWidth()) {
            Text(state.step.label, Modifier.weight(1f), style = MaterialTheme.typography.labelSmall, color = MusicDecodedColors.TextSecondary)
            Text("${state.progress}%", style = MaterialTheme.typography.labelSmall.tabular(), color = MusicDecodedColors.Primary)
        }
        LinearProgressIndicator(
            progress = { state.progress / 100f },
            modifier = Modifier.fillMaxWidth().height(Sizes.progressTrack).clip(CircleShape),
            color = MusicDecodedColors.Primary,
            trackColor = MusicDecodedColors.SurfaceSoft,
            drawStopIndicator = {},
        )
    }
}

@Composable
private fun StepRow(step: ProcessingStep, done: Boolean, active: Boolean) {
    Row(
        Modifier
            .fillMaxWidth()
            .background(if (done) MusicDecodedColors.PrimarySoft else MusicDecodedColors.Surface, Shapes.row)
            .border(Sizes.hairline, if (active) MusicDecodedColors.Primary else MusicDecodedColors.Line, Shapes.row)
            .padding(Spacing.sm),
        horizontalArrangement = Arrangement.spacedBy(Spacing.sm),
        verticalAlignment = Alignment.CenterVertically,
    ) {
        Box(
            Modifier
                .size(Sizes.smallIconButton)
                .background(if (done || active) MusicDecodedColors.Primary else MusicDecodedColors.SurfaceSoft, CircleShape),
            contentAlignment = Alignment.Center,
        ) {
            when {
                done -> Icon(Icons.Rounded.Check, contentDescription = "Done", tint = MusicDecodedColors.OnPrimary, modifier = Modifier.size(Sizes.smallIcon))
                active -> Icon(Icons.Rounded.GraphicEq, contentDescription = "In progress", tint = MusicDecodedColors.OnPrimary, modifier = Modifier.size(Sizes.smallIcon))
            }
        }
        Text(
            step.label,
            style = MaterialTheme.typography.titleSmall,
            color = if (done || active) MusicDecodedColors.Text else MusicDecodedColors.TextMuted,
        )
    }
}

@Preview
@Composable
private fun ProcessingPreview() = PreviewSurface {
    ProcessingContent(
        ProcessingUiState(ProcessingPhase.RUNNING, "Sample track", "Fixture artist", ProcessingStep.HARMONY, 58),
        onBackground = {},
        onOpen = {},
        onRetry = {},
    )
}
