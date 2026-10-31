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
import androidx.compose.foundation.layout.navigationBarsPadding
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
import androidx.compose.ui.semantics.heading
import androidx.compose.ui.semantics.semantics
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
import com.musicdecoded.mobile.library.TranscriptionRepository
import com.musicdecoded.mobile.ui.components.ErrorState
import com.musicdecoded.mobile.ui.components.MdIconButton
import com.musicdecoded.mobile.ui.components.PreviewSurface
import com.musicdecoded.mobile.ui.components.PrimaryButton
import com.musicdecoded.mobile.ui.components.SecondaryButton
import com.musicdecoded.mobile.ui.components.SongArt
import com.musicdecoded.mobile.ui.components.WrappedProgressBar
import com.musicdecoded.mobile.ui.components.block
import com.musicdecoded.mobile.ui.components.halftone
import com.musicdecoded.mobile.ui.components.hardShadow
import com.musicdecoded.mobile.ui.components.tilt
import com.musicdecoded.mobile.ui.theme.Borders
import com.musicdecoded.mobile.ui.theme.MusicDecodedColors
import com.musicdecoded.mobile.ui.theme.Shadows
import com.musicdecoded.mobile.ui.theme.TextStyles
import com.musicdecoded.mobile.ui.theme.Tilt
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
            .background(MusicDecodedColors.Violet)
            .halftone(MusicDecodedColors.White)
            .statusBarsPadding()
            .navigationBarsPadding()
            .padding(horizontal = Spacing.gutter, vertical = Spacing.xs),
    ) {
        // The steps scroll on short screens; the buttons stay pinned at the bottom, above the nav bar.
        Column(
            Modifier.weight(1f).fillMaxWidth().verticalScroll(rememberScrollState()),
            horizontalAlignment = Alignment.CenterHorizontally,
            verticalArrangement = Arrangement.spacedBy(Spacing.lg),
        ) {
            Box(Modifier.fillMaxWidth()) {
                MdIconButton(
                    Icons.Rounded.Close,
                    contentDescription = "Continue in background",
                    onClick = onBackground,
                    modifier = Modifier.align(Alignment.CenterEnd),
                    container = MusicDecodedColors.White,
                )
            }
            OrbitArt(state.title)
            Column(horizontalAlignment = Alignment.CenterHorizontally, verticalArrangement = Arrangement.spacedBy(Spacing.xxs)) {
                Text(
                    when (state.phase) {
                        ProcessingPhase.COMPLETE -> "Your transcription is ready"
                        ProcessingPhase.FAILED -> "We couldn't decode this song"
                        else -> "Decoding your music"
                    }.uppercase(),
                    style = MaterialTheme.typography.headlineMedium,
                    color = MusicDecodedColors.Text,
                    textAlign = TextAlign.Center,
                    modifier = Modifier.semantics { heading() },
                )
                val subtitle = listOfNotNull(state.title.takeIf { it.isNotBlank() }, state.artist).joinToString(" · ")
                if (subtitle.isNotEmpty()) {
                    Text(subtitle, style = MaterialTheme.typography.bodySmall, color = MusicDecodedColors.TextSecondaryOnViolet, maxLines = 1, overflow = TextOverflow.Ellipsis)
                }
            }

            if (state.phase == ProcessingPhase.FAILED) {
                ErrorState(state.error ?: TranscriptionRepository.GENERIC, onRetry = onRetry)
            } else {
                OverallProgress(state)
                // The step list sits on an ink card with a hard `shadowXl` (DESIGN.md §1 color map).
                Column(
                    Modifier
                        .fillMaxWidth()
                        .padding(end = Shadows.xl, bottom = Shadows.xl)
                        .block(MusicDecodedColors.Ink, shadow = Shadows.xl)
                        .padding(Spacing.xs),
                    verticalArrangement = Arrangement.spacedBy(Spacing.xxs),
                ) {
                    ProcessingStep.entries.forEach { StepRow(it, done = state.isDone(it), active = state.isActive(it)) }
                }
            }
        }
        Column(Modifier.fillMaxWidth().padding(top = Spacing.md), verticalArrangement = Arrangement.spacedBy(Spacing.sm)) {
            when (state.phase) {
                ProcessingPhase.COMPLETE -> PrimaryButton("Open transcription", onClick = onOpen, shadowColor = MusicDecodedColors.Border)
                ProcessingPhase.FAILED -> SecondaryButton("Back", onClick = onBackground)
                else -> SecondaryButton("Continue in background", onClick = onBackground)
            }
        }
    }
}

private val OrbitOuter = 208.dp
private val ArtSize = 120.dp

/** Art (the title's initial) as a tilted sticker card over a white ring, with a lime wave badge. */
@Composable
private fun OrbitArt(title: String) {
    Box(Modifier.size(OrbitOuter), contentAlignment = Alignment.Center) {
        Box(Modifier.size(OrbitOuter).border(Borders.thick, MusicDecodedColors.White.copy(alpha = 0.2f), CircleShape))
        SongArt(
            title,
            Modifier.size(ArtSize).tilt(Tilt.hero * 2).hardShadow(Shadows.lg),
            style = MaterialTheme.typography.displayMedium,
            border = MusicDecodedColors.Border,
        )
        Box(
            Modifier
                .align(Alignment.BottomEnd)
                .padding(Spacing.xl)
                .size(Sizes.iconButton)
                .tilt(Tilt.sticker)
                .block(MusicDecodedColors.Primary, shadow = Shadows.sm),
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
            Text(state.step.label.uppercase(), Modifier.weight(1f), style = TextStyles.overline, color = MusicDecodedColors.Text)
            Text("${state.progress}%", style = TextStyles.counter, color = MusicDecodedColors.Primary)
        }
        WrappedProgressBar(
            state.progress / 100f,
            Modifier.border(Borders.thick, MusicDecodedColors.Border),
            trackColor = MusicDecodedColors.Ink,
            height = Spacing.sm,
        )
    }
}

@Composable
private fun StepRow(step: ProcessingStep, done: Boolean, active: Boolean) {
    Row(
        Modifier
            .fillMaxWidth()
            .background(if (active) MusicDecodedColors.SurfaceRaised else MusicDecodedColors.Ink)
            .padding(Spacing.sm),
        horizontalArrangement = Arrangement.spacedBy(Spacing.sm),
        verticalAlignment = Alignment.CenterVertically,
    ) {
        Box(
            Modifier
                .size(Sizes.smallIconButton)
                .background(if (done || active) MusicDecodedColors.Primary else MusicDecodedColors.SurfaceRaised)
                .border(Borders.separator, if (done || active) MusicDecodedColors.Border else MusicDecodedColors.Line),
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
