package com.musicdecoded.mobile.sources.youtube

import androidx.compose.foundation.background
import androidx.compose.foundation.border
import androidx.compose.foundation.clickable
import androidx.compose.foundation.layout.Arrangement
import androidx.compose.foundation.layout.Box
import androidx.compose.foundation.layout.Column
import androidx.compose.foundation.layout.PaddingValues
import androidx.compose.foundation.layout.Row
import androidx.compose.foundation.layout.fillMaxSize
import androidx.compose.foundation.layout.fillMaxWidth
import androidx.compose.foundation.layout.padding
import androidx.compose.foundation.layout.size
import androidx.compose.foundation.lazy.LazyColumn
import androidx.compose.foundation.lazy.LazyRow
import androidx.compose.foundation.lazy.items
import androidx.compose.foundation.lazy.rememberLazyListState
import androidx.compose.foundation.text.KeyboardActions
import androidx.compose.foundation.text.KeyboardOptions
import androidx.compose.material.icons.Icons
import androidx.compose.material.icons.automirrored.rounded.ArrowBack
import androidx.compose.material.icons.rounded.Close
import androidx.compose.material.icons.rounded.Link
import androidx.compose.material.icons.rounded.Search
import androidx.compose.material3.CircularProgressIndicator
import androidx.compose.material3.Icon
import androidx.compose.material3.MaterialTheme
import androidx.compose.material3.OutlinedTextField
import androidx.compose.material3.OutlinedTextFieldDefaults
import androidx.compose.material3.Text
import androidx.compose.runtime.Composable
import androidx.compose.runtime.LaunchedEffect
import androidx.compose.runtime.derivedStateOf
import androidx.compose.runtime.getValue
import androidx.compose.runtime.mutableStateOf
import androidx.compose.runtime.remember
import androidx.compose.runtime.saveable.rememberSaveable
import androidx.compose.runtime.setValue
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.draw.clip
import androidx.compose.ui.platform.LocalContext
import androidx.compose.ui.semantics.Role
import androidx.compose.ui.text.input.ImeAction
import androidx.compose.ui.text.input.KeyboardType
import androidx.compose.ui.text.style.TextOverflow
import androidx.compose.ui.tooling.preview.Preview
import androidx.compose.ui.unit.dp
import androidx.lifecycle.compose.collectAsStateWithLifecycle
import androidx.lifecycle.viewmodel.compose.viewModel
import com.musicdecoded.mobile.MusicDecodedApplication
import com.musicdecoded.mobile.network.VideoSummary
import com.musicdecoded.mobile.ui.components.ErrorState
import com.musicdecoded.mobile.ui.components.MdFilterChip
import com.musicdecoded.mobile.ui.components.MdIconButton
import com.musicdecoded.mobile.ui.components.PreviewSurface
import com.musicdecoded.mobile.ui.components.RemoteImage
import com.musicdecoded.mobile.ui.components.SectionHeader
import com.musicdecoded.mobile.ui.components.SecondaryButton
import com.musicdecoded.mobile.ui.components.formatDuration
import com.musicdecoded.mobile.ui.theme.MusicDecodedColors
import com.musicdecoded.mobile.ui.theme.Shapes
import com.musicdecoded.mobile.ui.theme.Sizes
import com.musicdecoded.mobile.ui.theme.Spacing
import com.musicdecoded.mobile.ui.theme.TextStyles

/** DESIGN.md §3.4 result thumbnail: 132×78. */
private val ThumbWidth = 132.dp
private val ThumbHeight = 78.dp

/** Start loading the next page this many rows before the end. */
private const val PREFETCH_ROWS = 3

/** YouTube search (DESIGN.md §3.4). Pushed, no tab bar. */
@Composable
fun YouTubeSearchScreen(onBack: () -> Unit, onOpenVideo: (VideoSummary) -> Unit) {
    val app = LocalContext.current.applicationContext as MusicDecodedApplication
    val viewModel: YouTubeSearchViewModel = viewModel(factory = YouTubeSearchViewModel.factory(app))
    val state by viewModel.state.collectAsStateWithLifecycle()

    LaunchedEffect(state.openVideo) {
        state.openVideo?.let {
            viewModel.videoOpened()
            onOpenVideo(it)
        }
    }

    YouTubeSearchContent(
        state,
        onBack = onBack,
        onQueryChange = viewModel::onQueryChange,
        onSubmit = { viewModel.submit() },
        onSuggestion = viewModel::submit,
        onRetry = viewModel::retry,
        onLoadMore = viewModel::loadMore,
        onRetryLoadMore = viewModel::retryLoadMore,
        onOpenVideo = onOpenVideo,
        onPaste = viewModel::openPastedUrl,
        onPasteEdited = viewModel::clearPasteError,
    )
}

@Composable
private fun YouTubeSearchContent(
    state: YouTubeSearchUiState,
    onBack: () -> Unit,
    onQueryChange: (String) -> Unit,
    onSubmit: () -> Unit,
    onSuggestion: (String) -> Unit,
    onRetry: () -> Unit,
    onLoadMore: () -> Unit,
    onRetryLoadMore: () -> Unit,
    onOpenVideo: (VideoSummary) -> Unit,
    onPaste: (String) -> Unit,
    onPasteEdited: () -> Unit,
) {
    val listState = rememberLazyListState()
    val nearEnd by remember {
        derivedStateOf {
            val info = listState.layoutInfo
            val last = info.visibleItemsInfo.lastOrNull()?.index ?: 0
            info.totalItemsCount > 0 && last >= info.totalItemsCount - 1 - PREFETCH_ROWS
        }
    }
    LaunchedEffect(nearEnd, state.canLoadMore) { if (nearEnd && state.canLoadMore) onLoadMore() }

    Column(Modifier.fillMaxSize().background(MusicDecodedColors.Canvas)) {
        Row(
            Modifier.fillMaxWidth().padding(horizontal = Spacing.gutter, vertical = Spacing.xs),
            verticalAlignment = Alignment.CenterVertically,
            horizontalArrangement = Arrangement.spacedBy(Spacing.sm),
        ) {
            MdIconButton(Icons.AutoMirrored.Rounded.ArrowBack, contentDescription = "Back", onClick = onBack)
            Text("YouTube", style = MaterialTheme.typography.titleMedium, color = MusicDecodedColors.Text)
        }
        LazyColumn(
            Modifier.weight(1f),
            state = listState,
            contentPadding = PaddingValues(horizontal = Spacing.gutter, vertical = Spacing.xs),
            verticalArrangement = Arrangement.spacedBy(Spacing.sm),
        ) {
            item { SearchField(state.query, onQueryChange, onSubmit) }
            if (state.recent.isNotEmpty()) {
                item {
                    LazyRow(horizontalArrangement = Arrangement.spacedBy(Spacing.xs)) {
                        items(state.recent) { query ->
                            MdFilterChip(query, selected = query == state.resultsFor, onClick = { onSuggestion(query) })
                        }
                    }
                }
            }
            val error = state.error
            when {
                state.loading -> item {
                    Box(Modifier.fillMaxWidth().padding(Spacing.xl), contentAlignment = Alignment.Center) {
                        CircularProgressIndicator(color = MusicDecodedColors.Primary)
                    }
                }
                error != null -> item { ErrorCard(error, onRetry) }
                state.resultsFor != null && state.results.isEmpty() -> item {
                    Text(
                        "No videos found for \"${state.resultsFor}\".",
                        style = MaterialTheme.typography.bodySmall,
                        color = MusicDecodedColors.TextSecondary,
                    )
                }
                state.results.isNotEmpty() -> {
                    item { SectionHeader("Top results") }
                    items(state.results, key = { it.videoId }) { video -> ResultRow(video) { onOpenVideo(video) } }
                    if (state.loadingMore) {
                        item {
                            Box(Modifier.fillMaxWidth().padding(Spacing.sm), contentAlignment = Alignment.Center) {
                                CircularProgressIndicator(color = MusicDecodedColors.Primary, modifier = Modifier.size(Sizes.icon))
                            }
                        }
                    }
                    state.loadMoreError?.let { item { ErrorCard(it, onRetryLoadMore) } }
                }
            }
            item { PasteUrlCard(state.pasteError, onPaste, onPasteEdited) }
        }
    }
}

@Composable
private fun SearchField(query: String, onQueryChange: (String) -> Unit, onSubmit: () -> Unit) {
    OutlinedTextField(
        value = query,
        onValueChange = onQueryChange,
        modifier = Modifier.fillMaxWidth(),
        singleLine = true,
        placeholder = { Text("Search songs on YouTube", color = MusicDecodedColors.TextMuted) },
        leadingIcon = { Icon(Icons.Rounded.Search, contentDescription = null, tint = MusicDecodedColors.TextSecondary) },
        trailingIcon = if (query.isNotEmpty()) {
            { MdIconButton(Icons.Rounded.Close, contentDescription = "Clear search", onClick = { onQueryChange("") }, small = true) }
        } else null,
        keyboardOptions = KeyboardOptions(imeAction = ImeAction.Search),
        keyboardActions = KeyboardActions(onSearch = { onSubmit() }),
        shape = Shapes.control,
        colors = OutlinedTextFieldDefaults.colors(
            // Emphasized border (DESIGN.md §3.4).
            focusedBorderColor = MusicDecodedColors.Primary,
            unfocusedBorderColor = MusicDecodedColors.Primary.copy(alpha = 0.5f),
            focusedContainerColor = MusicDecodedColors.Surface,
            unfocusedContainerColor = MusicDecodedColors.Surface,
            focusedTextColor = MusicDecodedColors.Text,
            unfocusedTextColor = MusicDecodedColors.Text,
            cursorColor = MusicDecodedColors.Primary,
        ),
    )
}

/** Thumbnail (duration badge) · title (2 lines) · channel. */
@Composable
private fun ResultRow(video: VideoSummary, onClick: () -> Unit) {
    Row(
        Modifier.fillMaxWidth().clip(Shapes.row).clickable(role = Role.Button, onClick = onClick).padding(Spacing.xxs),
        horizontalArrangement = Arrangement.spacedBy(Spacing.sm),
    ) {
        Box(Modifier.size(ThumbWidth, ThumbHeight).clip(Shapes.art)) {
            RemoteImage(video.thumbnailUrl, Modifier.matchParentSize())
            video.durationSec?.let { DurationBadge(it, Modifier.align(Alignment.BottomEnd).padding(Spacing.xxs)) }
        }
        Column(Modifier.weight(1f), verticalArrangement = Arrangement.spacedBy(Spacing.xxs)) {
            Text(video.title, style = MaterialTheme.typography.titleSmall, color = MusicDecodedColors.Text, maxLines = 2, overflow = TextOverflow.Ellipsis)
            Text(video.channelTitle, style = MaterialTheme.typography.bodySmall, color = MusicDecodedColors.TextSecondary, maxLines = 1, overflow = TextOverflow.Ellipsis)
        }
    }
}

@Composable
internal fun DurationBadge(seconds: Int, modifier: Modifier = Modifier) {
    Text(
        formatDuration(seconds.toDouble()),
        style = MaterialTheme.typography.labelSmall,
        color = MusicDecodedColors.Text,
        modifier = modifier.background(MusicDecodedColors.Ink.copy(alpha = 0.8f), Shapes.chip).padding(horizontal = Spacing.xxs),
    )
}

@Composable
private fun ErrorCard(error: SearchError, onRetry: () -> Unit) {
    ErrorState(error.message, title = error.title, onRetry = onRetry.takeIf { error.retryable })
}

/** "Paste a video URL" → videoId → detail. Needs no search, so it works when search doesn't (R-6). */
@Composable
private fun PasteUrlCard(error: String?, onPaste: (String) -> Unit, onEdited: () -> Unit) {
    var text by rememberSaveable { mutableStateOf("") }
    Column(
        Modifier
            .fillMaxWidth()
            .padding(top = Spacing.xs)
            .background(MusicDecodedColors.Surface, Shapes.card)
            .border(Sizes.hairline, MusicDecodedColors.Line, Shapes.card)
            .padding(Spacing.md),
        verticalArrangement = Arrangement.spacedBy(Spacing.sm),
    ) {
        Row(horizontalArrangement = Arrangement.spacedBy(Spacing.xs), verticalAlignment = Alignment.CenterVertically) {
            Icon(Icons.Rounded.Link, contentDescription = null, tint = MusicDecodedColors.YouTube, modifier = Modifier.size(Sizes.smallIcon))
            Text("Paste a video URL", style = MaterialTheme.typography.titleSmall, color = MusicDecodedColors.Text)
        }
        OutlinedTextField(
            value = text,
            onValueChange = { text = it; onEdited() },
            modifier = Modifier.fillMaxWidth(),
            singleLine = true,
            placeholder = { Text("youtube.com/watch?v=… or youtu.be/…", color = MusicDecodedColors.TextMuted) },
            isError = error != null,
            supportingText = error?.let { { Text(it) } },
            keyboardOptions = KeyboardOptions(keyboardType = KeyboardType.Uri, imeAction = ImeAction.Go),
            keyboardActions = KeyboardActions(onGo = { onPaste(text) }),
            shape = Shapes.control,
            colors = OutlinedTextFieldDefaults.colors(
                focusedBorderColor = MusicDecodedColors.Primary,
                unfocusedBorderColor = MusicDecodedColors.Line,
                focusedTextColor = MusicDecodedColors.Text,
                unfocusedTextColor = MusicDecodedColors.Text,
                cursorColor = MusicDecodedColors.Primary,
            ),
        )
        SecondaryButton("Open video", onClick = { onPaste(text) }, enabled = text.isNotBlank())
    }
}

@Preview
@Composable
private fun YouTubeSearchPreview() = PreviewSurface {
    YouTubeSearchContent(
        YouTubeSearchUiState(
            query = "sunny days",
            recent = listOf("sunny days", "bossa nova"),
            resultsFor = "sunny days",
            results = listOf(VideoSummary("abcdefghijk", "Sunny Days (Official Video) — a long title that wraps", "Some Channel", "", 214)),
        ),
        {}, {}, {}, {}, {}, {}, {}, {}, {}, {},
    )
}

@Preview
@Composable
private fun YouTubeSearchErrorPreview() = PreviewSurface {
    YouTubeSearchContent(YouTubeSearchUiState(query = "x", resultsFor = "x", error = SearchError.QUOTA), {}, {}, {}, {}, {}, {}, {}, {}, {}, {})
}
