package com.musicdecoded.mobile.ui.library

import androidx.compose.foundation.background
import androidx.compose.foundation.border
import androidx.compose.foundation.horizontalScroll
import androidx.compose.foundation.layout.Arrangement
import androidx.compose.foundation.layout.Box
import androidx.compose.foundation.layout.Column
import androidx.compose.foundation.layout.PaddingValues
import androidx.compose.foundation.layout.Row
import androidx.compose.foundation.layout.fillMaxSize
import androidx.compose.foundation.layout.fillMaxWidth
import androidx.compose.foundation.layout.height
import androidx.compose.foundation.layout.padding
import androidx.compose.foundation.layout.size
import androidx.compose.foundation.lazy.LazyColumn
import androidx.compose.foundation.lazy.itemsIndexed
import androidx.compose.foundation.rememberScrollState
import androidx.compose.foundation.text.BasicTextField
import androidx.compose.foundation.text.KeyboardOptions
import androidx.compose.material.icons.Icons
import androidx.compose.material.icons.rounded.Close
import androidx.compose.material.icons.rounded.Download
import androidx.compose.material.icons.rounded.MusicNote
import androidx.compose.material.icons.rounded.Search
import androidx.compose.material3.Icon
import androidx.compose.material3.IconButton
import androidx.compose.material3.MaterialTheme
import androidx.compose.material3.Text
import androidx.compose.runtime.Composable
import androidx.compose.runtime.getValue
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.graphics.SolidColor
import androidx.compose.ui.graphics.vector.ImageVector
import androidx.compose.ui.platform.LocalContext
import androidx.compose.ui.semantics.contentDescription
import androidx.compose.ui.semantics.semantics
import androidx.compose.ui.text.input.ImeAction
import androidx.compose.ui.tooling.preview.Preview
import androidx.lifecycle.compose.collectAsStateWithLifecycle
import androidx.lifecycle.viewmodel.compose.viewModel
import com.musicdecoded.mobile.MusicDecodedApplication
import com.musicdecoded.mobile.library.LibraryCounts
import com.musicdecoded.mobile.ui.components.MdFilterChip
import com.musicdecoded.mobile.ui.components.PageHeader
import com.musicdecoded.mobile.ui.components.PreviewSurface
import com.musicdecoded.mobile.ui.components.SectionHeader
import androidx.compose.ui.unit.dp
import com.musicdecoded.mobile.ui.components.Sticker
import com.musicdecoded.mobile.ui.components.block
import com.musicdecoded.mobile.ui.components.stripes
import com.musicdecoded.mobile.ui.components.topRule
import com.musicdecoded.mobile.ui.theme.MusicDecodedColors
import com.musicdecoded.mobile.ui.theme.Shadows
import com.musicdecoded.mobile.ui.theme.Shapes
import com.musicdecoded.mobile.ui.theme.Sizes
import com.musicdecoded.mobile.ui.theme.Spacing

/** DESIGN.md §3.2. */
@Composable
fun LibraryScreen(
    onOpenPlayer: (trackId: String) -> Unit,
    onOpenProcessing: (trackId: String) -> Unit,
    modifier: Modifier = Modifier,
) {
    val app = LocalContext.current.applicationContext as MusicDecodedApplication
    val viewModel: LibraryViewModel = viewModel(factory = LibraryViewModel.factory(app))
    val state by viewModel.state.collectAsStateWithLifecycle()
    val searchText by viewModel.searchText.collectAsStateWithLifecycle()
    val message by viewModel.message.collectAsStateWithLifecycle()
    SongListMessages(message, viewModel::messageShown)

    LazyColumn(
        modifier.fillMaxSize(),
        contentPadding = PaddingValues(bottom = Spacing.lg),
        verticalArrangement = Arrangement.spacedBy(Spacing.xs),
    ) {
        item {
            Column(verticalArrangement = Arrangement.spacedBy(Spacing.lg)) {
                LibraryHeader {
                    SearchField(searchText, viewModel::setQuery)
                    Row(Modifier.horizontalScroll(rememberScrollState()), horizontalArrangement = Arrangement.spacedBy(Spacing.xs)) {
                        LibraryFilter.entries.forEach { filter ->
                            MdFilterChip(filter.label, selected = state.filter == filter, onClick = { viewModel.setFilter(filter) }, onColorBlock = true)
                        }
                    }
                }
                Column(Modifier.padding(horizontal = Spacing.gutter), verticalArrangement = Arrangement.spacedBy(Spacing.lg)) {
                    SummaryCard(state.counts)
                    SectionHeader("Recently added")
                }
            }
        }
        if (!state.loading && state.rows.isEmpty()) {
            item {
                Text(
                    if (state.libraryEmpty) "Your library is empty. Decode a song from Home to get started."
                    else "No songs match.",
                    style = MaterialTheme.typography.bodySmall,
                    color = MusicDecodedColors.TextSecondary,
                    modifier = Modifier.padding(horizontal = Spacing.gutter),
                )
            }
        }
        itemsIndexed(state.rows, key = { _, row -> row.trackId }) { index, row ->
            LibrarySongRow(row, viewModel, onOpenPlayer, onOpenProcessing, Modifier.padding(horizontal = Spacing.gutter), numeral = "%02d".format(index + 1))
        }
    }
}

/**
 * The lime header block (DESIGN.md §1 color map): ink eyebrow + LIBRARY title, a violet stripe
 * texture patch in the corner, then [controls] (search, filters).
 */
@Composable
private fun LibraryHeader(controls: @Composable () -> Unit) {
    Box(
        Modifier
            .fillMaxWidth()
            .background(MusicDecodedColors.Lime)
            .topRule(),
    ) {
        Box(Modifier.align(Alignment.TopEnd).size(StripePatch).stripes())
        Column(
            Modifier.padding(start = Spacing.gutter, end = Spacing.gutter, top = Spacing.xxl, bottom = Spacing.lg),
            verticalArrangement = Arrangement.spacedBy(Spacing.md),
        ) {
            Sticker("Your collection", fill = MusicDecodedColors.Violet, contentColor = MusicDecodedColors.White)
            PageHeader("Library", color = MusicDecodedColors.OnColor)
            controls()
        }
    }
}

/** The stripe texture patch in the Library header's corner (a decoration size). */
private val StripePatch = 120.dp

/** "Songs, artists, keys..." (44 tall): a white input with a black border and hard shadow, on the lime header. */
@Composable
private fun SearchField(text: String, onChange: (String) -> Unit) {
    Row(
        Modifier
            .fillMaxWidth()
            .height(Sizes.searchField)
            .block(MusicDecodedColors.White, shadow = Shadows.sm)
            .padding(start = Spacing.sm),
        verticalAlignment = Alignment.CenterVertically,
        horizontalArrangement = Arrangement.spacedBy(Spacing.xs),
    ) {
        Icon(Icons.Rounded.Search, contentDescription = null, tint = MusicDecodedColors.OnColor, modifier = Modifier.size(Sizes.smallIcon))
        Box(Modifier.weight(1f)) {
            if (text.isEmpty()) {
                Text("Songs, artists, keys...", style = MaterialTheme.typography.bodyMedium, color = MusicDecodedColors.TextSecondaryOnLime)
            }
            BasicTextField(
                value = text,
                onValueChange = onChange,
                singleLine = true,
                textStyle = MaterialTheme.typography.bodyMedium.copy(color = MusicDecodedColors.OnColor),
                cursorBrush = SolidColor(MusicDecodedColors.Violet),
                keyboardOptions = KeyboardOptions(imeAction = ImeAction.Search),
                // The placeholder is a sibling Text, so TalkBack needs the field's own label.
                modifier = Modifier.fillMaxWidth().semantics { contentDescription = "Search your library" },
            )
        }
        if (text.isNotEmpty()) {
            IconButton(onClick = { onChange("") }) {
                Icon(Icons.Rounded.Close, contentDescription = "Clear search", tint = MusicDecodedColors.OnColor)
            }
        }
    }
}

/** Decoded · available-offline counts (the grid/list toggle is post-v1, so it's hidden). */
@Composable
private fun SummaryCard(counts: LibraryCounts) {
    Row(
        Modifier
            .fillMaxWidth()
            .block(MusicDecodedColors.Surface, shadow = Shadows.md, shadowColor = MusicDecodedColors.Violet, border = MusicDecodedColors.BorderOnInk)
            .padding(Spacing.md),
        horizontalArrangement = Arrangement.spacedBy(Spacing.lg),
    ) {
        SummaryCount(Icons.Rounded.MusicNote, counts.decoded, if (counts.decoded == 1) "song decoded" else "songs decoded")
        SummaryCount(Icons.Rounded.Download, counts.offline, "available offline")
    }
}

@Composable
private fun SummaryCount(icon: ImageVector, count: Int, label: String) {
    Row(verticalAlignment = Alignment.CenterVertically, horizontalArrangement = Arrangement.spacedBy(Spacing.xs)) {
        Icon(icon, contentDescription = null, tint = MusicDecodedColors.Primary, modifier = Modifier.size(Sizes.smallIcon))
        Text("$count", style = MaterialTheme.typography.titleMedium, color = MusicDecodedColors.Primary)
        Text(label, style = MaterialTheme.typography.bodySmall, color = MusicDecodedColors.TextSecondary)
    }
}

@Preview
@Composable
private fun LibraryHeaderPreview() = PreviewSurface {
    Column(verticalArrangement = Arrangement.spacedBy(Spacing.md)) {
        SearchField("", {})
        SummaryCard(LibraryCounts(decoded = 12, offline = 9))
    }
}
