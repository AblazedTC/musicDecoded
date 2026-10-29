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
import androidx.compose.foundation.lazy.items
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
import com.musicdecoded.mobile.ui.theme.MusicDecodedColors
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
        contentPadding = PaddingValues(horizontal = Spacing.gutter, vertical = Spacing.lg),
        verticalArrangement = Arrangement.spacedBy(Spacing.xs),
    ) {
        item {
            Column(verticalArrangement = Arrangement.spacedBy(Spacing.md)) {
                PageHeader("Library", overline = "Your collection")
                SearchField(searchText, viewModel::setQuery)
                Row(Modifier.horizontalScroll(rememberScrollState()), horizontalArrangement = Arrangement.spacedBy(Spacing.xs)) {
                    LibraryFilter.entries.forEach { filter ->
                        MdFilterChip(filter.label, selected = state.filter == filter, onClick = { viewModel.setFilter(filter) })
                    }
                }
                SummaryCard(state.counts)
                SectionHeader("Recently added")
            }
        }
        if (!state.loading && state.rows.isEmpty()) {
            item {
                Text(
                    if (state.libraryEmpty) "Your library is empty. Decode a song from Home to get started."
                    else "No songs match.",
                    style = MaterialTheme.typography.bodySmall,
                    color = MusicDecodedColors.TextSecondary,
                )
            }
        }
        items(state.rows, key = { it.trackId }) { row ->
            LibrarySongRow(row, viewModel, onOpenPlayer, onOpenProcessing)
        }
    }
}

/** "Songs, artists, keys..." (44 tall, `surface` fill). */
@Composable
private fun SearchField(text: String, onChange: (String) -> Unit) {
    Row(
        Modifier
            .fillMaxWidth()
            .height(Sizes.searchField)
            .background(MusicDecodedColors.Surface, Shapes.control)
            .border(Sizes.hairline, MusicDecodedColors.Line, Shapes.control)
            .padding(start = Spacing.sm),
        verticalAlignment = Alignment.CenterVertically,
        horizontalArrangement = Arrangement.spacedBy(Spacing.xs),
    ) {
        Icon(Icons.Rounded.Search, contentDescription = null, tint = MusicDecodedColors.TextMuted, modifier = Modifier.size(Sizes.smallIcon))
        Box(Modifier.weight(1f)) {
            if (text.isEmpty()) {
                Text("Songs, artists, keys...", style = MaterialTheme.typography.bodyMedium, color = MusicDecodedColors.TextMuted)
            }
            BasicTextField(
                value = text,
                onValueChange = onChange,
                singleLine = true,
                textStyle = MaterialTheme.typography.bodyMedium.copy(color = MusicDecodedColors.Text),
                cursorBrush = SolidColor(MusicDecodedColors.Primary),
                keyboardOptions = KeyboardOptions(imeAction = ImeAction.Search),
                modifier = Modifier.fillMaxWidth(),
            )
        }
        if (text.isNotEmpty()) {
            IconButton(onClick = { onChange("") }) {
                Icon(Icons.Rounded.Close, contentDescription = "Clear search", tint = MusicDecodedColors.TextSecondary)
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
            .background(MusicDecodedColors.Surface, Shapes.row)
            .border(Sizes.hairline, MusicDecodedColors.Line, Shapes.row)
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
        Text("$count", style = MaterialTheme.typography.titleSmall, color = MusicDecodedColors.Text)
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
