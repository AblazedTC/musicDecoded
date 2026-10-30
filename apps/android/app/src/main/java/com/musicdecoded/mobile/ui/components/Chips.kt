package com.musicdecoded.mobile.ui.components

import androidx.compose.foundation.background
import androidx.compose.foundation.border
import androidx.compose.foundation.clickable
import androidx.compose.foundation.layout.Arrangement
import androidx.compose.foundation.layout.Box
import androidx.compose.foundation.layout.Column
import androidx.compose.foundation.layout.Row
import androidx.compose.foundation.layout.defaultMinSize
import androidx.compose.foundation.layout.fillMaxWidth
import androidx.compose.foundation.layout.padding
import androidx.compose.foundation.layout.size
import androidx.compose.foundation.selection.selectable
import androidx.compose.foundation.selection.selectableGroup
import androidx.compose.material.icons.Icons
import androidx.compose.material.icons.rounded.ExpandMore
import androidx.compose.material.icons.rounded.GraphicEq
import androidx.compose.material.icons.rounded.MusicNote
import androidx.compose.material.icons.rounded.Speed
import androidx.compose.material3.Icon
import androidx.compose.material3.MaterialTheme
import androidx.compose.material3.Text
import androidx.compose.runtime.Composable
import androidx.compose.runtime.getValue
import androidx.compose.runtime.mutableIntStateOf
import androidx.compose.runtime.remember
import androidx.compose.runtime.setValue
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.draw.clip
import androidx.compose.ui.graphics.vector.ImageVector
import androidx.compose.ui.semantics.Role
import androidx.compose.ui.semantics.semantics
import androidx.compose.ui.tooling.preview.Preview
import com.musicdecoded.mobile.core.chordDisplayName
import com.musicdecoded.mobile.ui.theme.MusicDecodedColors
import com.musicdecoded.mobile.ui.theme.Shapes
import com.musicdecoded.mobile.ui.theme.Sizes
import com.musicdecoded.mobile.ui.theme.Spacing
import com.musicdecoded.mobile.ui.theme.TextStyles
import kotlin.math.roundToInt

/** Library filter pill (All songs · Favorites · Offline · Recordings). Selected = `primary` fill. */
@Composable
fun MdFilterChip(label: String, selected: Boolean, onClick: () -> Unit, modifier: Modifier = Modifier) {
    Box(
        modifier
            .defaultMinSize(minHeight = Sizes.minTouchTarget)
            .clip(Shapes.pill)
            .selectable(selected = selected, role = Role.Tab, onClick = onClick),
        contentAlignment = Alignment.Center,
    ) {
        Text(
            label,
            style = MaterialTheme.typography.labelLarge,
            color = if (selected) MusicDecodedColors.OnPrimary else MusicDecodedColors.TextSecondary,
            modifier = Modifier
                .background(if (selected) MusicDecodedColors.Primary else MusicDecodedColors.Surface, Shapes.pill)
                .border(Sizes.hairline, if (selected) MusicDecodedColors.Primary else MusicDecodedColors.Line, Shapes.pill)
                .padding(horizontal = Spacing.md, vertical = Spacing.xs),
        )
    }
}

/** Player metadata chip: icon + overline label + value (Key · Time signature · BPM). */
@Composable
fun InfoChip(icon: ImageVector, label: String, value: String, modifier: Modifier = Modifier) {
    Row(
        modifier
            .semantics(mergeDescendants = true) {} // read as one: "Key, B minor"
            .background(MusicDecodedColors.Surface, Shapes.chip)
            .border(Sizes.hairline, MusicDecodedColors.Line, Shapes.chip)
            .padding(horizontal = Spacing.xs, vertical = Spacing.xs),
        horizontalArrangement = Arrangement.spacedBy(Spacing.xs),
        verticalAlignment = Alignment.CenterVertically,
    ) {
        Icon(icon, contentDescription = null, tint = MusicDecodedColors.Primary, modifier = Modifier.size(Sizes.smallIcon))
        Column {
            Text(label.uppercase(), style = TextStyles.overline)
            Text(value, style = MaterialTheme.typography.titleSmall, color = MusicDecodedColors.Text)
        }
    }
}

/** Player control chip: label above value, plus a chevron (Chords · Voicing · Transpose). */
@Composable
fun ControlChip(label: String, value: String, onClick: () -> Unit, modifier: Modifier = Modifier) {
    Row(
        modifier
            .defaultMinSize(minHeight = Sizes.minTouchTarget)
            .clip(Shapes.chip)
            .background(MusicDecodedColors.Surface)
            .border(Sizes.hairline, MusicDecodedColors.Line, Shapes.chip)
            .clickable(role = Role.Button, onClick = onClick)
            .padding(horizontal = Spacing.sm, vertical = Spacing.xs),
        verticalAlignment = Alignment.CenterVertically,
    ) {
        Column(Modifier.weight(1f)) {
            Text(label.uppercase(), style = TextStyles.overline)
            Text(value, style = MaterialTheme.typography.titleSmall, color = MusicDecodedColors.Text)
        }
        Icon(Icons.Rounded.ExpandMore, contentDescription = null, tint = MusicDecodedColors.TextMuted, modifier = Modifier.size(Sizes.smallIcon))
    }
}

/** Two or more equal segments (e.g. Piano | Video). The selected segment gets `surfaceSoft`. */
@Composable
fun SegmentedToggle(
    options: List<String>,
    selectedIndex: Int,
    onSelect: (Int) -> Unit,
    modifier: Modifier = Modifier,
    enabled: (Int) -> Boolean = { true },
) {
    Row(
        modifier
            .background(MusicDecodedColors.Surface, Shapes.control)
            .border(Sizes.hairline, MusicDecodedColors.Line, Shapes.control)
            .padding(Spacing.xxs)
            .selectableGroup(),
        horizontalArrangement = Arrangement.spacedBy(Spacing.xxs),
    ) {
        options.forEachIndexed { index, option ->
            val selected = index == selectedIndex
            val isEnabled = enabled(index)
            Box(
                Modifier
                    .weight(1f)
                    .defaultMinSize(minHeight = Sizes.minTouchTarget)
                    .clip(Shapes.chip)
                    .background(if (selected) MusicDecodedColors.SurfaceSoft else MusicDecodedColors.Surface)
                    .selectable(selected = selected, enabled = isEnabled, role = Role.Tab) { onSelect(index) },
                contentAlignment = Alignment.Center,
            ) {
                Text(
                    option,
                    style = MaterialTheme.typography.labelLarge,
                    color = when {
                        !isEnabled -> MusicDecodedColors.TextMuted
                        selected -> MusicDecodedColors.Text
                        else -> MusicDecodedColors.TextSecondary
                    },
                )
            }
        }
    }
}

@Preview
@Composable
private fun FilterChipPreview() = PreviewSurface {
    Row(horizontalArrangement = Arrangement.spacedBy(Spacing.xs)) {
        MdFilterChip("All songs", selected = true, onClick = {})
        MdFilterChip("Favorites", selected = false, onClick = {})
        MdFilterChip("Offline", selected = false, onClick = {})
    }
}

@Preview
@Composable
private fun InfoChipPreview() = PreviewSurface {
    Row(horizontalArrangement = Arrangement.spacedBy(Spacing.xs)) {
        InfoChip(Icons.Rounded.MusicNote, "Key", keyName(PreviewFixture.KEY), Modifier.weight(1f))
        InfoChip(Icons.Rounded.GraphicEq, "Time", PreviewFixture.TIME_SIGNATURE, Modifier.weight(1f))
        InfoChip(Icons.Rounded.Speed, "Tempo", "${PreviewFixture.BPM.roundToInt()} BPM", Modifier.weight(1f))
    }
}

@Preview
@Composable
private fun ControlChipPreview() = PreviewSurface {
    Row(horizontalArrangement = Arrangement.spacedBy(Spacing.xs)) {
        ControlChip("Chords", "Basic", onClick = {}, Modifier.weight(1f))
        ControlChip("Voicing", "Simple", onClick = {}, Modifier.weight(1f))
        ControlChip("Transpose", "0", onClick = {}, Modifier.weight(1f))
    }
}

@Preview
@Composable
private fun SegmentedTogglePreview() = PreviewSurface {
    var selected by remember { mutableIntStateOf(0) }
    Column(verticalArrangement = Arrangement.spacedBy(Spacing.sm)) {
        Text(chordDisplayName(PreviewFixture.CURRENT_CHORD), style = TextStyles.currentChord)
        SegmentedToggle(
            options = listOf("Piano", "Video"),
            selectedIndex = selected,
            onSelect = { selected = it },
            modifier = Modifier.fillMaxWidth(),
            enabled = { it == 0 }, // Video is disabled unless source = youtube; the fixture is a file.
        )
    }
}
