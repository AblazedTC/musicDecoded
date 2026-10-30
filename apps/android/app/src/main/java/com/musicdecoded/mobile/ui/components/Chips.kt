package com.musicdecoded.mobile.ui.components

import androidx.compose.foundation.background
import androidx.compose.foundation.border
import androidx.compose.foundation.clickable
import androidx.compose.foundation.layout.Arrangement
import androidx.compose.foundation.layout.Box
import androidx.compose.foundation.layout.Column
import androidx.compose.foundation.layout.FlowRow
import androidx.compose.foundation.layout.FlowRowScope
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
import androidx.compose.ui.platform.LocalDensity
import androidx.compose.ui.semantics.Role
import androidx.compose.ui.semantics.clearAndSetSemantics
import androidx.compose.ui.semantics.contentDescription
import androidx.compose.ui.semantics.semantics
import androidx.compose.ui.tooling.preview.Preview
import com.musicdecoded.mobile.core.chordDisplayName
import androidx.compose.ui.unit.Dp
import androidx.compose.ui.unit.dp
import com.musicdecoded.mobile.ui.theme.Borders
import com.musicdecoded.mobile.ui.theme.MusicDecodedColors
import com.musicdecoded.mobile.ui.theme.Shadows
import com.musicdecoded.mobile.ui.theme.Shapes
import com.musicdecoded.mobile.ui.theme.Sizes
import com.musicdecoded.mobile.ui.theme.Spacing
import com.musicdecoded.mobile.ui.theme.TextStyles
import kotlin.math.roundToInt

/**
 * Square filter chip (Library filters, YouTube suggestions). On ink: selected = lime block with a
 * violet shadow. With [onColorBlock] (e.g. the Library's lime header): white chips with black
 * borders, selected = violet with white text and a black shadow.
 */
@Composable
fun MdFilterChip(label: String, selected: Boolean, onClick: () -> Unit, modifier: Modifier = Modifier, onColorBlock: Boolean = false) {
    val fill = when {
        onColorBlock && selected -> MusicDecodedColors.Violet
        onColorBlock -> MusicDecodedColors.White
        selected -> MusicDecodedColors.Primary
        else -> MusicDecodedColors.Surface
    }
    val text = when {
        onColorBlock && selected -> MusicDecodedColors.White
        onColorBlock || selected -> MusicDecodedColors.OnColor
        else -> MusicDecodedColors.Text
    }
    Box(
        modifier
            .defaultMinSize(minHeight = Sizes.minTouchTarget)
            .selectable(selected = selected, interactionSource = null, indication = null, role = Role.Tab, onClick = onClick),
        contentAlignment = Alignment.Center,
    ) {
        Text(
            label.uppercase(),
            style = TextStyles.sticker,
            color = text,
            modifier = Modifier
                .block(
                    fill,
                    shadow = if (selected) Shadows.sm else 0.dp,
                    shadowColor = if (onColorBlock) MusicDecodedColors.Border else MusicDecodedColors.Violet,
                    border = if (selected || onColorBlock) MusicDecodedColors.Border else MusicDecodedColors.BorderOnInk,
                )
                .padding(horizontal = Spacing.sm, vertical = Spacing.xs),
        )
    }
}

/**
 * A row of equal-width chips (children use `Modifier.weight(1f)`). At large font scales the chips
 * stack one per line so their labels don't break mid-word.
 */
@Composable
fun ChipRow(modifier: Modifier = Modifier, content: @Composable FlowRowScope.() -> Unit) {
    val stacked = LocalDensity.current.fontScale >= STACKED_FONT_SCALE
    FlowRow(
        modifier,
        horizontalArrangement = Arrangement.spacedBy(Spacing.xs),
        verticalArrangement = Arrangement.spacedBy(Spacing.xs),
        maxItemsInEachRow = if (stacked) 1 else Int.MAX_VALUE,
        content = content,
    )
}

private const val STACKED_FONT_SCALE = 1.5f

/** Player metadata chip: icon + overline label + value (Key · Time signature · BPM). */
@Composable
fun InfoChip(icon: ImageVector, label: String, value: String, modifier: Modifier = Modifier) {
    Row(
        modifier
            .semantics(mergeDescendants = true) {} // read as one: "Key, B minor"
            .background(MusicDecodedColors.Surface)
            .border(Borders.thick, MusicDecodedColors.BorderOnInk)
            .padding(horizontal = Spacing.xs, vertical = Spacing.xs),
        horizontalArrangement = Arrangement.spacedBy(Spacing.xs),
        verticalAlignment = Alignment.CenterVertically,
    ) {
        Icon(icon, contentDescription = null, tint = MusicDecodedColors.Primary, modifier = Modifier.size(Sizes.smallIcon))
        Column {
            Text(label.uppercase(), style = TextStyles.chipLabel)
            Text(value, style = MaterialTheme.typography.titleSmall, color = MusicDecodedColors.Text)
        }
    }
}

/** Compact player metadata chip (DESIGN.md §3.7 item 2): small icon + value inline; [label] is read, not shown. */
@Composable
fun MetaChip(icon: ImageVector, label: String, value: String, modifier: Modifier = Modifier) {
    Row(
        modifier
            .clearAndSetSemantics { contentDescription = "$label, $value" }
            .background(MusicDecodedColors.Surface)
            .border(Borders.thick, MusicDecodedColors.BorderOnInk)
            .padding(horizontal = Spacing.xs, vertical = Spacing.xxs),
        horizontalArrangement = Arrangement.spacedBy(Spacing.xxs),
        verticalAlignment = Alignment.CenterVertically,
    ) {
        Icon(icon, contentDescription = null, tint = MusicDecodedColors.Primary, modifier = Modifier.size(Sizes.smallIcon))
        Text(value, style = MaterialTheme.typography.labelLarge, color = MusicDecodedColors.Text, maxLines = 1)
    }
}

/**
 * Player control chip: label above value, plus a chevron (Chords · Voicing · Transpose). [compact]
 * (the player's control row) puts label and value on one line when the width allows, with tighter padding.
 */
@Composable
fun ControlChip(label: String, value: String, onClick: () -> Unit, modifier: Modifier = Modifier, compact: Boolean = false) {
    Row(
        modifier
            .defaultMinSize(minHeight = if (compact) Sizes.compactControl else Sizes.minTouchTarget)
            .block(MusicDecodedColors.Surface, border = MusicDecodedColors.BorderOnInk, onClick = onClick)
            .padding(horizontal = if (compact) Spacing.xs else Spacing.sm, vertical = if (compact) Spacing.xxs else Spacing.xs),
        verticalAlignment = Alignment.CenterVertically,
    ) {
        if (compact) {
            FlowRow(Modifier.weight(1f), horizontalArrangement = Arrangement.spacedBy(Spacing.xxs)) {
                Text(label.uppercase(), style = TextStyles.chipLabel, modifier = Modifier.align(Alignment.CenterVertically))
                Text(value, style = MaterialTheme.typography.labelLarge, color = MusicDecodedColors.Text, modifier = Modifier.align(Alignment.CenterVertically))
            }
        } else {
            Column(Modifier.weight(1f)) {
                Text(label.uppercase(), style = TextStyles.chipLabel)
                Text(value, style = MaterialTheme.typography.titleSmall, color = MusicDecodedColors.Text)
            }
        }
        Icon(Icons.Rounded.ExpandMore, contentDescription = null, tint = MusicDecodedColors.Primary, modifier = Modifier.size(Sizes.smallIcon))
    }
}

/** Two or more equal segments (e.g. Piano | Video). The selected segment is a lime block. */
@Composable
fun SegmentedToggle(
    options: List<String>,
    selectedIndex: Int,
    onSelect: (Int) -> Unit,
    modifier: Modifier = Modifier,
    enabled: (Int) -> Boolean = { true },
    /** Minimum segment height; the player's compact tab header passes less than the touch target. */
    segmentHeight: Dp = Sizes.minTouchTarget,
) {
    Row(
        modifier
            .background(MusicDecodedColors.Surface)
            .border(Borders.thick, MusicDecodedColors.BorderOnInk)
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
                    .defaultMinSize(minHeight = segmentHeight)
                    .background(if (selected) MusicDecodedColors.Primary else MusicDecodedColors.Surface)
                    .selectable(selected = selected, enabled = isEnabled, role = Role.Tab) { onSelect(index) },
                contentAlignment = Alignment.Center,
            ) {
                Text(
                    option,
                    style = MaterialTheme.typography.labelLarge,
                    color = when {
                        !isEnabled -> MusicDecodedColors.TextMuted
                        selected -> MusicDecodedColors.OnPrimary
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
