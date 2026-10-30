package com.musicdecoded.mobile.player

import androidx.compose.foundation.Canvas
import androidx.compose.foundation.background
import androidx.compose.foundation.border
import androidx.compose.foundation.horizontalScroll
import androidx.compose.foundation.layout.Arrangement
import androidx.compose.foundation.layout.Box
import androidx.compose.foundation.layout.BoxWithConstraints
import androidx.compose.foundation.layout.Column
import androidx.compose.foundation.layout.IntrinsicSize
import androidx.compose.foundation.layout.Row
import androidx.compose.foundation.layout.fillMaxHeight
import androidx.compose.foundation.layout.aspectRatio
import androidx.compose.material3.ModalBottomSheet
import androidx.compose.ui.draw.clipToBounds
import com.musicdecoded.mobile.ui.components.SongArt
import com.musicdecoded.mobile.ui.theme.ArchivoBlack
import com.musicdecoded.mobile.ui.theme.Shapes
import androidx.compose.foundation.layout.fillMaxSize
import androidx.compose.foundation.layout.fillMaxWidth
import androidx.compose.foundation.layout.height
import androidx.compose.foundation.layout.padding
import androidx.compose.foundation.layout.size
import androidx.compose.foundation.rememberScrollState
import androidx.compose.foundation.verticalScroll
import androidx.compose.material.icons.Icons
import androidx.compose.material.icons.rounded.Delete
import androidx.compose.material.icons.rounded.Favorite
import androidx.compose.material.icons.rounded.FavoriteBorder
import androidx.compose.material.icons.rounded.GraphicEq
import androidx.compose.material.icons.rounded.MoreHoriz
import androidx.compose.material.icons.rounded.MusicNote
import androidx.compose.material.icons.rounded.OpenInFull
import androidx.compose.material.icons.rounded.Refresh
import androidx.compose.material.icons.rounded.Repeat
import androidx.compose.material.icons.rounded.SkipNext
import androidx.compose.material.icons.rounded.SkipPrevious
import androidx.compose.material.icons.rounded.Speed
import androidx.compose.material3.AlertDialog
import androidx.compose.material3.CircularProgressIndicator
import androidx.compose.material3.DropdownMenu
import androidx.compose.material3.DropdownMenuItem
import androidx.compose.material3.ExperimentalMaterial3Api
import androidx.compose.material3.Icon
import androidx.compose.material3.MaterialTheme
import androidx.compose.material3.Slider
import androidx.compose.material3.Text
import androidx.compose.material3.TextButton
import androidx.compose.runtime.Composable
import androidx.compose.runtime.LaunchedEffect
import androidx.compose.runtime.State
import androidx.compose.runtime.derivedStateOf
import androidx.compose.runtime.getValue
import androidx.compose.runtime.mutableStateOf
import androidx.compose.runtime.remember
import androidx.compose.runtime.setValue
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.geometry.Size
import androidx.compose.ui.graphics.graphicsLayer
import androidx.compose.ui.platform.LocalContext
import androidx.compose.ui.platform.LocalDensity
import androidx.compose.ui.semantics.contentDescription
import androidx.compose.ui.semantics.semantics
import androidx.compose.ui.semantics.stateDescription
import androidx.compose.ui.text.style.TextOverflow
import androidx.compose.ui.tooling.preview.Preview
import androidx.compose.ui.unit.dp
import androidx.lifecycle.compose.collectAsStateWithLifecycle
import com.musicdecoded.mobile.core.Analysis
import com.musicdecoded.mobile.core.currentState
import com.musicdecoded.mobile.ui.components.ControlChip
import com.musicdecoded.mobile.ui.components.MdIconButton
import com.musicdecoded.mobile.ui.components.MetaChip
import com.musicdecoded.mobile.ui.components.SegmentedToggle
import com.musicdecoded.mobile.ui.components.block
import com.musicdecoded.mobile.ui.components.formatDuration
import com.musicdecoded.mobile.ui.components.keyName
import com.musicdecoded.mobile.ui.components.topRule
import com.musicdecoded.mobile.ui.library.SongListMessages
import com.musicdecoded.mobile.ui.theme.Borders
import com.musicdecoded.mobile.ui.theme.MusicDecodedColors
import com.musicdecoded.mobile.ui.theme.MusicDecodedTheme
import com.musicdecoded.mobile.ui.theme.Shadows
import com.musicdecoded.mobile.ui.theme.Sizes
import com.musicdecoded.mobile.ui.theme.Spacing
import com.musicdecoded.mobile.ui.theme.tabular
import kotlin.math.roundToInt

/**
 * The Player (DESIGN.md §3.7 owner revision 2): header (back · view switcher · ⋯) · title line ·
 * metadata chips · the selected view (§3.7.1) · transport. The Timeline view fits one screen with no
 * vertical scrolling; at font scale ≥ [SCROLL_FONT_SCALE] it scrolls as a fallback.
 * [onExit] leaves after a ⋯ menu Delete / Re-analyze.
 */
@Composable
fun PlayerScreen(viewModel: PlayerViewModel, onBack: () -> Unit, onExit: (PlayerExit) -> Unit = { onBack() }) {
    val state by viewModel.state.collectAsStateWithLifecycle()
    val position = viewModel.position.collectAsStateWithLifecycle()
    PlaybackClock(viewModel, state.isPlaying)
    SongListMessages(state.message, viewModel::messageShown)
    LaunchedEffect(state.exit) {
        state.exit?.let {
            viewModel.exitHandled()
            onExit(it)
        }
    }

    Column(Modifier.fillMaxSize().background(MusicDecodedColors.Canvas)) {
        val track = state.track
        PlayerHeader(
            onBack = onBack,
            menu = if (track != null) {
                { PlayerMenu(track.title, state.favorite, viewModel::toggleFavorite, viewModel::reanalyze, viewModel::delete) }
            } else null,
        ) {
            if (track != null) ViewSwitcher(state.view, viewModel::selectView)
        }
        if (track == null) {
            PlayerPlaceholder(state.loading, Modifier.weight(1f))
            return@Column
        }
        PlayerTitleLine(track.title, track.artist)
        val analysis = track.analysis
        // Read only where it's used, so a chord change doesn't recompose the whole screen.
        val chordIndex = remember(analysis, position) { derivedStateOf { currentState(position.value, analysis).chordIndex } }
        val scrolls = LocalDensity.current.fontScale >= SCROLL_FONT_SCALE && state.view == PlayerView.TIMELINE

        Column(
            Modifier.weight(1f).then(if (scrolls) Modifier.verticalScroll(rememberScrollState()) else Modifier),
            verticalArrangement = Arrangement.spacedBy(Spacing.xs),
        ) {
            MetadataRow(analysis, Modifier.padding(horizontal = Spacing.gutter, vertical = Spacing.xxs))
            when (state.view) {
                PlayerView.TIMELINE -> TimelineView(state, track, chordIndex, position, viewModel, scrolls, if (scrolls) Modifier else Modifier.weight(1f))
                PlayerView.BEAT_GRID, PlayerView.CHORD_SHEET -> {
                    val chords = state.chords
                    val middle = Modifier.weight(1f)
                    if (state.view == PlayerView.BEAT_GRID) {
                        BeatGridView(analysis, chords, position, viewModel::seekTo, middle)
                    } else {
                        ChordSheetView(analysis, chords, position, viewModel::seekTo, middle)
                    }
                    ChordPanels(state, analysis, chordIndex, viewModel, compact = true)
                    // The video is the clock for youtube tracks without local audio, so it stays on screen.
                    if (state.videoIsClock) MediaStrip(state, track, viewModel)
                }
                PlayerView.PIANO_FOCUS -> {
                    PianoFocusView(state, analysis, position, viewModel, Modifier.weight(1f))
                    if (state.videoIsClock) MediaStrip(state, track, viewModel)
                }
            }
        }
        Column(horizontalAlignment = Alignment.CenterHorizontally) {
            state.loop?.let { LoopPill(it, onClear = viewModel::clearLoop, Modifier.padding(vertical = Spacing.xxs)) }
            TransportSheet(state, position, analysis.meta.durationSec, viewModel)
        }
    }
}

/** Timeline view (§3.7.1): timeline (~32% of the middle) · control row · piano panel · media panel (the rest). */
@Composable
private fun TimelineView(
    state: PlayerUiState,
    track: PlayerTrack,
    chordIndex: State<Int?>,
    position: State<Double>,
    viewModel: PlayerViewModel,
    scrolls: Boolean,
    modifier: Modifier,
) {
    val analysis = track.analysis
    BoxWithConstraints(modifier) {
        val timelineHeight = if (scrolls) TIMELINE_MAX else (maxHeight * TIMELINE_SHARE).coerceIn(TIMELINE_MIN, TIMELINE_MAX)
        Column(if (scrolls) Modifier else Modifier.fillMaxSize(), verticalArrangement = Arrangement.spacedBy(Spacing.xs)) {
            ChordTimeline(
                analysis = analysis,
                position = { position.value },
                chords = state.chords,
                currentIndex = { chordIndex.value },
                onSeek = viewModel::seekTo,
                modifier = Modifier.height(timelineHeight),
                loop = state.loop,
                onLoopSelected = viewModel::setLoop,
            )
            ChordPanels(state, analysis, chordIndex, viewModel)
            BoxWithConstraints(if (scrolls) Modifier.height(SCROLL_MEDIA_HEIGHT) else Modifier.weight(1f).fillMaxWidth()) {
                if (maxHeight < MEDIA_MIN) {
                    MediaStrip(state, track, viewModel)
                } else {
                    MediaContent(state, track, viewModel, Modifier.fillMaxSize())
                }
            }
        }
    }
}

/** Control row + piano panel for the chord under the playhead; the only part that recomposes on a chord change. */
@Composable
private fun ChordPanels(state: PlayerUiState, analysis: Analysis, chordIndex: State<Int?>, viewModel: PlayerViewModel, compact: Boolean = false) {
    val currentLabel = chordIndex.value?.let { analysis.chords[it].label }
    ControlRow(state, currentLabel, viewModel)
    PianoPanel(state, currentLabel, viewModel, compact)
}

/** Chords · Voicing · Transpose (DESIGN.md §3.7 item 4). */
@Composable
private fun ControlRow(state: PlayerUiState, currentLabel: String?, viewModel: PlayerViewModel) {
    Row(Modifier.padding(horizontal = Spacing.gutter).height(IntrinsicSize.Min), horizontalArrangement = Arrangement.spacedBy(Spacing.xs)) {
        ControlChip(
            "Chords",
            if (state.chordsMode == ChordsMode.BASIC) "Basic" else "Full",
            onClick = viewModel::toggleChordsMode,
            Modifier.weight(1f).fillMaxHeight(),
            compact = true,
        )
        val inversions = currentLabel?.let { state.chords.voicing(it)?.inversions } ?: 0
        VoicingChip(state.inversion, inversions, viewModel::setInversion, Modifier.weight(1f).fillMaxHeight(), compact = true)
        TransposeChip(state.transpose, viewModel::setTranspose, Modifier.weight(1f).fillMaxHeight(), compact = true)
    }
}

/**
 * Piano panel (§3.7): the active chord (name + numeral) over the fixed C3–B5 keyboard with fingering,
 * note names and inversion dots; swipe for inversions. [compact] = the 90-tall keyboard of the
 * Beat grid / Chord sheet views; otherwise 110–150 from the width.
 */
@Composable
private fun PianoPanel(state: PlayerUiState, label: String?, viewModel: PlayerViewModel, compact: Boolean = false) {
    val chords = state.chords
    // Same instance while the voicing inputs are unchanged, so the keyboard skips recomposing and redrawing.
    val voicing = remember(chords, label, state.inversion) { label?.let { chords.voicing(it, state.inversion) } }
    val description = remember(chords, label, state.inversion) { label?.let { chords.noteNames(it, state.inversion) } }
    BoxWithConstraints(
        Modifier
            .fillMaxWidth()
            .padding(start = Spacing.gutter, end = Spacing.gutter, bottom = Shadows.lg)
            // block() re-applies its receiver chain, so start it from a fresh Modifier (else the padding doubles).
            .then(Modifier.block(MusicDecodedColors.Surface, shadow = Shadows.lg, shadowColor = MusicDecodedColors.Violet))
            .padding(Spacing.xs),
    ) {
        val keysHeight = if (compact) {
            PianoKeyboardDefaults.CompactHeight
        } else {
            (maxWidth * PianoKeyboardDefaults.PANEL_HEIGHT_PER_WIDTH).coerceIn(PianoKeyboardDefaults.PanelMin, PianoKeyboardDefaults.PanelMax)
        }
        Column(verticalArrangement = Arrangement.spacedBy(Spacing.xxs)) {
            Row(
                Modifier.fillMaxWidth(),
                horizontalArrangement = Arrangement.spacedBy(Spacing.xs, Alignment.CenterHorizontally),
                verticalAlignment = Alignment.Bottom,
            ) {
                Text(
                    label?.let(chords::name)?.ifEmpty { null } ?: "—",
                    style = if (compact) MaterialTheme.typography.titleSmall.copy(fontFamily = ArchivoBlack) else MaterialTheme.typography.headlineSmall,
                    color = MusicDecodedColors.Chord,
                    maxLines = 1,
                    overflow = TextOverflow.Ellipsis,
                    modifier = Modifier.alignByBaseline(),
                )
                label?.let(chords::numeral)?.let {
                    Text(it, style = MaterialTheme.typography.titleSmall, color = MusicDecodedColors.TextSecondary, modifier = Modifier.alignByBaseline())
                }
            }
            FixedPianoKeyboard(
                voicing = voicing,
                noteName = chords::noteName,
                keysHeight = keysHeight,
                onStepInversion = viewModel::stepInversion,
                inversion = voicing?.let { state.inversion.coerceAtMost(it.inversions - 1) } ?: 0,
                showNoteNames = state.showNoteNames,
                description = description,
            )
        }
    }
}

/** Media panel content (§3.7): the YouTube video (16:9) or the artwork, letterboxed on ink. */
@Composable
private fun MediaContent(state: PlayerUiState, track: PlayerTrack, viewModel: PlayerViewModel, modifier: Modifier = Modifier) {
    Box(modifier.background(MusicDecodedColors.Ink).clipToBounds(), contentAlignment = Alignment.Center) {
        val videoId = track.videoId
        if (state.videoAvailable && videoId != null) {
            PlayerVideoPanel(viewModel, videoId, track.analysis.meta.durationSec)
        } else {
            SongArt(track.title, Modifier.fillMaxHeight().aspectRatio(1f, matchHeightConstraintsFirst = true), track.thumbnailUrl, style = MaterialTheme.typography.displayLarge)
        }
    }
}

/**
 * Collapsed media panel (§3.7: less than [MEDIA_MIN] left): a 56-tall strip with the artwork (or the
 * live video when it's the clock) + title + "Show video" / "Show artwork", which opens it as a sheet.
 */
@OptIn(ExperimentalMaterial3Api::class)
@Composable
private fun MediaStrip(state: PlayerUiState, track: PlayerTrack, viewModel: PlayerViewModel) {
    var sheet by remember { mutableStateOf(false) }
    val video = state.videoAvailable && track.videoId != null
    Row(
        Modifier.fillMaxWidth().height(Sizes.songArt).padding(horizontal = Spacing.gutter),
        verticalAlignment = Alignment.CenterVertically,
        horizontalArrangement = Arrangement.spacedBy(Spacing.sm),
    ) {
        val videoId = track.videoId
        // One WebView per video at a time: while the sheet is open the strip shows the artwork.
        if (video && state.videoIsClock && !sheet && videoId != null) {
            Box(Modifier.fillMaxHeight().aspectRatio(VIDEO_ASPECT)) { PlayerVideoPanel(viewModel, videoId, track.analysis.meta.durationSec, badge = false) }
        } else {
            SongArt(track.title, Modifier.size(Sizes.songArt), track.thumbnailUrl)
        }
        Text(track.title, style = MaterialTheme.typography.labelLarge, color = MusicDecodedColors.Text, maxLines = 1, overflow = TextOverflow.Ellipsis, modifier = Modifier.weight(1f))
        TextButton(onClick = { sheet = true }) {
            Text(if (video) "Show video" else "Show artwork", color = MusicDecodedColors.Primary, style = MaterialTheme.typography.labelLarge)
        }
    }
    if (sheet) {
        ModalBottomSheet(onDismissRequest = { sheet = false }, containerColor = MusicDecodedColors.Surface, shape = Shapes.sheet) {
            MediaContent(state, track, viewModel, Modifier.fillMaxWidth().aspectRatio(VIDEO_ASPECT).padding(bottom = Spacing.xl))
        }
    }
}

/** Timeline height: this share of the space between the metadata row and the transport, clamped (DESIGN.md §3.7). */
private const val TIMELINE_SHARE = 0.32f
private val TIMELINE_MIN = 150.dp
private val TIMELINE_MAX = 220.dp

/** Below this the media panel collapses to a strip (DESIGN.md §3.7). */
private val MEDIA_MIN = 120.dp
private const val VIDEO_ASPECT = 16f / 9f

/** At this font scale the Timeline view may scroll (DESIGN.md §3.7 accessibility); the media panel then gets a fixed height. */
private const val SCROLL_FONT_SCALE = 1.5f
private val SCROLL_MEDIA_HEIGHT = 200.dp

/** Header ⋯: Favorite · Re-analyze · Delete (with confirmation), like the song rows. */
@Composable
private fun PlayerMenu(title: String, favorite: Boolean, onFavorite: () -> Unit, onReanalyze: () -> Unit, onDelete: () -> Unit) {
    var open by remember { mutableStateOf(false) }
    var confirmDelete by remember { mutableStateOf(false) }
    Box {
        MdIconButton(Icons.Rounded.MoreHoriz, contentDescription = "More options", onClick = { open = true })
        DropdownMenu(expanded = open, onDismissRequest = { open = false }, containerColor = MusicDecodedColors.SurfaceRaised) {
            DropdownMenuItem(
                text = { Text(if (favorite) "Remove from favorites" else "Favorite") },
                leadingIcon = { Icon(if (favorite) Icons.Rounded.Favorite else Icons.Rounded.FavoriteBorder, null) },
                onClick = {
                    open = false
                    onFavorite()
                },
            )
            DropdownMenuItem(
                text = { Text("Re-analyze") },
                leadingIcon = { Icon(Icons.Rounded.Refresh, null) },
                onClick = {
                    open = false
                    onReanalyze()
                },
            )
            DropdownMenuItem(
                text = { Text("Delete", color = MusicDecodedColors.Record) },
                leadingIcon = { Icon(Icons.Rounded.Delete, null, tint = MusicDecodedColors.Record) },
                onClick = {
                    open = false
                    confirmDelete = true
                },
            )
        }
    }
    if (confirmDelete) {
        AlertDialog(
            onDismissRequest = { confirmDelete = false },
            containerColor = MusicDecodedColors.Surface,
            title = { Text("Delete “$title”?") },
            text = {
                Text(
                    "It's removed from your library and its chord timeline won't show up here anymore.",
                    style = MaterialTheme.typography.bodySmall,
                    color = MusicDecodedColors.TextSecondary,
                )
            },
            confirmButton = {
                TextButton(onClick = {
                    confirmDelete = false
                    onDelete()
                }) { Text("Delete", color = MusicDecodedColors.Record) }
            },
            dismissButton = {
                TextButton(onClick = { confirmDelete = false }) { Text("Cancel", color = MusicDecodedColors.TextSecondary) }
            },
        )
    }
}

@Composable
private fun PlayerPlaceholder(loading: Boolean, modifier: Modifier = Modifier) {
    Box(modifier.fillMaxSize().padding(Spacing.gutter), contentAlignment = Alignment.Center) {
        if (loading) {
            CircularProgressIndicator(color = MusicDecodedColors.Primary)
        } else {
            Text(
                "This transcription isn't on this device.",
                style = MaterialTheme.typography.bodySmall,
                color = MusicDecodedColors.TextSecondary,
            )
        }
    }
}

/** Key · Time signature · BPM as compact chips with the value inline (room for more; scrolls sideways if needed). */
@Composable
private fun MetadataRow(analysis: Analysis, modifier: Modifier = Modifier) {
    Row(modifier.fillMaxWidth().horizontalScroll(rememberScrollState()), horizontalArrangement = Arrangement.spacedBy(Spacing.xs)) {
        MetaChip(Icons.Rounded.MusicNote, "Key", analysis.key?.let(::keyName) ?: "—")
        MetaChip(Icons.Rounded.GraphicEq, "Time", analysis.timeSignature?.let { "${it.numerator}/${it.denominator}" } ?: "—")
        MetaChip(Icons.Rounded.Speed, "Tempo", analysis.bpm?.let { "${it.roundToInt()} BPM" } ?: "—")
    }
}

/** Seek bar with times on one line, then speed · previous chord · play/pause · next chord · loop (DESIGN.md §3.7 item 6). */
@Composable
private fun TransportSheet(state: PlayerUiState, position: State<Double>, duration: Double, viewModel: PlayerViewModel) {
    Column(
        Modifier
            .fillMaxWidth()
            // Own layer: the seek bar's updates re-record only the transport.
            .graphicsLayer()
            .background(MusicDecodedColors.Surface)
            .topRule()
            .padding(horizontal = Spacing.md, vertical = Spacing.xxs),
    ) {
        SeekBar(position, duration, viewModel::seekTo, inlineTimes = true)
        Row(Modifier.fillMaxWidth(), horizontalArrangement = Arrangement.SpaceBetween, verticalAlignment = Alignment.CenterVertically) {
            SpeedButton(state.speed, onToggle = viewModel::toggleSpeed, onSelect = viewModel::setSpeed)
            MdIconButton(Icons.Rounded.SkipPrevious, contentDescription = "Previous chord", onClick = viewModel::previousChord)
            PlayButton(state.isPlaying, onClick = viewModel::togglePlay)
            MdIconButton(Icons.Rounded.SkipNext, contentDescription = "Next chord", onClick = viewModel::nextChord)
            MdIconButton(
                Icons.Rounded.Repeat,
                contentDescription = if (state.loop != null) "Stop looping" else "Loop current bar",
                onClick = viewModel::toggleLoop,
                active = state.loop != null,
            )
        }
    }
}

/**
 * Seek bar per DESIGN.md §3.7: a thin square track (played = `primary`, rest = `surfaceRaised`) with
 * a small square lime thumb, plus current / total times (under the bar, or either side of it with
 * [inlineTimes]). The touch target keeps the Slider's 48dp height.
 */
@OptIn(ExperimentalMaterial3Api::class)
@Composable
internal fun SeekBar(position: State<Double>, duration: Double, onSeek: (Double) -> Unit, inlineTimes: Boolean = false) {
    val total = duration.toFloat().coerceAtLeast(1f)
    // The clock ticks at ~60 Hz; the bar only recomposes when the thumb moves a step (1/SEEK_STEPS of
    // the track, under a pixel) and the time text once a second.
    val step by remember(position, total) { derivedStateOf { seekStep(position.value, total) } }
    val elapsed by remember(position) { derivedStateOf { formatDuration(position.value) } }
    val value = step * total / SEEK_STEPS
    val timeStyle = MaterialTheme.typography.labelSmall.tabular()
    val slider = @Composable { modifier: Modifier ->
        Slider(
            value = value,
            onValueChange = { onSeek(it.toDouble()) },
            valueRange = 0f..total,
            modifier = modifier.semantics {
                contentDescription = "Seek"
                stateDescription = "$elapsed of ${formatDuration(duration)}"
            },
            thumb = { Box(Modifier.size(SeekThumb).background(MusicDecodedColors.Primary).border(Borders.separator, MusicDecodedColors.Border)) },
            track = {
                Canvas(Modifier.fillMaxWidth().height(Sizes.progressTrack)) {
                    drawRect(MusicDecodedColors.SurfaceRaised)
                    drawRect(MusicDecodedColors.Primary, size = Size(size.width * (value / total), size.height))
                }
            },
        )
    }
    if (inlineTimes) {
        Row(verticalAlignment = Alignment.CenterVertically, horizontalArrangement = Arrangement.spacedBy(Spacing.sm)) {
            Text(elapsed, style = timeStyle, color = MusicDecodedColors.TextSecondary)
            slider(Modifier.weight(1f))
            Text(formatDuration(duration), style = timeStyle, color = MusicDecodedColors.TextMuted)
        }
    } else {
        Column {
            slider(Modifier)
            Row(Modifier.fillMaxWidth(), horizontalArrangement = Arrangement.SpaceBetween) {
                Text(elapsed, style = timeStyle, color = MusicDecodedColors.TextSecondary)
                Text(formatDuration(duration), style = timeStyle, color = MusicDecodedColors.TextMuted)
            }
        }
    }
}

/** Seek bar resolution: the thumb moves in this many steps over the track. */
internal const val SEEK_STEPS = 1000

/** The seek bar step (0…[SEEK_STEPS]) for [position] seconds of [total]. */
internal fun seekStep(position: Double, total: Float): Int =
    (position / total * SEEK_STEPS).roundToInt().coerceIn(0, SEEK_STEPS)

/** The seek bar's square thumb (a component size, not spacing). */
private val SeekThumb = 14.dp

@Preview(widthDp = 412, heightDp = 860)
@Composable
private fun PlayerScreenPreview() = MusicDecodedTheme {
    val analysis = rememberPreviewAnalysis() ?: return@MusicDecodedTheme
    val context = LocalContext.current
    val viewModel = remember {
        PlayerViewModel(
            loadTrack = { PlayerTrack(analysis, "Sunny Days", "Fixture", favorite = true, audio = null) },
            voicings = runCatching { loadChordVoicings(context) }.getOrNull(),
            createEngine = { SilentPlaybackEngine(it.analysis.meta.durationSec) },
        ).apply { seekTo(16.0) }
    }
    PlayerScreen(viewModel, onBack = {})
}
