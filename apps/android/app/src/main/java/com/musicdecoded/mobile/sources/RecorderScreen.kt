package com.musicdecoded.mobile.sources

import android.Manifest
import android.content.Intent
import android.content.pm.PackageManager
import android.net.Uri
import android.provider.Settings
import androidx.activity.compose.rememberLauncherForActivityResult
import androidx.activity.result.contract.ActivityResultContracts
import androidx.compose.foundation.Canvas
import androidx.compose.foundation.background
import androidx.compose.foundation.border
import androidx.compose.foundation.clickable
import androidx.compose.foundation.layout.Arrangement
import androidx.compose.foundation.layout.Box
import androidx.compose.foundation.layout.Column
import androidx.compose.foundation.layout.Row
import androidx.compose.foundation.layout.Spacer
import androidx.compose.foundation.layout.aspectRatio
import androidx.compose.foundation.layout.fillMaxSize
import androidx.compose.foundation.layout.fillMaxWidth
import androidx.compose.foundation.layout.height
import androidx.compose.foundation.layout.padding
import androidx.compose.foundation.layout.size
import androidx.compose.foundation.rememberScrollState
import androidx.compose.foundation.shape.CircleShape
import androidx.compose.foundation.verticalScroll
import androidx.compose.material.icons.Icons
import androidx.compose.material.icons.automirrored.rounded.ArrowBack
import androidx.compose.material.icons.rounded.Check
import androidx.compose.material.icons.rounded.Delete
import androidx.compose.material.icons.rounded.Lightbulb
import androidx.compose.material.icons.rounded.Mic
import androidx.compose.material.icons.rounded.MicOff
import androidx.compose.material.icons.rounded.Pause
import androidx.compose.material3.AlertDialog
import androidx.compose.material3.CircularProgressIndicator
import androidx.compose.material3.Icon
import androidx.compose.material3.MaterialTheme
import androidx.compose.material3.OutlinedTextField
import androidx.compose.material3.OutlinedTextFieldDefaults
import androidx.compose.material3.Text
import androidx.compose.material3.TextButton
import androidx.compose.runtime.Composable
import androidx.compose.runtime.LaunchedEffect
import androidx.compose.runtime.getValue
import androidx.compose.runtime.mutableIntStateOf
import androidx.compose.runtime.mutableStateOf
import androidx.compose.runtime.saveable.rememberSaveable
import androidx.compose.runtime.setValue
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.geometry.CornerRadius
import androidx.compose.ui.geometry.Offset
import androidx.compose.ui.geometry.Size
import androidx.compose.ui.platform.LocalContext
import androidx.compose.ui.semantics.Role
import androidx.compose.ui.semantics.contentDescription
import androidx.compose.ui.semantics.semantics
import androidx.compose.ui.text.font.FontWeight
import androidx.compose.ui.tooling.preview.Preview
import androidx.core.content.ContextCompat
import androidx.lifecycle.Lifecycle
import androidx.lifecycle.compose.LifecycleEventEffect
import androidx.lifecycle.compose.collectAsStateWithLifecycle
import androidx.lifecycle.viewmodel.compose.viewModel
import com.musicdecoded.mobile.MusicDecodedApplication
import com.musicdecoded.mobile.ui.components.MdIconButton
import com.musicdecoded.mobile.ui.components.PolicyNote
import com.musicdecoded.mobile.ui.components.PreviewSurface
import com.musicdecoded.mobile.ui.components.PrimaryButton
import com.musicdecoded.mobile.ui.theme.MusicDecodedColors
import com.musicdecoded.mobile.ui.theme.Shapes
import com.musicdecoded.mobile.ui.theme.Sizes
import com.musicdecoded.mobile.ui.theme.Spacing
import com.musicdecoded.mobile.ui.theme.TextStyles
import com.musicdecoded.mobile.ui.theme.tabular
import kotlin.math.roundToInt

private const val NOISE_TIP =
    "Record in a quiet room, close to the instrument. Background noise and talking make chords harder to detect. Up to 10 minutes."

/** DESIGN.md §3.6. [onDecodeStarted] gets the trackId once the recording is uploaded and the job exists. */
@Composable
fun RecorderScreen(onClose: () -> Unit, onDecodeStarted: (trackId: String) -> Unit) {
    val context = LocalContext.current
    val app = context.applicationContext as MusicDecodedApplication
    val viewModel = viewModel { RecorderViewModel(AudioRecorder(app), app.transcriptionRepository, app.appScope) }
    val state by viewModel.state.collectAsStateWithLifecycle()

    fun granted() = ContextCompat.checkSelfPermission(context, Manifest.permission.RECORD_AUDIO) == PackageManager.PERMISSION_GRANTED
    var hasPermission by rememberSaveable { mutableStateOf(granted()) }
    var denials by rememberSaveable { mutableIntStateOf(0) }
    val requestPermission = rememberLauncherForActivityResult(ActivityResultContracts.RequestPermission()) { ok ->
        hasPermission = ok
        if (!ok) denials++
    }
    LaunchedEffect(Unit) { if (!hasPermission && denials == 0) requestPermission.launch(Manifest.permission.RECORD_AUDIO) }
    // Back from system settings: pick up a permission granted there.
    LifecycleEventEffect(Lifecycle.Event.ON_RESUME) { hasPermission = granted() }
    LifecycleEventEffect(Lifecycle.Event.ON_STOP) { viewModel.onBackgrounded() }
    LaunchedEffect(state.startedTrackId) { state.startedTrackId?.let(onDecodeStarted) }

    RecorderContent(
        state = state,
        hasPermission = hasPermission,
        // After a second denial Android stops showing the prompt, so send the user to settings.
        permissionAction = if (denials >= 2) "Open settings" else "Allow microphone",
        onRequestPermission = {
            if (denials >= 2) {
                context.startActivity(
                    Intent(Settings.ACTION_APPLICATION_DETAILS_SETTINGS, Uri.fromParts("package", context.packageName, null)),
                )
            } else {
                requestPermission.launch(Manifest.permission.RECORD_AUDIO)
            }
        },
        onClose = onClose,
        onToggleRecord = viewModel::toggleRecord,
        onDiscard = viewModel::discard,
        onFinish = viewModel::finish,
    )

    if (state.askTitle) {
        TitleDialog(state.defaultTitle, onSave = viewModel::save, onDismiss = viewModel::dismissTitle)
    }
}

@Composable
private fun RecorderContent(
    state: RecorderUiState,
    hasPermission: Boolean,
    permissionAction: String,
    onRequestPermission: () -> Unit,
    onClose: () -> Unit,
    onToggleRecord: () -> Unit,
    onDiscard: () -> Unit,
    onFinish: () -> Unit,
) {
    Column(
        Modifier
            .fillMaxSize()
            .background(MusicDecodedColors.Canvas)
            .verticalScroll(rememberScrollState())
            .padding(horizontal = Spacing.gutter, vertical = Spacing.xs),
        horizontalAlignment = Alignment.CenterHorizontally,
    ) {
        Row(Modifier.fillMaxWidth(), verticalAlignment = Alignment.CenterVertically) {
            MdIconButton(Icons.AutoMirrored.Rounded.ArrowBack, "Back", onClose)
            Text(
                "Record Audio",
                style = MaterialTheme.typography.titleMedium,
                color = MusicDecodedColors.Text,
                modifier = Modifier.weight(1f).padding(start = Spacing.xs),
            )
            TextButton(onClick = onFinish, enabled = state.hasRecording && state.phase != RecorderPhase.STARTING) {
                Text("Save", style = MaterialTheme.typography.labelLarge, color = if (state.hasRecording) MusicDecodedColors.Primary else MusicDecodedColors.TextMuted)
            }
        }
        Spacer(Modifier.height(Spacing.xxxl))

        if (!hasPermission) {
            PermissionDenied(permissionAction, onRequestPermission)
            return@Column
        }

        StateLabel(state.phase)
        Text(
            InputLevel.formatTimer(state.elapsedMs),
            style = MaterialTheme.typography.displayMedium.copy(fontWeight = FontWeight.Light).tabular(),
            color = MusicDecodedColors.Text,
            modifier = Modifier.padding(vertical = Spacing.md),
        )
        LiveWaveform(state.waveform, active = state.phase == RecorderPhase.RECORDING)
        Spacer(Modifier.height(Spacing.lg))
        LevelMeter(state.levelDb, active = state.phase == RecorderPhase.RECORDING)
        Spacer(Modifier.height(Spacing.xxl))

        Row(horizontalArrangement = Arrangement.spacedBy(Spacing.xxxl), verticalAlignment = Alignment.CenterVertically) {
            LabeledAction("Discard") {
                MdIconButton(Icons.Rounded.Delete, "Discard recording", onDiscard)
            }
            RecordButton(state.phase, onToggleRecord)
            LabeledAction("Finish") {
                MdIconButton(Icons.Rounded.Check, "Finish recording", onFinish, active = state.hasRecording)
            }
        }
        Spacer(Modifier.height(Spacing.xl))

        if (state.phase == RecorderPhase.STARTING) {
            Row(horizontalArrangement = Arrangement.spacedBy(Spacing.sm), verticalAlignment = Alignment.CenterVertically) {
                CircularProgressIndicator(Modifier.size(Sizes.icon), color = MusicDecodedColors.Primary)
                Text("Uploading…", style = MaterialTheme.typography.bodySmall, color = MusicDecodedColors.TextSecondary)
            }
            Spacer(Modifier.height(Spacing.md))
        }
        state.error?.let {
            Text(it, style = MaterialTheme.typography.bodySmall, color = MusicDecodedColors.Record)
            Spacer(Modifier.height(Spacing.md))
        }
        PolicyNote(NOISE_TIP, icon = Icons.Rounded.Lightbulb)
    }
}

@Composable
private fun StateLabel(phase: RecorderPhase) {
    val (label, color) = when (phase) {
        RecorderPhase.RECORDING -> "Recording" to MusicDecodedColors.Record
        RecorderPhase.PAUSED -> "Paused" to MusicDecodedColors.Chord
        RecorderPhase.STOPPED, RecorderPhase.STARTING -> "Recording finished" to MusicDecodedColors.TextSecondary
        RecorderPhase.IDLE -> "Microphone ready" to MusicDecodedColors.Primary
    }
    Row(horizontalArrangement = Arrangement.spacedBy(Spacing.xs), verticalAlignment = Alignment.CenterVertically) {
        Box(Modifier.size(Spacing.xs).background(color, CircleShape))
        Text(label, style = MaterialTheme.typography.bodySmall, color = color)
    }
}

/** Recent input levels as centered bars (dimmed when not recording). */
@Composable
private fun LiveWaveform(levels: List<Float>, active: Boolean) {
    val barColor = if (active) MusicDecodedColors.WavePlayed else MusicDecodedColors.WaveIdle
    Canvas(
        Modifier
            .fillMaxWidth()
            .aspectRatio(WAVEFORM_ASPECT)
            .background(MusicDecodedColors.TimelineBg, Shapes.card)
            .border(Sizes.hairline, MusicDecodedColors.TimelineBorder, Shapes.card)
            .padding(Spacing.md),
    ) {
        val slot = size.width / RecorderViewModel.WAVEFORM_BARS
        val barWidth = slot * 0.6f
        val minHeight = barWidth
        // Right-aligned: the newest level is at the right edge.
        val offset = RecorderViewModel.WAVEFORM_BARS - levels.size
        for (i in 0 until RecorderViewModel.WAVEFORM_BARS) {
            val level = levels.getOrNull(i - offset) ?: 0f
            val height = (level * size.height).coerceAtLeast(minHeight)
            drawRoundRect(
                color = barColor,
                topLeft = Offset(i * slot + (slot - barWidth) / 2, (size.height - height) / 2),
                size = Size(barWidth, height),
                cornerRadius = CornerRadius(barWidth / 2),
            )
        }
    }
}

/** 16-segment input level meter with the dB value. */
@Composable
private fun LevelMeter(db: Float, active: Boolean) {
    val lit = if (active) InputLevel.litSegments(db) else 0
    Column(Modifier.fillMaxWidth(), verticalArrangement = Arrangement.spacedBy(Spacing.xs)) {
        Row(Modifier.fillMaxWidth()) {
            Text("INPUT LEVEL", style = TextStyles.overline, modifier = Modifier.weight(1f))
            Text(
                if (active) "${db.roundToInt()} dB" else "— dB",
                style = MaterialTheme.typography.labelSmall.tabular(),
                color = MusicDecodedColors.TextSecondary,
            )
        }
        Row(
            Modifier.fillMaxWidth().semantics { contentDescription = "Input level ${db.roundToInt()} dB" },
            horizontalArrangement = Arrangement.spacedBy(Spacing.xxs),
        ) {
            repeat(InputLevel.SEGMENTS) { index ->
                val color = when {
                    index >= lit -> MusicDecodedColors.SurfaceSoft
                    index >= InputLevel.SEGMENTS - 2 -> MusicDecodedColors.Record // clipping
                    index >= InputLevel.SEGMENTS - 5 -> MusicDecodedColors.Chord
                    else -> MusicDecodedColors.Primary
                }
                Box(Modifier.weight(1f).aspectRatio(SEGMENT_ASPECT).background(color, Shapes.chip))
            }
        }
    }
}

/** The 84pt red ring: record, pause, or resume. */
@Composable
private fun RecordButton(phase: RecorderPhase, onClick: () -> Unit) {
    val enabled = phase == RecorderPhase.IDLE || phase == RecorderPhase.RECORDING || phase == RecorderPhase.PAUSED
    val label = when (phase) {
        RecorderPhase.RECORDING -> "Pause"
        RecorderPhase.PAUSED -> "Resume"
        else -> "Record"
    }
    LabeledAction(label) {
        Box(
            Modifier
                .size(Sizes.recordButton)
                .border(Spacing.xxs, MusicDecodedColors.RecordRing, CircleShape)
                .padding(Spacing.xs)
                .background(if (enabled) MusicDecodedColors.Record else MusicDecodedColors.SurfaceSoft, CircleShape)
                .clickable(enabled = enabled, role = Role.Button, onClickLabel = label, onClick = onClick),
            contentAlignment = Alignment.Center,
        ) {
            Icon(
                if (phase == RecorderPhase.RECORDING) Icons.Rounded.Pause else Icons.Rounded.Mic,
                contentDescription = label,
                tint = MusicDecodedColors.Text,
                modifier = Modifier.size(Sizes.icon),
            )
        }
    }
}

@Composable
private fun LabeledAction(label: String, content: @Composable () -> Unit) {
    Column(horizontalAlignment = Alignment.CenterHorizontally, verticalArrangement = Arrangement.spacedBy(Spacing.xxs)) {
        content()
        Text(label, style = MaterialTheme.typography.labelSmall, color = MusicDecodedColors.TextSecondary)
    }
}

@Composable
private fun PermissionDenied(action: String, onAction: () -> Unit) {
    Column(horizontalAlignment = Alignment.CenterHorizontally, verticalArrangement = Arrangement.spacedBy(Spacing.md)) {
        Box(
            Modifier.size(Sizes.recordButton).background(MusicDecodedColors.RecordSourceBg, CircleShape),
            contentAlignment = Alignment.Center,
        ) {
            Icon(Icons.Rounded.MicOff, contentDescription = null, tint = MusicDecodedColors.RecordSource)
        }
        Text("Microphone access is off", style = MaterialTheme.typography.titleMedium, color = MusicDecodedColors.Text)
        Text(
            "MusicDecoded needs the microphone to record a performance and decode its chords.",
            style = MaterialTheme.typography.bodySmall,
            color = MusicDecodedColors.TextSecondary,
        )
        PrimaryButton(action, onAction, icon = Icons.Rounded.Mic)
    }
}

@Composable
private fun TitleDialog(defaultTitle: String, onSave: (String) -> Unit, onDismiss: () -> Unit) {
    var title by rememberSaveable { mutableStateOf(defaultTitle) }
    AlertDialog(
        onDismissRequest = onDismiss,
        containerColor = MusicDecodedColors.TransportBg,
        title = { Text("Name your recording", style = MaterialTheme.typography.titleMedium, color = MusicDecodedColors.Text) },
        text = {
            OutlinedTextField(
                value = title,
                onValueChange = { title = it },
                singleLine = true,
                shape = Shapes.control,
                colors = OutlinedTextFieldDefaults.colors(
                    focusedTextColor = MusicDecodedColors.Text,
                    unfocusedTextColor = MusicDecodedColors.Text,
                    focusedBorderColor = MusicDecodedColors.Primary,
                    unfocusedBorderColor = MusicDecodedColors.Line,
                    cursorColor = MusicDecodedColors.Primary,
                ),
            )
        },
        confirmButton = {
            TextButton(onClick = { onSave(title) }, enabled = title.isNotBlank()) {
                Text("Decode", style = MaterialTheme.typography.labelLarge, color = MusicDecodedColors.Primary)
            }
        },
        dismissButton = {
            TextButton(onClick = onDismiss) {
                Text("Not now", style = MaterialTheme.typography.labelLarge, color = MusicDecodedColors.TextSecondary)
            }
        },
    )
}

/** Waveform container width:height (a component size, not spacing). */
private const val WAVEFORM_ASPECT = 3f
private const val SEGMENT_ASPECT = 0.5f

@Preview
@Composable
private fun RecorderRecordingPreview() = PreviewSurface {
    RecorderContent(
        state = RecorderUiState(
            phase = RecorderPhase.RECORDING,
            elapsedMs = 83_000,
            levelDb = -14f,
            waveform = List(RecorderViewModel.WAVEFORM_BARS) { ((it * 37) % 100) / 100f },
        ),
        hasPermission = true,
        permissionAction = "Allow microphone",
        onRequestPermission = {}, onClose = {}, onToggleRecord = {}, onDiscard = {}, onFinish = {},
    )
}

@Preview
@Composable
private fun RecorderDeniedPreview() = PreviewSurface {
    RecorderContent(
        state = RecorderUiState(),
        hasPermission = false,
        permissionAction = "Allow microphone",
        onRequestPermission = {}, onClose = {}, onToggleRecord = {}, onDiscard = {}, onFinish = {},
    )
}
