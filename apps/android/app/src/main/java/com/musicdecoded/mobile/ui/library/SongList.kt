package com.musicdecoded.mobile.ui.library

import com.musicdecoded.mobile.ui.paywall.LocalOpenPaywall
import com.musicdecoded.mobile.ui.paywall.opensPaywall
import android.widget.Toast
import androidx.compose.foundation.layout.Box
import androidx.compose.material.icons.Icons
import androidx.compose.material.icons.rounded.Delete
import androidx.compose.material.icons.rounded.Favorite
import androidx.compose.material.icons.rounded.FavoriteBorder
import androidx.compose.material.icons.rounded.Refresh
import androidx.compose.material3.AlertDialog
import androidx.compose.material3.DropdownMenu
import androidx.compose.material3.DropdownMenuItem
import androidx.compose.material3.Icon
import androidx.compose.material3.MaterialTheme
import androidx.compose.material3.Text
import androidx.compose.material3.TextButton
import androidx.compose.runtime.Composable
import androidx.compose.runtime.LaunchedEffect
import androidx.compose.runtime.getValue
import androidx.compose.runtime.mutableStateOf
import androidx.compose.runtime.remember
import androidx.compose.runtime.setValue
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.platform.LocalContext
import androidx.lifecycle.ViewModel
import androidx.lifecycle.viewModelScope
import com.musicdecoded.mobile.MusicDecodedApplication
import com.musicdecoded.mobile.library.LibraryItem
import com.musicdecoded.mobile.library.LibraryStatus
import com.musicdecoded.mobile.library.TranscriptionRepository
import com.musicdecoded.mobile.network.Job
import com.musicdecoded.mobile.ui.components.SongRow
import com.musicdecoded.mobile.ui.components.SongRowStatus
import com.musicdecoded.mobile.ui.components.formatDuration
import com.musicdecoded.mobile.ui.theme.MusicDecodedColors
import kotlinx.coroutines.CancellationException
import kotlinx.coroutines.flow.MutableStateFlow
import kotlinx.coroutines.flow.StateFlow
import kotlinx.coroutines.flow.asStateFlow
import kotlinx.coroutines.launch
import java.time.Instant
import kotlin.math.roundToInt

/** What one song row shows (Home and Library). */
data class SongRowState(
    val trackId: String,
    val title: String,
    val artist: String?,
    /** `duration · key · BPM`. */
    val meta: String,
    val status: SongRowStatus,
    /** 0–100 while analyzing, once the job has reported progress. */
    val progress: Int?,
    /** The audio is on this device. */
    val offline: Boolean,
    val favorite: Boolean,
    /** YouTube thumbnail; `null` shows the initial-letter art. */
    val thumbnailUrl: String? = null,
)

fun LibraryItem.toRowState(job: Job?): SongRowState = SongRowState(
    trackId = trackId,
    title = title,
    artist = artist,
    meta = songMeta(durationSec, key, bpm),
    status = when (status) {
        LibraryStatus.ANALYZING -> SongRowStatus.ANALYZING
        LibraryStatus.READY -> SongRowStatus.READY
        LibraryStatus.FAILED -> SongRowStatus.FAILED
    },
    progress = job?.progress?.coerceIn(0, 100),
    offline = localAudioPath != null,
    favorite = favorite,
    thumbnailUrl = thumbnailUrl,
)

/** `4:23 · F♯ minor · 136 BPM` from the list columns, skipping unknown parts. */
fun songMeta(durationSec: Double?, key: String?, bpm: Double?): String = listOfNotNull(
    durationSec?.let(::formatDuration),
    key?.let(::displayKey),
    bpm?.let { "${it.roundToInt()} BPM" },
).joinToString(" · ")

/** `F# minor` → `F♯ minor`, `Bb major` → `B♭ major` (only the tonic is touched). */
fun displayKey(key: String): String {
    val tonic = key.substringBefore(' ')
    return tonic.replace("#", "♯").replace("b", "♭") + key.removePrefix(tonic)
}

/** The ⋯-menu actions shared by Home and Library. Lambdas so ViewModel tests can use fakes. */
class SongActions(
    val delete: suspend (trackId: String) -> Unit,
    val setFavorite: suspend (trackId: String, favorite: Boolean) -> Unit,
    val reanalyze: suspend (trackId: String) -> Unit,
) {
    companion object {
        fun from(app: MusicDecodedApplication): SongActions {
            val dao = app.libraryDatabase.libraryDao()
            return SongActions(
                delete = {
                    dao.markDeleted(it, Instant.now())
                    app.nowPlaying.dismiss(it) // a deleted track can't stay in the mini player
                },
                setFavorite = { id, favorite -> dao.setFavorite(id, favorite, Instant.now()) },
                reanalyze = { app.transcriptionRepository.retry(it) },
            )
        }
    }
}

/** Base for screens that list songs: the row actions plus a one-shot error message. */
abstract class SongListViewModel(private val actions: SongActions) : ViewModel() {
    private val _message = MutableStateFlow<String?>(null)

    /** A user-facing error to show once (then call [messageShown]). */
    val message: StateFlow<String?> = _message.asStateFlow()

    fun messageShown() {
        _message.value = null
    }

    fun delete(trackId: String) = run { actions.delete(trackId) }

    fun setFavorite(trackId: String, favorite: Boolean) = run { actions.setFavorite(trackId, favorite) }

    /** Retry for a failed row, or Re-analyze from the menu. */
    fun reanalyze(trackId: String) = run { actions.reanalyze(trackId) }

    private fun run(block: suspend () -> Unit) {
        viewModelScope.launch {
            try {
                block()
            } catch (e: CancellationException) {
                throw e
            } catch (e: Exception) {
                _message.value = TranscriptionRepository.userMessage(e)
            }
        }
    }
}

/** Shows [SongListViewModel.message] as a toast. */
@Composable
fun SongListMessages(message: String?, onShown: () -> Unit) {
    val context = LocalContext.current
    val openPaywall = LocalOpenPaywall.current
    LaunchedEffect(message) {
        if (message != null) {
            // The daily plan limit (a re-analyze the user just asked for) opens the paywall.
            if (openPaywall != null && opensPaywall(message)) openPaywall()
            else Toast.makeText(context, message, Toast.LENGTH_LONG).show()
            onShown()
        }
    }
}

/**
 * A [SongRow] with its ⋯ menu (Delete with confirmation, Favorite, Re-analyze) wired to [viewModel].
 * Ready rows open the player; analyzing rows open the Processing screen.
 */
@Composable
fun LibrarySongRow(
    row: SongRowState,
    viewModel: SongListViewModel,
    onOpenPlayer: (trackId: String) -> Unit,
    onOpenProcessing: (trackId: String) -> Unit,
    modifier: Modifier = Modifier,
) {
    var menuOpen by remember { mutableStateOf(false) }
    var confirmDelete by remember { mutableStateOf(false) }
    Box(modifier) {
        SongRow(
            title = row.title,
            artist = row.artist,
            meta = row.meta,
            onClick = {
                when (row.status) {
                    SongRowStatus.READY -> onOpenPlayer(row.trackId)
                    SongRowStatus.ANALYZING -> onOpenProcessing(row.trackId)
                    SongRowStatus.FAILED -> Unit
                }
            },
            onMore = { menuOpen = true },
            status = row.status,
            progress = row.progress,
            offline = row.offline,
            onRetry = { viewModel.reanalyze(row.trackId) },
            thumbnailUrl = row.thumbnailUrl,
        )
        // Anchored to the row's end, under the ⋯ button.
        Box(Modifier.align(Alignment.TopEnd)) {
            DropdownMenu(expanded = menuOpen, onDismissRequest = { menuOpen = false }, containerColor = MusicDecodedColors.SurfaceRaised) {
                DropdownMenuItem(
                    text = { Text(if (row.favorite) "Remove from favorites" else "Favorite") },
                    leadingIcon = { Icon(if (row.favorite) Icons.Rounded.Favorite else Icons.Rounded.FavoriteBorder, null) },
                    onClick = {
                        menuOpen = false
                        viewModel.setFavorite(row.trackId, !row.favorite)
                    },
                )
                DropdownMenuItem(
                    text = { Text("Re-analyze") },
                    leadingIcon = { Icon(Icons.Rounded.Refresh, null) },
                    enabled = row.status != SongRowStatus.ANALYZING,
                    onClick = {
                        menuOpen = false
                        viewModel.reanalyze(row.trackId)
                    },
                )
                DropdownMenuItem(
                    text = { Text("Delete", color = MusicDecodedColors.Record) },
                    leadingIcon = { Icon(Icons.Rounded.Delete, null, tint = MusicDecodedColors.Record) },
                    onClick = {
                        menuOpen = false
                        confirmDelete = true
                    },
                )
            }
        }
    }
    if (confirmDelete) {
        AlertDialog(
            onDismissRequest = { confirmDelete = false },
            containerColor = MusicDecodedColors.Surface,
            title = { Text("Delete “${row.title}”?") },
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
                    viewModel.delete(row.trackId)
                }) { Text("Delete", color = MusicDecodedColors.Record) }
            },
            dismissButton = {
                TextButton(onClick = { confirmDelete = false }) { Text("Cancel", color = MusicDecodedColors.TextSecondary) }
            },
        )
    }
}
