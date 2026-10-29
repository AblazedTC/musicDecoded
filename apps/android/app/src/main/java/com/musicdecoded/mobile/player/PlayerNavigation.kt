package com.musicdecoded.mobile.player

import android.app.Activity
import androidx.compose.runtime.Composable
import androidx.compose.runtime.DisposableEffect
import androidx.compose.runtime.LaunchedEffect
import androidx.compose.runtime.remember
import androidx.compose.runtime.rememberUpdatedState
import androidx.compose.runtime.withFrameNanos
import androidx.compose.ui.platform.LocalContext
import androidx.lifecycle.Lifecycle
import androidx.lifecycle.LifecycleEventObserver
import androidx.lifecycle.LifecycleOwner
import androidx.lifecycle.viewmodel.compose.viewModel
import androidx.navigation.NavGraphBuilder
import androidx.navigation.NavHostController
import androidx.navigation.compose.composable
import com.musicdecoded.mobile.MusicDecodedApplication
import kotlinx.coroutines.isActive

/** Pushed player routes (DESIGN.md §2: no tab bar). */
object PlayerRoutes {
    const val PLAYER = "player/{trackId}"
    fun player(trackId: String) = "player/$trackId"
}

/** `player/{trackId}` and `player/{trackId}/piano`. The piano view shares the player's ViewModel. */
fun NavGraphBuilder.playerDestinations(navController: NavHostController) {
    composable(PlayerRoutes.PLAYER) { entry ->
        val trackId = entry.arguments?.getString("trackId").orEmpty()
        val viewModel = playerViewModel(trackId, entry)
        PlayerScreen(viewModel, onBack = { navController.popBackStack() })
    }
}

@Composable
private fun playerViewModel(trackId: String, owner: androidx.lifecycle.ViewModelStoreOwner): PlayerViewModel {
    val app = LocalContext.current.applicationContext as MusicDecodedApplication
    return viewModel(viewModelStoreOwner = owner, key = "player-$trackId", factory = PlayerViewModel.factory(app, trackId))
}

/**
 * Drives the clock: [PlayerViewModel.tick] every frame while playing (~60 Hz), and pauses when the
 * app goes to the background (no background playback service in v1; rotation doesn't pause).
 */
@Composable
internal fun PlaybackClock(viewModel: PlayerViewModel, isPlaying: Boolean) {
    LaunchedEffect(viewModel, isPlaying) {
        if (isPlaying) while (isActive) withFrameNanos { viewModel.tick() }
    }
    val activity = LocalContext.current as? Activity
    val currentViewModel = rememberUpdatedState(viewModel)
    DisposableEffect(activity) {
        val lifecycle = (activity as? LifecycleOwner)?.lifecycle
        val observer = LifecycleEventObserver { _, event ->
            if (event == Lifecycle.Event.ON_STOP && activity?.isChangingConfigurations != true) currentViewModel.value.pause()
        }
        lifecycle?.addObserver(observer)
        onDispose { lifecycle?.removeObserver(observer) }
    }
}
