package com.musicdecoded.mobile.sources.youtube

import android.net.Uri
import androidx.navigation.NavGraphBuilder
import androidx.navigation.NavHostController
import androidx.navigation.NavType
import androidx.navigation.compose.composable
import androidx.navigation.navArgument
import com.musicdecoded.mobile.network.VideoSummary

/** YouTube search and video detail (pushed, no tab bar). */
object YouTubeRoutes {
    const val SEARCH = "youtube"
    const val DETAIL = "youtube/video/{videoId}?title={title}&channel={channel}&thumb={thumb}&duration={duration}"

    fun detail(video: VideoSummary) = buildString {
        append("youtube/video/").append(Uri.encode(video.videoId))
        append("?title=").append(Uri.encode(video.title))
        append("&channel=").append(Uri.encode(video.channelTitle))
        append("&thumb=").append(Uri.encode(video.thumbnailUrl))
        append("&duration=").append(video.durationSec ?: -1)
    }
}

/**
 * Registers the YouTube screens. "Decode this song" calls [onDecodeStarted] with the trackId (the
 * app replaces the detail screen with Processing there).
 */
fun NavGraphBuilder.youTubeRoutes(
    navController: NavHostController,
    onDecodeStarted: (trackId: String) -> Unit,
    onOpenPlayer: (trackId: String) -> Unit,
    onOpenProcessing: (trackId: String) -> Unit,
) {
    composable(YouTubeRoutes.SEARCH) {
        YouTubeSearchScreen(
            onBack = { navController.popBackStack() },
            onOpenVideo = { navController.navigate(YouTubeRoutes.detail(it)) },
        )
    }
    composable(
        YouTubeRoutes.DETAIL,
        arguments = listOf(
            navArgument("videoId") { type = NavType.StringType },
            navArgument("title") { type = NavType.StringType; defaultValue = "" },
            navArgument("channel") { type = NavType.StringType; defaultValue = "" },
            navArgument("thumb") { type = NavType.StringType; defaultValue = "" },
            navArgument("duration") { type = NavType.IntType; defaultValue = -1 },
        ),
    ) { entry ->
        val args = entry.arguments
        val videoId = args?.getString("videoId").orEmpty()
        val video = VideoSummary(
            videoId = videoId,
            title = args?.getString("title")?.ifEmpty { null } ?: "YouTube video",
            channelTitle = args?.getString("channel").orEmpty(),
            thumbnailUrl = args?.getString("thumb")?.ifEmpty { null } ?: YouTubeSearchViewModel.pastedVideo(videoId).thumbnailUrl,
            durationSec = args?.getInt("duration")?.takeIf { it >= 0 },
        )
        YouTubeDetailScreen(
            video,
            onBack = { navController.popBackStack() },
            onDecodeStarted = onDecodeStarted,
            onOpenPlayer = onOpenPlayer,
            onOpenProcessing = onOpenProcessing,
        )
    }
}
