package com.musicdecoded.mobile.sources

import androidx.navigation.NavController
import androidx.navigation.NavGraphBuilder
import androidx.navigation.NavType
import androidx.navigation.compose.composable
import androidx.navigation.navArgument

/**
 * Entry points for the local sources. Home calls [navigateToImport] from the Files / Media
 * Library cards and [navigateToRecorder] from the Record card. Each flow ends once the job has
 * started; [onDecodeStarted] gets the trackId (the app opens the Processing screen there, T-211).
 */
object SourceRoutes {
    const val RECORD = "record"
    const val IMPORT = "import/{from}"

    fun import(from: ImportFrom) = "import/${from.name}"
}

fun NavController.navigateToImport(from: ImportFrom) = navigate(SourceRoutes.import(from))

fun NavController.navigateToRecorder() = navigate(SourceRoutes.RECORD)

/** Registers the source screens in the app's nav graph (no tab bar on these). */
fun NavGraphBuilder.sourceRoutes(
    navController: NavController,
    onDecodeStarted: (trackId: String) -> Unit = { navController.popBackStack() },
) {
    composable(SourceRoutes.IMPORT, arguments = listOf(navArgument("from") { type = NavType.StringType })) { entry ->
        val from = ImportFrom.valueOf(entry.arguments?.getString("from") ?: ImportFrom.FILES.name)
        ImportScreen(from) { trackId ->
            if (trackId != null) onDecodeStarted(trackId) else navController.popBackStack()
        }
    }
    composable(SourceRoutes.RECORD) {
        RecorderScreen(onClose = { navController.popBackStack() }, onDecodeStarted = onDecodeStarted)
    }
}
