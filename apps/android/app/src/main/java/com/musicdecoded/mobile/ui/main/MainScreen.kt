package com.musicdecoded.mobile.ui.main

import androidx.compose.foundation.gestures.detectTapGestures
import androidx.compose.foundation.layout.Box
import androidx.compose.foundation.layout.padding
import androidx.compose.material3.Scaffold
import androidx.compose.runtime.Composable
import androidx.compose.runtime.getValue
import androidx.compose.ui.Modifier
import androidx.compose.ui.input.pointer.pointerInput
import androidx.navigation.NavHostController
import androidx.navigation.compose.NavHost
import androidx.navigation.compose.composable
import androidx.navigation.compose.currentBackStackEntryAsState
import androidx.navigation.compose.rememberNavController
import com.musicdecoded.mobile.BuildConfig
import com.musicdecoded.mobile.player.DebugFixtures
import com.musicdecoded.mobile.player.PlayerRoutes
import com.musicdecoded.mobile.player.playerDestinations
import com.musicdecoded.mobile.sources.ImportFrom
import com.musicdecoded.mobile.sources.navigateToImport
import com.musicdecoded.mobile.sources.navigateToRecorder
import com.musicdecoded.mobile.sources.sourceRoutes
import com.musicdecoded.mobile.ui.components.AppTab
import com.musicdecoded.mobile.ui.components.MdBottomBar
import com.musicdecoded.mobile.ui.home.HomeNavigation
import com.musicdecoded.mobile.ui.home.HomeScreen
import com.musicdecoded.mobile.ui.library.LibraryScreen
import com.musicdecoded.mobile.ui.settings.SettingsScreen
import com.musicdecoded.mobile.ui.theme.MusicDecodedColors

/** App shell: Home · Library · Settings tabs (DESIGN.md §2). Pushed screens will hide the bar. */
@Composable
fun MainScreen() {
    val navController = rememberNavController()
    val backStackEntry by navController.currentBackStackEntryAsState()
    val currentTab = AppTab.entries.firstOrNull { it.route == backStackEntry?.destination?.route }

    Scaffold(
        containerColor = MusicDecodedColors.Canvas,
        bottomBar = {
            if (currentTab != null) MdBottomBar(selected = currentTab, onSelect = { navController.selectTab(it) })
        },
    ) { innerPadding ->
        NavHost(navController, startDestination = AppTab.HOME.route, Modifier.padding(innerPadding)) {
            composable(AppTab.HOME.route) {
                // Debug only: long-press the Home screen to open the player on the bundled fixture.
                val debugFixture = Modifier.pointerInput(Unit) {
                    detectTapGestures(onLongPress = { navController.navigate(PlayerRoutes.player(DebugFixtures.SUNNY_DAYS)) })
                }
                Box(if (BuildConfig.DEBUG) debugFixture else Modifier) {
                    HomeScreen(homeNavigation(navController))
                }
            }
            composable(AppTab.LIBRARY.route) {
                LibraryScreen(onOpenPlayer = { navController.navigate(PlayerRoutes.player(it)) }, onOpenProcessing = {})
            }
            composable(AppTab.SETTINGS.route) { SettingsScreen() }
            playerDestinations(navController)
            sourceRoutes(navController)
        }
    }
}

private fun homeNavigation(navController: NavHostController) = HomeNavigation(
    onSeeAll = { navController.selectTab(AppTab.LIBRARY) },
    // DESIGN.md §5: the header search opens Library search.
    onSearch = { navController.selectTab(AppTab.LIBRARY) },
    onProfile = { navController.selectTab(AppTab.SETTINGS) },
    onFiles = { navController.navigateToImport(ImportFrom.FILES) },
    onMediaLibrary = { navController.navigateToImport(ImportFrom.MEDIA_LIBRARY) },
    onRecord = { navController.navigateToRecorder() },
    onOpenPlayer = { navController.navigate(PlayerRoutes.player(it)) },
    onOpenProcessing = {},
)

private fun NavHostController.selectTab(tab: AppTab) {
    navigate(tab.route) {
        popUpTo(AppTab.HOME.route) { saveState = true }
        launchSingleTop = true
        restoreState = true
    }
}
