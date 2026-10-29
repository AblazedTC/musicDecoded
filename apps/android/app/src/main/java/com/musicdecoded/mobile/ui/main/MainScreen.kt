package com.musicdecoded.mobile.ui.main

import androidx.compose.foundation.layout.padding
import androidx.compose.material3.Scaffold
import androidx.compose.runtime.Composable
import androidx.compose.runtime.getValue
import androidx.compose.ui.Modifier
import androidx.navigation.NavHostController
import androidx.navigation.compose.NavHost
import androidx.navigation.compose.composable
import androidx.navigation.compose.currentBackStackEntryAsState
import androidx.navigation.compose.rememberNavController
import com.musicdecoded.mobile.ui.components.AppTab
import com.musicdecoded.mobile.ui.components.MdBottomBar
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
            composable(AppTab.HOME.route) { HomeScreen(onSeeAll = { navController.selectTab(AppTab.LIBRARY) }) }
            composable(AppTab.LIBRARY.route) { LibraryScreen() }
            composable(AppTab.SETTINGS.route) { SettingsScreen() }
        }
    }
}

private fun NavHostController.selectTab(tab: AppTab) {
    navigate(tab.route) {
        popUpTo(AppTab.HOME.route) { saveState = true }
        launchSingleTop = true
        restoreState = true
    }
}
