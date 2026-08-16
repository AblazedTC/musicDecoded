package com.musicdecoded.mobile.ui.main

import androidx.compose.foundation.layout.padding
import androidx.compose.material.icons.Icons
import androidx.compose.material.icons.automirrored.filled.ArrowBack
import androidx.compose.material.icons.filled.Add
import androidx.compose.material.icons.filled.LibraryMusic
import androidx.compose.material.icons.automirrored.filled.MenuBook
import androidx.compose.material.icons.filled.Settings
import androidx.compose.material3.ExperimentalMaterial3Api
import androidx.compose.material3.Icon
import androidx.compose.material3.IconButton
import androidx.compose.material3.NavigationBar
import androidx.compose.material3.NavigationBarItem
import androidx.compose.material3.Scaffold
import androidx.compose.material3.Text
import androidx.compose.material3.TopAppBar
import androidx.compose.runtime.Composable
import androidx.compose.runtime.getValue
import androidx.compose.ui.Modifier
import androidx.compose.ui.graphics.vector.ImageVector
import androidx.navigation.compose.NavHost
import androidx.navigation.compose.composable
import androidx.navigation.compose.currentBackStackEntryAsState
import androidx.navigation.compose.rememberNavController
import com.musicdecoded.mobile.auth.AuthViewModel
import com.musicdecoded.mobile.ui.add.AddScreen
import com.musicdecoded.mobile.ui.dictionary.DictionaryScreen
import com.musicdecoded.mobile.ui.library.LibraryScreen
import com.musicdecoded.mobile.ui.settings.SettingsScreen

private enum class TabDestination(val route: String, val label: String, val icon: ImageVector) {
    LIBRARY("library", "Library", Icons.Default.LibraryMusic),
    ADD("add", "Add", Icons.Default.Add),
    DICTIONARY("dictionary", "Dictionary", Icons.AutoMirrored.Filled.MenuBook),
}

@OptIn(ExperimentalMaterial3Api::class)
@Composable
fun MainScreen(authViewModel: AuthViewModel) {
    val navController = rememberNavController()
    val backStackEntry by navController.currentBackStackEntryAsState()
    val currentRoute = backStackEntry?.destination?.route ?: TabDestination.LIBRARY.route
    val onTabs = currentRoute in TabDestination.entries.map { it.route }

    Scaffold(
        topBar = {
            TopAppBar(
                title = { Text("MusicDecoded") },
                navigationIcon = {
                    if (!onTabs) {
                        IconButton(onClick = { navController.popBackStack() }) {
                            Icon(Icons.AutoMirrored.Filled.ArrowBack, contentDescription = "Back")
                        }
                    }
                },
                actions = {
                    if (onTabs) {
                        IconButton(onClick = { navController.navigate("settings") }) {
                            Icon(Icons.Default.Settings, contentDescription = "Settings")
                        }
                    }
                }
            )
        },
        bottomBar = {
            if (onTabs) {
                NavigationBar {
                    TabDestination.entries.forEach { dest ->
                        NavigationBarItem(
                            selected = currentRoute == dest.route,
                            onClick = {
                                navController.navigate(dest.route) {
                                    popUpTo(TabDestination.LIBRARY.route) { saveState = true }
                                    launchSingleTop = true
                                    restoreState = true
                                }
                            },
                            icon = { Icon(dest.icon, contentDescription = null) },
                            label = { Text(dest.label) }
                        )
                    }
                }
            }
        }
    ) { innerPadding ->
        NavHost(
            navController,
            startDestination = TabDestination.LIBRARY.route,
            Modifier.padding(innerPadding)
        ) {
            composable(TabDestination.LIBRARY.route) { LibraryScreen() }
            composable(TabDestination.ADD.route) { AddScreen() }
            composable(TabDestination.DICTIONARY.route) { DictionaryScreen() }
            composable("settings") { SettingsScreen(authViewModel) }
        }
    }
}
