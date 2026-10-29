package com.musicdecoded.mobile.ui.components

import androidx.compose.material.icons.Icons
import androidx.compose.material.icons.rounded.Home
import androidx.compose.material.icons.rounded.LibraryMusic
import androidx.compose.material.icons.rounded.Settings
import androidx.compose.material3.Icon
import androidx.compose.material3.MaterialTheme
import androidx.compose.material3.NavigationBar
import androidx.compose.material3.NavigationBarItem
import androidx.compose.material3.NavigationBarItemDefaults
import androidx.compose.material3.Text
import androidx.compose.runtime.Composable
import androidx.compose.ui.Modifier
import androidx.compose.ui.graphics.vector.ImageVector
import androidx.compose.ui.tooling.preview.Preview
import com.musicdecoded.mobile.ui.theme.MusicDecodedColors

/** v1 ships 3 tabs; Tuner awaits T-505 (DESIGN.md §2). */
enum class AppTab(val route: String, val label: String, val icon: ImageVector) {
    HOME("home", "Home", Icons.Rounded.Home),
    LIBRARY("library", "Library", Icons.Rounded.LibraryMusic),
    SETTINGS("settings", "Settings", Icons.Rounded.Settings),
}

/** Bottom navigation: selected = `primary`, inactive = `textMuted`, platform-default height. */
@Composable
fun MdBottomBar(selected: AppTab?, onSelect: (AppTab) -> Unit, modifier: Modifier = Modifier) {
    NavigationBar(modifier = modifier, containerColor = MusicDecodedColors.Ink, contentColor = MusicDecodedColors.TextMuted) {
        AppTab.entries.forEach { tab ->
            NavigationBarItem(
                selected = tab == selected,
                onClick = { onSelect(tab) },
                icon = { Icon(tab.icon, contentDescription = null) },
                label = { Text(tab.label, style = MaterialTheme.typography.labelSmall) },
                colors = NavigationBarItemDefaults.colors(
                    selectedIconColor = MusicDecodedColors.Primary,
                    selectedTextColor = MusicDecodedColors.Primary,
                    indicatorColor = MusicDecodedColors.PrimarySoft,
                    unselectedIconColor = MusicDecodedColors.TextMuted,
                    unselectedTextColor = MusicDecodedColors.TextMuted,
                ),
            )
        }
    }
}

@Preview
@Composable
private fun BottomBarPreview() = PreviewSurface {
    MdBottomBar(selected = AppTab.HOME, onSelect = {})
}
