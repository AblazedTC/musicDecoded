package com.musicdecoded.mobile

import android.os.Bundle
import androidx.activity.ComponentActivity
import androidx.activity.compose.setContent
import androidx.activity.enableEdgeToEdge
import com.musicdecoded.mobile.ui.main.MainScreen
import com.musicdecoded.mobile.ui.theme.MusicDecodedTheme

class MainActivity : ComponentActivity() {
    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        enableEdgeToEdge()
        // No login gate: everyone starts as a guest (PLAN.md §3.4); the app works offline.
        setContent {
            MusicDecodedTheme { MainScreen() }
        }
    }
}
