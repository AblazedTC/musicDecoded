package com.musicdecoded.mobile

import android.content.Intent
import android.graphics.Color
import android.os.Bundle
import androidx.activity.ComponentActivity
import androidx.activity.SystemBarStyle
import androidx.activity.compose.setContent
import androidx.activity.enableEdgeToEdge
import androidx.compose.runtime.getValue
import androidx.lifecycle.compose.collectAsStateWithLifecycle
import com.musicdecoded.mobile.ui.main.MainScreen
import com.musicdecoded.mobile.ui.theme.MusicDecodedTheme
import kotlinx.coroutines.flow.MutableStateFlow

class MainActivity : ComponentActivity() {
    /** Text shared to the app (a SoundCloud link, T-502a) until the UI has handled it. */
    private val sharedText = MutableStateFlow<String?>(null)

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        if (savedInstanceState == null) handleShare(intent)
        // Dark-only app: force light system-bar icons even when the device is in light mode.
        enableEdgeToEdge(
            statusBarStyle = SystemBarStyle.dark(Color.TRANSPARENT),
            navigationBarStyle = SystemBarStyle.dark(Color.TRANSPARENT),
        )
        // No login gate: everyone starts as a guest (PLAN.md §3.4); the app works offline.
        setContent {
            val shared by sharedText.collectAsStateWithLifecycle()
            MusicDecodedTheme { MainScreen(shared, onSharedTextHandled = { sharedText.value = null }) }
        }
    }

    override fun onNewIntent(intent: Intent) {
        super.onNewIntent(intent)
        setIntent(intent)
        handleShare(intent)
    }

    private fun handleShare(intent: Intent?) {
        if (intent?.action != Intent.ACTION_SEND || intent.type?.startsWith("text/") != true) return
        sharedText.value = intent.getStringExtra(Intent.EXTRA_TEXT)?.takeIf { it.isNotBlank() }
    }

    override fun onStart() {
        super.onStart()
        // Sync trigger: app foreground (PLAN.md §5.2).
        (application as MusicDecodedApplication).requestSync()
    }
}
