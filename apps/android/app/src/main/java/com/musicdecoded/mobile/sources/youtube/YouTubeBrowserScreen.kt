package com.musicdecoded.mobile.sources.youtube

import android.annotation.SuppressLint
import android.graphics.Bitmap
import android.os.Bundle
import android.content.ActivityNotFoundException
import android.content.Intent
import android.webkit.WebChromeClient
import android.webkit.WebResourceRequest
import android.webkit.WebView
import android.webkit.WebViewClient
import androidx.activity.compose.BackHandler
import androidx.compose.foundation.background
import androidx.compose.foundation.layout.Arrangement
import androidx.compose.foundation.layout.Column
import androidx.compose.foundation.layout.Row
import androidx.compose.foundation.layout.fillMaxSize
import androidx.compose.foundation.layout.fillMaxWidth
import androidx.compose.foundation.layout.padding
import androidx.compose.material.icons.Icons
import androidx.compose.material.icons.automirrored.rounded.ArrowBack
import androidx.compose.material.icons.automirrored.rounded.ArrowForward
import androidx.compose.material.icons.rounded.AutoAwesome
import androidx.compose.material.icons.rounded.Close
import androidx.compose.material.icons.rounded.Refresh
import androidx.compose.material3.MaterialTheme
import androidx.compose.material3.Text
import androidx.compose.runtime.Composable
import androidx.compose.runtime.DisposableEffect
import androidx.compose.runtime.LaunchedEffect
import androidx.compose.runtime.getValue
import androidx.compose.runtime.mutableStateOf
import androidx.compose.runtime.remember
import androidx.compose.runtime.setValue
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.draw.alpha
import androidx.compose.ui.platform.LocalContext
import androidx.compose.ui.viewinterop.AndroidView
import androidx.lifecycle.Lifecycle
import androidx.lifecycle.LifecycleEventObserver
import androidx.lifecycle.compose.LocalLifecycleOwner
import androidx.lifecycle.compose.collectAsStateWithLifecycle
import androidx.lifecycle.viewmodel.compose.viewModel
import com.musicdecoded.mobile.MusicDecodedApplication
import com.musicdecoded.mobile.ui.components.MdIconButton
import com.musicdecoded.mobile.ui.components.PrimaryButton
import com.musicdecoded.mobile.ui.theme.MusicDecodedColors
import com.musicdecoded.mobile.ui.theme.Spacing

/** Web back / forward when there's nowhere to go. */
private const val DISABLED_ALPHA = 0.4f

private const val HOME_URL = "https://m.youtube.com"

/** Pauses every `<video>` on the page (leaving the screen must stop playback). */
private const val PAUSE_MEDIA_JS = "document.querySelectorAll('video,audio').forEach(function(m){m.pause()})"

/**
 * In-app YouTube (T-308a): m.youtube.com in a WebView. On a single-video page "Transcribe this"
 * starts the `youtube` job and [onDecodeStarted] opens Processing. Nothing is downloaded on the
 * device (PLAN.md R-1): the server fetches the audio.
 */
@Composable
fun YouTubeBrowserScreen(onBack: () -> Unit, onDecodeStarted: (trackId: String) -> Unit) {
    val app = LocalContext.current.applicationContext as MusicDecodedApplication
    val viewModel: YouTubeBrowserViewModel = viewModel(factory = YouTubeBrowserViewModel.factory(app))
    val state by viewModel.state.collectAsStateWithLifecycle()
    var webView by remember { mutableStateOf<WebView?>(null) }

    LaunchedEffect(state.startedTrackId) {
        state.startedTrackId?.let {
            viewModel.navigated()
            onDecodeStarted(it)
        }
    }

    // System back walks the web history first, then leaves the screen.
    BackHandler(enabled = state.canGoBack) { webView?.goBack() }

    // Pause playback (and the page's timers) whenever the screen isn't in front.
    val lifecycleOwner = LocalLifecycleOwner.current
    DisposableEffect(lifecycleOwner, webView) {
        val view = webView
        val observer = LifecycleEventObserver { _, event ->
            when (event) {
                Lifecycle.Event.ON_PAUSE -> view?.pauseAll()
                Lifecycle.Event.ON_RESUME -> view?.onResume()
                else -> Unit
            }
        }
        lifecycleOwner.lifecycle.addObserver(observer)
        onDispose { lifecycleOwner.lifecycle.removeObserver(observer) }
    }

    Column(Modifier.fillMaxSize().background(MusicDecodedColors.Canvas)) {
        Row(
            Modifier.fillMaxWidth().padding(horizontal = Spacing.gutter, vertical = Spacing.xs),
            horizontalArrangement = Arrangement.spacedBy(Spacing.sm),
            verticalAlignment = Alignment.CenterVertically,
        ) {
            MdIconButton(Icons.Rounded.Close, contentDescription = "Back to app", onClick = onBack)
            Text("YouTube", style = MaterialTheme.typography.titleLarge, color = MusicDecodedColors.Text)
        }
        AndroidView(
            modifier = Modifier.weight(1f).fillMaxWidth(),
            factory = { context ->
                createBrowser(context, viewModel).also { view ->
                    val saved = viewModel.webState
                    if (saved == null || view.restoreState(saved) == null) view.loadUrl(HOME_URL)
                    webView = view
                }
            },
            onRelease = { view ->
                view.pauseAll()
                viewModel.webState = Bundle().also { view.saveState(it) }
                view.destroy()
            },
        )
        Column(
            Modifier.fillMaxWidth().padding(horizontal = Spacing.gutter, vertical = Spacing.xs),
            verticalArrangement = Arrangement.spacedBy(Spacing.xs),
        ) {
            state.error?.let { Text(it, style = MaterialTheme.typography.bodySmall, color = MusicDecodedColors.Record) }
            if (state.videoId != null) {
                PrimaryButton(
                    if (state.starting) "Starting…" else "Transcribe this",
                    onClick = viewModel::transcribe,
                    icon = Icons.Rounded.AutoAwesome,
                    loading = state.starting,
                )
            }
            Row(Modifier.fillMaxWidth(), horizontalArrangement = Arrangement.SpaceEvenly) {
                MdIconButton(
                    Icons.AutoMirrored.Rounded.ArrowBack,
                    contentDescription = "Previous page",
                    onClick = { webView?.goBack() },
                    modifier = Modifier.alpha(if (state.canGoBack) 1f else DISABLED_ALPHA),
                )
                MdIconButton(
                    Icons.AutoMirrored.Rounded.ArrowForward,
                    contentDescription = "Next page",
                    onClick = { webView?.goForward() },
                    modifier = Modifier.alpha(if (state.canGoForward) 1f else DISABLED_ALPHA),
                )
                MdIconButton(Icons.Rounded.Refresh, contentDescription = "Reload", onClick = { webView?.reload() })
            }
        }
    }
}

@SuppressLint("SetJavaScriptEnabled")
private fun createBrowser(context: android.content.Context, viewModel: YouTubeBrowserViewModel) = WebView(context).apply {
    setBackgroundColor(android.graphics.Color.BLACK)
    settings.javaScriptEnabled = true
    settings.domStorageEnabled = true
    settings.allowFileAccess = false
    settings.allowContentAccess = false
    // No downloads from the device (PLAN.md R-1).
    setDownloadListener { _, _, _, _, _ -> }
    webChromeClient = object : WebChromeClient() {
        override fun onReceivedTitle(view: WebView, title: String?) = viewModel.titleReceived(title)
    }
    webViewClient = object : WebViewClient() {
        // YouTube and Google sign-in load here; other web links open in the system browser and
        // app links (intent://, vnd.youtube:) are ignored.
        override fun shouldOverrideUrlLoading(view: WebView, request: WebResourceRequest): Boolean {
            val url = request.url
            if (url.scheme !in setOf("http", "https")) return true
            if (YouTubeUrl.isInAppHost(url.host)) return false
            try {
                context.startActivity(Intent(Intent.ACTION_VIEW, url).addFlags(Intent.FLAG_ACTIVITY_NEW_TASK))
            } catch (_: ActivityNotFoundException) {
                // Nothing can open it; stay on the page.
            }
            return true
        }

        override fun onPageStarted(view: WebView, url: String?, favicon: Bitmap?) =
            viewModel.pageChanged(url, view.canGoBack(), view.canGoForward())

        // YouTube is a single-page app: in-page navigations only show up here.
        override fun doUpdateVisitedHistory(view: WebView, url: String?, isReload: Boolean) =
            viewModel.pageChanged(url, view.canGoBack(), view.canGoForward())
    }
}

private fun WebView.pauseAll() {
    evaluateJavascript(PAUSE_MEDIA_JS, null)
    onPause()
}
