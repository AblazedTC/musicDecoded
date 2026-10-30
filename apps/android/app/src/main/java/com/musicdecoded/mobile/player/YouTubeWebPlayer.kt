package com.musicdecoded.mobile.player

import android.annotation.SuppressLint
import android.content.ActivityNotFoundException
import android.content.Context
import android.content.Intent
import android.graphics.Color
import android.os.Handler
import android.os.Looper
import android.webkit.JavascriptInterface
import android.webkit.WebResourceRequest
import android.webkit.WebView
import android.webkit.WebViewClient
import androidx.compose.runtime.Composable
import androidx.compose.runtime.DisposableEffect
import androidx.compose.runtime.getValue
import androidx.compose.runtime.mutableStateOf
import androidx.compose.runtime.remember
import androidx.compose.runtime.rememberUpdatedState
import androidx.compose.runtime.setValue
import androidx.compose.ui.Modifier
import androidx.compose.ui.viewinterop.AndroidView
import java.util.Locale

/**
 * The official YouTube IFrame Player API in a WebView (PLAN.md D-4; T-304a hero, T-305a panel).
 * A small local page is loaded with an https base URL so the embed gets a proper origin/referrer;
 * the page reports state and `currentTime` through a JS interface and takes commands through
 * `evaluateJavascript`. Commands sent before `onReady` are queued.
 */
class WebViewYouTubeBridge(
    private val webView: WebView,
    private val reportError: (code: Int) -> Unit = {},
) : YouTubePlayerBridge {
    override var listener: YouTubePlayerBridge.Listener? = null
    private val main = Handler(Looper.getMainLooper())
    private val pending = mutableListOf<String>()
    private var ready = false

    /** Called from the page (a JavaBridge thread); everything is forwarded to the main thread. */
    private inner class PageEvents {
        @JavascriptInterface fun onReady() {
            main.post {
                ready = true
                pending.forEach(::evaluate)
                pending.clear()
                listener?.onReady()
            }
        }

        @JavascriptInterface fun onState(code: Int) {
            main.post { listener?.onStateChange(YouTubePlayerState.of(code)) }
        }

        @JavascriptInterface fun onTime(seconds: Double) {
            main.post { listener?.onTime(seconds) }
        }

        @JavascriptInterface fun onRate(rate: Double) {
            main.post { listener?.onRateChange(rate.toFloat()) }
        }

        @JavascriptInterface fun onError(code: Int) {
            main.post { reportError(code) }
        }
    }

    @SuppressLint("JavascriptInterface", "AddJavascriptInterface")
    fun load(videoId: String, startSeconds: Int, controls: Boolean, autoplay: Boolean) {
        require(VIDEO_ID.matches(videoId)) { "Invalid videoId" }
        webView.addJavascriptInterface(PageEvents(), JS_NAME)
        webView.loadDataWithBaseURL(BASE_URL, page(videoId, startSeconds.coerceAtLeast(0), controls, autoplay), "text/html", "utf-8", null)
    }

    override fun play() = command("md.play()")
    override fun pause() = command("md.pause()")
    override fun seekTo(seconds: Double) = command("md.seek(${number(seconds.coerceAtLeast(0.0))})")
    override fun setPlaybackRate(rate: Float) = command("md.rate(${number(rate.toDouble())})")
    override fun mute() = command("md.mute()")

    private fun command(js: String) {
        if (ready) evaluate(js) else pending += js
    }

    private fun evaluate(js: String) = webView.evaluateJavascript(js, null)

    companion object {
        /** The embed's origin (also passed as the `origin` player var). */
        const val BASE_URL = "https://www.youtube-nocookie.com"
        private const val JS_NAME = "MusicDecodedBridge"
        private const val REPORT_INTERVAL_MS = 200
        private const val CUE_DELAY_MS = 300
        private val VIDEO_ID = Regex("[A-Za-z0-9_-]{11}")

        private fun number(value: Double) = String.format(Locale.US, "%.3f", value)

        private fun page(videoId: String, start: Int, controls: Boolean, autoplay: Boolean) = """
            <!DOCTYPE html>
            <html><head>
            <meta name="viewport" content="width=device-width, initial-scale=1">
            <style>html,body{margin:0;padding:0;width:100%;height:100%;background:#000;overflow:hidden}#player{position:absolute;top:0;left:0;width:100%;height:100%}</style>
            </head><body>
            <div id="player"></div>
            <script>
            var bridge = window.$JS_NAME, player = null, videoId = '$videoId', cuedAt = null, cueTimer = null;
            function report() { if (player && player.getCurrentTime) bridge.onTime(player.getCurrentTime() || 0); }
            var md = {
              play: function () {
                // A cue landing after playVideo() stops the video again, so a play right after an idle
                // seek cancels the pending cue and loads the video at that time instead.
                clearTimeout(cueTimer);
                if (cuedAt !== null) player.loadVideoById({ videoId: videoId, startSeconds: cuedAt });
                else player.playVideo();
                cuedAt = null;
              },
              pause: function () { player.pauseVideo(); },
              seek: function (t) {
                var s = player.getPlayerState();
                // seekTo() starts an unstarted/cued video, so cue it at t instead.
                if (s === -1 || s === 5) {
                  cuedAt = t;
                  clearTimeout(cueTimer);
                  cueTimer = setTimeout(function () { player.cueVideoById({ videoId: videoId, startSeconds: t }); }, $CUE_DELAY_MS);
                }
                else player.seekTo(t, true);
                // seekTo() also restarts an ended video; keep it paused like any other seek.
                if (s === 0) player.pauseVideo();
                report();
              },
              rate: function (r) { player.setPlaybackRate(r); },
              mute: function () { player.mute(); }
            };
            function onYouTubeIframeAPIReady() {
              player = new YT.Player('player', {
                host: '$BASE_URL',
                videoId: videoId,
                width: '100%', height: '100%',
                playerVars: { playsinline: 1, rel: 0, start: $start, controls: ${if (controls) 1 else 0},
                              autoplay: ${if (autoplay) 1 else 0}, origin: '$BASE_URL' },
                events: {
                  onReady: function () { bridge.onReady(); setInterval(report, $REPORT_INTERVAL_MS); },
                  onStateChange: function (e) { if (e.data === 1) cuedAt = null; report(); bridge.onState(e.data); },
                  onPlaybackRateChange: function (e) { bridge.onRate(e.data); },
                  onError: function (e) { bridge.onError(e.data); }
                }
              });
            }
            </script>
            <script src="https://www.youtube.com/iframe_api"></script>
            </body></html>
        """.trimIndent()
    }
}

/** YouTube errors 101/150: the owner doesn't allow embedding. */
fun youTubeErrorMessage(code: Int): String = when (code) {
    101, 150 -> "The owner of this video doesn't allow it to be played here."
    100 -> "This video is unavailable."
    else -> "This video can't be played right now."
}

/**
 * An embedded YouTube player. [onAttach] gets its bridge once the page starts loading and
 * [onDetach] the same bridge when it leaves the composition (the WebView is then destroyed).
 */
@SuppressLint("SetJavaScriptEnabled")
@Composable
fun YouTubeWebPlayer(
    videoId: String,
    modifier: Modifier = Modifier,
    startSeconds: Double = 0.0,
    controls: Boolean = true,
    autoplay: Boolean = false,
    onError: (code: Int) -> Unit = {},
    onAttach: (YouTubePlayerBridge) -> Unit = {},
    onDetach: (YouTubePlayerBridge) -> Unit = {},
) {
    var bridge by remember { mutableStateOf<YouTubePlayerBridge?>(null) }
    val currentOnError by rememberUpdatedState(onError)
    AndroidView(
        modifier = modifier,
        factory = { context ->
            WebView(context).apply {
                setBackgroundColor(Color.BLACK)
                settings.javaScriptEnabled = true
                settings.domStorageEnabled = true
                // Our transport starts playback from Kotlin, not from a tap inside the page.
                settings.mediaPlaybackRequiresUserGesture = false
                settings.allowFileAccess = false
                settings.allowContentAccess = false
                webViewClient = ExternalLinks(context)
                val created = WebViewYouTubeBridge(this) { currentOnError(it) }
                created.load(videoId, startSeconds.toInt(), controls, autoplay)
                bridge = created
            }
        },
        onRelease = { it.destroy() },
    )
    val attached = bridge
    DisposableEffect(attached) {
        attached?.let(onAttach)
        onDispose { attached?.let(onDetach) }
    }
}

/** Links tapped inside the embed (YouTube logo, title) open outside the app, never in this WebView. */
private class ExternalLinks(private val context: Context) : WebViewClient() {
    override fun shouldOverrideUrlLoading(view: WebView, request: WebResourceRequest): Boolean {
        if (!request.isForMainFrame) return false
        try {
            context.startActivity(Intent(Intent.ACTION_VIEW, request.url).addFlags(Intent.FLAG_ACTIVITY_NEW_TASK))
        } catch (_: ActivityNotFoundException) {
            // Nothing can open it; stay on the player.
        }
        return true
    }
}
