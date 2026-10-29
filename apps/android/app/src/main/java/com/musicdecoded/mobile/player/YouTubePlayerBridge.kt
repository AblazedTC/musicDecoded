package com.musicdecoded.mobile.player

/** `YT.PlayerState` of the official IFrame Player API. */
enum class YouTubePlayerState(val code: Int) {
    UNSTARTED(-1),
    ENDED(0),
    PLAYING(1),
    PAUSED(2),
    BUFFERING(3),
    CUED(5),
    ;

    /** Not loaded into a playing session yet: `getCurrentTime()` isn't meaningful. */
    val idle: Boolean get() = this == UNSTARTED || this == CUED

    companion object {
        fun of(code: Int): YouTubePlayerState = entries.firstOrNull { it.code == code } ?: UNSTARTED
    }
}

/**
 * Commands to and events from one embedded YouTube player (the official IFrame Player API in a
 * WebView, see [WebViewYouTubeBridge]). Kotlin-only so engines can be tested with a fake.
 * Commands sent before the player is ready are queued. Events arrive on the main thread.
 */
interface YouTubePlayerBridge {
    fun play()
    fun pause()

    /** Seeks without changing play/pause (an idle video is cued at [seconds] instead of starting). */
    fun seekTo(seconds: Double)

    /** YouTube may round to a supported rate; the actual rate comes back in [Listener.onRateChange]. */
    fun setPlaybackRate(rate: Float)
    fun mute()

    var listener: Listener?

    interface Listener {
        fun onReady() {}
        fun onStateChange(state: YouTubePlayerState)

        /** `getCurrentTime()`, reported a few times per second and on every state change. */
        fun onTime(seconds: Double)
        fun onRateChange(rate: Float) {}
    }
}
