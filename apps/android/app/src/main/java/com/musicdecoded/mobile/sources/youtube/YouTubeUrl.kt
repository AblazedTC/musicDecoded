package com.musicdecoded.mobile.sources.youtube

import java.net.URI
import java.net.URISyntaxException

/** "Paste a video URL" (DESIGN.md §3.4): pulls the videoId out of a pasted link. No network. */
object YouTubeUrl {
    private val VIDEO_ID = Regex("[A-Za-z0-9_-]{11}")
    private val WATCH_HOSTS = setOf("youtube.com", "m.youtube.com", "music.youtube.com", "youtube-nocookie.com")
    private val PATH_PREFIXES = setOf("shorts", "embed", "live", "v", "e")

    /**
     * The 11-character videoId from a watch / youtu.be / shorts / embed / live / music.youtube.com
     * link (with or without a scheme), or a bare videoId. `null` if it isn't a YouTube video link.
     */
    fun videoId(text: String): String? {
        val input = text.trim()
        if (VIDEO_ID.matches(input)) return input
        val uri = try {
            URI(if ("://" in input) input else "https://$input")
        } catch (_: URISyntaxException) {
            return null
        }
        if (uri.scheme?.lowercase() !in setOf("http", "https")) return null
        val host = uri.host?.lowercase()?.removePrefix("www.") ?: return null
        val segments = uri.rawPath.orEmpty().split('/').filter { it.isNotEmpty() }
        val candidate = when {
            host == "youtu.be" -> segments.firstOrNull()
            host in WATCH_HOSTS && segments.firstOrNull() == "watch" -> queryParam(uri.rawQuery, "v")
            host in WATCH_HOSTS && segments.firstOrNull() in PATH_PREFIXES -> segments.getOrNull(1)
            else -> null
        }
        return candidate?.takeIf { VIDEO_ID.matches(it) }
    }

    private fun queryParam(query: String?, name: String): String? =
        query?.split('&')?.firstNotNullOfOrNull { part ->
            val (key, value) = part.split('=', limit = 2).let { it[0] to it.getOrNull(1) }
            value.takeIf { key == name }
        }
}
