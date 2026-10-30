package com.musicdecoded.mobile.sources.soundcloud

import java.net.URI
import java.net.URISyntaxException

/** Pulls a SoundCloud track link out of pasted or shared text (T-502a). No network. */
object SoundCloudUrl {
    /** A soundcloud.com / on.soundcloud.com link anywhere in the text, with or without a scheme. */
    private val CANDIDATE = Regex("""(?i)(?<![\w.-])(?:https?://)?(?:(?:www|m|on)\.)?soundcloud\.com/[^\s<>"'()\[\]]+""")
    private val TRAILING_PUNCTUATION = ".,;:!?…"

    /**
     * The first SoundCloud track link in [text] (share-sheet text often wraps it in a sentence),
     * normalized to `https://soundcloud.com/...` or `https://on.soundcloud.com/...` without query or
     * fragment (tracking parameters). `null` if there's no link to a track: profiles, home and
     * other hosts are rejected.
     */
    fun extract(text: String): String? = CANDIDATE.findAll(text).firstNotNullOfOrNull { normalize(it.value) }

    /** A readable placeholder title from the link's slug until the analysis names the track. */
    fun titleFrom(url: String): String {
        val slug = url.substringAfter("soundcloud.com/").split('/').getOrNull(1)
        return slug?.replace('-', ' ')?.trim()?.replaceFirstChar { it.uppercase() }?.ifEmpty { null } ?: "SoundCloud track"
    }

    private fun normalize(match: String): String? {
        val trimmed = match.trimEnd { it in TRAILING_PUNCTUATION }
        val uri = try {
            URI(if ("://" in trimmed) trimmed else "https://$trimmed")
        } catch (_: URISyntaxException) {
            return null
        }
        val host = uri.host?.lowercase() ?: return null
        val segments = uri.rawPath.orEmpty().split('/').filter { it.isNotEmpty() }
        return when (host) {
            // Short share links: on.soundcloud.com/AbC123
            "on.soundcloud.com" -> if (segments.isNotEmpty()) "https://on.soundcloud.com/${segments.joinToString("/")}" else null
            // A track is /{user}/{track}[/{secret}]; /{user} alone is a profile.
            "soundcloud.com", "www.soundcloud.com", "m.soundcloud.com" ->
                if (segments.size >= 2) "https://soundcloud.com/${segments.joinToString("/")}" else null
            else -> null
        }
    }
}
