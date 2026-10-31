package com.musicdecoded.mobile.ui.components

import android.graphics.Bitmap
import android.graphics.BitmapFactory
import android.util.LruCache
import androidx.compose.foundation.Image
import androidx.compose.foundation.background
import androidx.compose.foundation.layout.Box
import androidx.compose.runtime.Composable
import androidx.compose.runtime.getValue
import androidx.compose.runtime.produceState
import androidx.compose.ui.Modifier
import androidx.compose.ui.graphics.asImageBitmap
import androidx.compose.ui.layout.ContentScale
import com.musicdecoded.mobile.ui.theme.MusicDecodedColors
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.withContext
import java.net.HttpURLConnection
import java.net.URL

/** Decoded thumbnails, ~8 MB. Small list images only (YouTube thumbnails), so no disk cache. */
private val thumbnails = object : LruCache<String, Bitmap>(8 * 1024 * 1024) {
    override fun sizeOf(key: String, value: Bitmap) = value.byteCount
}

/**
 * An https image (YouTube thumbnails) with a `surfaceRaised` placeholder while it loads or if it
 * fails. Platform only (HttpURLConnection + BitmapFactory): no image-loading library for a
 * handful of thumbnails.
 */
@Composable
fun RemoteImage(url: String?, modifier: Modifier = Modifier, contentScale: ContentScale = ContentScale.Crop) {
    val bitmap by produceState(url?.let(thumbnails::get), url) {
        if (value == null && url != null && url.startsWith("https://")) value = load(url)
    }
    Box(modifier.background(MusicDecodedColors.SurfaceRaised)) {
        bitmap?.let { Image(it.asImageBitmap(), contentDescription = null, contentScale = contentScale, modifier = Modifier.matchParentSize()) }
    }
}

private suspend fun load(url: String): Bitmap? = withContext(Dispatchers.IO) {
    runCatching {
        val connection = URL(url).openConnection() as HttpURLConnection
        connection.connectTimeout = TIMEOUT_MS
        connection.readTimeout = TIMEOUT_MS
        try {
            connection.inputStream.use(BitmapFactory::decodeStream)
        } finally {
            connection.disconnect()
        }
    }.getOrNull()?.also { thumbnails.put(url, it) }
}

private const val TIMEOUT_MS = 10_000
