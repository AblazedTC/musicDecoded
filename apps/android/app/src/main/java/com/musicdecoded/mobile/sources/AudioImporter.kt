package com.musicdecoded.mobile.sources

import android.content.Context
import android.media.MediaMetadataRetriever
import android.net.Uri
import android.provider.OpenableColumns
import android.webkit.MimeTypeMap
import com.musicdecoded.mobile.library.TranscriptionException
import com.musicdecoded.mobile.library.TranscriptionRepository
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.withContext
import java.io.File

/** A picked file, copied into app storage and waiting for "Import & decode" (DESIGN.md §3.3). */
data class ImportDraft(
    /** Pending copy in `audio/`; renamed to `audio/{trackId}.{ext}` once the job exists. */
    val file: File,
    val fileName: String,
    /** From the file's metadata, falling back to the file name. */
    val title: String,
    val sizeBytes: Long,
    val durationSec: Double?,
    /** Bits per second, if the container says. */
    val bitrate: Int?,
) {
    val typeLabel: String get() = file.extension.uppercase()
}

/** Copies a picked `content://` audio file into app storage and reads its metadata. */
class AudioImporter(private val context: Context, private val repository: TranscriptionRepository) {
    suspend fun import(uri: Uri): ImportDraft = withContext(Dispatchers.IO) {
        val resolver = context.contentResolver
        val fileName = displayName(uri)
            ?: uri.lastPathSegment?.substringAfterLast('/')?.substringAfterLast('\\')
            ?: "Audio"
        val extension = extensionFor(fileName, resolver.getType(uri))
            ?: throw TranscriptionException(UNSUPPORTED)
        val file = repository.newPendingFile(extension)
        try {
            val input = resolver.openInputStream(uri) ?: throw TranscriptionException(UNREADABLE)
            input.use { source -> file.outputStream().use { source.copyTo(it) } }
            val retriever = MediaMetadataRetriever()
            val (title, durationMs, bitrate) = try {
                retriever.setDataSource(file.absolutePath)
                Triple(
                    retriever.extractMetadata(MediaMetadataRetriever.METADATA_KEY_TITLE),
                    retriever.extractMetadata(MediaMetadataRetriever.METADATA_KEY_DURATION)?.toLongOrNull(),
                    retriever.extractMetadata(MediaMetadataRetriever.METADATA_KEY_BITRATE)?.toIntOrNull(),
                )
            } catch (_: RuntimeException) {
                Triple(null, null, null) // unreadable metadata: still importable, the server decides
            } finally {
                retriever.release()
            }
            ImportDraft(
                file = file,
                fileName = fileName,
                title = title?.trim()?.takeIf { it.isNotEmpty() } ?: fileName.substringBeforeLast('.'),
                sizeBytes = file.length(),
                durationSec = durationMs?.let { it / 1000.0 },
                bitrate = bitrate,
            )
        } catch (e: Exception) {
            file.delete()
            if (e is TranscriptionException) throw e
            throw TranscriptionException(UNREADABLE, e)
        }
    }

    private fun displayName(uri: Uri): String? =
        context.contentResolver.query(uri, arrayOf(OpenableColumns.DISPLAY_NAME), null, null, null)?.use { cursor ->
            if (cursor.moveToFirst()) cursor.getString(0) else null
        }

    companion object {
        const val UNSUPPORTED = "This audio format isn't supported. Try MP3, M4A, AAC, WAV, FLAC or OGG."
        const val UNREADABLE = "Couldn't read this file. Try another one."

        /** An extension the upload accepts, from the file name or else the MIME type. */
        fun extensionFor(fileName: String, mimeType: String?): String? {
            val fromName = fileName.substringAfterLast('.', "").lowercase()
            if (TranscriptionRepository.contentTypeFor(fromName) != null) return fromName
            val fromMime = when (mimeType) {
                "audio/x-wav", "audio/wave" -> "wav"
                "audio/x-flac" -> "flac"
                "audio/x-m4a", "audio/m4a" -> "m4a"
                else -> mimeType?.let { MimeTypeMap.getSingleton().getExtensionFromMimeType(it) }
            }
            return fromMime?.takeIf { TranscriptionRepository.contentTypeFor(it) != null }
        }

        /** `8.4 MB`, `640 KB`. */
        fun formatSize(bytes: Long): String = when {
            bytes >= 1_000_000 -> "%.1f MB".format(bytes / 1_000_000.0)
            else -> "${(bytes / 1000).coerceAtLeast(1)} KB"
        }
    }
}
