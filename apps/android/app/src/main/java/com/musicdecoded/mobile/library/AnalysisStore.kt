package com.musicdecoded.mobile.library

import com.musicdecoded.mobile.core.Analysis
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.withContext
import java.io.File

/**
 * Analysis files on disk: `analysis/{trackId}.json`, saved verbatim from the server (PLAN.md §5.1).
 * The player works offline whenever the file exists.
 */
class AnalysisStore(private val directory: File) {
    fun exists(trackId: String): Boolean = file(trackId).isFile

    /** Decodes `analysis/{trackId}.json`, or `null` if there is no file. Throws if it's corrupt. */
    suspend fun load(trackId: String): Analysis? = withContext(Dispatchers.IO) {
        val file = file(trackId)
        if (file.isFile) Analysis.decode(file.readText()) else null
    }

    /** Validates [json] against the models, then writes it unchanged (atomically, via a temp file). */
    suspend fun save(trackId: String, json: String): Unit = withContext(Dispatchers.IO) {
        Analysis.decode(json)
        directory.mkdirs()
        val target = file(trackId)
        val temp = File(directory, "$trackId.json.tmp")
        temp.writeText(json)
        if (!temp.renameTo(target)) {
            target.delete()
            check(temp.renameTo(target)) { "Couldn't write ${target.name}" }
        }
    }

    suspend fun delete(trackId: String): Unit = withContext(Dispatchers.IO) { file(trackId).delete() }

    private fun file(trackId: String): File {
        require(TRACK_ID.matches(trackId)) { "Invalid trackId" }
        return File(directory, "$trackId.json")
    }

    private companion object {
        /** Server ids are UUIDs; this also blocks path traversal. */
        val TRACK_ID = Regex("[A-Za-z0-9_-]+")
    }
}
