package com.musicdecoded.mobile.sources.youtube

import android.content.Context
import androidx.datastore.core.DataStore
import androidx.datastore.preferences.core.Preferences
import androidx.datastore.preferences.core.edit
import androidx.datastore.preferences.core.stringPreferencesKey
import androidx.datastore.preferences.preferencesDataStore
import kotlinx.coroutines.flow.first

/** Recent YouTube search queries, newest first (suggestion chips; PLAN.md R-6). Local only. */
interface RecentQueries {
    suspend fun load(): List<String>
    suspend fun save(queries: List<String>)

    companion object {
        const val MAX = 8

        /** [query] moved to the front (case-insensitive dedupe), capped at [MAX]. */
        fun add(queries: List<String>, query: String): List<String> =
            (listOf(query) + queries.filterNot { it.equals(query, ignoreCase = true) }).take(MAX)
    }
}

private val Context.youTubeStore: DataStore<Preferences> by preferencesDataStore("youtube")

/** DataStore-backed [RecentQueries]: one newline-separated string (queries never contain newlines). */
class RecentQueryStore(private val context: Context) : RecentQueries {
    private val key = stringPreferencesKey("recent_queries")

    override suspend fun load(): List<String> =
        context.youTubeStore.data.first()[key]?.split('\n')?.filter { it.isNotBlank() }.orEmpty()

    override suspend fun save(queries: List<String>) {
        context.youTubeStore.edit { it[key] = queries.joinToString("\n") }
    }
}
