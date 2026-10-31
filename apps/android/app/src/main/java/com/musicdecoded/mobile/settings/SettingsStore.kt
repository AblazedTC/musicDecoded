package com.musicdecoded.mobile.settings

import android.content.Context
import androidx.datastore.core.DataMigration
import androidx.datastore.core.DataStore
import androidx.datastore.preferences.core.Preferences
import androidx.datastore.preferences.core.edit
import androidx.datastore.preferences.core.floatPreferencesKey
import androidx.datastore.preferences.core.stringPreferencesKey
import androidx.datastore.preferences.preferencesDataStore
import com.musicdecoded.mobile.core.Accidentals
import com.musicdecoded.mobile.player.ChordsMode
import com.musicdecoded.mobile.player.PlayerView
import kotlinx.coroutines.flow.Flow
import kotlinx.coroutines.flow.first
import kotlinx.coroutines.flow.map

/** The Settings rows the player reads when it opens a track (DESIGN.md §3.9). */
data class PlayerPreferences(
    val defaultSpeed: Float = 1f,
    val chordsMode: ChordsMode = ChordsMode.FULL,
    val accidentals: Accidentals = Accidentals.AUTO,
    /** Settings "Default view": the view a player opens in; the header switcher updates it too (DESIGN.md §3.7: persists per app). */
    val playerView: PlayerView = PlayerView.TIMELINE,
) {
    companion object {
        /** Default speed choices (the player's long-press range is 0.50–1.25×). */
        val SPEEDS = listOf(0.5f, 0.75f, 0.9f, 1f, 1.25f)
    }
}

/** Drops the "Default visualization" value (T-216a); nothing read it and it has no [PlayerView] equivalent. */
internal object DropVisualizerPreference : DataMigration<Preferences> {
    private val key = stringPreferencesKey("visualizer")

    override suspend fun shouldMigrate(currentData: Preferences) = key in currentData
    override suspend fun migrate(currentData: Preferences) = currentData.toMutablePreferences().apply { remove(key) }.toPreferences()
    override suspend fun cleanUp() = Unit
}

private val Context.settingsDataStore: DataStore<Preferences> by preferencesDataStore(
    "settings",
    produceMigrations = { listOf(DropVisualizerPreference) },
)

/** DataStore-backed preferences. */
class SettingsStore(private val context: Context) {
    private val speedKey = floatPreferencesKey("default_speed")
    private val chordsKey = stringPreferencesKey("chords_mode")
    private val accidentalsKey = stringPreferencesKey("accidentals")
    private val viewKey = stringPreferencesKey("player_view")

    val preferences: Flow<PlayerPreferences> = context.settingsDataStore.data.map { it.read() }

    suspend fun current(): PlayerPreferences = preferences.first()

    suspend fun update(transform: (PlayerPreferences) -> PlayerPreferences) {
        context.settingsDataStore.edit { prefs ->
            val next = transform(prefs.read())
            prefs[speedKey] = next.defaultSpeed
            prefs[chordsKey] = next.chordsMode.name
            prefs[accidentalsKey] = next.accidentals.name
            prefs[viewKey] = next.playerView.name
        }
    }

    private fun Preferences.read() = PlayerPreferences(
        defaultSpeed = this[speedKey] ?: 1f,
        chordsMode = this[chordsKey].toEnum(ChordsMode.FULL),
        accidentals = this[accidentalsKey].toEnum(Accidentals.AUTO),
        playerView = this[viewKey].toEnum(PlayerView.TIMELINE),
    )
}

private inline fun <reified E : Enum<E>> String?.toEnum(default: E): E =
    this?.let { name -> enumValues<E>().firstOrNull { it.name == name } } ?: default
