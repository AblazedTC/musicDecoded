package com.musicdecoded.mobile.settings

import android.app.Application
import androidx.datastore.preferences.core.floatPreferencesKey
import androidx.datastore.preferences.core.preferencesOf
import androidx.datastore.preferences.core.stringPreferencesKey
import com.musicdecoded.mobile.player.PlayerView
import kotlinx.coroutines.test.runTest
import org.junit.Assert.assertEquals
import org.junit.Assert.assertFalse
import org.junit.Assert.assertTrue
import org.junit.Test
import org.junit.runner.RunWith
import org.robolectric.RobolectricTestRunner
import org.robolectric.RuntimeEnvironment
import org.robolectric.annotation.Config

@RunWith(RobolectricTestRunner::class)
@Config(application = Application::class)
class SettingsStoreTest {
    private val visualizer = stringPreferencesKey("visualizer")
    private val speed = floatPreferencesKey("default_speed")

    @Test fun defaultViewIsSavedAndReadBack() = runTest {
        val store = SettingsStore(RuntimeEnvironment.getApplication())
        assertEquals(PlayerView.TIMELINE, store.current().playerView)
        store.update { it.copy(playerView = PlayerView.PIANO_FOCUS) }
        assertEquals(PlayerView.PIANO_FOCUS, store.current().playerView)
    }

    @Test fun migrationDropsOnlyTheOldVisualizerPreference() = runTest {
        val old = preferencesOf(visualizer to "VIDEO", speed to 0.75f)
        assertTrue(DropVisualizerPreference.shouldMigrate(old))
        val migrated = DropVisualizerPreference.migrate(old)
        assertEquals(preferencesOf(speed to 0.75f), migrated)
        assertFalse(DropVisualizerPreference.shouldMigrate(migrated))
    }
}
