package com.musicdecoded.mobile.ui.auth

import com.musicdecoded.mobile.core.Source
import com.musicdecoded.mobile.library.LibraryItem
import com.musicdecoded.mobile.library.LibraryStatus
import com.musicdecoded.mobile.network.ApiError
import com.musicdecoded.mobile.network.ApiException
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.ExperimentalCoroutinesApi
import kotlinx.coroutines.test.UnconfinedTestDispatcher
import kotlinx.coroutines.test.resetMain
import kotlinx.coroutines.test.setMain
import org.junit.After
import org.junit.Assert.assertEquals
import org.junit.Assert.assertNull
import org.junit.Assert.assertTrue
import org.junit.Before
import org.junit.Rule
import org.junit.Test
import org.junit.rules.TemporaryFolder
import java.io.File
import java.time.Instant

@OptIn(ExperimentalCoroutinesApi::class)
class AccountTest {
    @get:Rule val folder = TemporaryFolder()

    @Before fun setUp() = Dispatchers.setMain(UnconfinedTestDispatcher())
    @After fun tearDown() = Dispatchers.resetMain()

    @Test fun passwordIs8CharactersTo72Utf8Bytes() {
        val v = AuthViewModel::validateCredentials
        assertTrue(v("a@b.co", "short") != null)
        assertNull(v("a@b.co", "12345678"))
        assertNull(v("a@b.co", "a".repeat(72)))
        assertTrue(v("a@b.co", "a".repeat(73)) != null)
        // 8 characters but 3 bytes each: valid length, 24 bytes.
        assertNull(v("a@b.co", "♯".repeat(8)))
        // 25 × 3 bytes = 75 > 72, although only 25 characters.
        assertTrue(v("a@b.co", "♯".repeat(25)) != null)
        assertTrue(v("not-an-email", "12345678") != null)
    }

    private fun viewModel(guestData: Boolean, fail: Exception? = null, syncs: MutableList<Unit> = mutableListOf()) = AuthViewModel(
        initialMode = AuthMode.LOG_IN,
        signUp = { _, _ -> },
        logIn = { _, _ -> fail?.let { throw it } },
        hasGuestData = { guestData },
        requestSync = { syncs += Unit },
    ).apply {
        setEmail("me@example.com")
        setPassword("password1")
    }

    @Test fun loginWithGuestDataGoesToMigrationWithoutSyncingYet() {
        val syncs = mutableListOf<Unit>()
        val vm = viewModel(guestData = true, syncs = syncs)
        vm.submit()
        assertEquals(AuthOutcome.MIGRATE, vm.state.value.outcome)
        assertTrue(syncs.isEmpty())
    }

    @Test fun loginWithoutGuestDataSyncsRightAway() {
        val syncs = mutableListOf<Unit>()
        val vm = viewModel(guestData = false, syncs = syncs)
        vm.submit()
        assertEquals(AuthOutcome.DONE, vm.state.value.outcome)
        assertEquals(1, syncs.size)
    }

    @Test fun showsTheApiErrorMessage() {
        val vm = viewModel(guestData = false, fail = ApiException(ApiError("unauthorized", "Invalid email or password"), 401))
        vm.submit()
        assertEquals("Invalid email or password", vm.state.value.error)
        assertNull(vm.state.value.outcome)
    }

    @Test fun migrationSummaryCountsTimelinesOwnAudioAndRestricted() {
        val files = folder.newFolder()
        File(files, "audio").mkdirs()
        File(files, "audio/a.mp3").writeBytes(ByteArray(1000))
        File(files, "audio/b.m4a").writeBytes(ByteArray(500))
        fun row(id: String, source: Source, path: String? = null) = LibraryItem(
            trackId = id, title = id, source = source, createdAt = Instant.EPOCH, updatedAt = Instant.EPOCH,
            status = LibraryStatus.READY, localAudioPath = path,
        )
        val summary = MigrationSummary.of(
            listOf(
                row("a", Source.FILE, "audio/a.mp3"),
                row("b", Source.RECORDING, "audio/b.m4a"),
                row("c", Source.YOUTUBE),
                row("d", Source.SOUNDCLOUD),
            ),
            files,
        )
        assertEquals(MigrationSummary(timelines = 4, ownAudio = 2, ownAudioBytes = 1500, restricted = 2), summary)
        // Free accounts: own audio isn't synced (Pro only, PLAN.md §9).
        assertEquals(false, MigrationSummary.of(listOf(row("a", Source.FILE, "audio/a.mp3")), files, pro = false).ownAudioSyncs)
    }
}
