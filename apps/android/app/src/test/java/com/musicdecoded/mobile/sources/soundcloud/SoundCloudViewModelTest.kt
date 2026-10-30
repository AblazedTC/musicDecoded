package com.musicdecoded.mobile.sources.soundcloud

import com.musicdecoded.mobile.library.TranscriptionException
import com.musicdecoded.mobile.library.TranscriptionRepository
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.ExperimentalCoroutinesApi
import kotlinx.coroutines.test.TestScope
import kotlinx.coroutines.test.UnconfinedTestDispatcher
import kotlinx.coroutines.test.resetMain
import kotlinx.coroutines.test.setMain
import org.junit.After
import org.junit.Assert.assertEquals
import org.junit.Assert.assertFalse
import org.junit.Assert.assertNull
import org.junit.Assert.assertTrue
import org.junit.Before
import org.junit.Test

@OptIn(ExperimentalCoroutinesApi::class)
class SoundCloudViewModelTest {
    private val dispatcher = UnconfinedTestDispatcher()
    private val started = mutableListOf<Pair<String, String>>()

    @Before fun setUp() = Dispatchers.setMain(dispatcher)
    @After fun tearDown() = Dispatchers.resetMain()

    private fun viewModel(
        text: String = "",
        enabled: Boolean? = true,
        start: suspend (String, String) -> String = { url, title -> started += url to title; "track-1" },
    ) = SoundCloudViewModel(text, enabled, start, TestScope(dispatcher))

    @Test fun sharedTextIsReducedToTheLink() {
        val vm = viewModel("Listen on #SoundCloud https://soundcloud.com/artist/sunny-days?utm_source=clipboard")
        assertEquals("https://soundcloud.com/artist/sunny-days", vm.state.value.text)
        assertNull(vm.state.value.linkError)
    }

    @Test fun sharedTextWithoutALinkShowsTheHint() {
        val vm = viewModel("just some text")
        assertEquals("just some text", vm.state.value.text)
        assertEquals(SoundCloudViewModel.NO_LINK, vm.state.value.linkError)
    }

    @Test fun decodeStartsASoundCloudJobAndOpensProcessing() {
        val vm = viewModel()
        vm.onTextChange("soundcloud.com/artist/sunny-days")
        vm.decode()
        assertEquals(listOf("https://soundcloud.com/artist/sunny-days" to "Sunny days"), started)
        assertEquals("track-1", vm.state.value.startedTrackId)
        vm.navigated()
        assertNull(vm.state.value.startedTrackId)
    }

    @Test fun decodeWithoutALinkDoesNothing() {
        val vm = viewModel()
        vm.onTextChange("https://soundcloud.com/artist")
        vm.decode()
        assertTrue(started.isEmpty())
        assertEquals(SoundCloudViewModel.NO_LINK, vm.state.value.linkError)
    }

    @Test fun networkErrorsCanBeRetried() {
        var fail = true
        val vm = viewModel(start = { _, _ -> if (fail) throw TranscriptionException(TranscriptionRepository.OFFLINE) else "track-1" })
        vm.onTextChange("https://soundcloud.com/a/b")
        vm.decode()
        assertEquals(TranscriptionRepository.OFFLINE, vm.state.value.error)
        assertTrue(vm.state.value.enabled)
        fail = false
        vm.decode()
        assertNull(vm.state.value.error)
        assertEquals("track-1", vm.state.value.startedTrackId)
    }

    @Test fun featureDisabledFromTheServerStopsRetrying() {
        val vm = viewModel(start = { _, _ -> throw TranscriptionException(TranscriptionRepository.SOUNDCLOUD_DISABLED) })
        vm.onTextChange("https://soundcloud.com/a/b")
        vm.decode()
        assertEquals(TranscriptionRepository.SOUNDCLOUD_DISABLED, vm.state.value.error)
        assertFalse(vm.state.value.enabled)
    }

    @Test fun disabledByConfigShowsTheMessageWithoutCallingTheServer() {
        val vm = viewModel("https://soundcloud.com/a/b", enabled = false)
        assertEquals(TranscriptionRepository.SOUNDCLOUD_DISABLED, vm.state.value.error)
        vm.decode()
        assertTrue(started.isEmpty())
    }
}
