package com.musicdecoded.mobile.sources.youtube

import com.musicdecoded.mobile.library.LibraryItem
import com.musicdecoded.mobile.library.TranscriptionException
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.ExperimentalCoroutinesApi
import kotlinx.coroutines.test.TestScope
import kotlinx.coroutines.test.UnconfinedTestDispatcher
import kotlinx.coroutines.test.resetMain
import kotlinx.coroutines.test.setMain
import org.junit.After
import org.junit.Assert.assertEquals
import org.junit.Assert.assertNull
import org.junit.Assert.assertTrue
import org.junit.Before
import org.junit.Test

@OptIn(ExperimentalCoroutinesApi::class)
class YouTubeBrowserViewModelTest {
    private val dispatcher = UnconfinedTestDispatcher()
    private val id = "dQw4w9WgXcQ"
    private val started = mutableListOf<Pair<String, String>>()

    @Before fun setUp() = Dispatchers.setMain(dispatcher)
    @After fun tearDown() = Dispatchers.resetMain()

    private fun viewModel(start: suspend (String, String) -> String = { v, t -> started += v to t; "track-1" }) =
        YouTubeBrowserViewModel(start, TestScope(dispatcher))

    @Test fun onlyVideoPagesOfferTranscribe() {
        val vm = viewModel()
        vm.pageChanged("https://m.youtube.com/", canGoBack = false, canGoForward = false)
        assertNull(vm.state.value.videoId)
        vm.transcribe("YouTube")
        assertTrue(started.isEmpty())
        vm.pageChanged("https://m.youtube.com/watch?v=$id&pp=x", canGoBack = true, canGoForward = false)
        assertEquals(id, vm.state.value.videoId)
        assertTrue(vm.state.value.canGoBack)
    }

    @Test fun transcribeStartsTheJobWithThePageTitle() {
        val vm = viewModel()
        vm.pageChanged("https://m.youtube.com/watch?v=$id", canGoBack = true, canGoForward = false)
        vm.transcribe("Sunny Days - YouTube")
        assertEquals(listOf(id to "Sunny Days"), started)
        assertEquals("track-1", vm.state.value.startedTrackId)
        vm.navigated()
        assertNull(vm.state.value.startedTrackId)
    }

    @Test fun noTitleFallsBackToThePlaceholder() {
        val vm = viewModel()
        vm.pageChanged("https://m.youtube.com/shorts/$id", canGoBack = false, canGoForward = false)
        vm.transcribe(null)
        assertEquals(listOf(id to LibraryItem.YOUTUBE_PLACEHOLDER_TITLE), started)
    }

    @Test fun aFailedStartShowsTheMessageUntilTheVideoChanges() {
        val vm = viewModel { _, _ -> throw TranscriptionException("You've reached today's decode limit") }
        vm.pageChanged("https://m.youtube.com/watch?v=$id", canGoBack = false, canGoForward = false)
        vm.transcribe("x")
        assertEquals("You've reached today's decode limit", vm.state.value.error)
        vm.pageChanged("https://m.youtube.com/", canGoBack = true, canGoForward = false)
        assertNull(vm.state.value.error)
    }
}
