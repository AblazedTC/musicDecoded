package com.musicdecoded.mobile.sources.youtube

import com.musicdecoded.mobile.core.Source
import com.musicdecoded.mobile.library.LibraryItem
import com.musicdecoded.mobile.library.LibraryStatus
import com.musicdecoded.mobile.library.TranscriptionException
import com.musicdecoded.mobile.network.VideoSummary
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.ExperimentalCoroutinesApi
import kotlinx.coroutines.flow.MutableStateFlow
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
import java.time.Instant

@OptIn(ExperimentalCoroutinesApi::class)
class YouTubeDetailViewModelTest {
    private val dispatcher = UnconfinedTestDispatcher()
    private val video = VideoSummary("dQw4w9WgXcQ", "Sunny Days", "Channel", "https://i.ytimg.com/x.jpg", 200)
    private val existing = MutableStateFlow<LibraryItem?>(null)
    private val started = mutableListOf<VideoSummary>()

    @Before fun setUp() = Dispatchers.setMain(dispatcher)
    @After fun tearDown() = Dispatchers.resetMain()

    private fun viewModel(start: suspend (VideoSummary) -> String = { started += it; "track-1" }) =
        YouTubeDetailViewModel(video, existing, start, TestScope(dispatcher))

    @Test fun decodeStartsTheYouTubeJobAndOpensProcessing() {
        val vm = viewModel()
        vm.decode()
        assertEquals(listOf(video), started)
        assertEquals("track-1", vm.state.value.startedTrackId)
        assertFalse(vm.state.value.starting)
        vm.navigated()
        assertNull(vm.state.value.startedTrackId)
    }

    @Test fun aFailedStartShowsTheMessage() {
        val vm = viewModel { throw TranscriptionException("You've reached today's decode limit") }
        vm.decode()
        assertEquals("You've reached today's decode limit", vm.state.value.error)
        assertNull(vm.state.value.startedTrackId)
    }

    @Test fun knowsWhenTheVideoWasAlreadyDecoded() {
        val vm = viewModel()
        assertFalse(vm.state.value.alreadyDecoded)
        existing.value = LibraryItem(
            trackId = "track-1", title = "Sunny Days", source = Source.YOUTUBE, sourceRef = video.videoId,
            createdAt = Instant.EPOCH, updatedAt = Instant.EPOCH, status = LibraryStatus.READY,
        )
        assertTrue(vm.state.value.alreadyDecoded)
    }

    @Test fun policyNoteIsPlanR1Verbatim() {
        assertEquals(
            "Offline audio is saved only when permitted. The chord timeline can remain available separately.",
            YOUTUBE_POLICY_NOTE,
        )
    }
}
