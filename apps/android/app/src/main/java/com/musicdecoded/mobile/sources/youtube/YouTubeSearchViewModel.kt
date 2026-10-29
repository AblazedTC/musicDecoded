package com.musicdecoded.mobile.sources.youtube

import androidx.lifecycle.ViewModel
import androidx.lifecycle.ViewModelProvider
import androidx.lifecycle.viewModelScope
import androidx.lifecycle.viewmodel.initializer
import androidx.lifecycle.viewmodel.viewModelFactory
import com.musicdecoded.mobile.BuildConfig
import com.musicdecoded.mobile.MusicDecodedApplication
import com.musicdecoded.mobile.auth.AuthRepository
import com.musicdecoded.mobile.network.ApiException
import com.musicdecoded.mobile.network.VideoSummary
import com.musicdecoded.mobile.network.YouTubeApiService
import kotlinx.coroutines.CancellationException
import kotlinx.coroutines.Job
import kotlinx.coroutines.delay
import kotlinx.coroutines.flow.MutableStateFlow
import kotlinx.coroutines.flow.StateFlow
import kotlinx.coroutines.flow.asStateFlow
import kotlinx.coroutines.flow.update
import kotlinx.coroutines.launch
import java.io.IOException

/** Why search can't show results. Quota is the server's YouTube quota: never the plan upsell. */
enum class SearchError(val title: String, val message: String) {
    UNAVAILABLE("Search is unavailable", "YouTube search isn't available right now. You can still paste a video URL below."),
    QUOTA("Search is busy", "YouTube search has reached its limit for now. Try again later, or paste a video URL below."),
    OFFLINE("You're offline", "Couldn't reach MusicDecoded. Check your connection and try again."),
    GENERIC("Something went wrong", "Search failed. Try again."),
    ;

    /** Retrying only helps when the problem may have gone away. */
    val retryable: Boolean get() = this != UNAVAILABLE

    companion object {
        fun from(e: Throwable): SearchError = when {
            e is ApiException && (e.error.code == "feature_disabled" || e.status == 403) -> UNAVAILABLE
            e is ApiException && (e.error.code == "rate_limited" || e.status == 429) -> QUOTA
            e is IOException -> OFFLINE
            else -> GENERIC
        }
    }
}

data class YouTubeSearchUiState(
    val query: String = "",
    /** Recent queries, newest first (max [RecentQueries.MAX]). */
    val recent: List<String> = emptyList(),
    /** The query [results] belong to; `null` before the first search. */
    val resultsFor: String? = null,
    val results: List<VideoSummary> = emptyList(),
    val nextPageToken: String? = null,
    val loading: Boolean = false,
    val loadingMore: Boolean = false,
    val error: SearchError? = null,
    /** A failed next page: the list stays, with a retry at the bottom. */
    val loadMoreError: SearchError? = null,
    val pasteError: String? = null,
    /** One-shot: open this video's detail screen (from a pasted URL). */
    val openVideo: VideoSummary? = null,
) {
    val canLoadMore: Boolean get() = nextPageToken != null && !loading && !loadingMore && loadMoreError == null
}

/**
 * YouTube search (DESIGN.md §3.4, T-304a). Typing searches after [DEBOUNCE_MS] of quiet; a chip or
 * the keyboard's search action searches at once. Each search costs server quota (PLAN.md R-6), so
 * the same query is never fetched twice in a row and only successful queries become chips.
 */
class YouTubeSearchViewModel(
    private val auth: AuthRepository,
    private val api: YouTubeApiService,
    private val recentQueries: RecentQueries,
) : ViewModel() {
    private val _state = MutableStateFlow(YouTubeSearchUiState())
    val state: StateFlow<YouTubeSearchUiState> = _state.asStateFlow()

    private var searchJob: Job? = null

    init {
        viewModelScope.launch {
            val recent = runCatching { recentQueries.load() }.getOrDefault(emptyList())
            _state.update { it.copy(recent = recent) }
        }
    }

    fun onQueryChange(query: String) {
        _state.update { it.copy(query = query) }
        searchJob?.cancel()
        searchJob = viewModelScope.launch {
            delay(DEBOUNCE_MS)
            search(normalize(query))
        }
    }

    /** Keyboard "search" or a suggestion chip: no debounce. */
    fun submit(query: String = _state.value.query) {
        _state.update { it.copy(query = query) }
        searchJob?.cancel()
        searchJob = viewModelScope.launch { search(normalize(query)) }
    }

    /** Re-runs the failed search. */
    fun retry() {
        val query = normalize(_state.value.query)
        searchJob?.cancel()
        searchJob = viewModelScope.launch { search(query, force = true) }
    }

    /** Infinite scroll: fetches the next page (the list calls this near its end). */
    fun loadMore() {
        val current = _state.value
        val token = current.nextPageToken ?: return
        val query = current.resultsFor ?: return
        if (!current.canLoadMore) return
        _state.update { it.copy(loadingMore = true) }
        viewModelScope.launch {
            try {
                val page = auth.authorized { api.search(it, query, token) }
                _state.update { state ->
                    if (state.resultsFor != query) return@update state.copy(loadingMore = false)
                    val known = state.results.mapTo(HashSet()) { it.videoId }
                    state.copy(
                        results = state.results + page.items.filter { it.videoId !in known },
                        nextPageToken = page.nextPageToken,
                        loadingMore = false,
                    )
                }
            } catch (e: CancellationException) {
                throw e
            } catch (e: Exception) {
                _state.update { it.copy(loadingMore = false, loadMoreError = SearchError.from(e)) }
            }
        }
    }

    fun retryLoadMore() {
        _state.update { it.copy(loadMoreError = null) }
        loadMore()
    }

    /** "Paste a video URL": opens the detail screen, or explains why the link isn't usable. */
    fun openPastedUrl(text: String) {
        val videoId = YouTubeUrl.videoId(text)
        if (videoId == null) {
            _state.update { it.copy(pasteError = "That isn't a YouTube video link. Try a youtube.com or youtu.be URL.") }
            return
        }
        _state.update { it.copy(pasteError = null, openVideo = pastedVideo(videoId)) }
    }

    fun clearPasteError() = _state.update { it.copy(pasteError = null) }

    fun videoOpened() = _state.update { it.copy(openVideo = null) }

    private suspend fun search(query: String, force: Boolean = false) {
        if (query.isEmpty()) {
            _state.update { it.copy(resultsFor = null, results = emptyList(), nextPageToken = null, loading = false, error = null, loadMoreError = null) }
            return
        }
        val current = _state.value
        if (!force && query == current.resultsFor && current.error == null) return
        _state.update { it.copy(loading = true, error = null, loadMoreError = null) }
        try {
            val page = auth.authorized { api.search(it, query) }
            val recent = RecentQueries.add(_state.value.recent, query)
            _state.update {
                it.copy(
                    resultsFor = query,
                    results = page.items.distinctBy { item -> item.videoId },
                    nextPageToken = page.nextPageToken,
                    loading = false,
                    recent = recent,
                )
            }
            runCatching { recentQueries.save(recent) }
        } catch (e: CancellationException) {
            throw e
        } catch (e: Exception) {
            _state.update {
                it.copy(resultsFor = query, results = emptyList(), nextPageToken = null, loading = false, error = SearchError.from(e))
            }
        }
    }

    companion object {
        const val DEBOUNCE_MS = 400L
        const val MAX_QUERY = 200

        /** One line, trimmed, within the contract's 1–200 characters. */
        fun normalize(query: String): String = query.replace(Regex("\\s+"), " ").trim().take(MAX_QUERY)

        /** A pasted link has only an id: the title is filled in by the embed / analysis later. */
        fun pastedVideo(videoId: String) = VideoSummary(
            videoId = videoId,
            title = "YouTube video",
            channelTitle = "",
            thumbnailUrl = "https://i.ytimg.com/vi/$videoId/hqdefault.jpg",
        )

        fun factory(app: MusicDecodedApplication): ViewModelProvider.Factory = viewModelFactory {
            initializer {
                YouTubeSearchViewModel(
                    app.authRepository,
                    YouTubeApiService(app.httpClient, BuildConfig.API_BASE_URL),
                    RecentQueryStore(app),
                )
            }
        }
    }
}
