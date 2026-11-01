import Foundation
import Observation

/// State for the YouTube search screen (DESIGN.md §3.4, T-304i): a 400 ms debounced query,
/// paged results (`nextPageToken`), recent-query chips and the error states.
@MainActor
@Observable
final class YouTubeSearchModel {
    /// Why a search failed. `quota` is the server's YouTube quota, never the user's plan, so it
    /// must not show the upsell (openapi `/youtube/search` 429).
    enum Failure: Equatable, Sendable {
        case unavailable, quota, offline, other

        init(_ error: any Error) {
            switch error {
            case let error as APIError where error.code == .featureDisabled || error.status == 403:
                self = .unavailable
            case let error as APIError where error.status == 429:
                self = .quota
            case is SessionStore.SessionError:
                self = .unavailable
            case is URLError:
                self = .offline
            default:
                self = .other
            }
        }

        var title: String {
            switch self {
            case .unavailable: "YouTube search is unavailable"
            case .quota: "Search is busy right now"
            case .offline: "You're offline"
            case .other: "Something went wrong"
            }
        }

        var message: String {
            switch self {
            case .unavailable: "Search isn't available right now. You can still paste a video link below."
            case .quota: "YouTube search has reached its limit for now. Try again later, or paste a video link below."
            case .offline: Transcription.offlineMessage
            case .other: Transcription.genericMessage
            }
        }

        /// Retrying makes sense unless search is switched off on the server.
        var canRetry: Bool { self != .unavailable }
    }

    enum Phase: Equatable, Sendable {
        case idle, loading, loaded, failed(Failure)
    }

    typealias Search = @MainActor (_ query: String, _ pageToken: String?) async throws -> YouTubeSearchPage

    /// The search field. Changing it schedules a search after the debounce.
    var query = "" {
        didSet {
            if query != oldValue {
                queryChanged()
            }
        }
    }

    private(set) var phase = Phase.idle
    private(set) var results: [VideoSummary] = []
    private(set) var isLoadingMore = false
    /// The last "load more" failed; the list footer offers a retry.
    private(set) var loadMoreFailed = false
    private(set) var recentQueries: [String]

    private var nextPageToken: String?
    /// The query `results` belong to.
    private var loadedQuery = ""

    private let search: Search
    private let defaults: UserDefaults
    private let debounce: Duration
    /// The running (or debouncing) first-page search. Tests await it.
    @ObservationIgnored private(set) var searchTask: Task<Void, Never>?
    /// The running next-page request. Tests await it.
    @ObservationIgnored private(set) var loadMoreTask: Task<Void, Never>?

    init(search: @escaping Search, defaults: UserDefaults = .standard, debounce: Duration = .milliseconds(400)) {
        self.search = search
        self.defaults = defaults
        self.debounce = debounce
        recentQueries = RecentQueries.load(from: defaults)
    }

    /// Searches through the API with the session's token.
    convenience init(session: SessionStore, defaults: UserDefaults = .standard, debounce: Duration = .milliseconds(400)) {
        self.init(
            search: { query, pageToken in
                try await session.authorized { api, token in
                    try await api.youtubeSearch(query: query, pageToken: pageToken, token: token)
                }
            },
            defaults: defaults,
            debounce: debounce
        )
    }

    var canLoadMore: Bool {
        phase == .loaded && nextPageToken != nil
    }

    // MARK: - Actions

    /// The return key: search right away and remember the query.
    func submit() {
        let text = trimmedQuery
        guard text.isEmpty == false else { return }
        remember(text)
        startSearch(text, after: nil)
    }

    /// A recent-query chip.
    func selectRecent(_ recent: String) {
        query = recent
        submit()
    }

    /// The Retry button of an error state.
    func retry() {
        let text = trimmedQuery
        guard text.isEmpty == false else { return }
        startSearch(text, after: nil)
    }

    /// A result was opened: its query is worth suggesting again.
    func didOpen(_ video: VideoSummary) {
        remember(trimmedQuery)
    }

    /// Call when a row appears; loads the next page once the last row is on screen.
    func loadMoreIfNeeded(after video: VideoSummary) {
        guard video.videoId == results.last?.videoId, loadMoreFailed == false else { return }
        loadMore()
    }

    /// Loads the next page (also the footer's retry).
    func loadMore() {
        guard canLoadMore, let pageToken = nextPageToken, loadMoreTask == nil else { return }
        let text = loadedQuery
        isLoadingMore = true
        loadMoreFailed = false
        loadMoreTask = Task { [weak self] in
            await self?.fetchMore(text, pageToken: pageToken)
        }
    }

    // MARK: - Private

    private var trimmedQuery: String {
        query.trimmingCharacters(in: .whitespacesAndNewlines)
    }

    private func queryChanged() {
        let text = trimmedQuery
        guard text.isEmpty == false else {
            cancelRequests()
            phase = .idle
            results = []
            nextPageToken = nil
            loadedQuery = ""
            return
        }
        startSearch(text, after: debounce)
    }

    private func cancelRequests() {
        searchTask?.cancel()
        searchTask = nil
        loadMoreTask?.cancel()
        loadMoreTask = nil
        isLoadingMore = false
        loadMoreFailed = false
    }

    private func startSearch(_ text: String, after delay: Duration?) {
        cancelRequests()
        searchTask = Task { [weak self] in
            if let delay {
                do {
                    try await Task.sleep(for: delay)
                } catch {
                    return
                }
            }
            await self?.fetchFirstPage(text)
        }
    }

    private func fetchFirstPage(_ text: String) async {
        guard Task.isCancelled == false else { return }
        phase = .loading
        results = []
        nextPageToken = nil
        do {
            let page = try await search(text, nil)
            guard Task.isCancelled == false else { return }
            loadedQuery = text
            results = Self.unique(page.items)
            nextPageToken = page.nextPageToken
            phase = .loaded
        } catch {
            // A newer query cancelled this one (URLSession reports it as URLError.cancelled).
            guard Task.isCancelled == false else { return }
            phase = .failed(Failure(error))
        }
    }

    private func fetchMore(_ text: String, pageToken: String) async {
        do {
            let page = try await search(text, pageToken)
            guard Task.isCancelled == false, text == loadedQuery else { return }
            let known = Set(results.map(\.videoId))
            results += Self.unique(page.items).filter { known.contains($0.videoId) == false }
            nextPageToken = page.nextPageToken
        } catch {
            guard Task.isCancelled == false else { return }
            loadMoreFailed = true
        }
        isLoadingMore = false
        loadMoreTask = nil
    }

    private func remember(_ text: String) {
        let updated = RecentQueries.adding(text, to: recentQueries)
        guard updated != recentQueries else { return }
        recentQueries = updated
        RecentQueries.save(updated, to: defaults)
    }

    /// The API can repeat a video across or within pages; the list needs unique ids.
    private static func unique(_ items: [VideoSummary]) -> [VideoSummary] {
        var seen = Set<String>()
        return items.filter { seen.insert($0.videoId).inserted }
    }
}
