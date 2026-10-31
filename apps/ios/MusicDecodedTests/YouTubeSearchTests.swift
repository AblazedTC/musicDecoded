import Foundation
import os
import Testing
@testable import MusicDecoded

/// The YouTube search model against a stubbed `GET /youtube/search` (T-304i).
@MainActor
struct YouTubeSearchTests {
    let defaults = UserDefaults(suiteName: "YouTubeSearchTests-\(UUID().uuidString)")!

    private func makeModel(_ handler: @escaping StubURLProtocol.Handler) -> YouTubeSearchModel {
        let session = SessionStore(
            api: StubURLProtocol.client(handler),
            secureStore: InMemorySecureStore(["deviceId": "device-1", "token": "token-1"]),
            defaults: defaults
        )
        return YouTubeSearchModel(session: session, defaults: defaults, debounce: .milliseconds(20))
    }

    @Test func debouncesTypingIntoOneRequest() async throws {
        let log = RequestLog()
        let queries = QueryLog()
        let model = makeModel { request in
            log.record(request)
            queries.append(queryItems(request))
            return (200, searchPage(["aaaaaaaaaaa"], next: nil))
        }

        model.query = "s"
        model.query = "su"
        model.query = "sunny"
        await model.searchTask?.value

        #expect(log.routes == ["GET /youtube/search"])
        #expect(queries.all == [["q": "sunny"]])
        #expect(log.all.first?.authorization == "Bearer token-1")
        #expect(model.phase == .loaded)
        #expect(model.results.map(\.videoId) == ["aaaaaaaaaaa"])
        #expect(model.results.first?.durationText == "3:05")
        // Typing alone doesn't add a suggestion chip.
        #expect(model.recentQueries.isEmpty)
    }

    @Test func clearingTheQueryResetsWithoutARequest() async throws {
        let log = RequestLog()
        let model = makeModel { request in
            log.record(request)
            return (200, searchPage(["aaaaaaaaaaa"], next: nil))
        }
        model.query = "sunny"
        await model.searchTask?.value
        model.query = "   "

        #expect(model.phase == .idle)
        #expect(model.results.isEmpty)
        #expect(log.routes.count == 1)
    }

    @Test func pagesWithNextPageToken() async throws {
        let queries = QueryLog()
        let model = makeModel { request in
            let items = queryItems(request)
            queries.append(items)
            if items["pageToken"] == "page-2" {
                // The repeated id is dropped.
                return (200, searchPage(["bbbbbbbbbbb", "ccccccccccc"], next: nil))
            }
            return (200, searchPage(["aaaaaaaaaaa", "bbbbbbbbbbb"], next: "page-2"))
        }

        model.query = "sunny"
        model.submit()
        await model.searchTask?.value
        #expect(model.canLoadMore)

        // Only the last row triggers the next page.
        model.loadMoreIfNeeded(after: model.results[0])
        #expect(model.loadMoreTask == nil)
        model.loadMoreIfNeeded(after: model.results[1])
        #expect(model.isLoadingMore)
        await model.loadMoreTask?.value

        #expect(queries.all == [["q": "sunny"], ["q": "sunny", "pageToken": "page-2"]])
        #expect(model.results.map(\.videoId) == ["aaaaaaaaaaa", "bbbbbbbbbbb", "ccccccccccc"])
        #expect(model.isLoadingMore == false)
        #expect(model.canLoadMore == false)
    }

    @Test func featureDisabledIsUnavailable() async throws {
        let model = makeModel { _ in
            (403, Data(#"{"code": "feature_disabled", "message": "No API key"}"#.utf8))
        }
        model.query = "sunny"
        model.submit()
        await model.searchTask?.value

        #expect(model.phase == .failed(.unavailable))
        #expect(YouTubeSearchModel.Failure.unavailable.canRetry == false)
    }

    @Test func quotaIsNotThePlanUpsell() async throws {
        let model = makeModel { _ in
            (429, Data(#"{"code": "rate_limited", "message": "YouTube quota exhausted"}"#.utf8))
        }
        model.query = "sunny"
        model.submit()
        await model.searchTask?.value

        #expect(model.phase == .failed(.quota))
        let message = YouTubeSearchModel.Failure.quota.message
        #expect(message != Transcription.rateLimitedMessage)
        #expect(message.localizedCaseInsensitiveContains("pro") == false)
        #expect(message.localizedCaseInsensitiveContains("upgrade") == false)
    }

    @Test func networkFailureIsOffline() async throws {
        let model = makeModel { _ in
            throw URLError(.notConnectedToInternet)
        }
        model.query = "sunny"
        model.submit()
        await model.searchTask?.value

        #expect(model.phase == .failed(.offline))
        #expect(YouTubeSearchModel.Failure.offline.canRetry)
    }

    @Test func recentQueriesKeepTheNewestEight() async throws {
        let model = makeModel { _ in (200, searchPage([], next: nil)) }
        for index in 1...9 {
            model.query = "query \(index)"
            model.submit()
        }
        await model.searchTask?.value
        model.query = "QUERY 3"
        model.submit()
        await model.searchTask?.value

        #expect(model.recentQueries == ["QUERY 3", "query 9", "query 8", "query 7", "query 6", "query 5", "query 4", "query 2"])
        // Persisted for the next visit.
        let reopened = makeModel { _ in (200, searchPage([], next: nil)) }
        #expect(reopened.recentQueries == model.recentQueries)
    }

    @Test func selectingARecentQuerySearchesItNow() async throws {
        let queries = QueryLog()
        let model = makeModel { request in
            queries.append(queryItems(request))
            return (200, searchPage(["aaaaaaaaaaa"], next: nil))
        }
        model.selectRecent("sunny days")
        await model.searchTask?.value

        #expect(model.query == "sunny days")
        #expect(queries.all == [["q": "sunny days"]])
        #expect(model.recentQueries == ["sunny days"])
    }

    @Test func addingRecentQueries() {
        #expect(RecentQueries.adding("  ", to: ["a"]) == ["a"])
        #expect(RecentQueries.adding(" b ", to: ["a", "B"]) == ["b", "a"])
    }
}

// Free functions so `@Sendable` stub handlers can call them.

/// The `q` and `pageToken` query items of a request.
private func queryItems(_ request: URLRequest) -> [String: String] {
    let items = request.url.flatMap { URLComponents(url: $0, resolvingAgainstBaseURL: false)?.queryItems } ?? []
    return Dictionary(items.map { ($0.name, $0.value ?? "") }, uniquingKeysWith: { _, last in last })
}

/// A `GET /youtube/search` body with one result per id.
private func searchPage(_ ids: [String], next: String?) -> Data {
    let items = ids.map { id in
        """
        {"videoId": "\(id)", "title": "Song \(id)", "channelTitle": "Channel", "thumbnailUrl": "https://i.ytimg.com/vi/\(id)/mqdefault.jpg", "durationSec": 185}
        """
    }
    let token = next.map { "\"\($0)\"" } ?? "null"
    return Data(#"{"nextPageToken": \#(token), "items": [\#(items.joined(separator: ","))]}"#.utf8)
}

/// Query items each stubbed request carried, safely from any thread.
private final class QueryLog: Sendable {
    private let entries = OSAllocatedUnfairLock<[[String: String]]>(initialState: [])

    func append(_ items: [String: String]) {
        entries.withLock { $0.append(items) }
    }

    var all: [[String: String]] {
        entries.withLock { $0 }
    }
}
