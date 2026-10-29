import Foundation
import os

/// Records the requests a stub received, safely from any thread.
final class RequestLog: Sendable {
    struct Entry: Sendable {
        let method: String
        let path: String
        let authorization: String?
        let body: Data?

        /// "METHOD /path", for compact assertions.
        var route: String { "\(method) \(path)" }
    }

    private let entries = OSAllocatedUnfairLock<[Entry]>(initialState: [])

    /// Records `request` and returns how many requests with the same route came before it.
    @discardableResult
    func record(_ request: URLRequest) -> Int {
        let entry = Entry(
            method: request.httpMethod ?? "GET",
            path: request.url?.path() ?? "",
            authorization: request.value(forHTTPHeaderField: "Authorization"),
            body: request.bodyData
        )
        return entries.withLock { entries in
            let earlier = entries.filter { $0.route == entry.route }.count
            entries.append(entry)
            return earlier
        }
    }

    var all: [Entry] {
        entries.withLock { $0 }
    }

    var routes: [String] {
        all.map(\.route)
    }
}
