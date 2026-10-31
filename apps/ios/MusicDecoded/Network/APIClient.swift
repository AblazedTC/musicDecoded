import Foundation

/// Calls the MusicDecoded API (contracts/openapi.yaml). Stateless: callers pass the bearer token,
/// and `SessionStore` owns signing in and retrying.
struct APIClient: Sendable {
    let baseURL: URL
    let session: URLSession

    init(baseURL: URL, session: URLSession = .shared) {
        self.baseURL = baseURL
        self.session = session
    }

    /// A client for `APIBaseURL` in Info.plist (from `API_BASE_URL` in Config/*.xcconfig),
    /// or `nil` when it's blank, as it is in Release until a host is chosen.
    static func configured(bundle: Bundle = .main) -> APIClient? {
        guard
            let value = bundle.object(forInfoDictionaryKey: "APIBaseURL") as? String,
            let url = URL(string: value),
            url.scheme != nil,
            url.host() != nil
        else { return nil }
        return APIClient(baseURL: url)
    }

    /// `POST /auth/anonymous`: a guest token for this device. Idempotent per `deviceId`.
    func anonymousAuth(deviceId: String) async throws -> AuthResponse {
        try await send("POST", "auth/anonymous", body: AnonymousAuthRequest(deviceId: deviceId))
    }

    /// `POST /auth/signup`. `guestToken` (optional bearer) links that guest's data to the account.
    func signUp(_ credentials: Credentials, guestToken: String?) async throws -> AuthResponse {
        try await send("POST", "auth/signup", body: credentials, token: guestToken)
    }

    /// `POST /auth/login`. `guestToken` (optional bearer) merges that guest's data into the account.
    func logIn(_ credentials: Credentials, guestToken: String?) async throws -> AuthResponse {
        try await send("POST", "auth/login", body: credentials, token: guestToken)
    }

    /// `GET /sync/library`: one page (up to 500 items). `since == nil` is a full pull.
    func pullLibrary(since: String?, token: String) async throws -> SyncPage {
        let query = since.map { [URLQueryItem(name: "since", value: $0)] } ?? []
        return try await send("GET", "sync/library", query: query, token: token)
    }

    /// `PUT /sync/library`: up to 500 items. Returns the server state of the pushed ids.
    func pushLibrary(_ items: [SyncItem], token: String) async throws -> SyncItems {
        try await send("PUT", "sync/library", body: SyncItems(items: items), token: token)
    }

    /// `GET /uploads/{uploadId}`: where to download the caller's own audio.
    func uploadLocation(id uploadId: String, token: String) async throws -> UploadLocation {
        try await send("GET", "uploads/\(uploadId)", token: token)
    }

    /// `GET /config`.
    func config(token: String) async throws -> ClientConfig {
        try await send("GET", "config", token: token)
    }

    /// `POST /uploads`: a presigned URL for the caller's own audio.
    func createUpload(_ body: UploadRequest, token: String) async throws -> UploadTarget {
        try await send("POST", "uploads", body: body, token: token)
    }

    /// PUTs `fileURL`'s bytes to a presigned `putUrl`. No bearer token: the URL carries its own auth.
    func uploadFile(_ fileURL: URL, to putUrl: URL, contentType: String) async throws {
        var request = URLRequest(url: putUrl)
        request.httpMethod = "PUT"
        request.setValue(contentType, forHTTPHeaderField: "Content-Type")
        let (data, response) = try await session.upload(for: request, fromFile: fileURL)
        try Self.check(response, data)
    }

    /// `POST /jobs`: `202` queued, or `200` when the content was already analyzed.
    func createJob(_ body: JobRequest, token: String) async throws -> Job {
        try await send("POST", "jobs", body: body, token: token)
    }

    /// `GET /jobs/{jobId}`.
    func job(id jobId: String, token: String) async throws -> Job {
        try await send("GET", "jobs/\(jobId)", token: token)
    }

    /// `GET /tracks/{trackId}/analysis`: where to download the analysis file.
    func analysisLocation(trackId: String, token: String) async throws -> AnalysisLocation {
        try await send("GET", "tracks/\(trackId)/analysis", token: token)
    }

    /// `GET /youtube/search`: one page of results. `403 feature_disabled` means search is
    /// unavailable; `429` means the server's YouTube quota is spent (not a plan limit).
    func youtubeSearch(query: String, pageToken: String?, token: String) async throws -> YouTubeSearchPage {
        var items = [URLQueryItem(name: "q", value: query)]
        if let pageToken {
            items.append(URLQueryItem(name: "pageToken", value: pageToken))
        }
        return try await send("GET", "youtube/search", query: items, token: token)
    }

    /// GETs a presigned URL (no bearer token) and returns the body.
    func download(_ url: URL) async throws -> Data {
        let (data, response) = try await session.data(from: url)
        try Self.check(response, data)
        return data
    }

    /// Downloads a presigned URL (no bearer token) to a temporary file and moves it to
    /// `destination`, replacing any file there.
    func download(_ url: URL, to destination: URL) async throws {
        let (temporary, response) = try await session.download(from: url)
        try Self.check(response, Data())
        let fileManager = FileManager.default
        if fileManager.fileExists(atPath: destination.path(percentEncoded: false)) {
            try fileManager.removeItem(at: destination)
        }
        try fileManager.moveItem(at: temporary, to: destination)
    }

    /// Sends a JSON request and decodes a JSON response. Throws `APIError` for non-2xx responses.
    private func send<Response: Decodable>(
        _ method: String,
        _ path: String,
        query: [URLQueryItem] = [],
        body: (any Encodable)? = nil,
        token: String? = nil
    ) async throws -> Response {
        var url = baseURL.appending(path: path)
        if query.isEmpty == false,
           var components = URLComponents(url: url, resolvingAgainstBaseURL: false) {
            components.queryItems = query
            // URLComponents leaves "+" as is, and servers decode it as a space ("c++" -> "c  ").
            components.percentEncodedQuery = components.percentEncodedQuery?
                .replacingOccurrences(of: "+", with: "%2B")
            url = components.url ?? url
        }
        var request = URLRequest(url: url)
        request.httpMethod = method
        request.setValue("application/json", forHTTPHeaderField: "Accept")
        if let token {
            request.setValue("Bearer \(token)", forHTTPHeaderField: "Authorization")
        }
        if let body {
            request.httpBody = try JSONEncoder().encode(body)
            request.setValue("application/json", forHTTPHeaderField: "Content-Type")
        }

        let (data, response) = try await session.data(for: request)
        try Self.check(response, data)
        return try JSONDecoder().decode(Response.self, from: data)
    }

    /// Throws `APIError` for a non-2xx response.
    private static func check(_ response: URLResponse, _ data: Data) throws {
        guard let http = response as? HTTPURLResponse else {
            throw URLError(.badServerResponse)
        }
        guard (200..<300).contains(http.statusCode) else {
            throw APIError(status: http.statusCode, body: data)
        }
    }
}
