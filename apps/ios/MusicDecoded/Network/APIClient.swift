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

    /// `GET /config`.
    func config(token: String) async throws -> ClientConfig {
        try await send("GET", "config", token: token)
    }

    /// Sends a JSON request and decodes a JSON response. Throws `APIError` for non-2xx responses.
    private func send<Response: Decodable>(
        _ method: String,
        _ path: String,
        body: (any Encodable)? = nil,
        token: String? = nil
    ) async throws -> Response {
        var request = URLRequest(url: baseURL.appending(path: path))
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
        guard let http = response as? HTTPURLResponse else {
            throw URLError(.badServerResponse)
        }
        guard (200..<300).contains(http.statusCode) else {
            throw APIError(status: http.statusCode, body: data)
        }
        return try JSONDecoder().decode(Response.self, from: data)
    }
}
