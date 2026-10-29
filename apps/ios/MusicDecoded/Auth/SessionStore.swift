import Foundation
import Observation

/// The guest session (PLAN.md §3.4): a `deviceId` made once per install, a guest token from
/// `POST /auth/anonymous`, and the last `/config`. Both secrets live in the Keychain.
@MainActor
@Observable
final class SessionStore {
    enum SessionError: LocalizedError {
        case missingBaseURL

        var errorDescription: String? {
            "The API address isn't set (API_BASE_URL in Config/*.xcconfig)."
        }
    }

    /// The last `/config` response. Cached across launches, as the contract asks, so flags
    /// are known offline.
    private(set) var config: ClientConfig? = nil
    /// Why the last `start()` failed, for display; `nil` after a success.
    private(set) var errorMessage: String? = nil

    /// `nil` when `API_BASE_URL` isn't set. Also used directly for presigned URLs, which take no token.
    let api: APIClient?
    private let secureStore: any SecureStore
    private let defaults: UserDefaults

    private static let deviceIdKey = "deviceId"
    private static let tokenKey = "token"
    private static let configKey = "clientConfig"

    init(api: APIClient?, secureStore: any SecureStore, defaults: UserDefaults = .standard) {
        self.api = api
        self.secureStore = secureStore
        self.defaults = defaults
        if let data = defaults.data(forKey: Self.configKey) {
            config = try? JSONDecoder().decode(ClientConfig.self, from: data)
        }
    }

    /// Run at launch (and on foreground): makes sure there's a deviceId and a token, then
    /// refreshes `/config`.
    func start() async {
        ensureDeviceId()
        do {
            let config = try await authorized { api, token in
                try await api.config(token: token)
            }
            self.config = config
            errorMessage = nil
            if let data = try? JSONEncoder().encode(config) {
                defaults.set(data, forKey: Self.configKey)
            }
        } catch {
            errorMessage = error.localizedDescription
        }
    }

    /// Runs `request` with the stored token, signing in first if there's none.
    /// A 401 means the token is no longer valid: sign in again once and retry once.
    func authorized<T>(_ request: @MainActor (APIClient, String) async throws -> T) async throws -> T {
        guard let api else { throw SessionError.missingBaseURL }

        let token: String
        if let stored = secureStore.string(for: Self.tokenKey) {
            token = stored
        } else {
            token = try await signIn(using: api)
        }

        do {
            return try await request(api, token)
        } catch let error as APIError where error.status == 401 {
            let freshToken = try await signIn(using: api)
            return try await request(api, freshToken)
        }
    }

    /// The install's deviceId, created on first use.
    @discardableResult
    func ensureDeviceId() -> String {
        if let existing = secureStore.string(for: Self.deviceIdKey) {
            return existing
        }
        let deviceId = UUID().uuidString.lowercased()
        secureStore.set(deviceId, for: Self.deviceIdKey)
        return deviceId
    }

    private func signIn(using api: APIClient) async throws -> String {
        let response = try await api.anonymousAuth(deviceId: ensureDeviceId())
        secureStore.set(response.token, for: Self.tokenKey)
        return response.token
    }
}
