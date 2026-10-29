import Foundation
import Observation

/// The signed-in identity (PLAN.md §3.4): a `deviceId` made once per install, a token from
/// `POST /auth/anonymous` (guest) or signup/login (account), and the last `/config`. The token,
/// deviceId and account details live in the Keychain.
@MainActor
@Observable
final class SessionStore {
    enum SessionError: LocalizedError {
        case missingBaseURL

        var errorDescription: String? {
            "The API address isn't set (API_BASE_URL in Config/*.xcconfig)."
        }
    }

    /// An account's token and userId, for `/sync/library` (accounts only).
    struct Account: Sendable, Equatable {
        let token: String
        let userId: String
    }

    /// The last `/config` response. Cached across launches, as the contract asks, so flags
    /// are known offline.
    private(set) var config: ClientConfig? = nil
    /// Why the last `start()` failed, for display; `nil` after a success.
    private(set) var errorMessage: String? = nil
    /// `nil` until there's a token. A token stored before accounts existed is a guest's.
    private(set) var kind: AuthResponse.Kind? = nil
    private(set) var userId: String? = nil
    /// The account's email, kept on this device for Settings (the API doesn't return it).
    private(set) var email: String? = nil

    /// `nil` when `API_BASE_URL` isn't set. Also used directly for presigned URLs, which take no token.
    let api: APIClient?
    private let secureStore: any SecureStore
    private let defaults: UserDefaults

    private static let deviceIdKey = "deviceId"
    private static let tokenKey = "token"
    private static let kindKey = "kind"
    private static let userIdKey = "userId"
    private static let emailKey = "email"
    private static let configKey = "clientConfig"

    init(api: APIClient?, secureStore: any SecureStore, defaults: UserDefaults = .standard) {
        self.api = api
        self.secureStore = secureStore
        self.defaults = defaults
        if let data = defaults.data(forKey: Self.configKey) {
            config = try? JSONDecoder().decode(ClientConfig.self, from: data)
        }
        loadIdentity()
    }

    var isAccount: Bool { kind == .account }

    /// The stored account session, without signing in (sync must never create a guest).
    var account: Account? {
        guard kind == .account,
              let token = secureStore.string(for: Self.tokenKey),
              let userId
        else { return nil }
        return Account(token: token, userId: userId)
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

    /// Runs `request` with the stored token, signing in as a guest first if there's none.
    /// A 401 for a guest means the token is no longer valid: sign in again once and retry once.
    /// Account tokens have no refresh endpoint yet, so their 401 goes to the caller.
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
        } catch let error as APIError where error.status == 401 && kind != .account {
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

    // MARK: - Accounts (T-403i)

    /// Creates an account. The guest token is sent, so the server links the guest's data to it.
    func signUp(email: String, password: String) async throws {
        guard let api else { throw SessionError.missingBaseURL }
        let response = try await api.signUp(Credentials(email: email, password: password), guestToken: guestToken)
        save(response, email: email)
    }

    /// Logs in. The guest token is sent, so the server merges the guest's data into the account.
    func logIn(email: String, password: String) async throws {
        guard let api else { throw SessionError.missingBaseURL }
        let response = try await api.logIn(Credentials(email: email, password: password), guestToken: guestToken)
        save(response, email: email)
    }

    /// Signs out of the account and falls back to this device's guest identity. The local library
    /// stays. If the guest sign-in fails (offline), the next request signs in instead.
    func logOut() async {
        clearIdentity()
        guard let api else { return }
        _ = try? await signIn(using: api)
    }

    // MARK: - Private

    private var guestToken: String? {
        kind == .guest ? secureStore.string(for: Self.tokenKey) : nil
    }

    private func signIn(using api: APIClient) async throws -> String {
        let response = try await api.anonymousAuth(deviceId: ensureDeviceId())
        save(response, email: nil)
        return response.token
    }

    private func save(_ response: AuthResponse, email: String?) {
        secureStore.set(response.token, for: Self.tokenKey)
        secureStore.set(response.kind.rawValue, for: Self.kindKey)
        secureStore.set(response.userId, for: Self.userIdKey)
        secureStore.set(response.kind == .account ? email : nil, for: Self.emailKey)
        loadIdentity()
    }

    private func clearIdentity() {
        for key in [Self.tokenKey, Self.kindKey, Self.userIdKey, Self.emailKey] {
            secureStore.set(nil, for: key)
        }
        loadIdentity()
    }

    private func loadIdentity() {
        guard secureStore.string(for: Self.tokenKey) != nil else {
            kind = nil
            userId = nil
            email = nil
            return
        }
        kind = secureStore.string(for: Self.kindKey).flatMap(AuthResponse.Kind.init(rawValue:)) ?? .guest
        userId = secureStore.string(for: Self.userIdKey)
        email = secureStore.string(for: Self.emailKey)
    }
}
