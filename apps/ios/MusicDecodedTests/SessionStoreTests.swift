import Foundation
import Testing
@testable import MusicDecoded

/// Guest session against a stubbed API: no network, no real Keychain.
@MainActor
struct SessionStoreTests {
    /// A throwaway defaults suite, so the cached config doesn't leak between tests.
    private func makeDefaults() -> UserDefaults {
        UserDefaults(suiteName: "SessionStoreTests-\(UUID().uuidString)")!
    }

    @Test func firstLaunchCreatesDeviceIdSignsInAndFetchesConfig() async throws {
        let log = RequestLog()
        let api = StubURLProtocol.client { request in
            log.record(request)
            switch (request.httpMethod, request.url?.path()) {
            case ("POST", "/auth/anonymous"): return (200, SampleJSON.authResponse)
            case ("GET", "/config"): return (200, SampleJSON.clientConfig)
            default: return (404, SampleJSON.notFound)
            }
        }
        let store = InMemorySecureStore()
        let session = SessionStore(api: api, secureStore: store, defaults: makeDefaults())

        await session.start()

        let deviceId = try #require(store.values["deviceId"])
        #expect(UUID(uuidString: deviceId) != nil)
        #expect(store.values["token"] == "guest-token-1")
        #expect(log.routes == ["POST /auth/anonymous", "GET /config"])

        let authBody = try #require(log.all.first?.body)
        let sent = try JSONDecoder().decode([String: String].self, from: authBody)
        #expect(sent == ["deviceId": deviceId, "platform": "ios"])
        #expect(log.all.last?.authorization == "Bearer guest-token-1")

        #expect(session.config?.limits.dailyJobs == 10)
        #expect(session.errorMessage == nil)
    }

    @Test func storedTokenSkipsSignIn() async {
        let log = RequestLog()
        let api = StubURLProtocol.client { request in
            log.record(request)
            return (200, SampleJSON.clientConfig)
        }
        let store = InMemorySecureStore(["deviceId": "device-1", "token": "stored-token"])
        let session = SessionStore(api: api, secureStore: store, defaults: makeDefaults())

        await session.start()

        #expect(log.routes == ["GET /config"])
        #expect(log.all.first?.authorization == "Bearer stored-token")
        #expect(store.values["deviceId"] == "device-1")
        #expect(session.config != nil)
    }

    @Test func unauthorizedReauthenticatesOnceAndRetries() async {
        let log = RequestLog()
        let api = StubURLProtocol.client { request in
            let earlier = log.record(request)
            switch (request.httpMethod, request.url?.path()) {
            case ("POST", "/auth/anonymous"): return (200, SampleJSON.authResponse)
            case ("GET", "/config"): return earlier == 0 ? (401, SampleJSON.unauthorized) : (200, SampleJSON.clientConfig)
            default: return (404, SampleJSON.notFound)
            }
        }
        let store = InMemorySecureStore(["deviceId": "device-1", "token": "expired-token"])
        let session = SessionStore(api: api, secureStore: store, defaults: makeDefaults())

        await session.start()

        #expect(log.routes == ["GET /config", "POST /auth/anonymous", "GET /config"])
        #expect(log.all.map(\.authorization) == ["Bearer expired-token", nil, "Bearer guest-token-1"])
        #expect(store.values["token"] == "guest-token-1")
        #expect(session.config != nil)
        #expect(session.errorMessage == nil)
    }

    @Test func secondUnauthorizedIsNotRetriedAgain() async throws {
        let log = RequestLog()
        let api = StubURLProtocol.client { request in
            log.record(request)
            switch (request.httpMethod, request.url?.path()) {
            case ("POST", "/auth/anonymous"): return (200, SampleJSON.authResponse)
            default: return (401, SampleJSON.unauthorized)
            }
        }
        let session = SessionStore(
            api: api,
            secureStore: InMemorySecureStore(["deviceId": "device-1", "token": "expired-token"]),
            defaults: makeDefaults()
        )

        do {
            _ = try await session.authorized { api, token in
                try await api.config(token: token)
            }
            Issue.record("Expected a 401")
        } catch let error as APIError {
            #expect(error.status == 401)
            #expect(error.code == .unauthorized)
        }

        #expect(log.routes == ["GET /config", "POST /auth/anonymous", "GET /config"])
    }

    @Test func configIsCachedForTheNextLaunch() async {
        let defaults = makeDefaults()
        let api = StubURLProtocol.client { _ in (200, SampleJSON.clientConfig) }
        let store = InMemorySecureStore(["deviceId": "device-1", "token": "stored-token"])

        await SessionStore(api: api, secureStore: store, defaults: defaults).start()
        let nextLaunch = SessionStore(api: nil, secureStore: store, defaults: defaults)

        #expect(nextLaunch.config?.maxUploadBytes == 104_857_600)
    }

    @Test func missingBaseURLReportsAnError() async {
        let session = SessionStore(api: nil, secureStore: InMemorySecureStore(), defaults: makeDefaults())

        await session.start()

        #expect(session.errorMessage != nil)
        #expect(session.config == nil)
    }
}
