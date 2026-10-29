import Foundation
import MusicDecodedCore
import SwiftData
import Testing
@testable import MusicDecoded

private let accountResponse = Data("""
{"token": "account-token", "userId": "account-1", "kind": "account", "plan": "free"}
""".utf8)

/// Sign up / log in, the migration summary and the account session. Ports
/// apps/android/.../ui/auth/AccountTest.kt.
@MainActor
struct AccountTests {
    private final class Recorder {
        var syncs = 0
    }

    @Test func passwordIs8CharactersTo72Utf8Bytes() {
        let validate = AuthModel.validate(email:password:)
        #expect(validate("a@b.co", "short") != nil)
        #expect(validate("a@b.co", "12345678") == nil)
        #expect(validate("a@b.co", String(repeating: "a", count: 72)) == nil)
        #expect(validate("a@b.co", String(repeating: "a", count: 73)) != nil)
        // 8 characters but 3 bytes each: valid length, 24 bytes.
        #expect(validate("a@b.co", String(repeating: "♯", count: 8)) == nil)
        // 25 × 3 bytes = 75 > 72, although only 25 characters.
        #expect(validate("a@b.co", String(repeating: "♯", count: 25)) != nil)
        #expect(validate("not-an-email", "12345678") != nil)
    }

    private func submit(guestData: Bool, failure: (any Error)? = nil, recorder: Recorder) async -> AuthModel {
        let model = AuthModel(mode: .logIn)
        model.email = "me@example.com"
        model.password = "password1"
        let actions = AuthActions(
            signUp: { _, _ in },
            logIn: { _, _ in
                if let failure { throw failure }
            },
            hasGuestData: { guestData },
            requestSync: { recorder.syncs += 1 }
        )
        await model.submit(using: actions)
        return model
    }

    @Test func loginWithGuestDataGoesToMigrationWithoutSyncingYet() async {
        let recorder = Recorder()
        let model = await submit(guestData: true, recorder: recorder)
        #expect(model.outcome == .migrate)
        #expect(recorder.syncs == 0)
    }

    @Test func loginWithoutGuestDataSyncsRightAway() async {
        let recorder = Recorder()
        let model = await submit(guestData: false, recorder: recorder)
        #expect(model.outcome == .done)
        #expect(recorder.syncs == 1)
    }

    @Test func showsTheApiErrorMessage() async {
        let error = APIError(code: .unauthorized, message: "Invalid email or password", status: 401)
        let model = await submit(guestData: false, failure: error, recorder: Recorder())
        #expect(model.error == "Invalid email or password")
        #expect(model.outcome == nil)
        #expect(model.isBusy == false)
    }

    @Test func migrationSummaryCountsTimelinesOwnAudioAndRestricted() throws {
        let directory = FileManager.default.temporaryDirectory
            .appending(path: "AccountTests-\(UUID().uuidString)", directoryHint: .isDirectory)
        defer { try? FileManager.default.removeItem(at: directory) }
        try FileManager.default.createDirectory(at: directory, withIntermediateDirectories: true)
        try Data(count: 1000).write(to: directory.appending(path: "a.mp3"))
        try Data(count: 500).write(to: directory.appending(path: "b.m4a"))

        func row(_ id: String, _ source: Analysis.Source, _ fileName: String? = nil) -> LibraryItem {
            let item = LibraryItem(trackId: id, title: id, source: source, status: .ready)
            item.localAudioPath = fileName
            return item
        }
        let summary = MigrationSummary.of(
            [
                row("a", .file, "a.mp3"),
                row("b", .recording, "b.m4a"),
                row("c", .youtube),
                row("d", .soundcloud),
            ],
            audioDirectory: directory
        )

        #expect(summary == MigrationSummary(timelines: 4, ownAudio: 2, ownAudioBytes: 1500, restricted: 2))
    }

    // MARK: - Session

    private func makeDefaults() -> UserDefaults {
        UserDefaults(suiteName: "AccountTests-\(UUID().uuidString)")!
    }

    @Test func loginSendsTheGuestTokenAndStoresTheAccount() async throws {
        let log = RequestLog()
        let api = StubURLProtocol.client { request in
            log.record(request)
            return (200, accountResponse)
        }
        let store = InMemorySecureStore(["deviceId": "device-1", "token": "guest-token", "kind": "guest", "userId": "guest-1"])
        let session = SessionStore(api: api, secureStore: store, defaults: makeDefaults())

        try await session.logIn(email: "me@example.com", password: "password1")

        #expect(log.routes == ["POST /auth/login"])
        // The guest bearer links the guest's data to the account.
        #expect(log.all.first?.authorization == "Bearer guest-token")
        let body = try #require(log.all.first?.body)
        #expect(try JSONDecoder().decode([String: String].self, from: body) == ["email": "me@example.com", "password": "password1"])
        #expect(session.isAccount)
        #expect(session.email == "me@example.com")
        #expect(session.account == SessionStore.Account(token: "account-token", userId: "account-1"))
        #expect(store.values["token"] == "account-token")
    }

    @Test func logoutFallsBackToAGuestSession() async throws {
        let api = StubURLProtocol.client { request in
            (200, request.url?.path() == "/auth/anonymous" ? SampleJSON.authResponse : accountResponse)
        }
        let store = InMemorySecureStore(["deviceId": "device-1"])
        let session = SessionStore(api: api, secureStore: store, defaults: makeDefaults())
        try await session.signUp(email: "me@example.com", password: "password1")
        #expect(session.isAccount)

        await session.logOut()

        #expect(session.isAccount == false)
        #expect(session.kind == .guest)
        #expect(session.email == nil)
        #expect(session.account == nil)
        #expect(store.values["token"] == "guest-token-1")
        #expect(store.values["deviceId"] == "device-1")
    }

    @Test func accountUnauthorizedIsNotReplacedByAGuest() async throws {
        let log = RequestLog()
        let api = StubURLProtocol.client { request in
            log.record(request)
            return (401, SampleJSON.unauthorized)
        }
        let store = InMemorySecureStore(["deviceId": "device-1", "token": "account-token", "kind": "account", "userId": "account-1"])
        let session = SessionStore(api: api, secureStore: store, defaults: makeDefaults())

        await #expect(throws: APIError.self) {
            try await session.authorized { api, token in
                try await api.config(token: token)
            }
        }
        #expect(log.routes == ["GET /config"])
        #expect(session.isAccount)
    }

    @Test func tokenFromBeforeAccountsIsAGuest() {
        let store = InMemorySecureStore(["deviceId": "device-1", "token": "old-token"])
        let session = SessionStore(api: nil, secureStore: store, defaults: makeDefaults())
        #expect(session.kind == .guest)
        #expect(session.account == nil)
    }

    // MARK: - Preferences

    @Test func preferencesPersist() {
        let defaults = makeDefaults()
        let settings = SettingsStore(defaults: defaults)
        #expect(settings.preferences == PlayerPreferences())

        settings.update {
            $0.defaultSpeed = 0.75
            $0.chordsMode = .basic
            $0.accidentals = .flat
            $0.visualizer = .video
        }

        let reopened = SettingsStore(defaults: defaults)
        #expect(reopened.preferences == PlayerPreferences(defaultSpeed: 0.75, chordsMode: .basic, accidentals: .flat, visualizer: .video))
    }
}
