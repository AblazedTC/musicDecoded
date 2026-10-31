import Foundation
import MusicDecodedCore
import Observation

enum AuthMode: String, CaseIterable, Hashable, Sendable {
    case signUp, logIn

    var title: String {
        switch self {
        case .signUp: "Sign up"
        case .logIn: "Log in"
        }
    }
}

/// Where to go after signing in: the migration screen when this device has guest data.
enum AuthOutcome: Equatable, Sendable {
    case migrate, done
}

/// Pushed account screens (DESIGN.md §2: no tab bar).
enum AccountRoute: Hashable {
    case auth(AuthMode)
    case migrate
}

/// What `AuthModel.submit` calls. The app passes `SessionStore` and `SyncService`; tests pass fakes.
struct AuthActions {
    var signUp: @MainActor (_ email: String, _ password: String) async throws -> Void
    var logIn: @MainActor (_ email: String, _ password: String) async throws -> Void
    var hasGuestData: @MainActor () -> Bool
    var requestSync: @MainActor () -> Void
}

/// Sign up / log in (T-403i). The guest token goes with the request, so the server moves the
/// guest's jobs and uploads to the account (`SessionStore`). Then: guest data → migration screen;
/// otherwise sync right away (PLAN.md §5.2 "after login"). Ports AuthViewModel.kt.
@MainActor
@Observable
final class AuthModel {
    static let minPasswordCharacters = 8
    static let maxPasswordBytes = 72

    var mode: AuthMode
    var email = ""
    var password = ""
    var error: String? = nil
    private(set) var isBusy = false
    private(set) var outcome: AuthOutcome? = nil

    init(mode: AuthMode) {
        self.mode = mode
    }

    func submit(using actions: AuthActions) async {
        guard isBusy == false else { return }
        let trimmedEmail = self.email.trimmingCharacters(in: .whitespacesAndNewlines)
        let currentPassword = self.password
        if let message = Self.validate(email: trimmedEmail, password: currentPassword) {
            self.error = message
            return
        }
        isBusy = true
        self.error = nil
        do {
            if mode == .signUp {
                try await actions.signUp(trimmedEmail, currentPassword)
            } else {
                try await actions.logIn(trimmedEmail, currentPassword)
            }
            if actions.hasGuestData() {
                outcome = .migrate
            } else {
                actions.requestSync()
                outcome = .done
            }
        } catch {
            self.error = Self.errorMessage(for: error)
        }
        isBusy = false
    }

    /// Client-side checks mirroring `Credentials` in openapi; `nil` when valid.
    static func validate(email: String, password: String) -> String? {
        if email.range(of: "^[^@\\s]+@[^@\\s]+\\.[^@\\s]+$", options: .regularExpression) == nil {
            return "Enter a valid email address."
        }
        if password.unicodeScalars.count < minPasswordCharacters {
            return "Use at least 8 characters for your password."
        }
        if password.utf8.count > maxPasswordBytes {
            return "This password is too long (72 bytes max; some characters count more than once)."
        }
        return nil
    }

    /// The API's own message (e.g. "Invalid email or password"), or a connection hint.
    static func errorMessage(for error: any Error) -> String {
        switch error {
        case let error as APIError:
            return error.message.isEmpty ? Transcription.genericMessage : error.message
        case is URLError:
            return Transcription.offlineMessage
        default:
            return Transcription.genericMessage
        }
    }
}

/// What the migration screen lists (DESIGN.md §3.10).
struct MigrationSummary: Equatable, Sendable {
    /// Chord timelines (analysis, tempo and key data): every guest row.
    var timelines: Int
    /// file/recording rows: their audio is eligible for private sync.
    var ownAudio: Int
    var ownAudioBytes: Int64
    /// youtube/soundcloud rows: the timeline syncs, the audio never does.
    var restricted: Int
    /// Own-audio sync is Pro only (PLAN.md §9); free accounts sync timelines only.
    var ownAudioSyncs = true

    @MainActor
    static func of(_ items: [LibraryItem], audioDirectory: URL, pro: Bool = true) -> MigrationSummary {
        let own = items.filter { $0.source == .file || $0.source == .recording }
        var bytes: Int64 = 0
        for item in own {
            guard let fileName = item.localAudioPath else { continue }
            let url = audioDirectory.appending(path: fileName, directoryHint: .notDirectory)
            let size = (try? url.resourceValues(forKeys: [.fileSizeKey]).fileSize) ?? 0
            bytes += Int64(size)
        }
        return MigrationSummary(
            timelines: items.count,
            ownAudio: own.count,
            ownAudioBytes: bytes,
            restricted: items.filter { $0.source == .youtube || $0.source == .soundcloud }.count,
            ownAudioSyncs: pro
        )
    }
}

/// Text helpers for the account and settings screens.
enum AccountFormat {
    /// `1 chord timeline`, `3 chord timelines`.
    static func plural(_ count: Int, _ noun: String) -> String {
        "\(count) \(noun)\(count == 1 ? "" : "s")"
    }

    /// The one file-size format (import sheet, migration, Settings storage): decimal units, one
    /// decimal for MB and GB. `0 KB`, `640 KB`, `6.0 MB`, `1.2 GB`.
    static func bytes(_ count: Int64) -> String {
        if count < 999_500 {
            return "\(Int((Double(count) / 1000).rounded())) KB"
        }
        if count < 999_950_000 {
            return String(format: "%.1f MB", Double(count) / 1_000_000)
        }
        return String(format: "%.1f GB", Double(count) / 1_000_000_000)
    }
}
