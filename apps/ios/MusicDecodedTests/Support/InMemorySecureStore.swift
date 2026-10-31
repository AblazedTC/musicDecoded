@testable import MusicDecoded

/// A `SecureStore` that keeps values in memory, so tests never touch the real Keychain.
@MainActor
final class InMemorySecureStore: SecureStore {
    var values: [String: String]

    init(_ values: [String: String] = [:]) {
        self.values = values
    }

    func string(for key: String) -> String? {
        values[key]
    }

    func set(_ value: String?, for key: String) {
        values[key] = value
    }
}
