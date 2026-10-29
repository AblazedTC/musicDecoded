/// Where the session keeps its secrets. `Keychain` in the app; tests use an in-memory store.
@MainActor
protocol SecureStore {
    func string(for key: String) -> String?
    /// Stores `value`, or removes the entry when it's `nil`.
    func set(_ value: String?, for key: String)
}
