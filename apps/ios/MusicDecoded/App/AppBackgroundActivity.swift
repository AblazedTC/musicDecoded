import UIKit

/// `BackgroundActivity` backed by `UIApplication` background tasks: iOS gives the app about
/// 30 seconds after it leaves the foreground, so a started upload can usually finish.
enum AppBackgroundActivity {
    @MainActor
    static func begin(_ name: String) -> @MainActor () -> Void {
        let token = Token()
        token.id = UIApplication.shared.beginBackgroundTask(withName: name) {
            MainActor.assumeIsolated { token.end() }
        }
        return { token.end() }
    }

    @MainActor
    private final class Token {
        var id: UIBackgroundTaskIdentifier = .invalid

        func end() {
            guard id != .invalid else { return }
            UIApplication.shared.endBackgroundTask(id)
            id = .invalid
        }
    }
}
