import SwiftUI

/// Opens the paywall from anywhere below `RootView`; `nil` when the user is already Pro (no upsell).
struct OpenPaywallAction: Sendable {
    let action: @MainActor @Sendable () -> Void

    @MainActor
    func callAsFunction() {
        action()
    }

    /// Whether an error message is the daily plan limit (429 `rate_limited` from `POST /jobs`), which
    /// offers "Go Pro". The YouTube search quota has its own message and never does (DESIGN.md §3.11).
    static func opens(for message: String?) -> Bool {
        message == Transcription.rateLimitedMessage
    }
}

private struct OpenPaywallKey: EnvironmentKey {
    static var defaultValue: OpenPaywallAction? { nil }
}

extension EnvironmentValues {
    var openPaywall: OpenPaywallAction? {
        get { self[OpenPaywallKey.self] }
        set { self[OpenPaywallKey.self] = newValue }
    }
}
