import SwiftUI

/// The one error state for anything that talks to the server (T-503i): icon, title, message and
/// an optional "Try again". VoiceOver announces it when it appears. The daily plan limit adds a
/// "Go Pro" action that opens the paywall (T-602i). Same copy as Android's ErrorState.
struct ErrorState: View {
    let message: String
    var title: String?
    /// `nil` hides "Try again" (when retrying can't help).
    var onRetry: (() -> Void)?

    @Environment(\.openPaywall) private var openPaywall

    var body: some View {
        let resolvedTitle = title ?? Self.title(for: message)
        VStack(alignment: .leading, spacing: Spacing.xs) {
            HStack(alignment: .firstTextBaseline, spacing: Spacing.xs) {
                Image(systemName: Self.isOffline(message) ? "wifi.slash" : "exclamationmark.circle")
                    .foregroundStyle(Palette.record)
                    .accessibilityHidden(true)
                Text(resolvedTitle)
                    .font(.rowTitle)
                    .foregroundStyle(Palette.text)
                    .fixedSize(horizontal: false, vertical: true)
            }
            Text(message)
                .font(.detail)
                .foregroundStyle(Palette.textSecondary)
                .fixedSize(horizontal: false, vertical: true)
            if let openPaywall, OpenPaywallAction.opens(for: message) {
                PrimaryButton(title: "Go Pro", systemImage: "sparkles") {
                    openPaywall()
                }
            }
            if let onRetry {
                Button(action: onRetry) {
                    Label("Try again", systemImage: "arrow.clockwise")
                        .font(.detail.weight(.semibold))
                        .foregroundStyle(Palette.primary)
                        .frame(minHeight: ComponentSize.minTouchTarget)
                        .contentShape(.rect)
                }
                .buttonStyle(.plain)
            }
        }
        .frame(maxWidth: .infinity, alignment: .leading)
        .padding(Spacing.md)
        .background(Palette.surface, in: .rect(cornerRadius: Radius.card))
        .overlay {
            RoundedRectangle(cornerRadius: Radius.card).strokeBorder(Palette.line, lineWidth: 1)
        }
        .accessibilityElement(children: .contain)
        .onAppear {
            AccessibilityNotification.Announcement("\(resolvedTitle). \(message)").post()
        }
    }

    /// "You're offline" / "Daily limit reached" for the shared copy, otherwise "Something went wrong".
    static func title(for message: String) -> String {
        if isOffline(message) { return "You're offline" }
        if message == Transcription.rateLimitedMessage { return "Daily limit reached" }
        return "Something went wrong"
    }

    static func isOffline(_ message: String) -> Bool {
        message == Transcription.offlineMessage
    }
}

#Preview {
    VStack(spacing: Spacing.sm) {
        ErrorState(message: Transcription.offlineMessage, onRetry: {})
        ErrorState(message: Transcription.rateLimitedMessage)
            .environment(\.openPaywall, OpenPaywallAction {})
    }
    .padding(Spacing.md)
    .background(Palette.canvas)
    .preferredColorScheme(.dark)
}
