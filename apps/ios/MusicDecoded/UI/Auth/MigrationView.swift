import SwiftUI

/// DESIGN.md §3.10 "Bring your music with you" → Confirm & sync. The server already linked the
/// guest data at login; confirming runs the first sync. Closing keeps everything local too.
/// Ports MigrationScreen.kt.
struct MigrationView: View {
    let onDone: () -> Void

    @Environment(SyncService.self) private var sync
    @Environment(SessionStore.self) private var session
    @Environment(\.openPaywall) private var openPaywall
    @State private var summary: MigrationSummary?

    var body: some View {
        ScrollView {
            VStack(spacing: Spacing.md) {
                Image(systemName: "icloud.and.arrow.up.fill")
                    .font(.title.weight(.bold))
                    .foregroundStyle(Palette.onColor)
                    .frame(width: ComponentSize.recordButton, height: ComponentSize.recordButton)
                    .block(Palette.yellow, shadow: HardShadow.md, shadowColor: Palette.violet)
                    .tilt(Tilt.sticker)
                    .padding(.bottom, Spacing.xs)
                    .accessibilityHidden(true)
                DisplayHeadline(text: "Bring your music with you", font: .sectionHeader)
                    .multilineTextAlignment(.center)
                Text("Move local music to your account.")
                    .font(.bodyText)
                    .foregroundStyle(Palette.textSecondary)

                if let summary {
                    VStack(spacing: Spacing.xs) {
                        MigrationCheckRow(
                            isChecked: true,
                            title: AccountFormat.plural(summary.timelines, "chord timeline"),
                            subtitle: "Analysis, tempo and key data"
                        )
                        if summary.ownAudioSyncs {
                            MigrationCheckRow(
                                isChecked: true,
                                title: AccountFormat.plural(summary.ownAudio, "local audio file"),
                                subtitle: "Eligible for private sync · \(AccountFormat.bytes(summary.ownAudioBytes))"
                            )
                        } else {
                            MigrationCheckRow(
                                isChecked: false,
                                title: AccountFormat.plural(summary.ownAudio, "local audio file"),
                                subtitle: "Private audio sync · \(AccountFormat.bytes(summary.ownAudioBytes))",
                                isPro: true
                            )
                        }
                        MigrationCheckRow(
                            isChecked: false,
                            title: AccountFormat.plural(summary.restricted, "restricted track"),
                            subtitle: "Timelines sync; source audio stays local"
                        )
                    }
                    if summary.ownAudioSyncs == false {
                        ProAudioNote(openPaywall: openPaywall)
                    }
                }

                PolicyNote(text: "Nothing is removed from this device. Restricted third-party audio never uploads.")
                PrimaryButton(title: "Confirm & sync") {
                    sync.requestSync()
                    onDone()
                }
                .disabled(summary == nil)
            }
            .padding(.horizontal, Spacing.md)
            .padding(.vertical, Spacing.lg)
        }
        .background(Palette.canvas)
        .navigationBarTitleDisplayMode(.inline)
        .navigationBarBackButtonHidden()
        .toolbarBackground(Palette.canvas, for: .navigationBar)
        .toolbar(.hidden, for: .tabBar)
        .toolbar {
            ToolbarItem(placement: .topBarLeading) {
                Button("Not now", systemImage: "xmark", action: onDone)
                    .tint(Palette.text)
            }
        }
        .task {
            summary = MigrationSummary.of(sync.unsyncedItems(), audioDirectory: sync.audioDirectory, pro: session.isPro)
        }
    }
}

/// Free accounts: own audio stays on this device. Tapping the note opens the paywall (when there
/// is one to open, i.e. not already Pro).
struct ProAudioNote: View {
    let openPaywall: OpenPaywallAction?

    static let text = "Syncing your own audio files is part of Pro. Their chord timelines still sync."

    var body: some View {
        if let openPaywall {
            Button {
                openPaywall()
            } label: {
                PolicyNote(text: Self.text + " Tap to see Pro.", systemImage: "sparkles")
            }
            .buttonStyle(.plain)
            .accessibilityHint("See Pro")
        } else {
            PolicyNote(text: Self.text, systemImage: "sparkles")
        }
    }
}

struct MigrationCheckRow: View {
    let isChecked: Bool
    let title: String
    let subtitle: String
    var isPro = false

    var body: some View {
        HStack(spacing: Spacing.sm) {
            Image(systemName: isChecked ? "checkmark.square.fill" : "square")
                .font(.title3)
                .foregroundStyle(isChecked ? Palette.primary : Palette.textMuted)
                .accessibilityHidden(true)
            VStack(alignment: .leading, spacing: Spacing.xxs) {
                Text(title)
                    .font(.rowTitle)
                    .foregroundStyle(isChecked ? Palette.text : Palette.textMuted)
                Text(subtitle)
                    .font(.detail)
                    .foregroundStyle(Palette.textSecondary)
            }
            .frame(maxWidth: .infinity, alignment: .leading)
            if isPro {
                Sticker(text: "Pro", fill: Palette.yellow)
            }
        }
        .padding(Spacing.md)
        .inkCard()
        .accessibilityElement(children: .ignore)
        .accessibilityLabel(Self.accessibilityLabel(title: title, subtitle: subtitle, isChecked: isChecked, isPro: isPro))
    }

    static func accessibilityLabel(title: String, subtitle: String, isChecked: Bool, isPro: Bool) -> String {
        "\(title), \(subtitle), \(isPro ? "Pro only, " : "")\(isChecked ? "included" : "not included")"
    }
}

#if DEBUG
#Preview {
    NavigationStack {
        MigrationView(onDone: {})
    }
    .environment(SyncService.preview())
    .environment(SessionStore.preview())
    .preferredColorScheme(.dark)
}
#endif
