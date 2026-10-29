import SwiftUI

/// DESIGN.md §3.10 "Bring your music with you" → Confirm & sync. The server already linked the
/// guest data at login; confirming runs the first sync. Closing keeps everything local too.
/// Ports MigrationScreen.kt.
struct MigrationView: View {
    let onDone: () -> Void

    @Environment(SyncService.self) private var sync
    @State private var summary: MigrationSummary?

    var body: some View {
        ScrollView {
            VStack(spacing: Spacing.md) {
                Image(systemName: "icloud.and.arrow.up")
                    .font(.title)
                    .foregroundStyle(Palette.primary)
                    .frame(width: ComponentSize.recordButton, height: ComponentSize.recordButton)
                    .background(Palette.primarySoft, in: .rect(cornerRadius: Radius.card))
                    .accessibilityHidden(true)
                Text("Bring your music with you")
                    .font(.sectionHeader)
                    .foregroundStyle(Palette.text)
                    .multilineTextAlignment(.center)
                    .accessibilityAddTraits(.isHeader)
                Text("Move local music to your account.")
                    .font(.detail)
                    .foregroundStyle(Palette.textSecondary)

                if let summary {
                    VStack(spacing: Spacing.xs) {
                        CheckRow(
                            isChecked: true,
                            title: AccountFormat.plural(summary.timelines, "chord timeline"),
                            subtitle: "Analysis, tempo and key data"
                        )
                        CheckRow(
                            isChecked: true,
                            title: AccountFormat.plural(summary.ownAudio, "local audio file"),
                            subtitle: "Eligible for private sync · \(AccountFormat.bytes(summary.ownAudioBytes))"
                        )
                        CheckRow(
                            isChecked: false,
                            title: AccountFormat.plural(summary.restricted, "restricted track"),
                            subtitle: "Timelines sync; source audio stays local"
                        )
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
            summary = MigrationSummary.of(sync.unsyncedItems(), audioDirectory: sync.audioDirectory)
        }
    }
}

private struct CheckRow: View {
    let isChecked: Bool
    let title: String
    let subtitle: String

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
        }
        .padding(Spacing.md)
        .background(Palette.surface, in: .rect(cornerRadius: Radius.row))
        .overlay {
            RoundedRectangle(cornerRadius: Radius.row).strokeBorder(Palette.line, lineWidth: 1)
        }
        .accessibilityElement(children: .ignore)
        .accessibilityLabel("\(title), \(subtitle), \(isChecked ? "included" : "not included")")
    }
}

#if DEBUG
#Preview {
    NavigationStack {
        MigrationView(onDone: {})
    }
    .environment(SyncService.preview())
    .preferredColorScheme(.dark)
}
#endif
