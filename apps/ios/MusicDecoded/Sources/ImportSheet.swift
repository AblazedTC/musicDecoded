import SwiftUI

/// DESIGN.md §3.3, with the corrected (server-side analysis) copy.
struct ImportSheet: View {
    static let note = "MusicDecoded uploads this file to our servers to analyze it and create an interactive chord timeline. A copy stays on your device for offline playback."

    let model: ImportModel
    let onImport: () -> Void
    let onChooseAnother: () -> Void

    var body: some View {
        ScrollView {
            VStack(alignment: .leading, spacing: Spacing.md) {
                if model.isPreparing {
                    HStack(spacing: Spacing.sm) {
                        ProgressView()
                            .tint(Palette.primary)
                        Text("Preparing file…")
                            .font(.detail)
                            .foregroundStyle(Palette.textSecondary)
                    }
                } else if let draft = model.draft {
                    FileHeader(draft: draft)
                    AdaptiveStack {
                        InfoChip(title: "Duration", value: durationText(draft), systemImage: "clock")
                        InfoChip(title: "Quality", value: qualityText(draft), systemImage: "waveform")
                        InfoChip(title: "Storage", value: "On device", systemImage: "iphone")
                    }
                    PolicyNote(text: Self.note, systemImage: "icloud.and.arrow.up")
                }

                if let errorMessage = model.errorMessage {
                    // Retry only once there's a file to send again.
                    ErrorState(message: errorMessage, onRetry: model.draft != nil && model.isStarting == false ? onImport : nil)
                }

                if model.draft != nil {
                    PrimaryButton(
                        title: model.isStarting ? "Uploading…" : "Import & decode",
                        systemImage: "waveform",
                        action: onImport
                    )
                    .disabled(model.isStarting)
                }

                SecondaryButton(title: "Choose another file", systemImage: "arrow.left.arrow.right", action: onChooseAnother)
                    .disabled(model.isStarting || model.isPreparing)
            }
            .padding(.horizontal, Spacing.md)
            .padding(.vertical, Spacing.xl)
        }
        .background(Palette.surface)
        .presentationDetents([.medium, .large])
        .presentationDragIndicator(.visible)
        .presentationCornerRadius(0)
        .presentationBackground(Palette.surface)
        .interactiveDismissDisabled(model.isStarting)
    }

    private func durationText(_ draft: ImportDraft) -> String {
        guard let seconds = draft.durationSec else { return "—" }
        return Duration.seconds(seconds).formatted(.time(pattern: .minuteSecond))
    }

    private func qualityText(_ draft: ImportDraft) -> String {
        guard let bitrate = draft.bitrate else { return draft.typeLabel }
        return "\(bitrate / 1000) kbps"
    }
}

/// File icon, name, and `TYPE · size`.
private struct FileHeader: View {
    let draft: ImportDraft

    var body: some View {
        HStack(spacing: Spacing.sm) {
            SourceIconTile(systemImage: "music.note", fill: SourceTint.files.fill)

            VStack(alignment: .leading, spacing: Spacing.xxs) {
                Text(draft.fileName)
                    .font(.rowTitle)
                    .foregroundStyle(Palette.text)
                    .lineLimit(2)
                Text("\(draft.typeLabel) · \(AccountFormat.bytes(Int64(draft.sizeBytes)))")
                    .font(.detail)
                    .foregroundStyle(Palette.textSecondary)
            }
        }
        .accessibilityElement(children: .combine)
    }
}

#Preview {
    Color.clear
        .sheet(isPresented: .constant(true)) {
            ImportSheet(
                model: ImportModel(
                    draft: ImportDraft(
                        fileURL: URL(filePath: "/tmp/pending.mp3"),
                        fileName: "Somehow.mp3",
                        title: "Somehow",
                        sizeBytes: 8_400_000,
                        durationSec: 263,
                        bitrate: 256_000
                    )
                ),
                onImport: {},
                onChooseAnother: {}
            )
        }
        .preferredColorScheme(.dark)
}
