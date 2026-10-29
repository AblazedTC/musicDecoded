import MusicDecodedCore
import SwiftUI

/// "Paste a video URL" (DESIGN.md §3.4): parses a watch / youtu.be / shorts / music link into a
/// videoId and opens its detail. Works without search, so it costs no quota (PLAN.md R-6).
struct PasteVideoURLCard: View {
    @State private var text = ""
    @State private var showInvalid = false
    let onOpen: (VideoSummary) -> Void

    var body: some View {
        VStack(alignment: .leading, spacing: Spacing.sm) {
            HStack(spacing: Spacing.sm) {
                Image(systemName: "link")
                    .font(.title3)
                    .foregroundStyle(SourceTint.youtube.icon)
                    .frame(width: ComponentSize.sourceIconTile, height: ComponentSize.sourceIconTile)
                    .background(SourceTint.youtube.background, in: .rect(cornerRadius: Radius.control))
                    .accessibilityHidden(true)
                VStack(alignment: .leading, spacing: Spacing.xxs) {
                    Text("Paste a video URL")
                        .font(.rowTitle)
                        .foregroundStyle(Palette.text)
                    Text("youtube.com, youtu.be, Shorts and YouTube Music links")
                        .font(.detail)
                        .foregroundStyle(Palette.textSecondary)
                }
            }

            HStack(spacing: Spacing.xs) {
                TextField("https://youtu.be/...", text: $text)
                    .font(.subheadline)
                    .foregroundStyle(Palette.text)
                    .tint(Palette.primary)
                    .keyboardType(.URL)
                    .textContentType(.URL)
                    .textInputAutocapitalization(.never)
                    .autocorrectionDisabled()
                    .submitLabel(.go)
                    .onSubmit(open)
                    .padding(.horizontal, Spacing.sm)
                    .frame(minHeight: ComponentSize.searchFieldHeight)
                    .background(Palette.canvas, in: .rect(cornerRadius: Radius.control))
                    .overlay {
                        RoundedRectangle(cornerRadius: Radius.control).strokeBorder(Palette.line, lineWidth: 1)
                    }
                Button("Open", action: open)
                    .font(.detail.weight(.semibold))
                    .foregroundStyle(Palette.primary)
                    .frame(minWidth: ComponentSize.minTouchTarget, minHeight: ComponentSize.minTouchTarget)
                    .disabled(text.isEmpty)
            }

            if showInvalid {
                Text("That doesn't look like a YouTube video link.")
                    .font(.detail)
                    .foregroundStyle(Palette.record)
            }
        }
        .padding(Spacing.md)
        .background(Palette.surface, in: .rect(cornerRadius: Radius.card))
        .overlay {
            RoundedRectangle(cornerRadius: Radius.card).strokeBorder(Palette.line, lineWidth: 1)
        }
        .onChange(of: text) {
            showInvalid = false
        }
    }

    private func open() {
        guard let videoId = YouTubeVideoID.parse(text) else {
            showInvalid = true
            return
        }
        onOpen(VideoSummary(pastedVideoId: videoId))
    }
}
