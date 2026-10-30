import SwiftUI

/// DESIGN.md §3.4 video detail: hero (plays the official embed inline), title, channel, the
/// availability card, "Decode this song" → `youtube` job → Processing, and the R-1 policy note.
struct VideoDetailScreen: View {
    let video: VideoSummary
    /// Gets the trackId once the job exists (the caller pushes Processing).
    let onDecodeStarted: (String) -> Void

    /// PLAN.md R-1, verbatim from DESIGN.md §3.4. Keep it.
    static let policyNote = "Offline audio is saved only when permitted. The chord timeline can remain available separately."

    @Environment(TranscriptionService.self) private var transcription
    @State private var isPlayingInline = false
    @State private var isStarting = false
    @State private var errorMessage: String?

    var body: some View {
        ScrollView {
            VStack(alignment: .leading, spacing: Spacing.lg) {
                hero

                VStack(alignment: .leading, spacing: Spacing.xxs) {
                    Text(video.displayTitle)
                        .font(.sectionHeader)
                        .foregroundStyle(Palette.text)
                        .accessibilityAddTraits(.isHeader)
                    let details = [video.channelTitle.isEmpty ? nil : video.channelTitle, video.durationText]
                        .compactMap { $0 }
                        .joined(separator: " · ")
                    if details.isEmpty == false {
                        Text(details)
                            .font(.detail)
                            .foregroundStyle(Palette.textSecondary)
                    }
                }

                AvailabilityCard()

                VStack(alignment: .leading, spacing: Spacing.sm) {
                    if let errorMessage {
                        ErrorState(message: errorMessage, onRetry: isStarting ? nil : { decode() })
                    }
                    PrimaryButton(title: isStarting ? "Starting…" : "Decode this song", systemImage: "waveform") {
                        decode()
                    }
                    .disabled(isStarting)
                    PolicyNote(text: Self.policyNote)
                }
            }
            .padding(.horizontal, Spacing.md)
            .padding(.vertical, Spacing.md)
        }
        .background(Palette.canvas)
        .navigationTitle("YouTube")
        .navigationBarTitleDisplayMode(.inline)
        .toolbarBackground(Palette.canvas, for: .navigationBar)
        .toolbar(.hidden, for: .tabBar)
    }

    private var hero: some View {
        ZStack {
            if isPlayingInline {
                YouTubeEmbedView(videoId: video.videoId)
            } else {
                VideoThumbnail(url: video.thumbnailUrl)
                Button {
                    isPlayingInline = true
                } label: {
                    Image(systemName: "play.fill")
                        .font(.title2)
                        .foregroundStyle(Palette.onPrimary)
                        .frame(width: ComponentSize.playButton, height: ComponentSize.playButton)
                        .background(Palette.primary, in: .circle)
                        .contentShape(.circle)
                }
                .buttonStyle(.plain)
                .accessibilityLabel("Play video")
            }
        }
        .frame(maxWidth: .infinity)
        .frame(height: ComponentSize.videoHeroHeight)
        .background(Palette.ink)
        .clipShape(.rect(cornerRadius: Radius.card))
    }

    private func decode() {
        guard isStarting == false else { return }
        isStarting = true
        errorMessage = nil
        Task {
            do {
                let trackId = try await transcription.startYouTube(video)
                isStarting = false
                onDecodeStarted(trackId)
            } catch {
                isStarting = false
                // A 429 here is the user's daily plan limit, so its message includes the upsell.
                errorMessage = Transcription.userMessage(for: error)
            }
        }
    }
}

/// "Analysis available": the chord timeline can be made for this video.
private struct AvailabilityCard: View {
    var body: some View {
        HStack(spacing: Spacing.sm) {
            Image(systemName: "checkmark.circle.fill")
                .font(.title3)
                .foregroundStyle(Palette.primary)
                .accessibilityHidden(true)
            VStack(alignment: .leading, spacing: Spacing.xxs) {
                Text("Analysis available")
                    .font(.rowTitle)
                    .foregroundStyle(Palette.text)
                Text("Chords, key and tempo, synced to the official player.")
                    .font(.detail)
                    .foregroundStyle(Palette.textSecondary)
            }
            Spacer(minLength: 0)
        }
        .padding(Spacing.md)
        .background(Palette.primarySoft, in: .rect(cornerRadius: Radius.card))
        .accessibilityElement(children: .combine)
    }
}

#if DEBUG
#Preview {
    NavigationStack {
        VideoDetailScreen(
            video: VideoSummary(pastedVideoId: "dQw4w9WgXcQ"),
            onDecodeStarted: { _ in }
        )
    }
    .environment(TranscriptionService.preview())
    .preferredColorScheme(.dark)
}
#endif
