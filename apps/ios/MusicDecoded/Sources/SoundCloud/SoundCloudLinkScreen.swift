import MusicDecodedCore
import SwiftUI
import UIKit

/// The SoundCloud source (T-502i): paste a track link (or take it from the clipboard) and decode
/// it as a `soundcloud` job → Processing. Home shows its card only while `soundcloudEnabled`.
/// iOS can't receive plain https links without universal links, so pasting is the entry point.
struct SoundCloudRoute: Hashable, Sendable {}

struct SoundCloudLinkScreen: View {
    /// Gets the trackId once the job exists (the caller pushes Processing).
    let onDecodeStarted: (String) -> Void

    @Environment(TranscriptionService.self) private var transcription
    @Environment(\.scenePhase) private var scenePhase
    @State private var text = ""
    @State private var isStarting = false
    @State private var errorMessage: String?
    /// The clipboard holds something that could be a link. Checked without reading it, so iOS
    /// doesn't ask for paste permission until the user taps "Paste from clipboard".
    @State private var clipboardHasContent = false

    var body: some View {
        ScrollView {
            VStack(alignment: .leading, spacing: Spacing.lg) {
                header

                VStack(alignment: .leading, spacing: Spacing.xs) {
                    TextField("https://soundcloud.com/...", text: $text)
                        .font(.subheadline)
                        .foregroundStyle(Palette.text)
                        .tint(Palette.primary)
                        .keyboardType(.URL)
                        .textContentType(.URL)
                        .textInputAutocapitalization(.never)
                        .autocorrectionDisabled()
                        .submitLabel(.go)
                        .onSubmit(decode)
                        .padding(.horizontal, Spacing.sm)
                        .frame(minHeight: ComponentSize.searchFieldHeight)
                        .background(Palette.surface, in: .rect(cornerRadius: Radius.control))
                        .overlay {
                            RoundedRectangle(cornerRadius: Radius.control).strokeBorder(Palette.line, lineWidth: 1)
                        }
                        .accessibilityLabel("SoundCloud link")

                    if clipboardHasContent {
                        Button {
                            pasteFromClipboard()
                        } label: {
                            Label("Paste from clipboard", systemImage: "doc.on.clipboard")
                                .font(.detail.weight(.semibold))
                                .foregroundStyle(Palette.primary)
                                .frame(minHeight: ComponentSize.minTouchTarget)
                                .contentShape(.rect)
                        }
                        .buttonStyle(.plain)
                    }

                    if let errorMessage {
                        Text(errorMessage)
                            .font(.detail)
                            .foregroundStyle(Palette.record)
                            .fixedSize(horizontal: false, vertical: true)
                    }
                }

                PrimaryButton(title: isStarting ? "Starting…" : "Decode this song", systemImage: "waveform") {
                    decode()
                }
                .disabled(isStarting || text.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty)

                PolicyNote(text: VideoDetailScreen.policyNote)
            }
            .padding(.horizontal, Spacing.md)
            .padding(.vertical, Spacing.md)
        }
        .scrollDismissesKeyboard(.interactively)
        .background(Palette.canvas)
        .navigationTitle("SoundCloud")
        .navigationBarTitleDisplayMode(.inline)
        .toolbarBackground(Palette.canvas, for: .navigationBar)
        .toolbar(.hidden, for: .tabBar)
        .onAppear(perform: refreshClipboard)
        .onChange(of: scenePhase) { _, phase in
            if phase == .active {
                refreshClipboard()
            }
        }
        .onChange(of: text) {
            errorMessage = nil
        }
    }

    private var header: some View {
        HStack(spacing: Spacing.sm) {
            Image(systemName: "cloud")
                .font(.title3)
                .foregroundStyle(SourceTint.soundcloud.icon)
                .frame(width: ComponentSize.sourceIconTile, height: ComponentSize.sourceIconTile)
                .background(SourceTint.soundcloud.background, in: .rect(cornerRadius: Radius.control))
                .accessibilityHidden(true)
            VStack(alignment: .leading, spacing: Spacing.xxs) {
                Text("Paste a SoundCloud link")
                    .font(.sectionHeader)
                    .foregroundStyle(Palette.text)
                    .accessibilityAddTraits(.isHeader)
                Text("Copy a track's link in SoundCloud (Share → Copy link), then paste it here.")
                    .font(.detail)
                    .foregroundStyle(Palette.textSecondary)
                    .fixedSize(horizontal: false, vertical: true)
            }
        }
    }

    private func refreshClipboard() {
        let pasteboard = UIPasteboard.general
        clipboardHasContent = pasteboard.hasURLs || pasteboard.hasStrings
    }

    private func pasteFromClipboard() {
        let pasteboard = UIPasteboard.general
        let pasted = pasteboard.url?.absoluteString ?? pasteboard.string ?? ""
        guard let url = SoundCloudURL.parse(pasted) else {
            errorMessage = "The clipboard doesn't hold a SoundCloud track link."
            return
        }
        text = url
    }

    private func decode() {
        guard isStarting == false else { return }
        guard SoundCloudURL.parse(text) != nil else {
            errorMessage = Transcription.soundcloudInvalidMessage
            return
        }
        isStarting = true
        errorMessage = nil
        let link = text
        Task {
            do {
                let trackId = try await transcription.startSoundCloud(url: link)
                isStarting = false
                onDecodeStarted(trackId)
            } catch {
                isStarting = false
                errorMessage = Transcription.userMessage(for: error)
            }
        }
    }
}

extension View {
    /// The SoundCloud link screen, pushed onto the caller's `NavigationStack`.
    func soundcloudDestinations(onDecodeStarted: @escaping (_ trackId: String) -> Void) -> some View {
        navigationDestination(for: SoundCloudRoute.self) { _ in
            SoundCloudLinkScreen(onDecodeStarted: onDecodeStarted)
        }
    }
}

#if DEBUG
#Preview {
    NavigationStack {
        SoundCloudLinkScreen(onDecodeStarted: { _ in })
    }
    .environment(TranscriptionService.preview())
    .preferredColorScheme(.dark)
}
#endif
