import MusicDecodedCore
import SwiftData
import SwiftUI

/// The pushed Player for `trackId` (DESIGN.md §3.7). Builds the `PlayerModel` from the library
/// row and the files on this device, then shows `PlayerView`. Works offline: it needs only
/// `analysis/{trackId}.json` (and plays `audio/{trackId}.*` when it's there).
struct PlayerScreen: View {
    let trackId: String

    @Environment(\.modelContext) private var modelContext
    @Environment(TranscriptionService.self) private var transcription
    @Environment(NowPlaying.self) private var nowPlaying
    @Environment(SettingsStore.self) private var settings
    @State private var model: PlayerModel?

    var body: some View {
        Group {
            if let model {
                PlayerView(model: model, actions: trackActions)
            } else {
                Palette.canvas
            }
        }
        .toolbar(.hidden, for: .tabBar)
        .task {
            guard model == nil else { return }
            let created = makeModel()
            model = created
            await created.load()
        }
    }

    /// The ⋯ menu's Re-analyze and Delete, for a track in the library.
    private var trackActions: PlayerTrackActions? {
        guard let item = try? modelContext.fetch(LibraryItem.withTrackId(trackId)).first else { return nil }
        let trackId = trackId
        let transcription = transcription
        let nowPlaying = nowPlaying
        let context = modelContext
        return PlayerTrackActions(
            canReanalyze: item.status != .analyzing,
            reanalyze: {
                try await transcription.retry(trackId: trackId)
            },
            delete: {
                LibraryActions.delete(item, nowPlaying: nowPlaying, context: context)
            }
        )
    }

    private func makeModel() -> PlayerModel {
        let item = try? modelContext.fetch(LibraryItem.withTrackId(trackId)).first
        let info = PlayerTrackLoader.Info(
            title: item?.title ?? trackId,
            artist: item?.artist,
            favorite: item?.favorite ?? false,
            audioURL: item.flatMap { transcription.audioURL(for: $0) }
        )
        let trackId = trackId
        let context = modelContext
        let nowPlaying = nowPlaying
        let preferences = settings.preferences
        return PlayerModel(
            loadTrack: {
                await PlayerTrackLoader.load(trackId: trackId, info: info)
            },
            voicings: BundledContracts.voicings,
            makeEngine: { track in
                // The app-level session owns the engine, so playback outlives this screen and
                // reopening the same track picks it up again.
                nowPlaying.engine(for: trackId, track: track) { track in
                    let engine = Self.makeEngine(for: track)
                    // A new engine starts at Settings → Default speed.
                    engine.setSpeed(preferences.defaultSpeed)
                    return engine
                }
            },
            saveFavorite: { favorite in
                guard let item else { return }
                LibraryActions.setFavorite(item, favorite, context: context)
            },
            markPlayed: {
                guard let item else { return }
                LibraryActions.markPlayed(item, context: context)
            },
            preferences: preferences
        )
    }

    /// The clock (DESIGN.md §4): local audio when it's on this device, otherwise the YouTube
    /// video for youtube tracks, otherwise a silent clock. A youtube track with local audio plays
    /// the audio and keeps the muted video in step.
    private static func makeEngine(for track: PlayerTrack) -> any PlaybackEngine {
        let meta = track.analysis.meta
        let audio = track.audioURL.map { AVPlaybackEngine(url: $0, durationSec: meta.durationSec) }
        if meta.source == .youtube, let videoId = meta.sourceRef, YouTubeVideoID.isValid(videoId) {
            let video = YouTubePlayerController(videoId: videoId)
            if let audio {
                return VideoFollowingEngine(audio: audio, video: video)
            }
            return YouTubePlaybackEngine(video: video, durationSec: meta.durationSec)
        }
        if let audio {
            return audio
        }
        return SilentPlaybackEngine(durationSec: meta.durationSec)
    }
}

/// The Player layout on one screen (DESIGN.md §3.7): metadata chips, the edge-to-edge timeline,
/// the control row, the visualizer panel filling the rest, and the transport pinned at the
/// bottom. Each section reads only the model state it shows, so the 60 Hz clock
/// redraws just the timeline and the seek bar.
struct PlayerView: View {
    let model: PlayerModel
    /// The ⋯ menu's library actions; `nil` hides Re-analyze and Delete (previews, fixtures).
    var actions: PlayerTrackActions?

    @Environment(\.scenePhase) private var scenePhase
    @Environment(\.dismiss) private var dismiss
    @Environment(\.dynamicTypeSize) private var dynamicTypeSize
    @Environment(\.openPaywall) private var openPaywall
    @State private var showPiano = false
    @State private var confirmDelete = false
    @State private var actionError: String?
    @State private var showActionError = false

    var body: some View {
        content
            .background(Palette.canvas)
            .navigationBarTitleDisplayMode(.inline)
            .toolbarBackground(Palette.canvas, for: .navigationBar)
            .toolbar(.hidden, for: .tabBar)
            .toolbar {
                ToolbarItem(placement: .principal) {
                    PlayerTitle(title: model.track?.title ?? "", artist: model.track?.artist)
                }
                ToolbarItem(placement: .topBarTrailing) {
                    HStack(spacing: 0) {
                        FavoriteButton(model: model)
                            .disabled(model.track == nil)
                        PlayerMoreMenu(
                            model: model,
                            actions: actions,
                            onReanalyze: reanalyze,
                            onDelete: { confirmDelete = true }
                        )
                    }
                }
            }
            .confirmationDialog(
                "Delete “\(model.track?.title ?? "this song")”?",
                isPresented: $confirmDelete,
                titleVisibility: .visible
            ) {
                Button("Delete", role: .destructive) {
                    model.pause()
                    actions?.delete()
                    dismiss()
                }
            } message: {
                Text("It's removed from your library and its chord timeline won't show up here anymore.")
            }
            .alert("Couldn't re-analyze", isPresented: $showActionError) {
                // The daily plan limit offers the paywall (DESIGN.md §3.11).
                if let openPaywall, OpenPaywallAction.opens(for: actionError) {
                    Button("Go Pro") {
                        openPaywall()
                    }
                }
                Button("OK", role: .cancel) {}
            } message: {
                Text(actionError ?? Transcription.genericMessage)
            }
            .onChange(of: scenePhase) { _, phase in
                // No background audio in v1: stop with the app.
                if phase == .background {
                    model.pause()
                }
            }
            .navigationDestination(isPresented: $showPiano) {
                // Shares this model, so playback carries on in the learning view.
                PianoView(model: model)
            }
    }

    /// Re-analyze: starts a new job and leaves the Player; the song's row shows the progress.
    private func reanalyze() {
        guard let actions else { return }
        Task {
            do {
                try await actions.reanalyze()
                model.pause()
                dismiss()
            } catch is CancellationError {
                return
            } catch {
                actionError = Transcription.userMessage(for: error)
                showActionError = true
            }
        }
    }

    /// The metadata row and the gap under it, taken off before sizing the timeline.
    private static let metadataRowAllowance: CGFloat = 44 + Spacing.xs
    private static let fallbackTimelineHeight: CGFloat = 220

    /// Metadata chips · timeline (edge to edge) · control row · visualizer panel (DESIGN.md §3.7).
    private func playerStack(_ analysis: Analysis, timelineHeight: CGFloat) -> some View {
        VStack(spacing: Spacing.xs) {
            MetadataRow(analysis: analysis)
                .padding(.horizontal, Spacing.md)
            PlayerTimeline(model: model)
                .frame(height: timelineHeight)
            ControlRow(model: model)
                .padding(.horizontal, Spacing.md)
            VisualizerPanel(model: model) {
                showPiano = true
            }
            .frame(maxHeight: .infinity)
            .padding(.horizontal, Spacing.md)
            // Room for the panel's hard shadow above the transport.
            .padding(.bottom, Spacing.xs)
        }
        .padding(.top, Spacing.xxs)
    }

    @ViewBuilder
    private var content: some View {
        if let analysis = model.analysis {
            Group {
                if dynamicTypeSize.isAccessibilitySize {
                    // Fallback at the largest text sizes: fixed sizes in a scroll view.
                    ScrollView {
                        playerStack(analysis, timelineHeight: Self.fallbackTimelineHeight)
                    }
                } else {
                    // One screen, no scrolling (DESIGN.md §3.7): the panel takes what's left.
                    GeometryReader { proxy in
                        let middle = proxy.size.height - Self.metadataRowAllowance
                        playerStack(analysis, timelineHeight: TimelineMath.timelineHeight(available: middle))
                            .frame(width: proxy.size.width, height: proxy.size.height, alignment: .top)
                    }
                }
            }
            .safeAreaInset(edge: .bottom, spacing: 0) {
                TransportPanel(model: model)
            }
        } else if model.isLoading {
            ProgressView()
                .tint(Palette.primary)
                .frame(maxWidth: .infinity, maxHeight: .infinity)
        } else {
            ContentUnavailableView(
                "Transcription Not Found",
                systemImage: "music.note",
                description: Text("This transcription isn't on this device.")
            )
            .frame(maxWidth: .infinity, maxHeight: .infinity)
        }
    }
}

/// Title (one line, ellipsized) and, only when it fits, the artist under it, centered in the
/// navigation bar (DESIGN.md §3.7 item 1).
struct PlayerTitle: View {
    let title: String
    let artist: String?

    var body: some View {
        ViewThatFits(in: .vertical) {
            VStack(spacing: 0) {
                titleText
                if let artist {
                    Text(artist)
                        .font(.detail)
                        .foregroundStyle(Palette.textSecondary)
                        .lineLimit(1)
                }
            }
            titleText
        }
        .accessibilityElement(children: .combine)
        .accessibilityAddTraits(.isHeader)
    }

    private var titleText: some View {
        Text(title)
            .font(.rowTitle)
            .foregroundStyle(Palette.text)
            .lineLimit(1)
    }
}

private struct FavoriteButton: View {
    let model: PlayerModel

    var body: some View {
        Button {
            model.toggleFavorite()
        } label: {
            Image(systemName: model.favorite ? "heart.fill" : "heart")
                .fontWeight(.bold)
                .foregroundStyle(model.favorite ? Palette.pink : Palette.text)
                .frame(minWidth: ComponentSize.minTouchTarget, minHeight: ComponentSize.minTouchTarget)
                .contentShape(.rect)
        }
        .accessibilityLabel(model.favorite ? "Remove from favorites" : "Add to favorites")
    }
}

/// Re-analyze and Delete for a library track, run by `PlayerScreen`.
struct PlayerTrackActions {
    /// `false` while a job is already running.
    var canReanalyze: Bool
    let reanalyze: @MainActor () async throws -> Void
    let delete: @MainActor () -> Void
}

/// The header ⋯ menu (DESIGN.md §3.7 item 1): Favorite, Re-analyze, Delete.
private struct PlayerMoreMenu: View {
    let model: PlayerModel
    let actions: PlayerTrackActions?
    let onReanalyze: () -> Void
    let onDelete: () -> Void

    var body: some View {
        Menu {
            Button(
                model.favorite ? "Remove from favorites" : "Favorite",
                systemImage: model.favorite ? "heart.slash" : "heart"
            ) {
                model.toggleFavorite()
            }
            .disabled(model.track == nil)
            if let actions {
                Button("Re-analyze", systemImage: "arrow.clockwise", action: onReanalyze)
                    .disabled(actions.canReanalyze == false)
                Button("Delete", systemImage: "trash", role: .destructive, action: onDelete)
            }
        } label: {
            Image(systemName: "ellipsis")
                .fontWeight(.bold)
                .foregroundStyle(Palette.text)
                .frame(minWidth: ComponentSize.minTouchTarget, minHeight: ComponentSize.minTouchTarget)
                .contentShape(.rect)
        }
        .accessibilityLabel("More actions")
    }
}

/// Key · Time · BPM as compact chips (DESIGN.md §3.7 item 2): icon and value inline, no caption.
private struct MetadataRow: View {
    let analysis: Analysis

    var body: some View {
        AdaptiveStack {
            MetaChip(title: "Key", value: analysis.key.map(PlayerFormat.key) ?? "—", systemImage: "music.note")
            MetaChip(
                title: "Time signature",
                value: analysis.timeSignature.map { "\($0.numerator)/\($0.denominator)" } ?? "—",
                systemImage: "metronome"
            )
            MetaChip(
                title: "Tempo",
                value: analysis.bpm.map { "\(Int($0.rounded())) BPM" } ?? "—",
                systemImage: "waveform.path"
            )
        }
        .frame(maxWidth: .infinity, alignment: .leading)
    }
}

/// One read-only metadata chip: lime icon + value on a square ink card.
private struct MetaChip: View {
    let title: String
    let value: String
    let systemImage: String

    var body: some View {
        HStack(spacing: Spacing.xs) {
            Image(systemName: systemImage)
                .font(.footnote.weight(.bold))
                .foregroundStyle(Palette.lime)
            Text(value)
                .font(.label)
                .monospacedDigit()
                .foregroundStyle(Palette.text)
                .lineLimit(1)
        }
        .padding(.horizontal, Spacing.sm)
        .frame(minHeight: ComponentSize.playerChipHeight)
        .inkCard()
        .accessibilityElement(children: .ignore)
        .accessibilityLabel(title)
        .accessibilityValue(value == "—" ? "Unknown" : ChordDisplay.spoken(value))
    }
}

/// The timeline, fed so that only it redraws on every frame while playing.
private struct PlayerTimeline: View {
    let model: PlayerModel

    var body: some View {
        if let analysis = model.analysis {
            ChordTimeline(
                analysis: analysis,
                chords: model.chords,
                isPlaying: model.isPlaying,
                // Read only while paused, so ticks don't invalidate this view while playing.
                pausedPosition: model.isPlaying ? 0 : model.position,
                livePosition: { model.livePosition },
                loop: model.loop,
                onSeek: { model.seek(to: $0) },
                onLoopSelected: { model.setLoop($0) },
                onPreviousChord: { model.previousChord() },
                onNextChord: { model.nextChord() }
            )
        }
    }
}

/// Chords (Basic/Full) · Voicing (Simple) · Transpose, label and value on one line where they fit
/// (DESIGN.md §3.7 item 4).
private struct ControlRow: View {
    let model: PlayerModel

    var body: some View {
        AdaptiveStack {
            ControlChip(title: "Chords", value: model.chordsMode == .basic ? "Basic" : "Full", inline: true) {
                model.toggleChordsMode()
            }
            // Only Simple exists in v1 (DESIGN.md §5).
            ControlChip(title: "Voicing", value: "Simple", inline: true) {}
                .disabled(true)
            TransposeChip(model: model, inline: true)
        }
    }
}

/// Piano | Video (DESIGN.md §3.7 item 5), taking all the height it's given. Video is disabled
/// unless the source is youtube.
private struct VisualizerPanel: View {
    @Bindable var model: PlayerModel
    let onExpand: () -> Void

    var body: some View {
        VStack(spacing: Spacing.xs) {
            HStack(spacing: Spacing.xs) {
                SegmentedToggle(
                    options: Visualizer.allCases,
                    selection: $model.visualizerChoice,
                    isEnabled: { option in option == .piano || model.videoAvailable },
                    label: { $0.title }
                )
                IconButton(title: "Open piano view", systemImage: "arrow.up.left.and.arrow.down.right", size: .small) {
                    onExpand()
                }
            }

            if model.visualizer == .piano {
                PanelPiano(model: model)
                    .frame(maxHeight: .infinity)
            }
            // Always in the hierarchy when there's a video (it may be the clock), shown only on
            // the Video tab, so switching tabs never reloads the embed. Letterboxed 16:9.
            if let controller = model.videoPlayer as? YouTubePlayerController {
                let visible = model.visualizer == .video
                VideoPanel(controller: controller, isVisible: visible)
                    .frame(maxWidth: .infinity, maxHeight: visible ? CGFloat.infinity : nil)
                    .background(visible ? Palette.black : Color.clear)
            } else if model.visualizer == .video {
                Text("The video isn't available for this song.")
                    .font(.detail)
                    .foregroundStyle(Palette.textSecondary)
                    .frame(maxWidth: .infinity, maxHeight: .infinity)
            }
        }
        .padding(Spacing.sm)
        .block(Palette.surface, shadow: HardShadow.lg, shadowColor: Palette.violet)
    }
}

/// The active chord name (lime) and its Roman numeral above a keyboard that fills the rest of
/// the panel: as many octaves as fit the width, centered on the active notes, with the note names
/// under the highlighted keys. Replaces the old Now Playing card.
private struct PanelPiano: View {
    let model: PlayerModel

    /// The keyboard's minimum (96) plus its note-name row.
    private static let minKeyboardHeight: CGFloat = ComponentSize.pianoCompactHeight + Spacing.md

    var body: some View {
        let chords = model.chords
        let label = model.currentLabel
        let name = label.map(chords.name) ?? ""
        let numeral = label.flatMap(chords.numeral)
        VStack(spacing: Spacing.xs) {
            HStack(alignment: .firstTextBaseline, spacing: Spacing.xs) {
                Text(name.isEmpty ? "—" : name)
                    .font(.sectionHeader)
                    .foregroundStyle(Palette.chord)
                    .lineLimit(1)
                    .minimumScaleFactor(0.6)
                if let numeral {
                    Text(numeral)
                        .font(.rowTitle)
                        .foregroundStyle(Palette.textSecondary)
                        .lineLimit(1)
                }
            }
            .frame(maxWidth: .infinity)
            .accessibilityElement(children: .ignore)
            .accessibilityLabel("Current chord")
            .accessibilityValue(Self.summary(chords, label: label, numeral: numeral))
            .accessibilityAddTraits(.updatesFrequently)

            GeometryReader { proxy in
                PianoKeyboard(
                    notes: label.map(chords.notes) ?? [],
                    bassNote: label.flatMap(chords.bassNote),
                    octaves: PianoLayout.octavesThatFit(width: Double(proxy.size.width)),
                    height: max(proxy.size.height, Self.minKeyboardHeight),
                    centersNotes: true,
                    namesBelow: { chords.noteName($0) },
                    notesDescription: chords.accessibilitySummary(label)
                )
                .clipShape(.rect)
            }
            .frame(minHeight: Self.minKeyboardHeight)
        }
    }

    /// "E major, three: E, G sharp, B".
    private static func summary(_ chords: ChordDisplay, label: String?, numeral: String?) -> String {
        guard let label, let numeral else { return chords.accessibilitySummary(label) }
        let notes = chords.spokenNoteNames(label)
        let head = "\(chords.spokenName(label)), \(spokenRomanNumeral(numeral))"
        return notes.isEmpty ? head : "\(head): \(notes)"
    }
}

#if DEBUG
#Preview {
    let model = PlayerModel(
        loadTrack: {
            DebugFixtures.load(trackId: DebugFixtures.sunnyDays).map {
                PlayerTrack(analysis: $0, title: "Sunny Days", artist: "Fixture", favorite: true, audioURL: nil)
            }
        },
        voicings: BundledContracts.voicings,
        makeEngine: { SilentPlaybackEngine(durationSec: $0.analysis.meta.durationSec) }
    )
    return NavigationStack {
        PlayerView(model: model)
            .task {
                await model.load()
                model.seek(to: 16)
            }
    }
    .preferredColorScheme(.dark)
}
#endif
