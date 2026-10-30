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
            audioURL: item.flatMap { transcription.audioURL(for: $0) },
            artworkURL: item?.thumbnailUrl.flatMap(URL.init(string:))
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

/// The Player on one screen (DESIGN.md §3.7, "Owner revision 2" and §3.7.1): header with the
/// view switcher and ⋯, a compact title line, the metadata chips, the selected view in the middle
/// and the transport pinned at the bottom. Each section reads only the model state it shows, so
/// the 60 Hz clock redraws just the timeline and the seek bar.
struct PlayerView: View {
    let model: PlayerModel
    /// The ⋯ menu's library actions; `nil` hides Re-analyze and Delete (previews, fixtures).
    var actions: PlayerTrackActions?

    @Environment(\.scenePhase) private var scenePhase
    @Environment(\.dismiss) private var dismiss
    @Environment(\.dynamicTypeSize) private var dynamicTypeSize
    @Environment(\.openPaywall) private var openPaywall
    /// The view switcher's choice, kept for the whole app (DESIGN.md §3.7).
    @AppStorage("player.viewMode") private var viewMode: PlayerViewMode = .timeline
    @State private var showMediaSheet = false
    @State private var confirmDelete = false
    @State private var actionError: String?
    @State private var showActionError = false

    var body: some View {
        content
            .overlayPreferenceValue(VideoSlotKey.self) { slot in
                // The one YouTube video view: over the media panel, or parked at 1 pt (it may be
                // the clock). While the media sheet is open, the sheet shows it instead.
                if showMediaSheet == false, let controller = model.videoPlayer as? YouTubePlayerController {
                    PlayerVideoHost(controller: controller, slot: slot)
                }
            }
            .background(Palette.canvas)
            .navigationBarTitleDisplayMode(.inline)
            .toolbarBackground(Palette.canvas, for: .navigationBar)
            .toolbar(.hidden, for: .tabBar)
            .toolbar {
                ToolbarItem(placement: .principal) {
                    PlayerViewSwitcher(selection: $viewMode)
                }
                ToolbarItem(placement: .topBarTrailing) {
                    PlayerMoreMenu(
                        model: model,
                        actions: actions,
                        onReanalyze: reanalyze,
                        onDelete: { confirmDelete = true }
                    )
                }
            }
            .sheet(isPresented: $showMediaSheet) {
                MediaSheet(model: model)
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

    /// Fixed middle-region height for the largest text sizes, where the screen scrolls.
    private static let fallbackMiddleHeight: CGFloat = 640

    /// Title line · metadata chips · the selected view.
    private func playerStack(_ analysis: Analysis, fillsHeight: Bool) -> some View {
        VStack(spacing: Spacing.xs) {
            VStack(alignment: .leading, spacing: Spacing.xxs) {
                PlayerTitleLine(title: model.track?.title ?? "", artist: model.track?.artist)
                MetadataRow(analysis: analysis)
            }
            .padding(.horizontal, Spacing.md)
            if fillsHeight {
                GeometryReader { proxy in
                    middle(analysis, height: proxy.size.height)
                        .frame(width: proxy.size.width, height: proxy.size.height, alignment: .top)
                }
            } else {
                middle(analysis, height: Self.fallbackMiddleHeight)
                    .frame(height: Self.fallbackMiddleHeight, alignment: .top)
            }
        }
        .padding(.top, Spacing.xxs)
    }

    @ViewBuilder
    private func middle(_ analysis: Analysis, height: CGFloat) -> some View {
        switch viewMode {
        case .timeline:
            TimelineRegion(model: model, height: height) {
                showMediaSheet = true
            }
        case .beatGrid:
            CompactRegion(model: model) {
                BeatGridView(model: model, analysis: analysis)
            }
        case .chordSheet:
            CompactRegion(model: model) {
                ChordSheetView(model: model, analysis: analysis)
            }
        case .pianoFocus:
            PianoFocusView(model: model)
        }
    }

    @ViewBuilder
    private var content: some View {
        if let analysis = model.analysis {
            Group {
                if dynamicTypeSize.isAccessibilitySize {
                    // Fallback at the largest text sizes: fixed sizes in a scroll view.
                    ScrollView {
                        playerStack(analysis, fillsHeight: false)
                    }
                } else {
                    // One screen, no scrolling (DESIGN.md §3.7): the middle takes what's left.
                    playerStack(analysis, fillsHeight: true)
                        .frame(maxWidth: .infinity, maxHeight: .infinity, alignment: .top)
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

/// Timeline view (DESIGN.md §3.7.1): timeline (edge to edge) · control row · piano panel ·
/// media panel, sized to `height` with no scrolling. The media panel collapses to a strip
/// (opening a sheet) when less than 120 points would be left for it.
private struct TimelineRegion: View {
    let model: PlayerModel
    let height: CGFloat
    let onOpenMedia: () -> Void

    /// Below the 150 minimum only when the panels wouldn't fit otherwise (small phones).
    private static let minTimelineHeight: CGFloat = 120

    var body: some View {
        let fixed = ComponentSize.minTouchTarget + Spacing.xs * 2
        let panelsMinimum = PlayerLayout.minimumPanels(gap: Spacing.xxs)
        let timeline = max(
            min(TimelineMath.timelineHeight(available: height), height - fixed - panelsMinimum),
            Self.minTimelineHeight
        )
        let split = PlayerLayout.split(available: height - timeline - fixed, gap: Spacing.xxs)
        VStack(spacing: Spacing.xs) {
            PlayerTimeline(model: model)
                .frame(height: timeline)
            ControlRow(model: model)
                .padding(.horizontal, Spacing.md)
            VStack(spacing: Spacing.xxs) {
                PianoPanel(model: model, keysHeight: split.keyboard)
                    .padding(.horizontal, Spacing.md)
                if split.mediaCollapsed {
                    MediaStrip(model: model, onOpen: onOpenMedia)
                } else {
                    MediaPanel(model: model)
                        .frame(height: max(split.media, 0))
                }
            }
        }
    }
}

/// Beat grid and Chord sheet (DESIGN.md §3.7.1): the view's content fills the space, then the
/// control row and a compact piano panel (keyboard 90).
private struct CompactRegion<Content: View>: View {
    let model: PlayerModel
    @ViewBuilder let content: () -> Content

    var body: some View {
        VStack(spacing: Spacing.xs) {
            content()
                .frame(maxWidth: .infinity, maxHeight: .infinity)
            ControlRow(model: model)
                .padding(.horizontal, Spacing.md)
            PianoPanel(model: model, keysHeight: ComponentSize.fixedKeyboardCompact)
                .padding(.horizontal, Spacing.md)
                .padding(.bottom, Spacing.xxs)
        }
    }
}

/// The header's view switcher (DESIGN.md §3.7 "Owner revision 2"): Timeline · Beat grid ·
/// Chord sheet · Piano focus as icon segments in a 3pt black frame; selected = lime, ink icon.
struct PlayerViewSwitcher: View {
    @Binding var selection: PlayerViewMode

    var body: some View {
        HStack(spacing: 0) {
            ForEach(PlayerViewMode.allCases, id: \.self) { mode in
                let isSelected = mode == selection
                Button {
                    selection = mode
                } label: {
                    Image(systemName: mode.systemImage)
                        .font(.footnote.weight(.bold))
                        .foregroundStyle(isSelected ? Palette.onPrimary : Palette.text)
                        .frame(width: ComponentSize.viewSwitcherSegmentWidth, height: ComponentSize.viewSwitcherHeight)
                        .background(isSelected ? Palette.primary : Palette.surfaceRaised, in: .rect)
                        .contentShape(.rect)
                }
                .buttonStyle(.plain)
                .accessibilityLabel(mode.title)
                .accessibilityAddTraits(isSelected ? .isSelected : [])
            }
        }
        .padding(BorderWidth.thick)
        .background(Palette.border, in: .rect)
        .dynamicTypeSize(...DynamicTypeSize.xxxLarge)
        .accessibilityElement(children: .contain)
        .accessibilityLabel("Player view")
    }
}

/// The song title on one compact line under the header (ellipsized), the artist after it.
private struct PlayerTitleLine: View {
    let title: String
    let artist: String?

    var body: some View {
        let artistText = artist.map { Text(" · \($0)").foregroundStyle(Palette.textSecondary) } ?? Text(verbatim: "")
        (Text(title).foregroundStyle(Palette.text) + artistText)
            .font(.labelSmall)
            .lineLimit(1)
            .truncationMode(.tail)
            .frame(maxWidth: .infinity, alignment: .leading)
            .accessibilityAddTraits(.isHeader)
    }
}

/// Re-analyze and Delete for a library track, run by `PlayerScreen`.
struct PlayerTrackActions {
    /// `false` while a job is already running.
    var canReanalyze: Bool
    let reanalyze: @MainActor () async throws -> Void
    let delete: @MainActor () -> Void
}

/// The header ⋯ menu (DESIGN.md §3.7 "Owner revision 2"): Favorite, Re-analyze, Delete.
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

/// Chords (Basic/Full) · Voicing (inversion) · Transpose, label and value on one line where they
/// fit (DESIGN.md §3.7 item 4).
private struct ControlRow: View {
    let model: PlayerModel

    var body: some View {
        AdaptiveStack {
            ControlChip(title: "Chords", value: model.chordsMode == .basic ? "Basic" : "Full", inline: true) {
                model.toggleChordsMode()
            }
            VoicingChip(model: model, inline: true)
            TransposeChip(model: model, inline: true)
        }
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
