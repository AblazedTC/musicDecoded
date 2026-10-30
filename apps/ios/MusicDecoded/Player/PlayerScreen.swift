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

/// The Player layout: metadata chips, timeline, Now Playing, control chips, visualizer card and
/// the transport sheet. Each section reads only the model state it shows, so the 60 Hz clock
/// redraws just the timeline and the seek bar.
struct PlayerView: View {
    let model: PlayerModel
    /// The ⋯ menu's library actions; `nil` hides Re-analyze and Delete (previews, fixtures).
    var actions: PlayerTrackActions?

    @Environment(\.scenePhase) private var scenePhase
    @Environment(\.dismiss) private var dismiss
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

    @ViewBuilder
    private var content: some View {
        if let analysis = model.analysis {
            ScrollView {
                // lg gaps leave room for the hard shadows and the tilted Now Playing card.
                VStack(spacing: Spacing.lg) {
                    MetadataRow(analysis: analysis)
                    PlayerTimeline(model: model)
                    NowPlayingChord(model: model)
                        .padding(.vertical, Spacing.xs)
                    AdaptiveStack {
                        ControlChip(title: "Chords", value: model.chordsMode == .basic ? "Basic" : "Full") {
                            model.toggleChordsMode()
                        }
                        // Only Simple exists in v1 (DESIGN.md §5).
                        ControlChip(title: "Voicing", value: "Simple") {}
                            .disabled(true)
                        TransposeChip(model: model)
                    }
                    VisualizerCard(model: model) {
                        showPiano = true
                    }
                }
                .padding(.horizontal, Spacing.md)
                .padding(.top, Spacing.xs)
                .padding(.bottom, Spacing.lg)
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

/// Title and artist, centered in the navigation bar.
struct PlayerTitle: View {
    let title: String
    let artist: String?

    var body: some View {
        VStack(spacing: 0) {
            Text(title)
                .font(.rowTitle)
                .foregroundStyle(Palette.text)
                .lineLimit(1)
            if let artist {
                Text(artist)
                    .font(.detail)
                    .foregroundStyle(Palette.textSecondary)
                    .lineLimit(1)
            }
        }
        .accessibilityElement(children: .combine)
        .accessibilityAddTraits(.isHeader)
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

/// Key · Time signature · BPM.
private struct MetadataRow: View {
    let analysis: Analysis

    var body: some View {
        AdaptiveStack {
            InfoChip(title: "Key", value: analysis.key.map(PlayerFormat.key) ?? "—", systemImage: "music.note")
            InfoChip(
                title: "Time",
                value: analysis.timeSignature.map { "\($0.numerator)/\($0.denominator)" } ?? "—",
                systemImage: "metronome"
            )
            InfoChip(
                title: "Tempo",
                value: analysis.bpm.map { "\(Int($0.rounded())) BPM" } ?? "—",
                systemImage: "waveform.path"
            )
        }
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

/// NOW PLAYING: the big lime chord name, its notes and the voicing caption, on the hero card
/// (`card`, 3pt black border, pink `shadowXl`, tilted -1.5°).
private struct NowPlayingChord: View {
    let model: PlayerModel

    var body: some View {
        let chords = model.chords
        let label = model.currentLabel
        let name = label.map(chords.name) ?? ""
        let notes = label.map(chords.noteNames) ?? ""
        VStack(alignment: .leading, spacing: Spacing.xs) {
            HStack(alignment: .top) {
                Eyebrow(text: "Now playing")
                Spacer(minLength: Spacing.xs)
                WaveformBars(bars: 5, color: Palette.lime, gap: Spacing.xxs)
                    .frame(width: 40, height: 20)
            }
            Text(name.isEmpty ? "—" : name)
                .font(.currentChord)
                .foregroundStyle(Palette.chord)
                .lineLimit(1)
                .minimumScaleFactor(0.6)
            if notes.isEmpty == false {
                Text(notes)
                    .font(.rowTitle)
                    .foregroundStyle(Palette.text)
                Text("Representative root-position voicing")
                    .font(.detail)
                    .foregroundStyle(Palette.textSecondary)
            }
        }
        .frame(maxWidth: .infinity, alignment: .leading)
        .padding(Spacing.lg)
        .block(Palette.card, shadow: HardShadow.xl, shadowColor: Palette.pink)
        .tilt(Tilt.hero)
        // "Now playing, D major: D, F sharp, A" instead of the glyph-heavy visible text.
        .accessibilityElement(children: .ignore)
        .accessibilityLabel("Now playing")
        .accessibilityValue(chords.accessibilitySummary(label))
        .accessibilityAddTraits([.isHeader, .updatesFrequently])
    }
}

/// Piano | Video (DESIGN.md §3.7 item 6). Video is disabled unless the source is youtube.
private struct VisualizerCard: View {
    @Bindable var model: PlayerModel
    let onExpand: () -> Void

    var body: some View {
        VStack(alignment: .leading, spacing: Spacing.sm) {
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
                CurrentChordSummary(model: model)
                CompactKeyboard(model: model)
            }
            // Always in the hierarchy when there's a video (it may be the clock), shown only on
            // the Video tab, so switching tabs never reloads the embed.
            if let controller = model.videoPlayer as? YouTubePlayerController {
                VideoPanel(controller: controller, isVisible: model.visualizer == .video)
            } else if model.visualizer == .video {
                Text("The video isn't available for this song.")
                    .font(.detail)
                    .foregroundStyle(Palette.textSecondary)
            }
        }
        .padding(Spacing.md)
        .block(Palette.surface, shadow: HardShadow.lg, shadowColor: Palette.violet)
    }
}

/// CURRENT CHORD + name + notes.
private struct CurrentChordSummary: View {
    let model: PlayerModel

    var body: some View {
        let chords = model.chords
        let label = model.currentLabel
        let name = label.map(chords.name) ?? ""
        VStack(alignment: .leading, spacing: Spacing.xxs) {
            Eyebrow(text: "Current chord")
            // Name and notes on one line, or stacked when large text doesn't fit.
            ViewThatFits(in: .horizontal) {
                HStack(alignment: .firstTextBaseline, spacing: Spacing.xs) {
                    chordName(name)
                    noteNames(label.map(chords.noteNames) ?? "")
                }
                VStack(alignment: .leading, spacing: Spacing.xxs) {
                    chordName(name)
                    noteNames(label.map(chords.noteNames) ?? "")
                }
            }
        }
        .accessibilityElement(children: .ignore)
        .accessibilityLabel("Current chord")
        .accessibilityValue(chords.accessibilitySummary(label))
        .accessibilityAddTraits(.updatesFrequently)
    }

    private func chordName(_ name: String) -> some View {
        Text(name.isEmpty ? "—" : name)
            .font(.sectionHeader)
            .foregroundStyle(Palette.chord)
    }

    private func noteNames(_ notes: String) -> some View {
        Text(notes)
            .font(.detail)
            .foregroundStyle(Palette.textSecondary)
    }
}

/// The 2-octave keyboard in the visualizer card.
private struct CompactKeyboard: View {
    let model: PlayerModel

    var body: some View {
        let chords = model.chords
        let label = model.currentLabel
        PianoKeyboard(
            notes: label.map(chords.notes) ?? [],
            bassNote: label.flatMap(chords.bassNote),
            notesDescription: chords.accessibilitySummary(label)
        )
        .clipShape(.rect)
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
