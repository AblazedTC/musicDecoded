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
    @State private var model: PlayerModel?

    var body: some View {
        Group {
            if let model {
                PlayerView(model: model)
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
        return PlayerModel(
            loadTrack: {
                await PlayerTrackLoader.load(trackId: trackId, info: info)
            },
            voicings: BundledContracts.voicings,
            makeEngine: { track in
                let duration = track.analysis.meta.durationSec
                if let url = track.audioURL {
                    return AVPlaybackEngine(url: url, durationSec: duration)
                }
                return SilentPlaybackEngine(durationSec: duration)
            },
            saveFavorite: { favorite in
                guard let item else { return }
                item.favorite = favorite
                item.updatedAt = .now
                try? context.save()
            }
        )
    }
}

/// The Player layout: metadata chips, timeline, Now Playing, control chips, visualizer card and
/// the transport sheet. Each section reads only the model state it shows, so the 60 Hz clock
/// redraws just the timeline and the seek bar.
struct PlayerView: View {
    let model: PlayerModel

    @Environment(\.scenePhase) private var scenePhase
    @State private var showPiano = false

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
                    FavoriteButton(model: model)
                        .disabled(model.track == nil)
                }
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

    @ViewBuilder
    private var content: some View {
        if let analysis = model.analysis {
            ScrollView {
                VStack(spacing: Spacing.md) {
                    MetadataRow(analysis: analysis)
                    PlayerTimeline(model: model)
                    NowPlaying(model: model)
                    HStack(spacing: Spacing.xs) {
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
                .padding(.vertical, Spacing.xs)
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
                .foregroundStyle(model.favorite ? Palette.primary : Palette.text)
        }
        .accessibilityLabel(model.favorite ? "Remove from favorites" : "Add to favorites")
    }
}

/// Key · Time signature · BPM.
private struct MetadataRow: View {
    let analysis: Analysis

    var body: some View {
        HStack(spacing: Spacing.xs) {
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
                onLoopSelected: { model.setLoop($0) }
            )
        }
    }
}

/// NOW PLAYING: the big chord name, its notes and the voicing caption.
private struct NowPlaying: View {
    let model: PlayerModel

    var body: some View {
        let chords = model.chords
        let label = model.currentLabel
        let name = label.map(chords.name) ?? ""
        let notes = label.map(chords.noteNames) ?? ""
        VStack(alignment: .leading, spacing: Spacing.xxs) {
            Text("Now playing")
                .overlineStyle()
            Text(name.isEmpty ? "—" : name)
                .font(.currentChord)
                .foregroundStyle(Palette.chord)
                .lineLimit(1)
                .minimumScaleFactor(0.6)
            if notes.isEmpty == false {
                Text(notes)
                    .font(.rowTitle)
                    .foregroundStyle(Palette.textSecondary)
                Text("Representative root-position voicing")
                    .font(.overline)
                    .foregroundStyle(Palette.textMuted)
            }
        }
        .frame(maxWidth: .infinity, alignment: .leading)
        .accessibilityElement(children: .combine)
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

            switch model.visualizer {
            case .piano:
                CurrentChordSummary(model: model)
                CompactKeyboard(model: model)
            case .video:
                // The synced YouTube embed arrives with T-305i.
                Text("Video playback is coming soon.")
                    .font(.detail)
                    .foregroundStyle(Palette.textSecondary)
            }
        }
        .padding(Spacing.md)
        .background(Palette.surface, in: .rect(cornerRadius: Radius.card))
        .overlay {
            RoundedRectangle(cornerRadius: Radius.card).strokeBorder(Palette.line, lineWidth: 1)
        }
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
            Text("Current chord")
                .overlineStyle()
            HStack(alignment: .firstTextBaseline, spacing: Spacing.xs) {
                Text(name.isEmpty ? "—" : name)
                    .font(.sectionHeader)
                    .foregroundStyle(Palette.chord)
                Text(label.map(chords.noteNames) ?? "")
                    .font(.detail)
                    .foregroundStyle(Palette.textSecondary)
            }
        }
        .accessibilityElement(children: .combine)
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
            notesDescription: label.map(chords.noteNames) ?? ""
        )
        .clipShape(.rect(cornerRadius: Radius.chip))
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
