import MusicDecodedCore
import SwiftUI

/// Piano learning view (DESIGN.md §3.8). Pushed from the player's visualizer card and shares its
/// `PlayerModel`, so playback, loop and settings carry over both ways.
struct PianoView: View {
    let model: PlayerModel

    @Environment(\.verticalSizeClass) private var verticalSizeClass
    @Environment(\.scenePhase) private var scenePhase

    /// Octaves on the large keyboard: 2 in portrait, wider in landscape.
    private static let portraitOctaves = 2
    private static let landscapeOctaves = 4

    var body: some View {
        let isLandscape = verticalSizeClass == .compact
        ScrollView {
            VStack(spacing: Spacing.md) {
                PianoHeadCard(model: model)
                    .padding(.horizontal, Spacing.md)
                PlayerSeekBar(model: model)
                    .padding(.horizontal, Spacing.md)
                LearningKeyboard(model: model, octaves: isLandscape ? Self.landscapeOctaves : Self.portraitOctaves)
                ProgressionStrip(model: model)
                LearningControls(model: model)
                    .padding(.horizontal, Spacing.md)
                LearningTransport(model: model)
                if isLandscape == false {
                    Text("Rotate your device for a wider keyboard")
                        .font(.overline)
                        .foregroundStyle(Palette.textMuted)
                }
            }
            .padding(.vertical, Spacing.md)
        }
        .background(Palette.canvas)
        .navigationBarTitleDisplayMode(.inline)
        .toolbarBackground(Palette.canvas, for: .navigationBar)
        .toolbar(.hidden, for: .tabBar)
        .toolbar {
            ToolbarItem(placement: .principal) {
                PlayerTitle(title: model.track?.title ?? "", artist: model.track?.artist)
            }
        }
        .onChange(of: scenePhase) { _, phase in
            if phase == .background {
                model.pause()
            }
        }
    }
}

/// CURRENT CHORD (name + notes) | UP NEXT (name + "in N beats").
private struct PianoHeadCard: View {
    let model: PlayerModel

    var body: some View {
        let chords = model.chords
        let current = model.currentLabel
        let currentName = current.map(chords.name) ?? ""
        let next = model.upcoming
        let nextLabel = next.flatMap { upcoming in
            model.analysis.map { $0.chords[upcoming.chordIndex].label }
        }
        let nextName = nextLabel.map(chords.name) ?? ""
        let nextSpoken = nextLabel.map(chords.spokenName) ?? ""
        let upNextSpoken = nextSpoken.isEmpty
            ? "None"
            : [nextSpoken, next.map { PlayerFormat.inBeats($0.inBeats) }].compactMap { $0 }.joined(separator: ", ")

        HStack(alignment: .top, spacing: Spacing.md) {
            VStack(alignment: .leading, spacing: Spacing.xxs) {
                Text("Current chord")
                    .overlineStyle()
                Text(currentName.isEmpty ? "—" : currentName)
                    .font(.currentChord)
                    .foregroundStyle(Palette.chord)
                    .lineLimit(1)
                    .minimumScaleFactor(0.5)
                Text(current.map(chords.noteNames) ?? "")
                    .font(.detail)
                    .foregroundStyle(Palette.textSecondary)
            }
            .frame(maxWidth: .infinity, alignment: .leading)
            .accessibilityElement(children: .ignore)
            .accessibilityLabel("Current chord")
            .accessibilityValue(chords.accessibilitySummary(current))
            .accessibilityAddTraits(.updatesFrequently)

            Rectangle()
                .fill(Palette.line)
                .frame(width: 1)
                .accessibilityHidden(true)

            VStack(alignment: .leading, spacing: Spacing.xxs) {
                Text("Up next")
                    .overlineStyle()
                Text(nextName.isEmpty ? "—" : nextName)
                    .font(.sectionHeader)
                    .foregroundStyle(Palette.text)
                    .lineLimit(1)
                if let next {
                    Text(PlayerFormat.inBeats(next.inBeats))
                        .font(.detail)
                        .foregroundStyle(Palette.textSecondary)
                }
            }
            .frame(maxWidth: .infinity, alignment: .leading)
            .accessibilityElement(children: .ignore)
            .accessibilityLabel("Up next")
            .accessibilityValue(upNextSpoken)
            .accessibilityAddTraits(.updatesFrequently)
        }
        .fixedSize(horizontal: false, vertical: true)
        .padding(Spacing.md)
        .background(Palette.surface, in: .rect(cornerRadius: Radius.card))
        .overlay {
            RoundedRectangle(cornerRadius: Radius.card).strokeBorder(Palette.line, lineWidth: 1)
        }
    }
}

/// The large keyboard, edge to edge, with optional note names.
private struct LearningKeyboard: View {
    let model: PlayerModel
    let octaves: Int

    var body: some View {
        let chords = model.chords
        let label = model.currentLabel
        let noteName: ((Int) -> String)?
        if model.showNoteNames {
            noteName = { chords.noteName($0) }
        } else {
            noteName = nil
        }
        return PianoKeyboard(
            notes: label.map(chords.notes) ?? [],
            bassNote: label.flatMap(chords.bassNote),
            octaves: octaves,
            height: ComponentSize.pianoLargeHeight,
            noteName: noteName,
            notesDescription: chords.accessibilitySummary(label)
        )
    }
}

/// Horizontally scrolling chord cards (chord + length in beats). The current one is filled and
/// kept in view; tapping a card plays from there.
private struct ProgressionStrip: View {
    let model: PlayerModel

    private struct Item: Identifiable {
        /// Index into `analysis.chords`.
        let id: Int
        let beats: Int
    }

    var body: some View {
        if let analysis = model.analysis {
            let items = Self.items(in: analysis)
            let chords = model.chords
            let current = model.chordIndex
            ScrollViewReader { proxy in
                ScrollView(.horizontal) {
                    LazyHStack(spacing: Spacing.xs) {
                        ForEach(items) { item in
                            let chord = analysis.chords[item.id]
                            StripCard(
                                name: chords.name(chord.label),
                                spokenName: chords.spokenName(chord.label),
                                beats: item.beats,
                                isCurrent: item.id == current
                            ) {
                                model.seek(to: chord.start)
                            }
                            .id(item.id)
                        }
                    }
                    .padding(.horizontal, Spacing.md)
                }
                .scrollIndicators(.hidden)
                .onAppear {
                    if let current {
                        proxy.scrollTo(current, anchor: .center)
                    }
                }
                .onChange(of: current) { _, index in
                    guard let index, items.contains(where: { $0.id == index }) else { return }
                    withAnimation {
                        proxy.scrollTo(index, anchor: .center)
                    }
                }
            }
        }
    }

    /// Every real chord with the number of beats inside it (`start ≤ t < end`).
    private static func items(in analysis: Analysis) -> [Item] {
        let beats = analysis.beats
        return analysis.chords.indices.compactMap { index in
            let chord = analysis.chords[index]
            guard isNoChord(chord.label) == false else { return nil }
            // Beats before `x` = upperBound(x − ε); the difference counts beats in [start, end).
            let count = upperBound(beats, chord.end - 1e-9, \.t) - upperBound(beats, chord.start - 1e-9, \.t)
            return Item(id: index, beats: count)
        }
    }
}

private struct StripCard: View {
    let name: String
    let spokenName: String
    let beats: Int
    let isCurrent: Bool
    let action: () -> Void

    var body: some View {
        Button(action: action) {
            VStack(alignment: .leading, spacing: Spacing.xxs) {
                Text(name)
                    .font(.rowTitle)
                    .foregroundStyle(isCurrent ? Palette.onChord : Palette.chord)
                Text(PlayerFormat.beats(beats))
                    .font(.overline)
                    .foregroundStyle(isCurrent ? Palette.onChord : Palette.textMuted)
            }
            .padding(.horizontal, Spacing.sm)
            .padding(.vertical, Spacing.xs)
            .frame(minHeight: ComponentSize.minTouchTarget)
            .background(isCurrent ? Palette.chord : Palette.surface, in: .rect(cornerRadius: Radius.chip))
            .overlay {
                RoundedRectangle(cornerRadius: Radius.chip)
                    .strokeBorder(isCurrent ? Palette.chord : Palette.line, lineWidth: 1)
            }
        }
        .buttonStyle(.plain)
        .accessibilityLabel("\(spokenName), \(PlayerFormat.beats(beats))")
        .accessibilityHint("Plays from this chord")
        .accessibilityAddTraits(isCurrent ? .isSelected : [])
    }
}

/// Speed · Loop · Note names · Transpose (2×2).
private struct LearningControls: View {
    let model: PlayerModel

    var body: some View {
        VStack(spacing: Spacing.xs) {
            HStack(spacing: Spacing.xs) {
                SpeedChip(model: model)
                ControlChip(title: "Loop", value: model.loop.map(PlayerFormat.loop) ?? "Off") {
                    model.toggleLoop()
                }
            }
            HStack(spacing: Spacing.xs) {
                ControlChip(title: "Note names", value: model.showNoteNames ? "On" : "Off") {
                    model.toggleNoteNames()
                }
                TransposeChip(model: model)
            }
        }
    }
}

/// Previous chord · play/pause · next chord.
private struct LearningTransport: View {
    let model: PlayerModel

    var body: some View {
        HStack(spacing: Spacing.xl) {
            IconButton(title: "Previous chord", systemImage: "backward.end.fill") { model.previousChord() }
            PlayButton(isPlaying: model.isPlaying, diameter: ComponentSize.learningPlayButton) { model.togglePlay() }
            IconButton(title: "Next chord", systemImage: "forward.end.fill") { model.nextChord() }
        }
    }
}

#if DEBUG
#Preview {
    let model = PlayerModel(
        loadTrack: {
            DebugFixtures.load(trackId: DebugFixtures.sunnyDays).map {
                PlayerTrack(analysis: $0, title: "Sunny Days", artist: "Fixture", favorite: false, audioURL: nil)
            }
        },
        voicings: BundledContracts.voicings,
        makeEngine: { SilentPlaybackEngine(durationSec: $0.analysis.meta.durationSec) }
    )
    return NavigationStack {
        PianoView(model: model)
            .task {
                await model.load()
                model.seek(to: 16)
            }
    }
    .preferredColorScheme(.dark)
}
#endif
