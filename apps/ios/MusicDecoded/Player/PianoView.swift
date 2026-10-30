import MusicDecodedCore
import SwiftUI

/// Piano focus (DESIGN.md §3.7.1): the Piano learning view (§3.8) inline in the player's middle
/// region. Shares the player's `PlayerModel` and transport: head card, the large fixed keyboard
/// (same voicing, inversion and fingering rules as the piano panel), the progression strip and
/// the learning controls.
struct PianoFocusView: View {
    let model: PlayerModel

    var body: some View {
        ScrollView {
            VStack(spacing: Spacing.md) {
                PianoHeadCard(model: model)
                    .padding(.horizontal, Spacing.md)
                // Inset by the gutter so the frame's hard shadow shows (DESIGN.md §1: lime shadowLg).
                PianoPanel(
                    model: model,
                    keysHeight: ComponentSize.pianoLargeHeight,
                    showsChordName: false,
                    showsNoteNames: model.showNoteNames,
                    shadow: HardShadow.lg
                )
                .padding(.horizontal, Spacing.md)
                .padding(.trailing, HardShadow.lg)
                ProgressionStrip(model: model)
                LearningControls(model: model)
                    .padding(.horizontal, Spacing.md)
            }
            .padding(.vertical, Spacing.md)
        }
        .scrollIndicators(.hidden)
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
                Eyebrow(text: "Current chord")
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
                .frame(width: BorderWidth.separator)
                .accessibilityHidden(true)

            VStack(alignment: .leading, spacing: Spacing.xxs) {
                Text("Up next")
                    .overlineStyle()
                Text(nextName.isEmpty ? "—" : nextName)
                    .font(.display(40, relativeTo: .title))
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
        .block(Palette.surface, shadow: HardShadow.lg, shadowColor: Palette.pink)
        .padding(.trailing, HardShadow.lg)
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
                    LazyHStack(spacing: Spacing.sm) {
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
                    // Room for the current card's hard shadow inside the scroll view's clip.
                    .padding(.bottom, HardShadow.sm)
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
                    .font(.detail)
                    .foregroundStyle(isCurrent ? Palette.textSecondaryOnLime : Palette.textSecondary)
            }
            .padding(.horizontal, Spacing.sm)
            .padding(.vertical, Spacing.xs)
            .frame(minHeight: ComponentSize.minTouchTarget)
            .block(
                isCurrent ? Palette.chord : Palette.surface,
                shadow: isCurrent ? HardShadow.sm : 0,
                shadowColor: Palette.violet,
                border: isCurrent ? Palette.border : Palette.borderOnInk
            )
        }
        .buttonStyle(PushButtonStyle())
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
