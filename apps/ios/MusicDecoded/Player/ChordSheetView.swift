import MusicDecodedCore
import SwiftUI

/// Chord sheet view (DESIGN.md §3.7.1): the whole progression as chord boxes (name + numeral,
/// the timeline label style), 4 bars per line with bar lines between them, repeated chords
/// within a bar shown once. Current chord = lime fill; auto-scrolls; tap a box to seek.
struct ChordSheetView: View {
    let model: PlayerModel
    let analysis: Analysis

    @State private var lines: [[ChordSheetBar]] = []

    var body: some View {
        let current = currentBox
        let chords = model.chords
        ScrollViewReader { proxy in
            ScrollView {
                LazyVStack(alignment: .leading, spacing: Spacing.sm) {
                    ForEach(lines.indices, id: \.self) { lineIndex in
                        let line = lines[lineIndex]
                        // Plain values (no closures), and only the line holding the current box gets
                        // it, so a chord change redraws just the lines it enters or leaves.
                        SheetLine(
                            bars: line,
                            labels: analysis.chords,
                            chords: chords,
                            current: current.flatMap { box in line.contains { $0.number == box.barNumber } ? box : nil },
                            model: model
                        )
                        .id(lineIndex)
                    }
                }
                .padding(.horizontal, Spacing.md)
                .padding(.vertical, Spacing.xs)
            }
            .scrollIndicators(.hidden)
            .onAppear {
                if lines.isEmpty {
                    lines = ChordSheet.lines(ChordSheet.bars(for: analysis))
                }
            }
            .onChange(of: currentLine) { _, line in
                guard let line else { return }
                withAnimation {
                    proxy.scrollTo(line, anchor: UnitPoint(x: 0.5, y: 0.3))
                }
            }
            .onChange(of: lines.count) { _, _ in
                if let line = currentLine {
                    proxy.scrollTo(line, anchor: UnitPoint(x: 0.5, y: 0.3))
                }
            }
        }
    }

    /// The time the current box is looked up at: the current beat (so this only changes on beats),
    /// or the current chord's start for songs without beats.
    private var lookupTime: Double? {
        if let beat = model.beatIndex, analysis.beats.indices.contains(beat) {
            return analysis.beats[beat].t
        }
        return model.chordIndex.map { analysis.chords[$0].start }
    }

    /// Bar number + entry index of the current box.
    private var currentBox: SheetPosition? {
        guard let t = lookupTime else { return nil }
        let bars = lines.flatMap { $0 }
        guard let found = ChordSheet.current(at: t, chordIndex: model.chordIndex, in: bars) else { return nil }
        return SheetPosition(barNumber: bars[found.bar].number, entry: found.entry)
    }

    private var currentLine: Int? {
        guard let box = currentBox else { return nil }
        return lines.firstIndex { line in line.contains { $0.number == box.barNumber } }
    }
}

struct SheetPosition: Equatable {
    let barNumber: Int
    let entry: Int
}

/// One line: up to 4 equal bar columns, a bar line before each.
private struct SheetLine: View {
    let bars: [ChordSheetBar]
    let labels: [Analysis.Chord]
    let chords: ChordDisplay
    let current: SheetPosition?
    /// For seeking only; the line reads no model state.
    let model: PlayerModel

    var body: some View {
        HStack(alignment: .center, spacing: Spacing.xxs) {
            ForEach(0..<ChordSheet.barsPerLine, id: \.self) { column in
                Rectangle()
                    .fill(Palette.line)
                    .frame(width: BorderWidth.separator)
                    .accessibilityHidden(true)
                if column < bars.count {
                    bar(bars[column])
                } else {
                    Color.clear.frame(maxWidth: .infinity)
                }
            }
            Rectangle()
                .fill(Palette.line)
                .frame(width: BorderWidth.separator)
                .accessibilityHidden(true)
        }
        .fixedSize(horizontal: false, vertical: true)
    }

    private func bar(_ bar: ChordSheetBar) -> some View {
        HStack(spacing: Spacing.xxs) {
            if bar.entries.isEmpty {
                Text("—")
                    .font(.detail)
                    .foregroundStyle(Palette.textMuted)
                    .frame(maxWidth: .infinity, minHeight: ComponentSize.minTouchTarget)
                    .accessibilityLabel("Bar \(bar.number), no chord")
            }
            ForEach(bar.entries.indices, id: \.self) { index in
                let entry = bar.entries[index]
                SheetBox(
                    name: chords.name(entry.label),
                    numeral: chords.numeral(entry.label),
                    spoken: "Bar \(bar.number), \(chords.spokenName(entry.label))",
                    isCurrent: current == SheetPosition(barNumber: bar.number, entry: index)
                ) {
                    model.seek(to: entry.time)
                }
            }
        }
        .frame(maxWidth: .infinity)
    }
}

/// A chord box in the timeline label style: 2pt black border, padding 4/6; current = lime fill,
/// ink text and `shadowSm`; others `cardRaised` with white text.
private struct SheetBox: View {
    let name: String
    let numeral: String?
    let spoken: String
    let isCurrent: Bool
    let action: () -> Void

    var body: some View {
        Button(action: action) {
            VStack(spacing: 0) {
                Text(name)
                    .font(.display(17, relativeTo: .headline))
                    .lineLimit(1)
                    .minimumScaleFactor(0.5)
                if let numeral {
                    Text(numeral)
                        .font(.labelSmall)
                        .lineLimit(1)
                        .minimumScaleFactor(0.7)
                }
            }
            .foregroundStyle(isCurrent ? Palette.onChord : Palette.text)
            .padding(.vertical, Spacing.xxs)
            .padding(.horizontal, Spacing.xxs)
            .frame(maxWidth: .infinity, minHeight: ComponentSize.minTouchTarget)
            .background(isCurrent ? Palette.chord : Palette.surfaceRaised, in: .rect)
            .overlay {
                Rectangle().strokeBorder(Palette.border, lineWidth: BorderWidth.separator)
            }
            .hardShadow(isCurrent ? HardShadow.sm : 0, color: Palette.violet)
            .contentShape(.rect)
        }
        .buttonStyle(PushButtonStyle())
        .accessibilityLabel(numeral.map { "\(spoken), \(spokenRomanNumeral($0))" } ?? spoken)
        .accessibilityHint("Plays from this chord")
        .accessibilityAddTraits(isCurrent ? .isSelected : [])
    }
}
