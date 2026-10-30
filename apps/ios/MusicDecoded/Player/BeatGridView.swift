import MusicDecodedCore
import SwiftUI

/// Beat grid view (DESIGN.md §3.7.1): one row per bar, one cell per beat, bar numbers on the
/// left. A cell shows the chord that starts in it; continuation cells get a thin lime underline.
/// Current beat = lime fill, current bar = 3pt lime outline; keeps the current bar in the upper
/// third; tap a cell to seek to its beat.
struct BeatGridView: View {
    let model: PlayerModel
    let analysis: Analysis

    @State private var bars: [BeatGridBar] = []

    var body: some View {
        Group {
            if analysis.beats.isEmpty {
                ContentUnavailableView(
                    "No beat grid for this song",
                    systemImage: "square.grid.2x2",
                    description: Text("The analysis didn't find any beats.")
                )
                .frame(maxWidth: .infinity, maxHeight: .infinity)
            } else {
                grid
            }
        }
        .onAppear {
            if bars.isEmpty {
                bars = BeatGrid.bars(for: analysis)
            }
        }
    }

    private var grid: some View {
        let perBar = BeatGrid.beatsPerBar(analysis)
        let currentBeat = model.beatIndex
        let currentBar = currentBeat.flatMap { BeatGrid.barIndex(ofBeat: $0, in: bars) }
        let chords = model.chords
        return ScrollViewReader { proxy in
            ScrollView {
                LazyVStack(spacing: Spacing.xxs) {
                    ForEach(bars, id: \.number) { bar in
                        BeatGridRow(
                            bar: bar,
                            perBar: perBar,
                            labels: analysis.chords,
                            chords: chords,
                            currentBeat: currentBeat,
                            isCurrentBar: currentBar.map { bars[$0].number == bar.number } ?? false,
                            onSeek: { model.seek(to: $0) }
                        )
                        .id(bar.number)
                    }
                }
                .padding(.horizontal, Spacing.md)
                .padding(.vertical, Spacing.xs)
            }
            .scrollIndicators(.hidden)
            .onAppear {
                scroll(proxy, to: currentBar, animated: false)
            }
            .onChange(of: currentBar) { _, bar in
                scroll(proxy, to: bar, animated: true)
            }
            .onChange(of: bars.count) { _, _ in
                scroll(proxy, to: currentBar, animated: false)
            }
        }
    }

    /// Keeps the current bar in the upper third.
    private func scroll(_ proxy: ScrollViewProxy, to barIndex: Int?, animated: Bool) {
        guard let barIndex, bars.indices.contains(barIndex) else { return }
        let id = bars[barIndex].number
        let anchor = UnitPoint(x: 0.5, y: 0.25)
        if animated {
            withAnimation {
                proxy.scrollTo(id, anchor: anchor)
            }
        } else {
            proxy.scrollTo(id, anchor: anchor)
        }
    }
}

private struct BeatGridRow: View {
    let bar: BeatGridBar
    let perBar: Int
    let labels: [Analysis.Chord]
    let chords: ChordDisplay
    let currentBeat: Int?
    let isCurrentBar: Bool
    let onSeek: (Double) -> Void

    var body: some View {
        HStack(spacing: Spacing.xxs) {
            Text("\(bar.number)")
                .font(.detail)
                .monospacedDigit()
                .foregroundStyle(isCurrentBar ? Palette.lime : Palette.textMuted)
                .lineLimit(1)
                .minimumScaleFactor(0.6)
                .frame(width: ComponentSize.barNumberWidth, alignment: .trailing)
                .accessibilityHidden(true)
            HStack(spacing: Spacing.xxs) {
                ForEach(0..<perBar, id: \.self) { slot in
                    if slot < bar.cells.count {
                        cell(bar.cells[slot], beat: slot + 1)
                    } else {
                        Color.clear
                            .frame(maxWidth: .infinity, minHeight: ComponentSize.beatCellHeight)
                    }
                }
            }
            .padding(BorderWidth.thick)
            .overlay {
                if isCurrentBar {
                    Rectangle()
                        .strokeBorder(Palette.lime, lineWidth: BorderWidth.thick)
                }
            }
        }
    }

    private func cell(_ cell: BeatGridCell, beat: Int) -> some View {
        let isCurrent = cell.beatIndex == currentBeat
        let label = cell.chordIndex.map { labels[$0].label }
        let name = label.map(chords.name) ?? ""
        let numeral = label.flatMap(chords.numeral)
        let spoken = label.map(chords.spokenName) ?? (cell.continues ? "Chord continues" : "No chord")
        return Button {
            onSeek(cell.time)
        } label: {
            VStack(spacing: 0) {
                if name.isEmpty == false {
                    Text(name)
                        .font(.display(15, relativeTo: .headline))
                        .lineLimit(1)
                        .minimumScaleFactor(0.5)
                    if let numeral {
                        Text(numeral)
                            .font(.overline)
                            .lineLimit(1)
                    }
                }
            }
            .foregroundStyle(isCurrent ? Palette.onChord : Palette.text)
            .padding(.horizontal, Spacing.xxs)
            .frame(maxWidth: .infinity, minHeight: ComponentSize.beatCellHeight)
            .background(isCurrent ? Palette.chord : Palette.surfaceRaised, in: .rect)
            .overlay(alignment: .bottom) {
                if cell.continues {
                    Rectangle()
                        .fill(Palette.lime)
                        .frame(height: BorderWidth.separator)
                        .padding(.horizontal, Spacing.xxs)
                        .padding(.bottom, Spacing.xxs)
                }
            }
            .contentShape(.rect)
        }
        .buttonStyle(PushButtonStyle())
        .accessibilityLabel("Bar \(bar.number), beat \(beat)")
        .accessibilityValue(numeral.map { "\(spoken), \(spokenRomanNumeral($0))" } ?? spoken)
        .accessibilityHint("Plays from this beat")
        .accessibilityAddTraits(isCurrent ? .isSelected : [])
    }
}
