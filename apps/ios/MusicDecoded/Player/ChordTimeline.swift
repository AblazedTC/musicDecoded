import MusicDecodedCore
import SwiftUI

/// The player timeline (DESIGN.md §3.7 item 3), edge to edge with no card: a time-label row,
/// compact chord labels (name + Roman numeral) at each chord change with a tick down to the
/// waveform, downbeat gridlines and a fixed center playhead. Content scrolls under the playhead.
/// Drag the waveform to scrub, drag the first label row to select a loop (snapped to beats), tap
/// a label to seek to its chord, pinch to zoom (60–400 pt/s). Only the visible window is drawn.
/// It fills the height its parent gives it. The label band has a second row only when a label needs
/// it at the current zoom; otherwise the waveform takes that height.
///
/// While playing, a `TimelineView(.animation)` redraws every frame from `livePosition` (the
/// engine clock, not observed); while paused it draws `pausedPosition`, so seeks still redraw.
struct ChordTimeline: View {
    let analysis: Analysis
    let chords: ChordDisplay
    let isPlaying: Bool
    /// The playhead while paused (observed, so scrubbing and seeking redraw).
    let pausedPosition: Double
    /// Reads the playback clock. Called every frame while playing and at the start of gestures.
    let livePosition: () -> Double
    /// The active loop, drawn as a band.
    var loop: Range<Double>?
    let onSeek: (Double) -> Void
    var onLoopSelected: (Range<Double>) -> Void = { _ in }
    /// VoiceOver actions: the chord labels are drawn, so these stand in for tapping one.
    var onPreviousChord: () -> Void = {}
    var onNextChord: () -> Void = {}

    @Environment(\.dynamicTypeSize) private var dynamicTypeSize
    @State private var zoom = TimelineZoom.standard
    @State private var zoomAtGestureStart: CGFloat?
    @State private var dragMode: DragMode?
    @State private var loopDraft: Range<Double>?
    /// Label sizes, the label layout and the last drawn label frames. Not observed.
    @State private var labels = TimelineLabelCache()

    private enum DragMode: Equatable {
        /// Content follows the finger; `startPlayhead` is the position when the drag began.
        case scrub(startPlayhead: Double)
        /// Selecting a loop on the label row, measured against the viewport at the start.
        case loop(startPlayhead: Double)
        /// Mostly vertical, or a pinch started: ignore the rest of this drag.
        case ignore
    }

    var body: some View {
        GeometryReader { proxy in
            let size = proxy.size
            TimelineView(.animation(minimumInterval: nil, paused: !isPlaying)) { _ in
                let renderer = TimelineRenderer(
                    analysis: analysis,
                    chords: chords,
                    viewport: TimelineViewport(
                        width: size.width,
                        playhead: isPlaying ? livePosition() : pausedPosition,
                        pointsPerSecond: zoom
                    ),
                    height: size.height,
                    loop: loopDraft ?? loop,
                    labels: labels,
                    typeSize: dynamicTypeSize
                )
                Canvas { context, _ in
                    renderer.draw(in: &context)
                }
            }
            .contentShape(.rect)
            .gesture(dragGesture(width: size.width).exclusively(before: tapGesture(width: size.width)))
            .simultaneousGesture(zoomGesture)
        }
        .clipped()
        .accessibilityElement()
        .accessibilityLabel("Chord timeline")
        .accessibilityValue(accessibilityValue)
        .accessibilityAdjustableAction { direction in
            let step = direction == .increment ? Self.accessibilityStep : -Self.accessibilityStep
            seek(livePosition() + step)
        }
        // The visible hint row is gone (DESIGN.md §3.7); its text lives here.
        .accessibilityHint("Pinch to zoom, drag to scrub. Swipe up or down to move 5 seconds.")
        .accessibilityAction(named: "Previous chord") {
            onPreviousChord()
        }
        .accessibilityAction(named: "Next chord") {
            onNextChord()
        }
        .accessibilityAction(named: "Zoom in") {
            zoom = TimelineZoom.clamp(zoom * Self.zoomStep)
        }
        .accessibilityAction(named: "Zoom out") {
            zoom = TimelineZoom.clamp(zoom / Self.zoomStep)
        }
    }

    /// VoiceOver swipes up/down move by this many seconds.
    private static let accessibilityStep: Double = 5
    /// One VoiceOver "Zoom in" / "Zoom out" step.
    private static let zoomStep: CGFloat = 1.5

    /// "Current chord E major, three. Next A major, six. 0:17"
    private var accessibilityValue: String {
        let t = isPlaying ? livePosition() : pausedPosition
        var parts: [String] = []
        if let index = currentState(at: t, in: analysis).chordIndex, isNoChord(analysis.chords[index].label) == false {
            parts.append("Current chord \(spokenChord(analysis.chords[index].label))")
        }
        if let next = upNext(at: t, in: analysis) {
            parts.append("Next \(spokenChord(analysis.chords[next.chordIndex].label))")
        }
        parts.append(PlayerFormat.time(t))
        return parts.joined(separator: ". ")
    }

    private func spokenChord(_ label: String) -> String {
        let name = chords.spokenName(label)
        guard let numeral = chords.numeral(label) else { return name }
        return "\(name), \(spokenRomanNumeral(numeral))"
    }

    // MARK: - Gestures

    private func dragGesture(width: CGFloat) -> some Gesture {
        DragGesture(minimumDistance: Spacing.xs)
            .onChanged { value in
                dragChanged(value, width: width)
            }
            .onEnded { _ in
                if case .loop = dragMode, let loopDraft {
                    onLoopSelected(loopDraft)
                }
                dragMode = nil
                loopDraft = nil
            }
    }

    private func dragChanged(_ value: DragGesture.Value, width: CGFloat) {
        if zoomAtGestureStart != nil {
            dragMode = .ignore
            loopDraft = nil
        }
        if dragMode == nil {
            let dx = value.translation.width
            let dy = value.translation.height
            if abs(dy) > abs(dx) {
                dragMode = .ignore
            } else if labels.loopBand.contains(value.startLocation.y) {
                dragMode = .loop(startPlayhead: livePosition())
            } else {
                dragMode = .scrub(startPlayhead: livePosition())
            }
        }
        switch dragMode {
        case .scrub(let startPlayhead):
            seek(startPlayhead - Double(value.translation.width / zoom))
        case .loop(let startPlayhead):
            let viewport = TimelineViewport(width: width, playhead: startPlayhead, pointsPerSecond: zoom)
            loopDraft = TimelineMath.snappedLoop(
                viewport.time(value.startLocation.x),
                viewport.time(value.location.x),
                in: analysis
            )
        case .ignore, .none:
            break
        }
    }

    /// Tapping a label seeks to its chord (the frames drawn last, i.e. what's on screen).
    private func tapGesture(width: CGFloat) -> some Gesture {
        SpatialTapGesture()
            .onEnded { value in
                if let index = labels.chordIndex(at: value.location) {
                    onSeek(analysis.chords[index].start)
                }
            }
    }

    private var zoomGesture: some Gesture {
        MagnifyGesture()
            .onChanged { value in
                let base = zoomAtGestureStart ?? zoom
                zoomAtGestureStart = base
                zoom = TimelineZoom.clamp(base * value.magnification)
            }
            .onEnded { _ in
                zoomAtGestureStart = nil
            }
    }

    private func seek(_ t: Double) {
        onSeek(min(max(t, 0), analysis.meta.durationSec))
    }
}

/// Fixed drawing sizes of the timeline (component sizes, not spacing).
private enum TimelineMetrics {
    static let timeRow: CGFloat = 16
    static let labelSpacing: CGFloat = 56
    static let barMaxWidth: CGFloat = 3
    static let hairline: CGFloat = 1
    static let playhead: CGFloat = 2
    /// Chord label box: 2pt black border, padding 4 vertical / 6 horizontal (DESIGN.md §3.7).
    static let labelBorder: CGFloat = 2
    static let labelPaddingV: CGFloat = 4
    static let labelPaddingH: CGFloat = 6
    static let minWaveform: CGFloat = 32
    /// Labels can start this far left of the viewport and still show.
    static let labelLookBehind: CGFloat = 160

    static var nameFont: Font { .display(20, relativeTo: .title3) }
    static var numeralFont: Font { .label }

    static let labelTop: CGFloat = timeRow + Spacing.xxs
}

/// Label sizes and layout, reused across frames: sizes are measured once per text; the boxes, the
/// layout and the band's row count once per zoom and chord display. A chord change re-runs the
/// layout only when that chord's label had been dropped (otherwise placing it changes nothing).
/// Also keeps the label frames of the last frame for tap hit-testing.
@MainActor
final class TimelineLabelCache {
    struct LayoutKey: Equatable {
        var pointsPerSecond: CGFloat
        var transpose: Int
        var basic: Bool
        var accidentals: Accidentals
        var chordCount: Int
        var typeSize: DynamicTypeSize
    }

    private var sizes: [String: CGSize] = [:]
    private var sizesTypeSize: DynamicTypeSize?
    private var layoutKey: LayoutKey?
    private var boxes: [ChordLabelBox] = []
    private var bandRows = 1
    /// The layout with no current chord.
    private var baseRows: [Int: Int] = [:]
    /// The layout for `rowsCurrent`, a chord dropped from `baseRows`; `nil` until needed.
    private var rows: [Int: Int]?
    private var rowsCurrent: Int?
    /// Label frames drawn last, by chord index.
    var frames: [Int: CGRect] = [:]
    /// The first label row, where a drag selects a loop.
    var loopBand: ClosedRange<CGFloat> = TimelineMetrics.labelTop...(TimelineMetrics.labelTop + 48)

    func size(of key: String, typeSize: DynamicTypeSize, measure: () -> CGSize) -> CGSize {
        if sizesTypeSize != typeSize {
            sizes = [:]
            sizesTypeSize = typeSize
        }
        if let size = sizes[key] { return size }
        let size = measure()
        sizes[key] = size
        return size
    }

    func layout(
        for key: LayoutKey,
        current: Int?,
        boxes makeBoxes: () -> [ChordLabelBox]
    ) -> (rows: [Int: Int], bandRows: Int) {
        if layoutKey != key {
            layoutKey = key
            boxes = makeBoxes()
            baseRows = ChordLabelLayout.layout(boxes, current: nil)
            bandRows = ChordLabelLayout.rowCount(boxes)
            rows = nil
        }
        guard let current, baseRows[current] == nil else {
            return (baseRows, bandRows)
        }
        if let rows, rowsCurrent == current {
            return (rows, bandRows)
        }
        let placed = ChordLabelLayout.layout(boxes, current: current)
        rows = placed
        rowsCurrent = current
        return (placed, bandRows)
    }

    func chordIndex(at point: CGPoint) -> Int? {
        // Later labels are drawn on top.
        frames.filter { $0.value.contains(point) }.max { $0.key < $1.key }?.key
    }
}

/// Draws one frame of the timeline. Only what's inside the viewport is drawn.
@MainActor
private struct TimelineRenderer {
    let analysis: Analysis
    let chords: ChordDisplay
    let viewport: TimelineViewport
    let height: CGFloat
    let loop: Range<Double>?
    let labels: TimelineLabelCache
    let typeSize: DynamicTypeSize

    private var showsNumerals: Bool { chords.key != nil }

    func draw(in context: inout GraphicsContext) {
        let boxHeight = labelBoxHeight(context)
        let current: Int? = currentState(at: viewport.playhead, in: analysis).chordIndex.flatMap { index in
            isNoChord(analysis.chords[index].label) ? nil : index
        }
        let layout = labelLayout(current: current, context)
        let band = TimelineMath.labelBandHeight(rows: layout.bandRows, boxHeight: boxHeight, rowGap: CGFloat(ChordLabelLayout.gap))
        let waveTop = min(TimelineMetrics.labelTop + band + Spacing.xs, height - TimelineMetrics.minWaveform)
        let waveBottom = height - Spacing.xxs
        labels.loopBand = TimelineMetrics.labelTop...(TimelineMetrics.labelTop + boxHeight)

        drawTimeLabels(&context)
        drawDownbeats(&context, top: TimelineMetrics.labelTop, bottom: waveBottom)
        drawWaveform(&context, top: waveTop, bottom: waveBottom)
        if let loop {
            drawLoop(loop, in: &context, bottom: waveBottom)
        }
        drawChordLabels(&context, rows: layout.rows, current: current, boxHeight: boxHeight, waveTop: waveTop)
        drawPlayhead(&context, bottom: waveBottom)
    }

    // MARK: Time labels, gridlines, waveform

    private func drawTimeLabels(_ context: inout GraphicsContext) {
        let step = TimelineMath.timeLabelStep(pointsPerSecond: viewport.pointsPerSecond, minSpacing: TimelineMetrics.labelSpacing)
        let last = min(viewport.endTime, analysis.meta.durationSec)
        var second = Int((max(0, viewport.startTime) / Double(step)).rounded(.up)) * step
        var ticks = Path()
        let row = TimelineMetrics.timeRow
        while Double(second) <= last {
            let x = viewport.x(Double(second))
            let label = context.resolve(
                Text(PlayerFormat.time(Double(second)))
                    .font(.overline)
                    .monospacedDigit()
                    .foregroundStyle(Palette.textMuted)
            )
            context.draw(label, at: CGPoint(x: x + Spacing.xxs, y: (row - Spacing.xxs) / 2), anchor: .leading)
            ticks.move(to: CGPoint(x: x, y: 0))
            ticks.addLine(to: CGPoint(x: x, y: row))
            second += step
        }
        context.stroke(ticks, with: .color(Palette.textMuted), lineWidth: TimelineMetrics.hairline)
    }

    private func drawDownbeats(_ context: inout GraphicsContext, top: CGFloat, bottom: CGFloat) {
        var lines = Path()
        let beats = analysis.beats
        for index in TimelineMath.visibleBeatIndices(beats, from: viewport.startTime, to: viewport.endTime) where beats[index].downbeat {
            let x = viewport.x(beats[index].t)
            lines.move(to: CGPoint(x: x, y: top))
            lines.addLine(to: CGPoint(x: x, y: bottom))
        }
        context.stroke(lines, with: .color(Palette.line.opacity(0.5)), lineWidth: TimelineMetrics.hairline)
    }

    /// Waveform bars from `waveformPeaks` (played = lime), or beat ticks when there are no peaks.
    private func drawWaveform(_ context: inout GraphicsContext, top: CGFloat, bottom: CGFloat) {
        let middle = (top + bottom) / 2
        let height = max(bottom - top, 0)
        let minBar = TimelineMetrics.hairline
        var played = Path()
        var idle = Path()
        let lineWidth: CGFloat

        if let peaks = analysis.waveformPeaks, peaks.peaks.isEmpty == false, peaks.fps > 0 {
            let pitch = viewport.pointsPerSecond / CGFloat(peaks.fps)
            lineWidth = min(max(pitch * 0.6, minBar), TimelineMetrics.barMaxWidth)
            for index in TimelineMath.visiblePeakIndices(peaks, from: viewport.startTime, to: viewport.endTime) {
                let t = Double(index) / peaks.fps
                let x = viewport.x(t)
                let half = max(minBar, CGFloat(peaks.peaks[index]) * height) / 2
                if t < viewport.playhead {
                    played.move(to: CGPoint(x: x, y: middle - half))
                    played.addLine(to: CGPoint(x: x, y: middle + half))
                } else {
                    idle.move(to: CGPoint(x: x, y: middle - half))
                    idle.addLine(to: CGPoint(x: x, y: middle + half))
                }
            }
        } else {
            lineWidth = TimelineMetrics.hairline * 2
            let beats = analysis.beats
            for index in TimelineMath.visibleBeatIndices(beats, from: viewport.startTime, to: viewport.endTime) {
                let beat = beats[index]
                let x = viewport.x(beat.t)
                let half = height * (beat.downbeat ? 0.35 : 0.18)
                if beat.t < viewport.playhead {
                    played.move(to: CGPoint(x: x, y: middle - half))
                    played.addLine(to: CGPoint(x: x, y: middle + half))
                } else {
                    idle.move(to: CGPoint(x: x, y: middle - half))
                    idle.addLine(to: CGPoint(x: x, y: middle + half))
                }
            }
        }
        let style = StrokeStyle(lineWidth: lineWidth, lineCap: .round)
        context.stroke(played, with: .color(Palette.wavePlayed), style: style)
        context.stroke(idle, with: .color(Palette.waveIdle), style: style)
    }

    // MARK: Chord labels

    private func nameText(_ name: String, current: Bool) -> Text {
        Text(name)
            .font(TimelineMetrics.nameFont)
            .foregroundStyle(current ? Palette.onChord : Palette.text)
    }

    private func numeralText(_ numeral: String, current: Bool) -> Text {
        Text(numeral)
            .font(TimelineMetrics.numeralFont)
            .foregroundStyle(current ? Palette.onChord : Palette.text)
    }

    private func measure(_ text: Text, _ context: GraphicsContext) -> CGSize {
        context.resolve(text).measure(in: CGSize(width: CGFloat.greatestFiniteMagnitude, height: CGFloat.greatestFiniteMagnitude))
    }

    private func nameSize(_ name: String, _ context: GraphicsContext) -> CGSize {
        labels.size(of: "n:" + name, typeSize: typeSize) { measure(nameText(name, current: false), context) }
    }

    private func numeralSize(_ numeral: String, _ context: GraphicsContext) -> CGSize {
        labels.size(of: "r:" + numeral, typeSize: typeSize) { measure(numeralText(numeral, current: false), context) }
    }

    /// Every box has the same height: one name line, plus one numeral line when there's a key.
    private func labelBoxHeight(_ context: GraphicsContext) -> CGFloat {
        var textHeight = nameSize("A♯", context).height
        if showsNumerals {
            textHeight += numeralSize("vii°", context).height
        }
        return textHeight + TimelineMetrics.labelPaddingV * 2 + TimelineMetrics.labelBorder * 2
    }

    private func labelWidth(_ index: Int, _ context: GraphicsContext) -> CGFloat {
        let label = analysis.chords[index].label
        var textWidth = nameSize(chords.name(label), context).width
        if let numeral = chords.numeral(label) {
            textWidth = max(textWidth, numeralSize(numeral, context).width)
        }
        return textWidth + TimelineMetrics.labelPaddingH * 2 + TimelineMetrics.labelBorder * 2
    }

    private func labelLayout(current: Int?, _ context: GraphicsContext) -> (rows: [Int: Int], bandRows: Int) {
        let chordList = analysis.chords
        let key = TimelineLabelCache.LayoutKey(
            pointsPerSecond: viewport.pointsPerSecond,
            transpose: chords.transpose,
            basic: chords.basic,
            accidentals: chords.accidentals,
            chordCount: chordList.count,
            typeSize: typeSize
        )
        return labels.layout(for: key, current: current) {
            ChordLabelLayout.boxes(for: chordList, pointsPerSecond: Double(viewport.pointsPerSecond)) { index in
                Double(labelWidth(index, context))
            }
        }
    }

    private func drawChordLabels(
        _ context: inout GraphicsContext,
        rows: [Int: Int],
        current: Int?,
        boxHeight: CGFloat,
        waveTop: CGFloat
    ) {
        let chordList = analysis.chords
        let measuring = context
        let lookBehind = Double(TimelineMetrics.labelLookBehind / viewport.pointsPerSecond)
        var frames: [Int: CGRect] = [:]
        var ticks = Path()
        var boxes: [(index: Int, rect: CGRect)] = []
        for index in TimelineMath.visibleChordIndices(chordList, from: viewport.startTime - lookBehind, to: viewport.endTime) {
            let chord = chordList[index]
            if isNoChord(chord.label) { continue }
            let x = viewport.x(chord.start)
            guard let row = rows[index] else {
                // Dropped at this zoom: the tick stays.
                ticks.move(to: CGPoint(x: x, y: TimelineMetrics.labelTop))
                ticks.addLine(to: CGPoint(x: x, y: waveTop))
                continue
            }
            let top = TimelineMetrics.labelTop + CGFloat(row) * (boxHeight + CGFloat(ChordLabelLayout.gap))
            let rect = CGRect(x: x, y: top, width: labelWidth(index, measuring), height: boxHeight)
            guard rect.maxX >= 0, rect.minX <= viewport.width else { continue }
            ticks.move(to: CGPoint(x: x, y: rect.maxY))
            ticks.addLine(to: CGPoint(x: x, y: max(waveTop, rect.maxY + Spacing.xs)))
            boxes.append((index: index, rect: rect))
            frames[index] = rect
        }
        context.stroke(ticks, with: .color(Palette.textMuted), lineWidth: TimelineMetrics.hairline)

        // The current box last, so its shadow sits over its neighbors.
        boxes.sort { a, b in
            (a.index == current ? 1 : 0, a.index) < (b.index == current ? 1 : 0, b.index)
        }
        for box in boxes {
            drawLabel(index: box.index, in: box.rect, current: box.index == current, context: &context)
        }
        labels.frames = frames
    }

    /// Square box, 2pt black border; current = lime + ink text + `shadowSm`, others `cardRaised`.
    private func drawLabel(index: Int, in rect: CGRect, current: Bool, context: inout GraphicsContext) {
        let label = analysis.chords[index].label
        if current {
            context.fill(
                Path(rect.offsetBy(dx: HardShadow.sm, dy: HardShadow.sm)),
                with: .color(Palette.violet)
            )
        }
        context.fill(Path(rect), with: .color(current ? Palette.chord : Palette.cardRaised))
        let half = TimelineMetrics.labelBorder / 2
        context.stroke(Path(rect.insetBy(dx: half, dy: half)), with: .color(Palette.border), lineWidth: TimelineMetrics.labelBorder)

        let inner = rect.insetBy(dx: 0, dy: TimelineMetrics.labelBorder + TimelineMetrics.labelPaddingV)
        let name = context.resolve(nameText(chords.name(label), current: current))
        context.draw(name, at: CGPoint(x: rect.midX, y: inner.minY), anchor: .top)
        if let numeral = chords.numeral(label) {
            let resolved = context.resolve(numeralText(numeral, current: current))
            context.draw(resolved, at: CGPoint(x: rect.midX, y: inner.maxY), anchor: .bottom)
        }
    }

    // MARK: Loop, playhead

    private func drawLoop(_ loop: Range<Double>, in context: inout GraphicsContext, bottom: CGFloat) {
        let left = viewport.x(loop.lowerBound)
        let right = viewport.x(loop.upperBound)
        guard right >= 0, left <= viewport.width else { return }
        let top = TimelineMetrics.labelTop
        context.fill(
            Path(CGRect(x: left, y: top, width: right - left, height: bottom - top)),
            with: .color(Palette.chord.opacity(0.14))
        )
        var edges = Path()
        edges.move(to: CGPoint(x: left, y: top))
        edges.addLine(to: CGPoint(x: left, y: bottom))
        edges.move(to: CGPoint(x: right, y: top))
        edges.addLine(to: CGPoint(x: right, y: bottom))
        context.stroke(edges, with: .color(Palette.chord), lineWidth: TimelineMetrics.hairline * 2)
    }

    /// A 2pt lime line at the center, through the labels and the waveform (DESIGN.md §3.7).
    private func drawPlayhead(_ context: inout GraphicsContext, bottom: CGFloat) {
        var line = Path()
        line.move(to: CGPoint(x: viewport.center, y: TimelineMetrics.timeRow))
        line.addLine(to: CGPoint(x: viewport.center, y: bottom))
        context.stroke(line, with: .color(Palette.primary), lineWidth: TimelineMetrics.playhead)
    }
}

#if DEBUG
private struct ChordTimelinePreview: View {
    @State private var position: Double = 16
    @State private var loop: Range<Double>? = 12.841..<18.947
    private let analysis = DebugFixtures.load(trackId: DebugFixtures.sunnyDays)

    var body: some View {
        if let analysis {
            ChordTimeline(
                analysis: analysis,
                chords: ChordDisplay(key: analysis.key, voicings: BundledContracts.voicings),
                isPlaying: false,
                pausedPosition: position,
                livePosition: { position },
                loop: loop,
                onSeek: { position = $0 },
                onLoopSelected: { loop = $0 }
            )
            .frame(height: 180)
        } else {
            Text("Fixture not bundled")
        }
    }
}

#Preview {
    ChordTimelinePreview()
        .frame(maxHeight: .infinity)
        .background(Palette.canvas)
        .preferredColorScheme(.dark)
}
#endif
