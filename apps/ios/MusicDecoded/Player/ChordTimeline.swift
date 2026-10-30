import MusicDecodedCore
import SwiftUI

/// The player timeline (DESIGN.md §3.7 item 3): time labels, a chord track over the waveform,
/// downbeat gridlines and a fixed center playhead. Content scrolls under the playhead. Drag the
/// waveform to scrub, drag the chord track to select a loop (snapped to beats), tap a chord block
/// to seek to its start, pinch to zoom (60–400 pt/s). Only the visible window is drawn.
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
    /// VoiceOver actions: the chord blocks are drawn, so these stand in for tapping one.
    var onPreviousChord: () -> Void = {}
    var onNextChord: () -> Void = {}

    @State private var zoom = TimelineZoom.standard
    @State private var zoomAtGestureStart: CGFloat?
    @State private var dragMode: DragMode?
    @State private var loopDraft: Range<Double>?

    private enum DragMode: Equatable {
        /// Content follows the finger; `startPlayhead` is the position when the drag began.
        case scrub(startPlayhead: Double)
        /// Selecting a loop on the chord track, measured against the viewport at the start.
        case loop(startPlayhead: Double)
        /// Mostly vertical, or a pinch started: ignore the rest of this drag.
        case ignore
    }

    var body: some View {
        VStack(alignment: .leading, spacing: Spacing.xs) {
            GeometryReader { proxy in
                let width = proxy.size.width
                TimelineView(.animation(minimumInterval: nil, paused: !isPlaying)) { _ in
                    let renderer = TimelineRenderer(
                        analysis: analysis,
                        chords: chords,
                        viewport: TimelineViewport(
                            width: width,
                            playhead: isPlaying ? livePosition() : pausedPosition,
                            pointsPerSecond: zoom
                        ),
                        loop: loopDraft ?? loop
                    )
                    Canvas { context, _ in
                        renderer.draw(in: &context)
                    }
                }
                .contentShape(.rect)
                .gesture(dragGesture(width: width).exclusively(before: tapGesture(width: width)))
                .simultaneousGesture(zoomGesture)
            }
            .frame(height: TimelineMetrics.height)
            .clipped()
            .accessibilityElement()
            .accessibilityLabel("Chord timeline")
            .accessibilityValue(accessibilityValue)
            .accessibilityAdjustableAction { direction in
                let step = direction == .increment ? Self.accessibilityStep : -Self.accessibilityStep
                seek(livePosition() + step)
            }
            .accessibilityHint("Swipe up or down to move 5 seconds.")
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

            Text("Pinch to zoom · drag to scrub")
                .font(.overline)
                .foregroundStyle(Palette.textMuted)
                .padding(.horizontal, Spacing.md)
                .accessibilityHidden(true)
        }
        .padding(.vertical, Spacing.sm)
        .background(Palette.timelineBackground, in: .rect(cornerRadius: Radius.card))
        .overlay {
            RoundedRectangle(cornerRadius: Radius.card).strokeBorder(Palette.timelineBorder, lineWidth: 1)
        }
    }

    /// VoiceOver swipes up/down move by this many seconds.
    private static let accessibilityStep: Double = 5
    /// One VoiceOver "Zoom in" / "Zoom out" step.
    private static let zoomStep: CGFloat = 1.5

    private var accessibilityValue: String {
        let t = isPlaying ? livePosition() : pausedPosition
        let name = currentState(at: t, in: analysis).chordIndex.map { chords.spokenName(analysis.chords[$0].label) } ?? ""
        let time = PlayerFormat.time(t)
        return name.isEmpty ? time : "\(name), \(time)"
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
            } else if TimelineMetrics.chordTrackRange.contains(value.startLocation.y) {
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

    private func tapGesture(width: CGFloat) -> some Gesture {
        SpatialTapGesture()
            .onEnded { value in
                guard TimelineMetrics.chordTrackRange.contains(value.location.y) else { return }
                let viewport = TimelineViewport(width: width, playhead: livePosition(), pointsPerSecond: zoom)
                let t = viewport.time(value.location.x)
                if let chord = analysis.chords.first(where: { t >= $0.start && t < $0.end && isNoChord($0.label) == false }) {
                    onSeek(chord.start)
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
    static let labels: CGFloat = 16
    static let chordTrack: CGFloat = 32
    static let waveform: CGFloat = 72
    static let labelSpacing: CGFloat = 56
    static let barMaxWidth: CGFloat = 3
    static let hairline: CGFloat = 1
    static let playhead: CGFloat = 1.5
    static let playheadGlow: CGFloat = 6
    static let playheadDot: CGFloat = 4
    static let blockRadius: CGFloat = 6
    static let minLabelWidth: CGFloat = 16

    static let chordTop: CGFloat = labels + Spacing.xxs
    static let chordBottom: CGFloat = chordTop + chordTrack
    static let waveTop: CGFloat = chordBottom + Spacing.xs
    static let waveBottom: CGFloat = waveTop + waveform
    static let height: CGFloat = waveBottom
    static let chordTrackRange: ClosedRange<CGFloat> = chordTop...chordBottom
}

/// Draws one frame of the timeline. Only what's inside the viewport is drawn.
@MainActor
private struct TimelineRenderer {
    let analysis: Analysis
    let chords: ChordDisplay
    let viewport: TimelineViewport
    let loop: Range<Double>?

    func draw(in context: inout GraphicsContext) {
        drawTimeLabels(&context)
        drawDownbeats(&context)
        drawWaveform(&context)
        drawChordBlocks(&context)
        if let loop {
            drawLoop(loop, in: &context)
        }
        drawPlayhead(&context)
    }

    private func drawTimeLabels(_ context: inout GraphicsContext) {
        let step = TimelineMath.timeLabelStep(pointsPerSecond: viewport.pointsPerSecond, minSpacing: TimelineMetrics.labelSpacing)
        let last = min(viewport.endTime, analysis.meta.durationSec)
        var second = Int((max(0, viewport.startTime) / Double(step)).rounded(.up)) * step
        var ticks = Path()
        while Double(second) <= last {
            let x = viewport.x(Double(second))
            let label = context.resolve(
                Text(PlayerFormat.time(Double(second)))
                    .font(.caption2)
                    .monospacedDigit()
                    .foregroundStyle(Palette.textMuted)
            )
            context.draw(label, at: CGPoint(x: x, y: (TimelineMetrics.labels - Spacing.xxs) / 2), anchor: .center)
            ticks.move(to: CGPoint(x: x, y: TimelineMetrics.labels - Spacing.xxs / 2))
            ticks.addLine(to: CGPoint(x: x, y: TimelineMetrics.labels))
            second += step
        }
        context.stroke(ticks, with: .color(Palette.line), lineWidth: TimelineMetrics.hairline)
    }

    private func drawDownbeats(_ context: inout GraphicsContext) {
        var lines = Path()
        let beats = analysis.beats
        for index in TimelineMath.visibleBeatIndices(beats, from: viewport.startTime, to: viewport.endTime) where beats[index].downbeat {
            let x = viewport.x(beats[index].t)
            lines.move(to: CGPoint(x: x, y: TimelineMetrics.chordTop))
            lines.addLine(to: CGPoint(x: x, y: TimelineMetrics.waveBottom))
        }
        context.stroke(lines, with: .color(Palette.line.opacity(0.5)), lineWidth: TimelineMetrics.hairline)
    }

    /// Waveform bars from `waveformPeaks` (played = primary), or beat ticks when there are no peaks.
    private func drawWaveform(_ context: inout GraphicsContext) {
        let middle = (TimelineMetrics.waveTop + TimelineMetrics.waveBottom) / 2
        let height = TimelineMetrics.waveform
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

    private func drawChordBlocks(_ context: inout GraphicsContext) {
        let inset = Spacing.xxs / 2
        let padding = Spacing.xs
        let chordList = analysis.chords
        for index in TimelineMath.visibleChordIndices(chordList, from: viewport.startTime, to: viewport.endTime) {
            let chord = chordList[index]
            if isNoChord(chord.label) { continue }
            let left = viewport.x(chord.start) + inset
            let width = viewport.x(chord.end) - inset - left
            if width <= 0 { continue }
            let isCurrent = viewport.playhead >= chord.start && viewport.playhead < chord.end
            let rect = CGRect(x: left, y: TimelineMetrics.chordTop, width: width, height: TimelineMetrics.chordTrack)
            let block = Path(roundedRect: rect, cornerRadius: TimelineMetrics.blockRadius)
            if isCurrent {
                context.fill(block, with: .color(Palette.chord))
            } else {
                context.fill(block, with: .color(Palette.chordMuted))
                context.stroke(block, with: .color(Palette.chordMutedBorder), lineWidth: TimelineMetrics.hairline)
            }

            let labelRect = rect.insetBy(dx: padding, dy: 0)
            guard labelRect.width >= TimelineMetrics.minLabelWidth else { continue }
            let label = context.resolve(
                Text(chords.name(chord.label))
                    .font(.footnote.weight(.semibold))
                    .foregroundStyle(isCurrent ? Palette.onChord : Palette.chord)
            )
            // Keep the name readable while a long block scrolls off the left edge.
            let labelX = max(labelRect.minX, padding)
            context.drawLayer { layer in
                layer.clip(to: Path(labelRect))
                layer.draw(label, at: CGPoint(x: labelX, y: labelRect.midY), anchor: .leading)
            }
        }
    }

    private func drawLoop(_ loop: Range<Double>, in context: inout GraphicsContext) {
        let left = viewport.x(loop.lowerBound)
        let right = viewport.x(loop.upperBound)
        guard right >= 0, left <= viewport.width else { return }
        let top = TimelineMetrics.chordTop
        let bottom = TimelineMetrics.waveBottom
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

    /// A 1.5pt primary line with a dot on top and a soft glow (DESIGN.md §3.7).
    private func drawPlayhead(_ context: inout GraphicsContext) {
        let x = viewport.center
        let top = TimelineMetrics.chordTop - TimelineMetrics.playheadDot
        var line = Path()
        line.move(to: CGPoint(x: x, y: top))
        line.addLine(to: CGPoint(x: x, y: TimelineMetrics.waveBottom))
        context.stroke(
            line,
            with: .color(Palette.primary.opacity(0.22)),
            style: StrokeStyle(lineWidth: TimelineMetrics.playheadGlow, lineCap: .round)
        )
        context.stroke(line, with: .color(Palette.primary), lineWidth: TimelineMetrics.playhead)
        let dot = TimelineMetrics.playheadDot
        context.fill(
            Path(ellipseIn: CGRect(x: x - dot, y: top - dot, width: dot * 2, height: dot * 2)),
            with: .color(Palette.primary)
        )
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
            .padding(Spacing.md)
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
