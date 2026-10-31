import MusicDecodedCore
import SwiftUI

/// The piano panel (DESIGN.md §3.7 "Owner revision 2"): the active chord name and numeral, the
/// fixed C3–B5 keyboard (bass `blue`, right hand `lime`, finger numbers, note names under the
/// keys) and one page dot per inversion. Swipe left/right = next/previous inversion; VoiceOver
/// gets the same as adjustable actions. Natural height: `keysHeight + PlayerLayout.pianoPanelChrome`.
/// Reads only chord-level model state (current chord, display settings, inversion), never the
/// playhead, so the clock doesn't redraw it.
struct PianoPanel: View {
    let model: PlayerModel
    var keysHeight: CGFloat = ComponentSize.fixedKeyboardCompact
    var showsChordName = true
    var showsNoteNames = true
    /// The Piano focus view's frame: lime hard shadow (DESIGN.md §1).
    var shadow: CGFloat = 0

    /// A swipe must travel this far, mostly sideways, to change the inversion.
    private static let swipeThreshold: CGFloat = 40

    @State private var haptic = 0

    var body: some View {
        let chords = model.chords
        let label = model.currentLabel
        let voicing = model.currentVoicing
        let numeral = label.flatMap(chords.numeral)
        VStack(spacing: 0) {
            if showsChordName {
                chordRow(name: label.map(chords.name) ?? "", numeral: numeral)
                    .frame(height: ComponentSize.pianoPanelChordRow)
                    .padding(.bottom, Spacing.xxs)
            }
            FixedKeyboard(
                voicing: voicing,
                keysHeight: keysHeight,
                namesHeight: showsNoteNames ? ComponentSize.pianoPanelNamesRow : 0,
                names: FixedKeyboard.names(voicing, chords: chords)
            )
            PageDots(count: voicing?.inversions ?? 0, current: voicing?.inversion ?? 0)
                .frame(height: ComponentSize.pianoPanelDotsRow)
        }
        .padding(.horizontal, Spacing.xs)
        .padding(.vertical, Spacing.xxs)
        .block(Palette.surface, shadow: shadow, shadowColor: Palette.lime, border: Palette.borderOnInk)
        .contentShape(.rect)
        .simultaneousGesture(swipe)
        .sensoryFeedback(.impact(weight: .light), trigger: haptic)
        .accessibilityElement(children: .ignore)
        .accessibilityLabel("Piano")
        .accessibilityValue(Self.summary(chords, label: label, numeral: numeral, voicing: voicing))
        .accessibilityHint("Swipe up or down to change the inversion.")
        .accessibilityAddTraits(.updatesFrequently)
        .accessibilityAdjustableAction { direction in
            switch direction {
            case .increment:
                shift(1)
            case .decrement:
                shift(-1)
            @unknown default:
                break
            }
        }
    }

    private func chordRow(name: String, numeral: String?) -> some View {
        HStack(alignment: .firstTextBaseline, spacing: Spacing.xs) {
            Text(name.isEmpty ? "—" : name)
                .font(.display(22, relativeTo: .title2))
                .foregroundStyle(Palette.chord)
                .lineLimit(1)
                .minimumScaleFactor(0.5)
            if let numeral {
                Text(numeral)
                    .font(.label)
                    .foregroundStyle(Palette.textSecondary)
                    .lineLimit(1)
            }
        }
        .frame(maxWidth: .infinity)
    }

    private var swipe: some Gesture {
        DragGesture(minimumDistance: Spacing.md)
            .onEnded { value in
                let dx = value.translation.width
                guard abs(dx) >= Self.swipeThreshold, abs(dx) > abs(value.translation.height) else { return }
                // Swipe left = next inversion (like turning a page).
                shift(dx < 0 ? 1 : -1)
            }
    }

    private func shift(_ delta: Int) {
        if model.shiftInversion(by: delta) {
            haptic += 1
        }
    }

    /// "E major, three. Root position. Bass E; right hand E, G sharp, B."
    private static func summary(_ chords: ChordDisplay, label: String?, numeral: String?, voicing: PianoVoicing?) -> String {
        guard let label, let voicing else { return "No chord" }
        var head = chords.spokenName(label)
        if let numeral {
            head += ", \(spokenRomanNumeral(numeral))"
        }
        let rightHand = voicing.rightHand.map { ChordDisplay.spoken(chords.noteName($0)) }.joined(separator: ", ")
        let bass = ChordDisplay.spoken(chords.noteName(voicing.bass))
        return "\(head). \(PlayerFormat.spokenInversion(voicing.inversion)), \(voicing.inversion + 1) of \(voicing.inversions). "
            + "Bass \(bass); right hand \(rightHand)."
    }
}

/// The fixed C3–B5 keyboard (DESIGN.md §3.7): 21 white keys spanning the width. Its layout never
/// scrolls or re-centers; only the highlights change. Decorative for VoiceOver (the panel speaks).
/// Its inputs are plain values (no closures), so SwiftUI skips it while they're unchanged.
struct FixedKeyboard: View {
    let voicing: PianoVoicing?
    let keysHeight: CGFloat
    var namesHeight: CGFloat = ComponentSize.pianoPanelNamesRow
    /// Name per highlighted MIDI note.
    let names: [Int: String]

    var body: some View {
        let renderer = FixedKeyboardRenderer(voicing: voicing, keysHeight: keysHeight, names: names)
        Canvas { context, size in
            renderer.draw(in: &context, size: size)
        }
        .frame(height: keysHeight + namesHeight)
        .accessibilityHidden(true)
    }

    static func highlighted(_ voicing: PianoVoicing?) -> [Int] {
        guard let voicing else { return [] }
        return [voicing.bass] + voicing.rightHand
    }

    static func names(_ voicing: PianoVoicing?, chords: ChordDisplay) -> [Int: String] {
        Dictionary(highlighted(voicing).map { ($0, chords.noteName($0)) }, uniquingKeysWith: { first, _ in first })
    }
}

/// Draws the fixed keyboard for one chord.
@MainActor
private struct FixedKeyboardRenderer {
    let voicing: PianoVoicing?
    let keysHeight: CGFloat
    /// Name per highlighted MIDI note.
    let names: [Int: String]

    private static let firstWhite = PianoLayout.whiteKeyIndex(PianoVoicing.keyboardRange.lowerBound)
    private static let whiteCount = PianoLayout.whiteKeyIndex(PianoVoicing.keyboardRange.upperBound) - firstWhite + 1
    private static let blackKeyHeight: CGFloat = 0.62
    private static let blackKeyWidth: CGFloat = 0.6
    /// Minimum space between two note names under the keys.
    private static let nameGap: CGFloat = 2

    private var fingers: [Int: Int] {
        guard let voicing else { return [:] }
        var fingers = [voicing.bass: voicing.bassFinger]
        for (note, finger) in zip(voicing.rightHand, voicing.rightHandFingers) {
            fingers[note] = finger
        }
        return fingers
    }

    func draw(in context: inout GraphicsContext, size: CGSize) {
        let keys = CGRect(x: 0, y: 0, width: size.width, height: min(keysHeight, size.height))
        let whiteWidth = keys.width / CGFloat(Self.whiteCount)
        let fingers = fingers
        context.fill(Path(keys), with: .color(Palette.ink))

        for offset in 0..<Self.whiteCount {
            let midi = PianoLayout.midiOfWhiteKey(Self.firstWhite + offset)
            let rect = CGRect(x: CGFloat(offset) * whiteWidth, y: 0, width: whiteWidth, height: keys.height)
            context.fill(Path(rect), with: .color(color(of: midi, black: false)))
            context.stroke(Path(rect), with: .color(Palette.ink), lineWidth: BorderWidth.key)
            if let finger = fingers[midi] {
                drawFinger(finger, font: .labelSmall, in: rect, context: &context)
            }
        }

        let blackWidth = whiteWidth * Self.blackKeyWidth
        let blackHeight = keys.height * Self.blackKeyHeight
        for offset in 0..<Self.whiteCount {
            let midi = PianoLayout.midiOfWhiteKey(Self.firstWhite + offset) + 1
            guard PianoLayout.isBlackKey(midi), PianoVoicing.keyboardRange.contains(midi) else { continue }
            let rect = CGRect(x: CGFloat(offset + 1) * whiteWidth - blackWidth / 2, y: 0, width: blackWidth, height: blackHeight)
            context.fill(Path(rect), with: .color(color(of: midi, black: true)))
            if let finger = fingers[midi] {
                drawFinger(finger, font: .overline, in: rect, context: &context)
            }
        }

        context.stroke(Path(keys.insetBy(dx: BorderWidth.key / 2, dy: BorderWidth.key / 2)), with: .color(Palette.border), lineWidth: BorderWidth.key)

        if size.height > keys.height {
            drawNames(whiteWidth: whiteWidth, top: keys.height, height: size.height - keys.height, context: &context)
        }
    }

    private func color(of midi: Int, black: Bool) -> Color {
        guard let voicing else { return black ? Palette.pianoBlackKey : Palette.pianoWhiteKey }
        if midi == voicing.bass { return Palette.pianoBass }
        if voicing.rightHand.contains(midi) { return black ? Palette.pianoActiveBlackKey : Palette.pianoActiveWhiteKey }
        return black ? Palette.pianoBlackKey : Palette.pianoWhiteKey
    }

    /// The finger number near the bottom of the key, in ink.
    private func drawFinger(_ finger: Int, font: Font, in rect: CGRect, context: inout GraphicsContext) {
        let text = context.resolve(Text("\(finger)").font(font).foregroundStyle(Palette.onColor))
        context.draw(text, at: CGPoint(x: rect.midX, y: rect.maxY - Spacing.xxs), anchor: .bottom)
    }

    /// Each highlighted note's name under its key (black keys sit on the white-key boundary), in
    /// the key's color; a name that would overlap the previous one is nudged right.
    private func drawNames(whiteWidth: CGFloat, top: CGFloat, height: CGFloat, context: inout GraphicsContext) {
        guard let voicing else { return }
        var previousRight = -CGFloat.greatestFiniteMagnitude
        for midi in FixedKeyboard.highlighted(voicing).sorted() {
            guard let name = names[midi] else { continue }
            let white = PianoLayout.whiteKeyIndex(midi) - Self.firstWhite
            let center = PianoLayout.isBlackKey(midi)
                ? CGFloat(white + 1) * whiteWidth
                : (CGFloat(white) + 0.5) * whiteWidth
            let text = context.resolve(
                Text(name)
                    .font(.overline)
                    .foregroundStyle(midi == voicing.bass ? Palette.pianoBass : Palette.chord)
            )
            let width = text.measure(in: CGSize(width: CGFloat.greatestFiniteMagnitude, height: height)).width
            let x = max(center, previousRight + Self.nameGap + width / 2)
            context.draw(text, at: CGPoint(x: x, y: top + height / 2), anchor: .center)
            previousRight = x + width / 2
        }
    }
}

/// One dot per inversion of the current chord; the current one is lime.
struct PageDots: View {
    let count: Int
    let current: Int

    var body: some View {
        HStack(spacing: Spacing.xs) {
            ForEach(0..<max(count, 0), id: \.self) { index in
                Circle()
                    .fill(index == current ? Palette.lime : Palette.textMuted)
                    .frame(width: ComponentSize.pageDot, height: ComponentSize.pageDot)
            }
        }
        .frame(maxWidth: .infinity)
        .accessibilityHidden(true)
    }
}

/// The Voicing chip: shows the inversion and opens a picker (DESIGN.md §3.7 "Inversions").
struct VoicingChip: View {
    @Bindable var model: PlayerModel
    var inline = false

    var body: some View {
        // The current chord's inversion (the setting clamped to it).
        let shown = model.currentVoicing?.inversion ?? model.inversion
        Menu {
            Picker("Voicing", selection: $model.inversionChoice) {
                ForEach(Array(0...PlayerModel.maxInversion), id: \.self) { inversion in
                    Text(PlayerFormat.inversion(inversion)).tag(inversion)
                }
            }
        } label: {
            ControlChipLabel(title: "Voicing", value: PlayerFormat.inversion(shown), inline: inline)
        }
        .buttonStyle(.plain)
        .accessibilityLabel("Voicing")
        .accessibilityValue(PlayerFormat.spokenInversion(shown))
    }
}
