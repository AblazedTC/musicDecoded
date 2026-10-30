import SwiftUI

/// A Canvas piano that highlights `notes` (MIDI) in the DESIGN.md §1 piano colors, with the slash
/// `bassNote` in pink. It shows `octaves` octaves and moves to keep the active keys in
/// view (holding its place while there's no chord). `noteName`, when set, labels the white keys
/// and the active keys.
struct PianoKeyboard: View {
    let notes: [Int]
    var bassNote: Int?
    var octaves = 2
    var height = ComponentSize.pianoCompactHeight
    var noteName: ((Int) -> String)?
    /// Read by VoiceOver, e.g. "D major: D, F sharp, A" (`ChordDisplay.accessibilitySummary`).
    var notesDescription = ""

    @State private var lastStart = PianoLayout.defaultStart

    var body: some View {
        let start = PianoLayout.keyboardStart(notes, octaves: octaves) ?? lastStart
        let renderer = KeyboardRenderer(
            firstWhite: PianoLayout.whiteKeyIndex(start),
            octaves: octaves,
            active: Set(notes),
            bassNote: bassNote,
            noteName: noteName
        )
        Canvas { context, size in
            renderer.draw(in: &context, size: size)
        }
        .frame(height: height)
        .onChange(of: start) { _, newStart in
            lastStart = newStart
        }
        .accessibilityElement()
        .accessibilityLabel("Piano")
        .accessibilityValue(notes.isEmpty ? "No chord" : (notesDescription.isEmpty ? "\(notes.count) keys highlighted" : notesDescription))
    }
}

/// Draws the keys for one frame.
@MainActor
private struct KeyboardRenderer {
    let firstWhite: Int
    let octaves: Int
    let active: Set<Int>
    let bassNote: Int?
    let noteName: ((Int) -> String)?

    private static let separator = BorderWidth.key
    private static let blackKeyHeight: CGFloat = 0.62
    private static let blackKeyWidth: CGFloat = 0.6

    func draw(in context: inout GraphicsContext, size: CGSize) {
        context.fill(Path(CGRect(origin: .zero, size: size)), with: .color(Palette.ink))
        let whiteCount = octaves * PianoLayout.whiteKeysPerOctave
        let whiteWidth = size.width / CGFloat(whiteCount)
        let indices = (firstWhite - 1)...(firstWhite + whiteCount + 1)

        for index in indices {
            let midi = PianoLayout.midiOfWhiteKey(index)
            guard PianoLayout.range.contains(midi) else { continue }
            let rect = CGRect(x: CGFloat(index - firstWhite) * whiteWidth, y: 0, width: whiteWidth, height: size.height)
            let key = Path(rect)
            context.fill(key, with: .color(color(of: midi, black: false)))
            context.stroke(key, with: .color(Palette.ink), lineWidth: Self.separator)
            if let noteName {
                let label = active.contains(midi) ? noteName(midi) : PianoLayout.whiteLabel(midi)
                drawLabel(label, color: Palette.pianoBlackKey, in: rect, context: &context)
            }
        }

        let blackWidth = whiteWidth * Self.blackKeyWidth
        let blackHeight = size.height * Self.blackKeyHeight
        for index in indices {
            let midi = PianoLayout.midiOfWhiteKey(index) + 1
            guard PianoLayout.isBlackKey(midi), PianoLayout.range.contains(midi) else { continue }
            let rect = CGRect(
                x: CGFloat(index + 1 - firstWhite) * whiteWidth - blackWidth / 2,
                y: 0,
                width: blackWidth,
                height: blackHeight
            )
            context.fill(Path(rect), with: .color(color(of: midi, black: true)))
            if let noteName, active.contains(midi) {
                drawLabel(noteName(midi), color: Palette.onColor, in: rect, context: &context)
            }
        }
    }

    private func color(of midi: Int, black: Bool) -> Color {
        if midi == bassNote { return Palette.pianoBass }
        if active.contains(midi) { return black ? Palette.pianoActiveBlackKey : Palette.pianoActiveWhiteKey }
        return black ? Palette.pianoBlackKey : Palette.pianoWhiteKey
    }

    /// A centered label at the bottom of the key; skipped when it doesn't fit.
    private func drawLabel(_ text: String, color: Color, in rect: CGRect, context: inout GraphicsContext) {
        let resolved = context.resolve(
            Text(text)
                .font(.overline)
                .foregroundStyle(color)
        )
        let size = resolved.measure(in: CGSize(width: CGFloat.greatestFiniteMagnitude, height: rect.height))
        guard size.width <= rect.width else { return }
        context.draw(resolved, at: CGPoint(x: rect.midX, y: rect.maxY - Spacing.xxs), anchor: .bottom)
    }
}

#Preview {
    VStack(spacing: Spacing.md) {
        // D/F♯: bass F♯2, chord D3 F♯3 A3.
        PianoKeyboard(notes: [42, 50, 54, 57], bassNote: 42)
        PianoKeyboard(
            notes: [50, 54, 57, 61],
            height: ComponentSize.pianoLargeHeight,
            noteName: { ["C", "C♯", "D", "E♭", "E", "F", "F♯", "G", "A♭", "A", "B♭", "B"][$0 % 12] }
        )
    }
    .padding(Spacing.md)
    .background(Palette.canvas)
    .preferredColorScheme(.dark)
}
