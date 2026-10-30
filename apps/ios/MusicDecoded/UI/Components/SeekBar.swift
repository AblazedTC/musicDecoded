import SwiftUI

/// The seek bar from DESIGN.md §3.7: a thin `surfaceSoft` track, a `primary` fill and a round
/// thumb. Tap or drag anywhere on its 44pt-tall area. VoiceOver, Voice Control and Switch Control
/// get a native `Slider` through `accessibilityRepresentation`, so it stays adjustable.
struct SeekBar: View {
    @Binding var value: Double
    let range: ClosedRange<Double>
    /// Read by VoiceOver.
    var label = "Position"
    /// e.g. "1:04 of 3:20".
    var valueDescription: String

    @State private var isDragging = false

    var body: some View {
        GeometryReader { proxy in
            let thumb = isDragging ? ComponentSize.seekThumbActive : ComponentSize.seekThumb
            let usable = max(proxy.size.width - thumb, 0)
            let offset = usable * Self.fraction(value, in: range)
            ZStack(alignment: .leading) {
                Capsule()
                    .fill(Palette.surfaceSoft)
                    .frame(height: ComponentSize.seekTrackHeight)
                Capsule()
                    .fill(Palette.primary)
                    .frame(width: offset + thumb / 2, height: ComponentSize.seekTrackHeight)
                Circle()
                    .fill(Palette.primary)
                    .frame(width: thumb, height: thumb)
                    .offset(x: offset)
            }
            .frame(width: proxy.size.width, height: proxy.size.height)
            .contentShape(.rect)
            .gesture(
                DragGesture(minimumDistance: 0)
                    .onChanged { gesture in
                        isDragging = true
                        let position = usable > 0 ? (gesture.location.x - thumb / 2) / usable : 0
                        value = Self.value(atFraction: Double(position), in: range)
                    }
                    .onEnded { _ in
                        isDragging = false
                    }
            )
            .animation(.easeOut(duration: 0.15), value: isDragging)
        }
        .frame(height: ComponentSize.minTouchTarget)
        .accessibilityRepresentation {
            Slider(value: $value, in: range) {
                Text(label)
            }
            .accessibilityValue(valueDescription)
        }
    }

    /// Where `value` sits in `range`, 0…1.
    static func fraction(_ value: Double, in range: ClosedRange<Double>) -> Double {
        let span = range.upperBound - range.lowerBound
        guard span > 0 else { return 0 }
        return min(max((value - range.lowerBound) / span, 0), 1)
    }

    /// The value at `fraction` (clamped to 0…1) of `range`.
    static func value(atFraction fraction: Double, in range: ClosedRange<Double>) -> Double {
        let clamped = min(max(fraction, 0), 1)
        return range.lowerBound + clamped * (range.upperBound - range.lowerBound)
    }
}

#Preview {
    SeekBar(value: .constant(64), range: 0...200, valueDescription: "1:04 of 3:20")
        .padding(Spacing.md)
        .background(Palette.transportBackground)
        .preferredColorScheme(.dark)
}
