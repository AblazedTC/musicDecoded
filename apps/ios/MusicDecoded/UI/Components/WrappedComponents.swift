import Foundation
import SwiftUI

// DESIGN.md §1 "Wrapped" signature elements: stickers, eyebrows, display headlines, big numerals,
// waveform bars and textures. The shadow, border, tilt and pressed-state modifiers live in
// DesignSystem/WrappedStyle.swift.

/// Sticker (portfolio `Sticker.tsx`): accent fill, 3pt black border, `shadowMd`, uppercase heavy
/// label, tilted -3°. For eyebrows, status badges and "PRO".
struct Sticker: View {
    let text: String
    var fill: Color = Palette.lime
    var foreground: Color = Palette.onColor
    var rotation: Double = Tilt.sticker

    var body: some View {
        Text(text)
            .font(.sticker)
            .textCase(.uppercase)
            .tracking(Tracking.sticker)
            .foregroundStyle(foreground)
            .lineLimit(1)
            .padding(.horizontal, Spacing.sm)
            .padding(.vertical, Spacing.xxs)
            .block(fill, shadow: HardShadow.md)
            .tilt(rotation)
    }
}

/// Eyebrow label: uppercase, 11pt bold, +0.25em. `lime` on ink (default), `white` on violet.
struct Eyebrow: View {
    let text: String
    var color: Color = Palette.lime

    var body: some View {
        Text(text)
            .overlineStyle(color)
    }
}

/// Display headline: Archivo Black, uppercase, tight tracking. Page title size by default.
struct DisplayHeadline: View {
    let text: String
    var font: Font = .pageTitle
    var color: Color = Palette.text

    var body: some View {
        Text(text)
            .font(font)
            .textCase(.uppercase)
            .tracking(Tracking.display)
            .foregroundStyle(color)
            .fixedSize(horizontal: false, vertical: true)
            .accessibilityAddTraits(.isHeader)
    }
}

/// An oversized decorative number at 8% opacity ("01", "#1") behind a section title or row.
/// Fixed size and hidden from accessibility.
struct BigNumeral: View {
    let text: String
    var size: CGFloat = 120
    var color: Color = Palette.white

    var body: some View {
        Text(text)
            .font(.custom(FontName.display, fixedSize: size))
            .foregroundStyle(color.opacity(0.08))
            .lineLimit(1)
            .fixedSize()
            .allowsHitTesting(false)
            .accessibilityHidden(true)
    }
}

/// Flat equalizer bars (portfolio `Waveform.tsx`) for idle/now-playing indicators. Decorative.
struct WaveformBars: View {
    var bars = 24
    var color: Color = Palette.lime
    var gap: CGFloat = Spacing.xxs

    var body: some View {
        let count = max(bars, 1)
        let fill = color
        let spacing = gap
        return Canvas { context, size in
            let width = max((size.width - spacing * CGFloat(count - 1)) / CGFloat(count), 1)
            for index in 0..<count {
                let factor = CGFloat(0.25 + 0.70 * abs(sin(Double(index) * 1.7 + 1)))
                let height = size.height * factor
                let rect = CGRect(
                    x: CGFloat(index) * (width + spacing),
                    y: size.height - height,
                    width: width,
                    height: height
                )
                context.fill(Path(rect), with: .color(fill))
            }
        }
        .allowsHitTesting(false)
        .accessibilityHidden(true)
    }
}

/// Square-ended progress bar: lime on `surfaceRaised`, 4pt tall by default. Decorative (the
/// percentage is read from the text next to it).
struct WrappedProgressBar: View {
    /// 0…1.
    let fraction: Double
    var color: Color = Palette.primary
    var track: Color = Palette.surfaceRaised
    var height: CGFloat = ComponentSize.seekTrackHeight

    var body: some View {
        let clamped = CGFloat(min(max(fraction, 0), 1))
        return Rectangle()
            .fill(track)
            .frame(height: height)
            .overlay(alignment: .leading) {
                GeometryReader { proxy in
                    Rectangle()
                        .fill(color)
                        .frame(width: proxy.size.width * clamped)
                }
            }
            .accessibilityHidden(true)
    }
}

/// Halftone dot grid texture: 2pt dots on a 12pt pitch at 20% opacity. Put it in a `background`.
struct HalftoneTexture: View {
    var color: Color = Palette.white
    var opacity: Double = 0.2

    var body: some View {
        let fill = color.opacity(opacity)
        return Canvas { context, size in
            let pitch: CGFloat = 12
            let radius: CGFloat = 1
            var dots = Path()
            var y = pitch / 2
            while y < size.height {
                var x = pitch / 2
                while x < size.width {
                    dots.addEllipse(in: CGRect(x: x - radius, y: y - radius, width: radius * 2, height: radius * 2))
                    x += pitch
                }
                y += pitch
            }
            context.fill(dots, with: .color(fill))
        }
        .allowsHitTesting(false)
        .accessibilityHidden(true)
    }
}

/// Diagonal stripe texture (violet at 50% on lime blocks). Put it in a `background`.
struct StripeTexture: View {
    var color: Color = Palette.violet
    var opacity: Double = 0.5

    var body: some View {
        let stroke = color.opacity(opacity)
        return Canvas { context, size in
            let pitch: CGFloat = 28
            var stripes = Path()
            var x = -size.height
            while x < size.width {
                stripes.move(to: CGPoint(x: x, y: size.height))
                stripes.addLine(to: CGPoint(x: x + size.height, y: 0))
                x += pitch
            }
            context.stroke(stripes, with: .color(stroke), lineWidth: 12)
        }
        .clipped()
        .allowsHitTesting(false)
        .accessibilityHidden(true)
    }
}

#Preview {
    VStack(alignment: .leading, spacing: Spacing.lg) {
        HStack(spacing: Spacing.md) {
            Sticker(text: "Your library")
            Sticker(text: "Pro", fill: Palette.yellow)
            Sticker(text: "Now playing", fill: Palette.pink, rotation: Tilt.stickerMax)
        }
        Eyebrow(text: "Now playing")
        DisplayHeadline(text: "Library")
        WaveformBars()
            .frame(height: 32)
        Color.clear
            .frame(width: 200, height: 80)
            .background { StripeTexture() }
            .block(Palette.lime, shadow: HardShadow.lg)
    }
    .padding(Spacing.md)
    .frame(maxWidth: .infinity, maxHeight: .infinity)
    .background { HalftoneTexture() }
    .background(Palette.canvas)
    .preferredColorScheme(.dark)
}
