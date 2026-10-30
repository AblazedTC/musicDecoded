import SwiftUI

/// DESIGN.md §1 borders: 3pt on blocks, cards and art; 2pt separators inside lists.
/// Corners are square everywhere; only circular play/record buttons and avatars are round.
enum BorderWidth {
    static let thick: CGFloat = 3
    static let separator: CGFloat = 2
    /// Piano key separators.
    static let key: CGFloat = 1.5
}

/// DESIGN.md §1 hard offset shadows (no blur, x = y).
enum HardShadow {
    /// Chips, small buttons.
    static let sm: CGFloat = 3
    /// Stickers, buttons, rows on color.
    static let md: CGFloat = 5
    /// Cards, timeline, visualizer.
    static let lg: CGFloat = 8
    /// Hero cards: player now-playing card, paywall.
    static let xl: CGFloat = 10
    /// The pressed state moves a block by this and shrinks its shadow by the same amount.
    static let press: CGFloat = 2
}

/// DESIGN.md §1 tilt in degrees. Decoration only (never the timeline, piano, inputs or anything
/// draggable), and none when Reduce Motion is on.
enum Tilt {
    static let hero: Double = -1.5
    static let listCard: Double = 1
    static let sticker: Double = -3
    static let stickerMax: Double = -6
}

extension View {
    /// A solid shadow offset `offset` down-right, no blur, drawn behind the view (and outside its
    /// bounds, so leave room).
    func hardShadow(_ offset: CGFloat, color: Color = Palette.border) -> some View {
        background {
            Rectangle()
                .fill(color)
                .offset(x: offset, y: offset)
                .opacity(offset > 0 ? 1 : 0)
        }
    }

    /// A circular hard shadow, for the round play and record buttons.
    func hardShadowCircle(_ offset: CGFloat, color: Color = Palette.border) -> some View {
        background {
            Circle()
                .fill(color)
                .offset(x: offset, y: offset)
                .opacity(offset > 0 ? 1 : 0)
        }
    }

    /// A 3pt square border drawn inside the view's bounds.
    func squareBorder(_ color: Color = Palette.border, width: CGFloat = BorderWidth.thick) -> some View {
        overlay {
            Rectangle().strokeBorder(color, lineWidth: width)
        }
    }

    /// A Wrapped color block: fill, 3pt border (`nil` = none) and a hard shadow.
    func block(
        _ fill: Color,
        shadow: CGFloat = 0,
        shadowColor: Color = Palette.border,
        border: Color? = Palette.border
    ) -> some View {
        background(fill, in: .rect)
            .overlay {
                if let border {
                    Rectangle().strokeBorder(border, lineWidth: BorderWidth.thick)
                }
            }
            .hardShadow(shadow, color: shadowColor)
    }

    /// A card on ink: `surface` fill, a 3pt white-20% border and an optional accent shadow.
    func inkCard(shadow: CGFloat = 0, shadowColor: Color = Palette.violet, border: Color = Palette.borderOnInk) -> some View {
        block(Palette.surface, shadow: shadow, shadowColor: shadowColor, border: border)
    }

    /// A 3pt rule along the top edge, over the content (tab bar, transport, sheets).
    func topRule(_ color: Color = Palette.border) -> some View {
        overlay(alignment: .top) {
            Rectangle()
                .fill(color)
                .frame(height: BorderWidth.thick)
                .accessibilityHidden(true)
        }
    }

    /// Rotates by `degrees` for decoration, unless Reduce Motion is on.
    func tilt(_ degrees: Double) -> some View {
        modifier(TiltModifier(degrees: degrees))
    }
}

private struct TiltModifier: ViewModifier {
    let degrees: Double

    @Environment(\.accessibilityReduceMotion) private var reduceMotion

    func body(content: Content) -> some View {
        content.rotationEffect(.degrees(reduceMotion ? 0 : degrees))
    }
}

/// The pushed-sticker pressed state (DESIGN.md §1) for any tappable color block: fill, 3pt border
/// and hard shadow; while pressed the block moves (+2, +2) and the shadow shrinks by 2, so the
/// shadow stays put. No tint.
struct BlockButtonStyle: ButtonStyle {
    var fill: Color
    var shadow: CGFloat = HardShadow.md
    var shadowColor: Color = Palette.border
    var border: Color? = Palette.border
    var isCircle = false

    func makeBody(configuration: Configuration) -> some View {
        let push: CGFloat = configuration.isPressed && shadow > 0 ? HardShadow.press : 0
        return Group {
            if isCircle {
                configuration.label
                    .background(fill, in: .circle)
                    .overlay {
                        if let border {
                            Circle().strokeBorder(border, lineWidth: BorderWidth.thick)
                        }
                    }
                    .hardShadowCircle(shadow - push, color: shadowColor)
            } else {
                configuration.label
                    .block(fill, shadow: shadow - push, shadowColor: shadowColor, border: border)
            }
        }
        .offset(x: push, y: push)
        .animation(.easeOut(duration: 0.08), value: configuration.isPressed)
    }
}

/// The pressed state for buttons that already draw their own block: only the (+2, +2) push.
struct PushButtonStyle: ButtonStyle {
    func makeBody(configuration: Configuration) -> some View {
        configuration.label
            .offset(x: configuration.isPressed ? HardShadow.press : 0, y: configuration.isPressed ? HardShadow.press : 0)
            .animation(.easeOut(duration: 0.08), value: configuration.isPressed)
    }
}
