import SwiftUI

/// A section title (uppercase display) with an optional eyebrow, a decorative big numeral behind
/// it ("01") and a trailing action ("See all").
struct SectionHeader: View {
    let title: String
    var eyebrow: String?
    var numeral: String?
    var color: Color = Palette.text
    var actionTitle: String?
    var action: (() -> Void)?

    var body: some View {
        HStack(alignment: .bottom, spacing: Spacing.xs) {
            VStack(alignment: .leading, spacing: Spacing.xxs) {
                if let eyebrow {
                    Eyebrow(text: eyebrow)
                }
                DisplayHeadline(text: title, font: .sectionHeader, color: color)
            }
            .frame(maxWidth: .infinity, alignment: .leading)

            if let actionTitle, let action {
                Button(action: action) {
                    Text(actionTitle)
                        .font(.sticker)
                        .textCase(.uppercase)
                        .tracking(Tracking.sticker)
                        .foregroundStyle(Palette.primary)
                        .frame(minHeight: ComponentSize.minTouchTarget)
                        .contentShape(.rect)
                }
                .buttonStyle(PushButtonStyle())
            }
        }
        .background(alignment: .trailing) {
            if let numeral {
                BigNumeral(text: numeral, size: 72, color: color)
            }
        }
    }
}

#Preview {
    VStack(spacing: Spacing.lg) {
        SectionHeader(title: "Recently Decoded", numeral: "01", actionTitle: "See all") {}
        SectionHeader(title: "Transcribe Music", numeral: "02")
    }
    .padding(Spacing.md)
    .background(Palette.canvas)
    .preferredColorScheme(.dark)
}
