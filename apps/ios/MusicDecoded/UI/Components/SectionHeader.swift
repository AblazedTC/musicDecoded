import SwiftUI

/// A section title with an optional trailing action ("See all").
struct SectionHeader: View {
    let title: String
    var actionTitle: String?
    var action: (() -> Void)?

    var body: some View {
        HStack(alignment: .firstTextBaseline, spacing: Spacing.xs) {
            Text(title)
                .font(.sectionHeader)
                .foregroundStyle(Palette.text)
                .accessibilityAddTraits(.isHeader)

            Spacer(minLength: Spacing.xs)

            if let actionTitle, let action {
                Button(action: action) {
                    Text(actionTitle)
                        .font(.detail.weight(.semibold))
                        .foregroundStyle(Palette.primary)
                        .frame(minHeight: ComponentSize.minTouchTarget)
                        .contentShape(.rect)
                }
                .buttonStyle(.plain)
            }
        }
    }
}

#Preview {
    VStack(spacing: Spacing.md) {
        SectionHeader(title: "Recently Decoded", actionTitle: "See all") {}
        SectionHeader(title: "Transcribe Music")
    }
    .padding(Spacing.md)
    .background(Palette.canvas)
    .preferredColorScheme(.dark)
}
