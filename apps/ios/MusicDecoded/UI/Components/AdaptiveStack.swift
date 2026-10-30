import SwiftUI

/// A row of equal items (chips, cards) that becomes a column at accessibility text sizes, so
/// nothing clips or truncates (T-503i). A fixed rule rather than `ViewThatFits`, so repeated rows
/// switch together.
struct AdaptiveStack<Content: View>: View {
    var spacing: CGFloat = Spacing.xs
    @ViewBuilder let content: () -> Content

    @Environment(\.dynamicTypeSize) private var dynamicTypeSize

    var body: some View {
        let layout = dynamicTypeSize.isAccessibilitySize
            ? AnyLayout(VStackLayout(alignment: .leading, spacing: spacing))
            : AnyLayout(HStackLayout(alignment: .top, spacing: spacing))
        layout {
            content()
        }
    }
}
