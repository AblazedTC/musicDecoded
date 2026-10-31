import SwiftUI

/// A small boxed note that explains a policy or what happens to the user's data: a `surface` card
/// with a blue info icon.
struct PolicyNote: View {
    let text: String
    var systemImage = "info.circle.fill"

    var body: some View {
        HStack(alignment: .firstTextBaseline, spacing: Spacing.xs) {
            Image(systemName: systemImage)
                .foregroundStyle(Palette.info)
                .accessibilityHidden(true)
            Text(text)
                .foregroundStyle(Palette.textSecondary)
                .frame(maxWidth: .infinity, alignment: .leading)
        }
        .font(.detail)
        .padding(Spacing.sm)
        .inkCard()
        .accessibilityElement(children: .combine)
    }
}

#Preview {
    PolicyNote(text: "Offline audio is saved only when permitted. The chord timeline can remain available separately.")
        .padding(Spacing.md)
        .background(Palette.canvas)
        .preferredColorScheme(.dark)
}
