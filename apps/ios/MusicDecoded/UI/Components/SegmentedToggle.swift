import SwiftUI

/// A compact segmented control in the app's style (visualizer: Piano | Video): a `surface` box
/// with a white-20% border; the selected segment is a lime block with ink text.
struct SegmentedToggle<Option: Hashable>: View {
    let options: [Option]
    @Binding var selection: Option
    /// Disabled options are dimmed and can't be selected (visualizer: Video unless youtube).
    let isEnabled: (Option) -> Bool
    let label: (Option) -> String

    init(
        options: [Option],
        selection: Binding<Option>,
        isEnabled: @escaping (Option) -> Bool = { _ in true },
        label: @escaping (Option) -> String
    ) {
        self.options = options
        _selection = selection
        self.isEnabled = isEnabled
        self.label = label
    }

    var body: some View {
        HStack(spacing: Spacing.xxs) {
            ForEach(options, id: \.self) { option in
                let isSelected = option == selection
                let enabled = isEnabled(option)
                Button {
                    selection = option
                } label: {
                    Text(label(option))
                        .font(.label)
                        .foregroundStyle(enabled ? (isSelected ? Palette.onPrimary : Palette.textSecondary) : Palette.textMuted)
                        .padding(.horizontal, Spacing.sm)
                        .frame(maxWidth: .infinity, minHeight: ComponentSize.minTouchTarget)
                        .background(isSelected ? Palette.primary : Palette.surface, in: .rect)
                        .contentShape(.rect)
                }
                .buttonStyle(.plain)
                .disabled(enabled == false)
                .accessibilityAddTraits(isSelected ? .isSelected : [])
            }
        }
        .padding(Spacing.xxs)
        .inkCard()
    }
}

#Preview {
    SegmentedToggle(options: ["Piano", "Video"], selection: .constant("Piano")) { $0 }
        .padding(Spacing.md)
        .background(Palette.canvas)
        .preferredColorScheme(.dark)
}
