import SwiftUI

/// A compact segmented control in the app's style (visualizer: Piano | Video).
struct SegmentedToggle<Option: Hashable>: View {
    let options: [Option]
    @Binding var selection: Option
    let label: (Option) -> String

    var body: some View {
        HStack(spacing: Spacing.xxs) {
            ForEach(options, id: \.self) { option in
                let isSelected = option == selection
                Button {
                    selection = option
                } label: {
                    Text(label(option))
                        .font(.detail.weight(.semibold))
                        .foregroundStyle(isSelected ? Palette.text : Palette.textSecondary)
                        .padding(.horizontal, Spacing.sm)
                        .frame(maxWidth: .infinity, minHeight: ComponentSize.minTouchTarget)
                        .background(isSelected ? Palette.surfaceSoft : Color.clear, in: .rect(cornerRadius: Radius.chip))
                        .contentShape(.rect)
                }
                .buttonStyle(.plain)
                .accessibilityAddTraits(isSelected ? .isSelected : [])
            }
        }
        .padding(Spacing.xxs)
        .background(Palette.surface, in: .rect(cornerRadius: Radius.control))
    }
}

#Preview {
    SegmentedToggle(options: ["Piano", "Video"], selection: .constant("Piano")) { $0 }
        .padding(Spacing.md)
        .background(Palette.canvas)
        .preferredColorScheme(.dark)
}
