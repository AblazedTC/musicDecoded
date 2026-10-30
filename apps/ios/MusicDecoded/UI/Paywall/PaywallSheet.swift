import SwiftUI

/// The Pro sheet (DESIGN.md §3.11), presented by `RootView` over the whole app. Ports PaywallSheet.kt.
struct PaywallSheet: View {
    let model: PaywallModel

    @Environment(\.dismiss) private var dismiss

    var body: some View {
        PaywallContent(
            phase: model.phase,
            plans: model.plans,
            selectedId: model.selectedId,
            isGuest: model.isGuest,
            freeDailyJobs: model.freeDailyJobs,
            error: model.error,
            onSelect: { model.select($0) },
            onSubscribe: { Task { await model.subscribe() } },
            onRestore: { Task { await model.restore() } },
            onRetry: { Task { await model.retry() } },
            onDone: { dismiss() }
        )
        .task {
            await model.loadIfNeeded()
        }
        .interactiveDismissDisabled(model.isDismissible == false)
        .presentationDetents([.large])
        .presentationDragIndicator(.visible)
        .presentationCornerRadius(0)
        .presentationBackground(Palette.violet)
    }
}

/// The sheet's content for every state; takes plain values so it can be previewed and tested.
/// DESIGN.md §1 color map: violet with a halftone texture, white plan cards with black borders and
/// `shadowLg`, a yellow "PRO" sticker and the lime CTA.
struct PaywallContent: View {
    static let finePrint = "Subscriptions renew automatically until cancelled. Cancel anytime in your App Store "
        + "account settings at least 24 hours before the end of the period. * Fair-use limit applies."

    let phase: PaywallModel.Phase
    let plans: [PlanCard]
    let selectedId: String?
    let isGuest: Bool
    let freeDailyJobs: Int?
    let error: String?
    let onSelect: (String) -> Void
    let onSubscribe: () -> Void
    let onRestore: () -> Void
    let onRetry: () -> Void
    let onDone: () -> Void

    /// "Create an account to subscribe" for guests (DESIGN.md §3.11).
    static func subscribeTitle(isGuest: Bool) -> String {
        isGuest ? "Create an account to subscribe" : "Subscribe"
    }

    var body: some View {
        ScrollView {
            VStack(spacing: Spacing.lg) {
                if phase == .success {
                    PaywallSuccess(onDone: onDone)
                } else {
                    form
                }
            }
            .padding(.horizontal, Spacing.md)
            .padding(.top, Spacing.xl)
            .padding(.bottom, Spacing.lg)
        }
        .background {
            HalftoneTexture()
                .ignoresSafeArea()
        }
        .background(Palette.violet)
    }

    @ViewBuilder
    private var form: some View {
        PaywallHeader()
        PaywallComparison(freeDailyJobs: freeDailyJobs)
        switch phase {
        case .unavailable:
            PolicyNote(text: BillingError.unavailable.message)
        case .loading:
            VStack(spacing: Spacing.md) {
                PlanSkeleton()
                PlanSkeleton()
            }
            .accessibilityElement(children: .ignore)
            .accessibilityLabel("Loading plans")
        default:
            VStack(spacing: Spacing.md) {
                ForEach(plans, id: \.option.id) { card in
                    PlanCardRow(card: card, isSelected: card.option.id == selectedId) {
                        onSelect(card.option.id)
                    }
                    .disabled(phase != .ready)
                }
            }
        }
        if let error {
            PaywallErrorNote(message: error, onRetry: phase == .ready ? onRetry : nil)
        }
        if phase != .unavailable {
            let ready = phase == .ready && plans.isEmpty == false && selectedId != nil
            ZStack {
                PrimaryButton(
                    title: phase == .purchasing ? "" : Self.subscribeTitle(isGuest: isGuest),
                    shadowColor: Palette.border,
                    action: onSubscribe
                )
                if phase == .purchasing {
                    ProgressView()
                        .tint(Palette.onPrimary)
                        .accessibilityLabel("Purchasing")
                }
            }
            .disabled(ready == false)
            Button(action: onRestore) {
                Text("Restore purchases")
                    .font(.label)
                    .foregroundStyle(Palette.lime)
                    .frame(maxWidth: .infinity, minHeight: ComponentSize.minTouchTarget)
                    .contentShape(.rect)
            }
            .buttonStyle(PushButtonStyle())
            .disabled(phase != .ready)
        }
        Text(Self.finePrint)
            .font(.detail)
            .foregroundStyle(Palette.textSecondaryOnViolet)
            .multilineTextAlignment(.center)
            .fixedSize(horizontal: false, vertical: true)
    }
}

private struct PaywallHeader: View {
    var body: some View {
        VStack(spacing: Spacing.sm) {
            Sticker(text: "Pro", fill: Palette.yellow, rotation: Tilt.stickerMax)
                .accessibilityHidden(true)
            DisplayHeadline(text: "MusicDecoded Pro", font: .display(36, relativeTo: .largeTitle), color: Palette.white)
                .multilineTextAlignment(.center)
            Text("Decode more, practice deeper.")
                .font(.rowTitle)
                .foregroundStyle(Palette.lime)
                .multilineTextAlignment(.center)
        }
    }
}

/// Free vs Pro (PLAN.md §9).
private struct PaywallComparison: View {
    let freeDailyJobs: Int?

    var body: some View {
        VStack(spacing: Spacing.sm) {
            ComparisonRow(label: "", free: "Free", pro: "Pro", isHeader: true)
            ComparisonRow(
                label: "Daily transcriptions",
                free: freeDailyJobs.map { "\($0) / day" } ?? "Limited",
                pro: "Unlimited*"
            )
            ComparisonRow(label: "Analysis models", free: "Standard", pro: "Premium")
            ComparisonRow(label: "Sync your own audio files", free: "—", pro: "Included")
        }
        .padding(Spacing.md)
        .block(Palette.white, shadow: HardShadow.lg)
    }
}

/// A row of the comparison table, on the white card (ink text).
private struct ComparisonRow: View {
    let label: String
    let free: String
    let pro: String
    var isHeader = false

    var body: some View {
        HStack(spacing: Spacing.xs) {
            Text(label)
                .font(.detail)
                .foregroundStyle(Palette.onColor)
                .frame(maxWidth: .infinity, alignment: .leading)
                .layoutPriority(1)
            Group {
                Text(free)
                    .foregroundStyle(Palette.textSecondaryOnLime)
                Text(pro)
                    .foregroundStyle(isHeader ? Palette.violet : Palette.onColor)
            }
            .font(isHeader ? Font.overline : Font.labelSmall)
            .textCase(isHeader ? .uppercase : nil)
            .multilineTextAlignment(.center)
            .frame(maxWidth: .infinity)
        }
        .accessibilityElement(children: .ignore)
        .accessibilityLabel(isHeader ? "Free versus Pro" : "\(label): Free \(free), Pro \(pro)")
        .accessibilityAddTraits(isHeader ? .isHeader : [])
    }
}

private struct PlanCardRow: View {
    let card: PlanCard
    let isSelected: Bool
    let action: () -> Void

    var body: some View {
        Button(action: action) {
            HStack(spacing: Spacing.xs) {
                VStack(alignment: .leading, spacing: Spacing.xxs) {
                    Text(card.title)
                        .font(.rowTitle)
                    Text(card.price)
                        .font(.detail)
                }
                .foregroundStyle(Palette.onColor)
                .frame(maxWidth: .infinity, alignment: .leading)
                if let saving = card.saving {
                    Sticker(text: saving, fill: Palette.pink)
                }
                Image(systemName: isSelected ? "checkmark.square.fill" : "square")
                    .font(.title3.weight(.bold))
                    .foregroundStyle(Palette.onColor)
                    .accessibilityHidden(true)
            }
            .padding(.horizontal, Spacing.md)
            .padding(.vertical, Spacing.sm)
            .frame(minHeight: ComponentSize.playButton + Spacing.md)
            .contentShape(.rect)
        }
        .buttonStyle(BlockButtonStyle(fill: isSelected ? Palette.lime : Palette.white, shadow: HardShadow.lg))
        .accessibilityElement(children: .combine)
        .accessibilityAddTraits(isSelected ? .isSelected : [])
    }
}

private struct PlanSkeleton: View {
    var body: some View {
        Rectangle()
            .fill(Palette.white.opacity(0.2))
            .frame(maxWidth: .infinity)
            .frame(height: ComponentSize.playButton + Spacing.md)
            .squareBorder(Palette.border)
    }
}

/// PolicyNote-style inline error with Retry (DESIGN.md §3.11).
private struct PaywallErrorNote: View {
    let message: String
    let onRetry: (() -> Void)?

    var body: some View {
        VStack(alignment: .leading, spacing: Spacing.xs) {
            HStack(alignment: .firstTextBaseline, spacing: Spacing.xs) {
                Image(systemName: "exclamationmark.circle")
                    .foregroundStyle(Palette.record)
                    .accessibilityHidden(true)
                Text(message)
                    .foregroundStyle(Palette.textSecondary)
                    .frame(maxWidth: .infinity, alignment: .leading)
                    .fixedSize(horizontal: false, vertical: true)
            }
            .font(.detail)
            if let onRetry {
                SecondaryButton(title: "Retry", systemImage: "arrow.clockwise", action: onRetry)
            }
        }
        .padding(Spacing.sm)
        .block(Palette.surface, shadow: HardShadow.md, shadowColor: Palette.orange, border: Palette.orange)
        .accessibilityElement(children: .contain)
        .onAppear {
            AccessibilityNotification.Announcement(message).post()
        }
    }
}

private struct PaywallSuccess: View {
    let onDone: () -> Void

    var body: some View {
        VStack(spacing: Spacing.md) {
            Image(systemName: "checkmark")
                .font(.title.weight(.bold))
                .foregroundStyle(Palette.onColor)
                .frame(width: ComponentSize.songArt, height: ComponentSize.songArt)
                .block(Palette.yellow, shadow: HardShadow.md)
                .tilt(Tilt.sticker)
                .accessibilityHidden(true)
            DisplayHeadline(text: "You're Pro", font: .display(36, relativeTo: .largeTitle), color: Palette.white)
            Text("Thanks for supporting MusicDecoded.")
                .font(.bodyText)
                .foregroundStyle(Palette.textSecondaryOnViolet)
            PrimaryButton(title: "Done", shadowColor: Palette.border, action: onDone)
        }
        .padding(.vertical, Spacing.xl)
        .onAppear {
            AccessibilityNotification.Announcement("You're Pro").post()
        }
    }
}

#Preview {
    PaywallContent(
        phase: .ready,
        plans: PaywallModel.planCards([
            PlanOption(id: "monthly", period: .monthly, price: "$4.99", amount: Decimal(string: "4.99") ?? 0),
            PlanOption(id: "annual", period: .annual, price: "$39.99", amount: Decimal(string: "39.99") ?? 0),
        ]),
        selectedId: "annual",
        isGuest: false,
        freeDailyJobs: 50,
        error: nil,
        onSelect: { _ in },
        onSubscribe: {},
        onRestore: {},
        onRetry: {},
        onDone: {}
    )
    .preferredColorScheme(.dark)
}
