import SwiftData
import SwiftUI

/// Pushes the Processing screen for a running job (DESIGN.md §2: pushed, no tab bar).
struct ProcessingRoute: Hashable, Sendable {
    let trackId: String
}

/// DESIGN.md §3.5. Watches one track's row and job. × and "Continue in background" both just
/// leave: the job keeps running and the row keeps showing progress. Leaving never cancels it.
struct ProcessingScreen: View {
    let trackId: String
    /// Replaces this screen with the Player.
    let onOpenTranscription: () -> Void

    @Environment(\.dismiss) private var dismiss
    @Environment(TranscriptionService.self) private var transcription
    @Query private var items: [LibraryItem]
    @State private var state = ProcessingState()
    @State private var isRetrying = false

    init(trackId: String, onOpenTranscription: @escaping () -> Void) {
        self.trackId = trackId
        self.onOpenTranscription = onOpenTranscription
        _items = Query(LibraryItem.withTrackId(trackId))
    }

    private struct Input: Equatable {
        var row: ProcessingState.Row?
        var job: Job?
    }

    var body: some View {
        let input = Input(row: items.first.map { ProcessingState.Row($0) }, job: transcription.jobs[trackId])
        ScrollView {
            VStack(spacing: Spacing.lg) {
                OrbitArt(title: state.title)
                    .padding(.top, Spacing.xxxl)

                VStack(spacing: Spacing.xxs) {
                    Text(headline)
                        .font(.pageTitle)
                        .foregroundStyle(Palette.text)
                        .multilineTextAlignment(.center)
                        .accessibilityAddTraits(.isHeader)
                    let subtitle = [state.title.isEmpty ? nil : state.title, state.artist]
                        .compactMap { $0 }
                        .joined(separator: " · ")
                    if subtitle.isEmpty == false {
                        Text(subtitle)
                            .font(.detail)
                            .foregroundStyle(Palette.textSecondary)
                            .lineLimit(1)
                    }
                }

                if state.phase == .failed {
                    Text(state.error ?? Transcription.genericMessage)
                        .font(.detail)
                        .foregroundStyle(Palette.record)
                        .multilineTextAlignment(.center)
                } else {
                    OverallProgress(state: state)
                    VStack(spacing: Spacing.xs) {
                        ForEach(ProcessingStep.allCases, id: \.self) { step in
                            StepRow(step: step, done: state.isDone(step), active: state.isActive(step))
                        }
                    }
                }

                actions
            }
            .padding(.horizontal, Spacing.md)
            .padding(.bottom, Spacing.lg)
        }
        .background(Palette.canvas)
        .navigationBarBackButtonHidden()
        .toolbar(.hidden, for: .tabBar)
        .toolbarBackground(Palette.canvas, for: .navigationBar)
        .toolbar {
            ToolbarItem(placement: .topBarTrailing) {
                Button("Continue in background", systemImage: "xmark") {
                    dismiss()
                }
                .labelStyle(.iconOnly)
                .foregroundStyle(Palette.text)
            }
        }
        .onChange(of: input, initial: true) { _, newInput in
            state = ProcessingState.reduce(state, row: newInput.row, job: newInput.job)
        }
    }

    private var headline: String {
        switch state.phase {
        case .complete: "Your transcription is ready"
        case .failed: "We couldn't decode this song"
        case .loading, .running: "Decoding your music"
        }
    }

    @ViewBuilder
    private var actions: some View {
        VStack(spacing: Spacing.sm) {
            switch state.phase {
            case .complete:
                PrimaryButton(title: "Open transcription", systemImage: "music.note", action: onOpenTranscription)
            case .failed:
                PrimaryButton(title: isRetrying ? "Starting…" : "Try again", systemImage: "arrow.clockwise", action: retry)
                    .disabled(isRetrying)
                SecondaryButton(title: "Back", systemImage: "chevron.left") {
                    dismiss()
                }
            case .loading, .running:
                SecondaryButton(title: "Continue in background", systemImage: "arrow.down.right.and.arrow.up.left") {
                    dismiss()
                }
            }
        }
    }

    /// "Try again". A failure to even start shows up as the row's error.
    private func retry() {
        guard isRetrying == false else { return }
        isRetrying = true
        Task {
            try? await transcription.retry(trackId: trackId)
            isRetrying = false
        }
    }
}

/// Art (the title's initial) inside two orbit rings, with a wave badge.
private struct OrbitArt: View {
    let title: String

    private static let outer: CGFloat = 208
    private static let inner: CGFloat = 164
    private static let art: CGFloat = 120

    var body: some View {
        ZStack {
            Circle()
                .strokeBorder(Palette.line, lineWidth: 1)
                .frame(width: Self.outer, height: Self.outer)
            Circle()
                .strokeBorder(Palette.primarySoft, lineWidth: 1)
                .frame(width: Self.inner, height: Self.inner)
            InitialArt(title: title, size: Self.art, cornerRadius: Radius.card, font: .pageTitle)
            Image(systemName: "waveform")
                .font(.footnote.weight(.bold))
                .foregroundStyle(Palette.onPrimary)
                .frame(width: ComponentSize.iconButton, height: ComponentSize.iconButton)
                .background(Palette.primary, in: .circle)
                .frame(width: Self.outer, height: Self.outer, alignment: .bottomTrailing)
                .padding(Spacing.xl)
        }
        .frame(width: Self.outer, height: Self.outer)
        .accessibilityHidden(true)
    }
}

private struct OverallProgress: View {
    let state: ProcessingState

    var body: some View {
        VStack(spacing: Spacing.xs) {
            HStack {
                Text(state.step.title)
                    .font(.overline)
                    .foregroundStyle(Palette.textSecondary)
                Spacer(minLength: Spacing.xs)
                Text("\(state.progress)%")
                    .font(.overline.weight(.semibold))
                    .monospacedDigit()
                    .foregroundStyle(Palette.primary)
            }
            ProgressView(value: Double(state.progress), total: 100)
                .tint(Palette.primary)
        }
        .accessibilityElement(children: .ignore)
        .accessibilityLabel("Overall progress")
        .accessibilityValue("\(state.progress) percent, \(state.step.title)")
    }
}

private struct StepRow: View {
    let step: ProcessingStep
    let done: Bool
    let active: Bool

    var body: some View {
        HStack(spacing: Spacing.sm) {
            ZStack {
                Circle()
                    .fill(done || active ? Palette.primary : Palette.surfaceSoft)
                if done {
                    Image(systemName: "checkmark")
                        .font(.footnote.weight(.bold))
                        .foregroundStyle(Palette.onPrimary)
                } else if active {
                    Image(systemName: "waveform")
                        .font(.footnote.weight(.bold))
                        .foregroundStyle(Palette.onPrimary)
                }
            }
            .frame(width: ComponentSize.smallIconButton, height: ComponentSize.smallIconButton)

            Text(step.title)
                .font(.rowTitle)
                .foregroundStyle(done || active ? Palette.text : Palette.textMuted)
                .frame(maxWidth: .infinity, alignment: .leading)
        }
        .padding(Spacing.sm)
        .background(done ? Palette.primarySoft : Palette.surface, in: .rect(cornerRadius: Radius.row))
        .overlay {
            RoundedRectangle(cornerRadius: Radius.row)
                .strokeBorder(active ? Palette.primary : Palette.line, lineWidth: 1)
        }
        .accessibilityElement(children: .combine)
        .accessibilityValue(done ? "Done" : active ? "In progress" : "Waiting")
    }
}
