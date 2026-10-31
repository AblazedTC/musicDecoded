import SwiftData
import SwiftUI

/// Pushes the Processing screen for a running job (DESIGN.md §2: pushed, no tab bar).
struct ProcessingRoute: Hashable, Sendable {
    let trackId: String
}

/// DESIGN.md §3.5. Watches one track's row and job. × and "Continue in background" both just
/// leave: the job keeps running and the row keeps showing progress. Leaving never cancels it.
/// DESIGN.md §1 color map: violet, with the step list on an ink card with `shadowXl`.
struct ProcessingScreen: View {
    let trackId: String
    /// Replaces this screen with the Player.
    let onOpenTranscription: () -> Void

    @Environment(\.dismiss) private var dismiss
    @Environment(\.dynamicTypeSize) private var dynamicTypeSize
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
            VStack(spacing: Spacing.xl) {
                ProcessingArt(title: state.title)
                    .padding(.top, Spacing.xxl)

                VStack(spacing: Spacing.xs) {
                    Eyebrow(text: eyebrow, color: Palette.white)
                    DisplayHeadline(text: headline, font: .display(36, relativeTo: .largeTitle), color: Palette.white)
                        .multilineTextAlignment(.center)
                    let subtitle = [state.title.isEmpty ? nil : state.title, state.artist]
                        .compactMap { $0 }
                        .joined(separator: " · ")
                    if subtitle.isEmpty == false {
                        Text(subtitle)
                            .font(.detail)
                            .foregroundStyle(Palette.textSecondaryOnViolet)
                            .lineLimit(dynamicTypeSize.isAccessibilitySize ? 3 : 1)
                    }
                }

                // The "Try again" button below is this error state's retry.
                if state.phase == .failed {
                    ErrorState(message: state.error ?? Transcription.genericMessage)
                } else {
                    VStack(spacing: Spacing.md) {
                        OverallProgress(state: state)
                        VStack(spacing: Spacing.xs) {
                            ForEach(ProcessingStep.allCases, id: \.self) { step in
                                StepRow(step: step, done: state.isDone(step), active: state.isActive(step))
                            }
                        }
                    }
                    .padding(Spacing.md)
                    .block(Palette.ink, shadow: HardShadow.xl)
                }

                actions
            }
            .padding(.horizontal, Spacing.md)
            .padding(.bottom, Spacing.xl)
        }
        .background {
            HalftoneTexture()
                .ignoresSafeArea()
        }
        .background(Palette.violet)
        .navigationBarBackButtonHidden()
        .toolbar(.hidden, for: .tabBar)
        .toolbarBackground(Palette.violet, for: .navigationBar)
        .toolbar {
            ToolbarItem(placement: .topBarTrailing) {
                Button("Continue in background", systemImage: "xmark") {
                    dismiss()
                }
                .labelStyle(.iconOnly)
                .fontWeight(.bold)
                .foregroundStyle(Palette.white)
            }
        }
        .onChange(of: input, initial: true) { _, newInput in
            state = ProcessingState.reduce(state, row: newInput.row, job: newInput.job)
        }
    }

    private var eyebrow: String {
        switch state.phase {
        case .complete: "Done"
        case .failed: "Stopped"
        case .loading, .running: "Processing"
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
        VStack(spacing: Spacing.md) {
            switch state.phase {
            case .complete:
                PrimaryButton(title: "Open transcription", systemImage: "music.note", shadowColor: Palette.border, action: onOpenTranscription)
            case .failed:
                PrimaryButton(title: isRetrying ? "Starting…" : "Try again", systemImage: "arrow.clockwise", shadowColor: Palette.border, action: retry)
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

/// The art (the title's initial) as a tilted card with a black border and hard shadow, a lime
/// waveform sticker on its corner.
private struct ProcessingArt: View {
    let title: String

    private static let art: CGFloat = 132

    var body: some View {
        InitialArt(title: title, size: Self.art, font: .display(64, relativeTo: .largeTitle), border: Palette.border)
            .hardShadow(HardShadow.lg)
            .overlay(alignment: .bottomTrailing) {
                Image(systemName: "waveform")
                    .font(.footnote.weight(.bold))
                    .foregroundStyle(Palette.onColor)
                    .frame(width: ComponentSize.iconButton, height: ComponentSize.iconButton)
                    .block(Palette.lime, shadow: HardShadow.sm)
                    .tilt(Tilt.stickerMax)
                    .offset(x: Spacing.md, y: Spacing.md)
            }
            .tilt(Tilt.hero)
            .padding(Spacing.md)
            .accessibilityHidden(true)
    }
}

private struct OverallProgress: View {
    let state: ProcessingState

    var body: some View {
        VStack(spacing: Spacing.xs) {
            HStack(alignment: .firstTextBaseline) {
                Text(state.step.title)
                    .overlineStyle(Palette.lime)
                Spacer(minLength: Spacing.xs)
                Text("\(state.progress)%")
                    .font(.display(28, relativeTo: .title2))
                    .monospacedDigit()
                    .foregroundStyle(Palette.primary)
            }
            WrappedProgressBar(fraction: Double(state.progress) / 100, height: Spacing.xs)
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
                Rectangle()
                    .fill(done || active ? Palette.primary : Palette.surfaceRaised)
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
        .block(
            done ? Palette.surfaceRaised : Palette.surface,
            border: active ? Palette.primary : Palette.borderOnInk
        )
        .accessibilityElement(children: .combine)
        .accessibilityValue(done ? "Done" : active ? "In progress" : "Waiting")
    }
}
