import SwiftUI

/// Seek bar + speed · previous chord · play/pause · next chord · loop, with the loop pill floating
/// above it while a loop is on (DESIGN.md §3.7 item 7).
struct TransportPanel: View {
    let model: PlayerModel

    var body: some View {
        VStack(spacing: Spacing.xs) {
            if let loop = model.loop {
                LoopPill(loop: loop) { model.clearLoop() }
            }
            // Compact (~104): one-line seek bar, then the buttons (DESIGN.md §3.7 item 6).
            VStack(spacing: 0) {
                PlayerSeekBar(model: model, compact: true)
                HStack {
                    SpeedButton(model: model)
                    Spacer(minLength: Spacing.xxs)
                    IconButton(title: "Previous chord", systemImage: "backward.end.fill") { model.previousChord() }
                    Spacer(minLength: Spacing.xxs)
                    PlayButton(isPlaying: model.isPlaying) { model.togglePlay() }
                    Spacer(minLength: Spacing.xxs)
                    IconButton(title: "Next chord", systemImage: "forward.end.fill") { model.nextChord() }
                    Spacer(minLength: Spacing.xxs)
                    IconButton(
                        title: model.loop == nil ? "Loop current bar" : "Stop looping",
                        systemImage: "repeat",
                        isActive: model.loop != nil
                    ) {
                        model.toggleLoop()
                    }
                }
            }
            .padding(.horizontal, Spacing.md)
            .padding(.top, Spacing.xs)
            .padding(.bottom, Spacing.xxs)
            .background {
                // A square sheet on ink with a 3pt black top rule.
                Palette.surface
                    .topRule()
                    .ignoresSafeArea(edges: .bottom)
            }
        }
    }
}

/// Seek slider with current / total times.
struct PlayerSeekBar: View {
    @Bindable var model: PlayerModel
    /// Current time · bar · total time on one line (the player's compact transport).
    var compact = false

    var body: some View {
        let duration = model.duration
        if compact {
            HStack(spacing: Spacing.xs) {
                Text(PlayerFormat.time(model.position))
                    .foregroundStyle(Palette.textSecondary)
                    .accessibilityHidden(true)
                seekBar(duration: duration)
                Text(PlayerFormat.time(duration))
                    .foregroundStyle(Palette.textMuted)
                    .accessibilityHidden(true)
            }
            .font(.detail)
            .monospacedDigit()
        } else {
            stacked(duration: duration)
        }
    }

    private func seekBar(duration: Double) -> some View {
        SeekBar(
            value: $model.seekPosition,
            range: 0...max(duration, 1),
            valueDescription: "\(PlayerFormat.time(model.position)) of \(PlayerFormat.time(duration))"
        )
    }

    private func stacked(duration: Double) -> some View {
        VStack(spacing: Spacing.xxs) {
            seekBar(duration: duration)

            HStack {
                Text(PlayerFormat.time(model.position))
                    .foregroundStyle(Palette.textSecondary)
                Spacer()
                Text(PlayerFormat.time(duration))
                    .foregroundStyle(Palette.textMuted)
            }
            .font(.detail)
            .monospacedDigit()
            .accessibilityHidden(true)
        }
    }
}

/// The primary play/pause circle (56pt in the transport; larger in the learning view): pink fill,
/// ink icon, 3pt black border and a black hard shadow, pushed while pressed.
struct PlayButton: View {
    let isPlaying: Bool
    var diameter: CGFloat = ComponentSize.playButton
    let action: () -> Void

    var body: some View {
        Button(action: action) {
            Image(systemName: isPlaying ? "pause.fill" : "play.fill")
                .font(.title2)
                .foregroundStyle(Palette.onSecondary)
                .frame(width: diameter, height: diameter)
                .contentShape(.circle)
        }
        .buttonStyle(BlockButtonStyle(fill: Palette.secondary, shadow: HardShadow.sm, isCircle: true))
        .dynamicTypeSize(...DynamicTypeSize.xxxLarge)
        .accessibilityShowsLargeContentViewer {
            Label(isPlaying ? "Pause" : "Play", systemImage: isPlaying ? "pause.fill" : "play.fill")
        }
        .accessibilityLabel(isPlaying ? "Pause" : "Play")
    }
}

/// Speed: tap toggles 1.00× ⇄ 0.75×; long-press picks 0.50–1.25× in 0.05 steps (DESIGN.md §3.7).
struct SpeedButton: View {
    let model: PlayerModel

    var body: some View {
        Menu {
            SpeedPicker(model: model)
        } label: {
            Text(PlayerFormat.speed(model.speed))
                .font(.rowTitle)
                .monospacedDigit()
                .foregroundStyle(model.speed != 1 ? Palette.primary : Palette.textSecondary)
                .frame(minWidth: ComponentSize.minTouchTarget, minHeight: ComponentSize.minTouchTarget)
                .contentShape(.rect)
        } primaryAction: {
            model.toggleSpeed()
        }
        .dynamicTypeSize(...DynamicTypeSize.xxxLarge)
        .accessibilityLabel("Speed")
        .accessibilityValue(PlayerFormat.speed(model.speed))
        .accessibilityHint("Toggles practice speed. Touch and hold to choose a speed.")
    }
}

/// The 0.50…1.25× choices, as a menu section.
struct SpeedPicker: View {
    @Bindable var model: PlayerModel

    var body: some View {
        Picker("Speed", selection: $model.speedChoice) {
            ForEach(PlayerModel.speedSteps, id: \.self) { step in
                Text(PlayerFormat.speed(step)).tag(step)
            }
        }
    }
}

/// Transpose −6…+6 (display only), as a control chip that opens a menu.
struct TransposeChip: View {
    @Bindable var model: PlayerModel
    var inline = false

    var body: some View {
        Menu {
            Picker("Transpose", selection: $model.transposeChoice) {
                ForEach(Array(-PlayerModel.maxTranspose...PlayerModel.maxTranspose), id: \.self) { semitones in
                    Text(PlayerFormat.transpose(semitones)).tag(semitones)
                }
            }
        } label: {
            ControlChipLabel(title: "Transpose", value: PlayerFormat.transpose(model.transpose), inline: inline)
        }
        .buttonStyle(.plain)
        .accessibilityLabel("Transpose")
        .accessibilityValue(PlayerFormat.transpose(model.transpose))
    }
}

/// Speed as a control chip that opens the speed menu (learning view).
struct SpeedChip: View {
    let model: PlayerModel

    var body: some View {
        Menu {
            SpeedPicker(model: model)
        } label: {
            ControlChipLabel(title: "Speed", value: PlayerFormat.speed(model.speed))
        }
        .buttonStyle(.plain)
        .accessibilityLabel("Speed")
        .accessibilityValue(PlayerFormat.speed(model.speed))
    }
}

/// "Looping 1:04–1:20 · Clear", floating above the transport: a pink sticker with ink text.
struct LoopPill: View {
    let loop: Range<Double>
    let onClear: () -> Void

    var body: some View {
        HStack(spacing: Spacing.xxs) {
            Text("Looping \(PlayerFormat.loop(loop)) ·")
                .font(.labelSmall)
                .monospacedDigit()
            Button("Clear", action: onClear)
                .font(.detail.weight(.bold))
                .buttonStyle(.plain)
                .frame(minHeight: ComponentSize.minTouchTarget)
                .contentShape(.rect)
                .accessibilityLabel("Clear loop")
        }
        .foregroundStyle(Palette.onSecondary)
        .padding(.horizontal, Spacing.md)
        .block(Palette.secondary, shadow: HardShadow.sm)
    }
}
