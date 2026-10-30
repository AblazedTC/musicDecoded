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
            VStack(spacing: Spacing.xxs) {
                PlayerSeekBar(model: model)
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
            .padding(.top, Spacing.sm)
            .padding(.bottom, Spacing.xs)
            .background {
                let shape = UnevenRoundedRectangle(topLeadingRadius: Radius.sheet, topTrailingRadius: Radius.sheet)
                ZStack {
                    shape.fill(Palette.transportBackground)
                    shape.stroke(Palette.transportBorder, lineWidth: 1)
                }
                .playerShadow()
                .ignoresSafeArea(edges: .bottom)
            }
        }
    }
}

/// Seek slider with current / total times.
struct PlayerSeekBar: View {
    @Bindable var model: PlayerModel

    var body: some View {
        let duration = model.duration
        VStack(spacing: Spacing.xxs) {
            SeekBar(
                value: $model.seekPosition,
                range: 0...max(duration, 1),
                valueDescription: "\(PlayerFormat.time(model.position)) of \(PlayerFormat.time(duration))"
            )

            HStack {
                Text(PlayerFormat.time(model.position))
                    .foregroundStyle(Palette.textSecondary)
                Spacer()
                Text(PlayerFormat.time(duration))
                    .foregroundStyle(Palette.textMuted)
            }
            .font(.overline)
            .monospacedDigit()
            .accessibilityHidden(true)
        }
    }
}

/// The primary play/pause circle (56pt in the transport; larger in the learning view).
struct PlayButton: View {
    let isPlaying: Bool
    var diameter: CGFloat = ComponentSize.playButton
    let action: () -> Void

    var body: some View {
        Button(action: action) {
            Image(systemName: isPlaying ? "pause.fill" : "play.fill")
                .font(.title2)
                .foregroundStyle(Palette.onPrimary)
                .frame(width: diameter, height: diameter)
                .background(Palette.primary, in: .circle)
                .contentShape(.circle)
        }
        .buttonStyle(.plain)
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

    var body: some View {
        Menu {
            Picker("Transpose", selection: $model.transposeChoice) {
                ForEach(Array(-PlayerModel.maxTranspose...PlayerModel.maxTranspose), id: \.self) { semitones in
                    Text(PlayerFormat.transpose(semitones)).tag(semitones)
                }
            }
        } label: {
            ControlChipLabel(title: "Transpose", value: PlayerFormat.transpose(model.transpose))
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

/// "Looping 1:04–1:20 · Clear", floating above the transport (chord fill).
struct LoopPill: View {
    let loop: Range<Double>
    let onClear: () -> Void

    var body: some View {
        HStack(spacing: Spacing.xxs) {
            Text("Looping \(PlayerFormat.loop(loop)) ·")
                .font(.detail.weight(.semibold))
                .monospacedDigit()
            Button("Clear", action: onClear)
                .font(.detail.weight(.bold))
                .buttonStyle(.plain)
                .frame(minHeight: ComponentSize.minTouchTarget)
                .contentShape(.rect)
                .accessibilityLabel("Clear loop")
        }
        .foregroundStyle(Palette.onChord)
        .padding(.horizontal, Spacing.md)
        .background(Palette.chord, in: .capsule)
    }
}
