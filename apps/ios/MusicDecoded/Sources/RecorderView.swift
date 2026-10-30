import SwiftUI
import UIKit

/// DESIGN.md §3.6. Pushed from Home (no tab bar). `onDecodeStarted` gets the trackId once the
/// recording is uploaded and the job exists.
struct RecorderView: View {
    static let tip = "Record in a quiet room, close to the instrument. Background noise and talking make chords harder to detect. Up to 10 minutes."

    let onDecodeStarted: (_ trackId: String) -> Void

    @Environment(TranscriptionService.self) private var transcription
    @Environment(\.scenePhase) private var scenePhase
    @Environment(\.openURL) private var openURL
    @State private var model = RecorderModel()
    @State private var title = ""

    var body: some View {
        ScrollView {
            VStack(spacing: Spacing.lg) {
                switch model.permission {
                case .granted:
                    recorder
                case .undetermined:
                    PermissionCard(
                        title: "Allow the microphone",
                        message: "MusicDecoded needs the microphone to record a performance and decode its chords.",
                        actionTitle: "Allow microphone"
                    ) {
                        Task { await model.requestPermission() }
                    }
                case .denied:
                    PermissionCard(
                        title: "Microphone access is off",
                        message: "Turn on the microphone for MusicDecoded in Settings to record a performance.",
                        actionTitle: "Open Settings"
                    ) {
                        if let url = URL(string: UIApplication.openSettingsURLString) {
                            openURL(url)
                        }
                    }
                }
            }
            .padding(.horizontal, Spacing.md)
            .padding(.top, Spacing.xxxl)
            .padding(.bottom, Spacing.lg)
        }
        .background(Palette.canvas)
        .navigationTitle("Record Audio")
        .navigationBarTitleDisplayMode(.inline)
        .toolbar(.hidden, for: .tabBar)
        .toolbar {
            ToolbarItem(placement: .confirmationAction) {
                Button("Save") { model.finish() }
                    .disabled(!model.hasRecording || model.phase == .starting)
            }
        }
        .task {
            await model.requestPermission()
        }
        .onChange(of: scenePhase) { _, phase in
            switch phase {
            case .background: model.onBackgrounded()
            case .active: model.refreshPermission()
            default: break
            }
        }
        .onChange(of: model.askTitle) { _, ask in
            if ask { title = model.defaultTitle }
        }
        .onChange(of: model.startedTrackId) { _, trackId in
            if let trackId { onDecodeStarted(trackId) }
        }
        .onDisappear {
            model.close()
        }
        .alert("Name your recording", isPresented: $model.askTitle) {
            TextField("Title", text: $title)
            Button("Decode") {
                Task { await model.save(title: title, using: transcription) }
            }
            Button("Not now", role: .cancel) {}
        }
    }

    @ViewBuilder
    private var recorder: some View {
        StateLabel(phase: model.phase)

        Text(InputLevel.formatTimer(model.elapsed))
            .font(.display(56, relativeTo: .largeTitle))
            .monospacedDigit()
            .lineLimit(1)
            .minimumScaleFactor(0.5)
            .foregroundStyle(Palette.text)
            .accessibilityLabel("Recorded time \(InputLevel.formatTimer(model.elapsed))")

        LiveWaveform(levels: model.waveform, isActive: model.phase == .recording)
        LevelMeter(db: model.levelDb, isActive: model.phase == .recording)

        HStack(alignment: .top, spacing: Spacing.xxl) {
            LabeledAction(label: "Discard") {
                IconButton(title: "Discard recording", systemImage: "trash") { model.discard() }
                    .disabled(!model.hasRecording || model.phase == .starting)
            }
            RecordButton(phase: model.phase) {
                model.toggleRecord(using: transcription)
            }
            LabeledAction(label: "Finish") {
                IconButton(title: "Finish recording", systemImage: "checkmark", isActive: model.hasRecording) { model.finish() }
                    .disabled(!model.hasRecording || model.phase == .starting)
            }
        }
        .padding(.top, Spacing.md)

        if model.phase == .starting, model.startedTrackId == nil {
            HStack(spacing: Spacing.sm) {
                ProgressView()
                    .tint(Palette.primary)
                Text("Uploading…")
                    .font(.detail)
                    .foregroundStyle(Palette.textSecondary)
            }
        }

        if let errorMessage = model.errorMessage {
            Text(errorMessage)
                .font(.detail)
                .foregroundStyle(Palette.record)
                .multilineTextAlignment(.center)
        }

        PolicyNote(text: Self.tip, systemImage: "lightbulb")
    }
}

/// "Microphone ready" / "Recording" / ... with a colored dot.
private struct StateLabel: View {
    let phase: RecorderModel.Phase

    var body: some View {
        let (text, color): (String, Color) = switch phase {
        case .idle: ("Microphone ready", Palette.primary)
        case .recording: ("Recording", Palette.record)
        case .paused: ("Paused", Palette.chord)
        case .stopped, .starting: ("Recording finished", Palette.textSecondary)
        }
        HStack(spacing: Spacing.xs) {
            Circle()
                .fill(color)
                .frame(width: Spacing.xs, height: Spacing.xs)
                .accessibilityHidden(true)
            Text(text)
                .overlineStyle(color)
        }
    }
}

/// Recent input levels as centered bars, newest on the right (dimmed when not recording).
private struct LiveWaveform: View {
    let levels: [Float]
    let isActive: Bool

    var body: some View {
        let values = levels
        let color = isActive ? Palette.wavePlayed : Palette.waveIdle
        let bars = InputLevel.waveformBars
        Canvas { context, size in
            let slot = size.width / CGFloat(bars)
            let barWidth = slot * 0.6
            let offset = bars - values.count
            for index in 0..<bars {
                let valueIndex = index - offset
                let level = values.indices.contains(valueIndex) ? CGFloat(values[valueIndex]) : 0
                let height = max(barWidth, level * size.height)
                let rect = CGRect(
                    x: CGFloat(index) * slot + (slot - barWidth) / 2,
                    y: (size.height - height) / 2,
                    width: barWidth,
                    height: height
                )
                context.fill(Path(rect), with: .color(color))
            }
        }
        .frame(height: ComponentSize.recorderWaveformHeight)
        .padding(Spacing.md)
        .block(Palette.surface, shadow: HardShadow.lg, shadowColor: Palette.violet)
        .accessibilityHidden(true)
    }
}

/// The 16-segment input level meter with its dB value.
private struct LevelMeter: View {
    let db: Float
    let isActive: Bool

    var body: some View {
        let lit = isActive ? InputLevel.litSegments(db) : 0
        let dbText = isActive ? "\(Int(db.rounded())) dB" : "— dB"
        VStack(alignment: .leading, spacing: Spacing.xs) {
            HStack {
                Eyebrow(text: "Input level")
                Spacer()
                Text(dbText)
                    .font(.detail)
                    .monospacedDigit()
                    .foregroundStyle(Palette.textSecondary)
            }
            HStack(spacing: Spacing.xxs) {
                ForEach(0..<InputLevel.segments, id: \.self) { index in
                    Rectangle()
                        .fill(color(for: index, lit: lit))
                        .frame(height: ComponentSize.meterSegmentHeight)
                }
            }
        }
        .accessibilityElement(children: .ignore)
        .accessibilityLabel("Input level")
        .accessibilityValue(isActive ? dbText : "Not recording")
    }

    private func color(for index: Int, lit: Int) -> Color {
        if index >= lit { return Palette.surfaceRaised }
        if index >= InputLevel.segments - 2 { return Palette.record } // clipping
        if index >= InputLevel.segments - 5 { return Palette.yellow }
        return Palette.primary
    }
}

/// The 84pt orange record button with a black border and violet hard shadow: record, pause or resume.
private struct RecordButton: View {
    let phase: RecorderModel.Phase
    let action: () -> Void

    private var label: String {
        switch phase {
        case .recording: "Pause"
        case .paused: "Resume"
        default: "Record"
        }
    }

    private var isEnabled: Bool {
        phase == .idle || phase == .recording || phase == .paused
    }

    var body: some View {
        LabeledAction(label: label) {
            Button(action: action) {
                Image(systemName: phase == .recording ? "pause.fill" : "mic.fill")
                    .font(.title2)
                    .foregroundStyle(isEnabled ? Palette.onColor : Palette.textMuted)
                    .frame(width: ComponentSize.recordButton, height: ComponentSize.recordButton)
                    .contentShape(.circle)
            }
            .buttonStyle(
                BlockButtonStyle(
                    fill: isEnabled ? Palette.record : Palette.surfaceRaised,
                    shadow: isEnabled ? HardShadow.md : 0,
                    shadowColor: Palette.violet,
                    isCircle: true
                )
            )
            .disabled(!isEnabled)
            .accessibilityLabel(label)
        }
    }
}

private struct LabeledAction<Content: View>: View {
    let label: String
    @ViewBuilder let content: Content

    var body: some View {
        VStack(spacing: Spacing.xxs) {
            content
            Text(label)
                .font(.labelSmall)
                .foregroundStyle(Palette.textSecondary)
                .accessibilityHidden(true)
        }
    }
}

/// Shown instead of the recorder until the microphone is allowed.
private struct PermissionCard: View {
    let title: String
    let message: String
    let actionTitle: String
    let action: () -> Void

    var body: some View {
        VStack(spacing: Spacing.md) {
            Image(systemName: "mic.slash.fill")
                .font(.title)
                .foregroundStyle(Palette.onColor)
                .frame(width: ComponentSize.recordButton, height: ComponentSize.recordButton)
                .background(SourceTint.record.fill, in: .circle)
                .overlay {
                    Circle().strokeBorder(Palette.border, lineWidth: BorderWidth.thick)
                }
                .hardShadowCircle(HardShadow.md, color: Palette.violet)
                .accessibilityHidden(true)
            DisplayHeadline(text: title, font: .sectionHeader)
                .multilineTextAlignment(.center)
            Text(message)
                .font(.detail)
                .foregroundStyle(Palette.textSecondary)
                .multilineTextAlignment(.center)
            PrimaryButton(title: actionTitle, systemImage: "mic", action: action)
        }
    }
}

#Preview {
    NavigationStack {
        RecorderView { _ in }
    }
    .environment(TranscriptionService.preview())
    .preferredColorScheme(.dark)
}
