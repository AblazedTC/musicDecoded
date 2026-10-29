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
            .font(.largeTitle.weight(.light))
            .monospacedDigit()
            .foregroundStyle(Palette.text)
            .accessibilityLabel("Recorded time \(InputLevel.formatTimer(model.elapsed))")

        LiveWaveform(levels: model.waveform, isActive: model.phase == .recording)
        LevelMeter(db: model.levelDb, isActive: model.phase == .recording)

        HStack(alignment: .top, spacing: Spacing.xxxl) {
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
                .font(.detail)
                .foregroundStyle(color)
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
                context.fill(Path(roundedRect: rect, cornerRadius: barWidth / 2), with: .color(color))
            }
        }
        .frame(height: ComponentSize.recorderWaveformHeight)
        .padding(Spacing.md)
        .background(Palette.timelineBackground, in: .rect(cornerRadius: Radius.card))
        .overlay {
            RoundedRectangle(cornerRadius: Radius.card).strokeBorder(Palette.timelineBorder, lineWidth: 1)
        }
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
                Text("Input level")
                    .overlineStyle()
                Spacer()
                Text(dbText)
                    .font(.overline)
                    .monospacedDigit()
                    .foregroundStyle(Palette.textSecondary)
            }
            HStack(spacing: Spacing.xxs) {
                ForEach(0..<InputLevel.segments, id: \.self) { index in
                    Capsule()
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
        if index >= lit { return Palette.surfaceSoft }
        if index >= InputLevel.segments - 2 { return Palette.record } // clipping
        if index >= InputLevel.segments - 5 { return Palette.chord }
        return Palette.primary
    }
}

/// The 84pt red ring: record, pause or resume.
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
                    .foregroundStyle(Palette.text)
                    .frame(maxWidth: .infinity, maxHeight: .infinity)
                    .background(isEnabled ? Palette.record : Palette.surfaceSoft, in: .circle)
                    .padding(Spacing.xs)
                    .overlay {
                        Circle().strokeBorder(Palette.recordRing, lineWidth: Spacing.xxs)
                    }
                    .frame(width: ComponentSize.recordButton, height: ComponentSize.recordButton)
                    .contentShape(.circle)
            }
            .buttonStyle(.plain)
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
                .font(.overline)
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
            Image(systemName: "mic.slash")
                .font(.title)
                .foregroundStyle(SourceTint.record.icon)
                .frame(width: ComponentSize.recordButton, height: ComponentSize.recordButton)
                .background(SourceTint.record.background, in: .circle)
                .accessibilityHidden(true)
            Text(title)
                .font(.sectionHeader)
                .foregroundStyle(Palette.text)
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
