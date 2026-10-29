import SwiftUI

/// Entry points for the local sources. Home sets a route from its source cards; `sourceRoutes`
/// opens the right picker or screen. Each flow ends once the job has started, and
/// `onDecodeStarted` gets the trackId (the Processing screen, T-211, will hook in there).
enum SourceRoute: Hashable, Sendable {
    case files
    case mediaLibrary
}

extension View {
    /// Presents the source flows for `route` (and sets it back to `nil`).
    func sourceRoutes(
        _ route: Binding<SourceRoute?>,
        onDecodeStarted: @escaping (_ trackId: String) -> Void = { _ in }
    ) -> some View {
        modifier(SourceRoutesModifier(route: route, onDecodeStarted: onDecodeStarted))
    }
}

@MainActor
private struct SourceRoutesModifier: ViewModifier {
    /// The sheet, which first holds the Media Library picker and then the Import sheet.
    private enum Sheet: String, Identifiable {
        case mediaPicker, importSheet

        var id: String { rawValue }
    }

    /// What to present once the current sheet has finished dismissing (SwiftUI can't present
    /// while dismissing).
    private enum Next {
        case filePicker
        case mediaPicker
        /// Show the Import sheet as it is (e.g. with an error).
        case importSheet
        case importMediaItem(MediaLibraryPick)
    }

    @Binding var route: SourceRoute?
    let onDecodeStarted: (String) -> Void

    @Environment(TranscriptionService.self) private var transcription
    @State private var importModel = ImportModel()
    @State private var importOrigin = SourceRoute.files
    @State private var showFilePicker = false
    @State private var sheet: Sheet?
    @State private var next: Next?

    func body(content: Content) -> some View {
        content
            .onChange(of: route) { _, newRoute in
                guard let newRoute else { return }
                route = nil
                open(newRoute)
            }
            .fileImporter(isPresented: $showFilePicker, allowedContentTypes: [.audio]) { result in
                guard case .success(let url) = result else { return }
                sheet = .importSheet
                Task {
                    await importModel.importFile(url, using: transcription)
                }
            }
            .sheet(item: $sheet, onDismiss: presentNext) { sheet in
                switch sheet {
                case .mediaPicker:
                    MediaLibraryPicker { outcome in
                        mediaPickerFinished(outcome)
                    }
                    .ignoresSafeArea()
                case .importSheet:
                    ImportSheet(model: importModel, onImport: importAndDecode, onChooseAnother: chooseAnother)
                }
            }
    }

    private func open(_ newRoute: SourceRoute) {
        importModel.reset()
        importOrigin = newRoute
        switch newRoute {
        case .files:
            showFilePicker = true
        case .mediaLibrary:
            Task {
                if await MediaLibraryAccess.request() {
                    sheet = .mediaPicker
                } else {
                    importModel.errorMessage = "MusicDecoded can't see your music. Allow Media & Apple Music access in Settings."
                    sheet = .importSheet
                }
            }
        }
    }

    private func mediaPickerFinished(_ outcome: MediaLibraryPicker.Outcome) {
        switch outcome {
        case .picked(let pick):
            next = .importMediaItem(pick)
        case .unavailable:
            importModel.errorMessage = AudioImporter.protectedMessage
            next = .importSheet
        case .cancelled:
            next = nil
        }
        sheet = nil
    }

    private func presentNext() {
        guard let pending = next else {
            // Closed by the user, or decoding started: drop any pending copy.
            importModel.reset()
            return
        }
        next = nil
        switch pending {
        case .filePicker:
            showFilePicker = true
        case .mediaPicker:
            sheet = .mediaPicker
        case .importSheet:
            sheet = .importSheet
        case .importMediaItem(let pick):
            sheet = .importSheet
            Task {
                await importModel.importMediaItem(pick, using: transcription)
            }
        }
    }

    private func importAndDecode() {
        Task {
            if let trackId = await importModel.importAndDecode(using: transcription) {
                sheet = nil
                onDecodeStarted(trackId)
            }
        }
    }

    /// "Choose another file": close the sheet, then reopen the picker it came from.
    private func chooseAnother() {
        importModel.reset()
        next = importOrigin == .files ? .filePicker : .mediaPicker
        sheet = nil
    }
}
