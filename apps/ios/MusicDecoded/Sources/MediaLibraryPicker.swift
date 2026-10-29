import MediaPlayer
import SwiftUI

/// The system Media Library picker, limited to local, unprotected songs. The simulator has no
/// music library, so this only does something on a device.
struct MediaLibraryPicker: UIViewControllerRepresentable {
    enum Outcome: Sendable {
        case picked(MediaLibraryPick)
        /// The song has no `assetURL` (DRM-protected or not downloaded).
        case unavailable
        case cancelled
    }

    let onFinish: (Outcome) -> Void

    func makeUIViewController(context: Context) -> MPMediaPickerController {
        let picker = MPMediaPickerController(mediaTypes: .music)
        picker.allowsPickingMultipleItems = false
        picker.showsCloudItems = false
        picker.showsItemsWithProtectedAssets = false
        picker.prompt = "Choose a song to decode"
        picker.delegate = context.coordinator
        return picker
    }

    func updateUIViewController(_ uiViewController: MPMediaPickerController, context: Context) {
        context.coordinator.onFinish = onFinish
    }

    func makeCoordinator() -> Coordinator {
        Coordinator(onFinish: onFinish)
    }

    @MainActor
    final class Coordinator: NSObject, @preconcurrency MPMediaPickerControllerDelegate {
        var onFinish: (Outcome) -> Void

        init(onFinish: @escaping (Outcome) -> Void) {
            self.onFinish = onFinish
        }

        func mediaPicker(_ mediaPicker: MPMediaPickerController, didPickMediaItems mediaItemCollection: MPMediaItemCollection) {
            guard let item = mediaItemCollection.items.first, let assetURL = item.assetURL else {
                onFinish(.unavailable)
                return
            }
            onFinish(.picked(MediaLibraryPick(assetURL: assetURL, title: item.title)))
        }

        func mediaPickerDidCancel(_ mediaPicker: MPMediaPickerController) {
            onFinish(.cancelled)
        }
    }
}

enum MediaLibraryAccess {
    /// Asks for Media Library access if it hasn't been decided yet. Nonisolated, so the
    /// system's callback (on a background queue) never runs a main-actor closure.
    static func request() async -> Bool {
        var status = MPMediaLibrary.authorizationStatus()
        if status == .notDetermined {
            status = await withCheckedContinuation { continuation in
                MPMediaLibrary.requestAuthorization { newStatus in
                    continuation.resume(returning: newStatus)
                }
            }
        }
        return status == .authorized
    }
}
