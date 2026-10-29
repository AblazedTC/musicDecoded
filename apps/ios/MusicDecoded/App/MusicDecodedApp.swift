import SwiftData
import SwiftUI

@main
struct MusicDecodedApp: App {
    @Environment(\.scenePhase) private var scenePhase
    @State private var session: SessionStore
    @State private var transcription: TranscriptionService
    private let container: ModelContainer

    init() {
        let session = SessionStore(api: APIClient.configured(), secureStore: Keychain())
        let container: ModelContainer
        let audioDirectory: URL
        let analysisStore: AnalysisStore
        do {
            container = try ModelContainer(for: LibraryItem.self)
            audioDirectory = try AppDirectories.audio()
            analysisStore = try AnalysisStore.appDefault()
        } catch {
            fatalError("Couldn't open the on-device library: \(error)")
        }
        self.container = container
        _session = State(initialValue: session)
        _transcription = State(
            initialValue: TranscriptionService(
                session: session,
                context: container.mainContext,
                analysisStore: analysisStore,
                audioDirectory: audioDirectory,
                backgroundActivity: { name in AppBackgroundActivity.begin(name) }
            )
        )
    }

    var body: some Scene {
        WindowGroup {
            RootView()
                .environment(session)
                .environment(transcription)
                .task {
                    transcription.resumePending()
                    await session.start()
                }
        }
        .modelContainer(container)
        .onChange(of: scenePhase) { _, phase in
            // Polling is suspended with the app; pick up running jobs again on return.
            if phase == .active {
                transcription.resumePending()
            }
        }
    }
}
