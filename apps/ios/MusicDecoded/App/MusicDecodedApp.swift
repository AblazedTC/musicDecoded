import SwiftData
import SwiftUI

@main
struct MusicDecodedApp: App {
    @Environment(\.scenePhase) private var scenePhase
    @State private var session: SessionStore
    @State private var transcription: TranscriptionService
    @State private var sync: SyncService
    @State private var settings = SettingsStore()
    /// Owns the playing track's engine so playback continues across tabs (mini player).
    @State private var nowPlaying = NowPlaying()
    /// The Pro paywall (T-602i), shared by every screen.
    @State private var paywall: PaywallModel
    /// App Store purchases via RevenueCat; unavailable when the build has no key.
    private let billing: any BillingService
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
        let sync = SyncService(
            session: session,
            context: container.mainContext,
            analysisStore: analysisStore,
            audioDirectory: audioDirectory
        )
        _session = State(initialValue: session)
        let billing = RevenueCatBillingService.configured()
        self.billing = billing
        _paywall = State(
            initialValue: PaywallModel(
                billing: billing,
                accountUserId: { session.isAccount ? session.userId : nil },
                config: { session.config },
                refreshConfig: { await session.refreshConfig() }
            )
        )
        _sync = State(initialValue: sync)
        _transcription = State(
            initialValue: TranscriptionService(
                session: session,
                context: container.mainContext,
                analysisStore: analysisStore,
                audioDirectory: audioDirectory,
                backgroundActivity: { name in AppBackgroundActivity.begin(name) },
                // Sync trigger: after a job completes (PLAN.md §5.2).
                onJobComplete: { sync.requestSync() }
            )
        )
    }

    var body: some Scene {
        WindowGroup {
            RootView()
                .environment(session)
                .environment(transcription)
                .environment(sync)
                .environment(settings)
                .environment(nowPlaying)
                .environment(paywall)
                .task {
                    transcription.resumePending()
                    await session.start()
                    sync.requestSync()
                }
                // The store identity follows the session: appUserID = our userId for accounts (PLAN.md §9).
                .task(id: storeUserId) {
                    await updateStoreIdentity(storeUserId)
                }
        }
        .modelContainer(container)
        .onChange(of: scenePhase) { _, phase in
            // Polling is suspended with the app; pick up running jobs again on return.
            if phase == .active {
                transcription.resumePending()
                // Sync trigger: app foreground (PLAN.md §5.2). A no-op for guests.
                sync.requestSync()
            } else if phase == .background {
                // No background audio in v1: playback stops with the app.
                nowPlaying.pause()
            }
        }
    }

    /// Our userId for accounts; `nil` for guests (and before there's a session).
    private var storeUserId: String? {
        session.isAccount ? session.userId : nil
    }

    private func updateStoreIdentity(_ userId: String?) async {
        guard billing.isAvailable else { return }
        do {
            if let userId {
                try await billing.logIn(userId: userId)
            } else {
                try await billing.logOut()
            }
        } catch {
            // Purchases log in again before buying, so a failure here only delays the identity switch.
        }
    }
}
