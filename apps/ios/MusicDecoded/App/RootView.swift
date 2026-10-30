import SwiftUI

/// The bottom tabs (DESIGN.md §2; 3 tabs until the Tuner decision, T-505).
enum AppTab: Hashable {
    case home, library, settings
}

/// The three tabs from PLAN.md §3.1. Each tab owns its `NavigationStack`.
struct RootView: View {
    @Environment(SessionStore.self) private var session
    @Environment(PaywallModel.self) private var paywall
    @State private var tab = AppTab.home
    /// Sign-up for a guest who tapped Subscribe (T-602i); the purchase continues afterwards.
    @State private var showsAccountFlow = false
    @State private var accountPath: [AccountRoute] = []

    var body: some View {
        TabView(selection: $tab) {
            HomeView(
                onShowLibrary: { tab = .library },
                onShowSettings: { tab = .settings }
            )
            .tabItem { Label("Home", systemImage: "house") }
            .tag(AppTab.home)

            LibraryView()
                .tabItem { Label("Library", systemImage: "music.note.list") }
                .tag(AppTab.library)

            SettingsView()
                .tabItem { Label("Settings", systemImage: "gearshape") }
                .tag(AppTab.settings)
        }
        .tint(Palette.primary)
        .environment(\.openPaywall, openPaywall)
        .sheet(isPresented: Bindable(paywall).isPresented, onDismiss: paywallDismissed) {
            PaywallSheet(model: paywall)
        }
        .sheet(isPresented: $showsAccountFlow, onDismiss: accountFlowDismissed) {
            NavigationStack(path: $accountPath) {
                AuthView(
                    mode: .signUp,
                    onMigrate: { accountPath = [.migrate] },
                    onDone: { showsAccountFlow = false }
                )
                .navigationDestination(for: AccountRoute.self) { route in
                    switch route {
                    case .auth(let mode):
                        AuthView(mode: mode, onMigrate: { accountPath = [.migrate] }, onDone: { showsAccountFlow = false })
                    case .migrate:
                        MigrationView(onDone: { showsAccountFlow = false })
                    }
                }
            }
        }
        .preferredColorScheme(.dark) // v1 is dark-only (DESIGN.md §1)
    }

    /// No upsell once `/config` says Pro (DESIGN.md §3.11).
    private var openPaywall: OpenPaywallAction? {
        if session.isPro { return nil }
        let paywall = paywall
        return OpenPaywallAction { paywall.open() }
    }

    /// A guest who tapped Subscribe goes to sign-up once the paywall is gone.
    private func paywallDismissed() {
        paywall.dismissed()
        if paywall.needsAccount {
            paywall.accountFlowStarted()
            accountPath = []
            showsAccountFlow = true
        }
    }

    private func accountFlowDismissed() {
        Task {
            await paywall.accountFlowFinished()
        }
    }
}

#if DEBUG
#Preview {
    RootView()
        .environment(TranscriptionService.preview())
        .environment(SessionStore.preview())
        .environment(SyncService.preview())
        .environment(SettingsStore())
        .environment(NowPlaying())
        .environment(PaywallModel.preview())
        .modelContainer(for: LibraryItem.self, inMemory: true)
}
#endif
