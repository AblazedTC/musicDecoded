import SwiftUI

@main
struct MusicDecodedApp: App {
    @State private var session = SessionStore(api: APIClient.configured(), secureStore: Keychain())

    var body: some Scene {
        WindowGroup {
            RootView()
                .environment(session)
                .task {
                    await session.start()
                }
        }
    }
}
