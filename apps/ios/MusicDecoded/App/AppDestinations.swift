import SwiftUI

extension View {
    /// The screens a tab pushes onto its `NavigationStack(path:)`: the Player and Processing.
    /// "Open transcription" replaces Processing with the Player.
    func appDestinations(path: Binding<NavigationPath>) -> some View {
        navigationDestination(for: PlayerRoute.self) { route in
            PlayerScreen(trackId: route.trackId)
        }
        .navigationDestination(for: ProcessingRoute.self) { route in
            ProcessingScreen(trackId: route.trackId) {
                if path.wrappedValue.isEmpty == false {
                    path.wrappedValue.removeLast()
                }
                path.wrappedValue.append(PlayerRoute(trackId: route.trackId))
            }
        }
    }
}
