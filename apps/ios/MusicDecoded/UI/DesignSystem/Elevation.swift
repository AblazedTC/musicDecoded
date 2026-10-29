import SwiftUI

extension View {
    /// `shadowCard`: y 12, blur 30, black 20%.
    func cardShadow() -> some View {
        shadow(color: .black.opacity(0.2), radius: 15, y: 12)
    }

    /// `shadowPlayer`: y -14, blur 35, black 28%.
    func playerShadow() -> some View {
        shadow(color: .black.opacity(0.28), radius: 17.5, y: -14)
    }
}
