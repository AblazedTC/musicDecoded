import Testing
import UIKit

/// If a bundled font isn't copied or listed in UIAppFonts, SwiftUI silently falls back to the
/// system font. These run inside the host app, so a missing registration fails here instead.
struct FontRegistrationTests {
    @Test(arguments: ["ArchivoBlack-Regular", "SpaceGrotesk-Regular", "SpaceGrotesk-Medium", "SpaceGrotesk-Bold"])
    func bundledFontIsRegistered(postScriptName: String) {
        #expect(UIFont(name: postScriptName, size: 12) != nil)
    }
}
