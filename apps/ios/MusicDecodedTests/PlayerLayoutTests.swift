import CoreGraphics
import Testing
@testable import MusicDecoded

/// Timeline view panel split (DESIGN.md §3.7 "Owner revision 2").
struct PlayerLayoutTests {
    private let chrome = PlayerLayout.pianoPanelChrome
    private let gap: CGFloat = 4

    @Test func roomyScreensGetTheLargestKeyboardAndTheRestForMedia() {
        let split = PlayerLayout.split(available: 500, gap: gap)
        #expect(split.keyboard == ComponentSize.fixedKeyboardMax)
        #expect(split.mediaCollapsed == false)
        #expect(split.media == 500 - chrome - gap - ComponentSize.fixedKeyboardMax)
    }

    @Test func theKeyboardShrinksBeforeTheMediaCollapses() {
        let available = chrome + gap + ComponentSize.mediaPanelMin + 120
        let split = PlayerLayout.split(available: available, gap: gap)
        #expect(split.keyboard == 120)
        #expect(split.media == ComponentSize.mediaPanelMin)
        #expect(split.mediaCollapsed == false)
    }

    @Test func underTheMinimumTheMediaBecomesAStrip() {
        let available = chrome + gap + ComponentSize.mediaPanelMin + 100
        let split = PlayerLayout.split(available: available, gap: gap)
        #expect(split.mediaCollapsed)
        #expect(split.media == ComponentSize.mediaStripHeight)
        #expect(split.keyboard == ComponentSize.fixedKeyboardMax)

        let tiny = PlayerLayout.split(available: 100, gap: gap)
        #expect(tiny.keyboard == ComponentSize.fixedKeyboardCompact)
        #expect(tiny.mediaCollapsed)
    }
}
