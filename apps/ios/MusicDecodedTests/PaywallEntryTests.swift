import Foundation
import Testing
@testable import MusicDecoded

/// Where the paywall opens from (DESIGN.md §3.11) and the sheet's guest/unavailable copy.
/// Ports PaywallEntryTest.kt.
@MainActor
struct PaywallEntryTests {
    private let dailyLimit = APIError(code: .rateLimited, message: "Daily limit reached", status: 429)

    @Test func dailyLimitFromJobsOffersGoPro() {
        let message = Transcription.userMessage(for: dailyLimit)
        #expect(message == Transcription.rateLimitedMessage)
        #expect(OpenPaywallAction.opens(for: message))
        #expect(ErrorState.title(for: message) == "Daily limit reached")
    }

    @MainActor
    final class Counter {
        var value = 0
    }

    @Test func goProActionRunsTheOpener() {
        let opened = Counter()
        let action = OpenPaywallAction { opened.value += 1 }
        action()
        #expect(opened.value == 1)
    }

    @Test func youTubeSearchQuotaNeverOpensThePaywall() {
        let failure = YouTubeSearchModel.Failure(dailyLimit)
        #expect(failure == .quota)
        #expect(OpenPaywallAction.opens(for: failure.message) == false)
    }

    @Test func otherErrorsHaveNoGoPro() {
        #expect(OpenPaywallAction.opens(for: Transcription.offlineMessage) == false)
        #expect(OpenPaywallAction.opens(for: Transcription.genericMessage) == false)
        #expect(OpenPaywallAction.opens(for: nil) == false)
    }

    @Test func guestsAreAskedToCreateAnAccount() {
        #expect(PaywallContent.subscribeTitle(isGuest: true) == "Create an account to subscribe")
        #expect(PaywallContent.subscribeTitle(isGuest: false) == "Subscribe")
        let cards = PaywallModel.planCards(FakeBillingService().plans)
        #expect(cards.compactMap(\.saving) == ["Save 33%"])
    }

    @Test func unavailableBuildSaysSo() async {
        let model = PaywallModel(
            billing: FakeBillingService(isAvailable: false),
            accountUserId: { "user-1" },
            config: { nil },
            refreshConfig: { nil }
        )
        model.open()
        await model.loadIfNeeded()
        #expect(model.phase == .unavailable)
        #expect(BillingError.unavailable.message == "Purchases aren't available in this build.")
    }

    @Test func revenueCatWithoutAKeyIsUnavailable() async {
        let service = RevenueCatBillingService(apiKey: "  ")
        #expect(service.isAvailable == false)
        await #expect(throws: BillingError.unavailable) {
            _ = try await service.offerings()
        }
    }
}
