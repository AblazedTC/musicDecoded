import Foundation
import Testing
@testable import MusicDecoded

/// The Pro paywall's states against a fake store (T-602i). Ports PaywallViewModelTest.kt.
@MainActor
struct PaywallModelTests {
    /// The session and `/config` the model reads; the next `/config` answers `serverPlans` in turn
    /// (the API flips the plan after RevenueCat's webhook).
    @MainActor
    final class Server {
        var userId: String? = "user-1"
        var config: ClientConfig? = Server.config(.free)
        var serverPlans: [Plan] = [.free]
        private(set) var refreshes = 0

        func refresh() -> ClientConfig? {
            refreshes += 1
            let plan = serverPlans.count > 1 ? serverPlans.removeFirst() : serverPlans[0]
            config = Server.config(plan)
            return config
        }

        nonisolated static func config(_ plan: Plan) -> ClientConfig {
            ClientConfig(
                offlineStreamAudio: false,
                maxUploadBytes: 1,
                soundcloudEnabled: nil,
                plan: plan,
                limits: ClientConfig.Limits(dailyJobs: plan == .pro ? nil : 50, jobsUsedToday: 50)
            )
        }
    }

    let billing: FakeBillingService
    let server: Server

    init() {
        billing = FakeBillingService()
        server = Server()
    }

    private func makeModel(_ service: FakeBillingService? = nil) -> PaywallModel {
        let server = server
        return PaywallModel(
            billing: service ?? billing,
            accountUserId: { server.userId },
            config: { server.config },
            refreshConfig: { server.refresh() },
            configRetryDelay: .zero
        )
    }

    /// Opens the sheet and loads the plans, as the sheet's `.task` does.
    private func openModel(_ service: FakeBillingService? = nil) async -> PaywallModel {
        let model = makeModel(service)
        model.open()
        await model.loadIfNeeded()
        return model
    }

    @Test func offeringsBecomeCardsWithYearlySavingPreselected() async {
        let model = await openModel()

        #expect(model.isPresented)
        #expect(model.phase == .ready)
        #expect(model.plans.map(\.title) == ["Monthly", "Yearly"])
        #expect(model.plans[0].price == "$4.99 / month")
        #expect(model.plans[1].price == "$39.99 / year")
        #expect(model.plans[0].saving == nil)
        #expect(model.plans[1].saving == "Save 33%") // 39.99 vs 12 × 4.99 = 59.88
        #expect(model.selectedId == "annual")
        #expect(model.freeDailyJobs == 50)
    }

    @Test func noMonthlyMeansNoSavingLabel() throws {
        let card = try #require(PaywallModel.planCards([FakeBillingService.annual]).first)
        #expect(card.option.period == .annual)
        #expect(card.saving == nil)
    }

    @Test func purchaseSuccessRefreshesConfigAndShowsSuccess() async {
        let model = await openModel()
        model.select("monthly")
        server.serverPlans = [.pro]

        await model.subscribe()

        #expect(billing.calls == ["offerings", "logIn:user-1", "purchase:monthly"])
        #expect(model.phase == .success)
        #expect(server.refreshes == 1)
        #expect(server.config?.plan == .pro) // every screen reads the plan from /config
        model.dismissed()
        #expect(model.isPresented == false)
    }

    @Test func purchaseRetriesConfigUntilTheWebhookLands() async {
        server.serverPlans = [.free, .free, .pro]
        let model = await openModel()

        await model.subscribe()

        #expect(server.refreshes == 3)
        #expect(model.phase == .success)
    }

    @Test func cancelledPurchaseIsSilent() async {
        billing.purchaseResult = { .cancelled }
        let model = await openModel()

        await model.subscribe()

        #expect(model.phase == .ready)
        #expect(model.error == nil)
        #expect(server.refreshes == 0)
    }

    @Test func purchaseErrorShowsMessageAndRetryWorks() async {
        billing.purchaseResult = { throw BillingError.offline }
        let model = await openModel()

        await model.subscribe()

        #expect(model.error == BillingError.offline.message)
        #expect(model.failed == .purchase)
        #expect(model.phase == .ready)

        billing.purchaseResult = { .purchased }
        server.serverPlans = [.pro]
        await model.retry() // Retry re-runs the failed action
        #expect(model.phase == .success)
        #expect(model.error == nil)
        #expect(billing.calls.filter { $0.hasPrefix("purchase") }.count == 2)
    }

    @Test func offeringsErrorRetriesLoad() async {
        billing.offeringsError = BillingError.offline
        let model = await openModel()

        #expect(model.failed == .load)
        #expect(model.plans.isEmpty)

        billing.offeringsError = nil
        await model.retry()
        #expect(model.error == nil)
        #expect(model.plans.count == 2)
    }

    @Test func sheetCantBeDismissedWhilePurchasing() async {
        let (gate, open) = AsyncStream<Void>.makeStream()
        billing.purchaseGate = {
            for await _ in gate { break }
        }
        let model = await openModel()

        let purchase = Task { await model.subscribe() }
        while billing.calls.last?.hasPrefix("purchase") != true {
            await Task.yield()
        }
        #expect(model.phase == .purchasing)
        #expect(model.isDismissible == false)

        open.yield()
        open.finish()
        await purchase.value
        #expect(model.phase == .success)
        #expect(model.isDismissible)
    }

    @Test func guestSignsUpFirstThenThePurchaseContinues() async {
        server.userId = nil
        let model = await openModel()
        #expect(model.isGuest)
        model.select("monthly")

        await model.subscribe()

        #expect(model.needsAccount)
        #expect(model.isPresented == false)
        #expect(billing.calls.contains { $0.hasPrefix("purchase") } == false) // guests can't buy (PLAN.md §9)
        model.dismissed()
        model.accountFlowStarted()
        #expect(model.needsAccount == false)

        server.userId = "user-1" // signed up
        server.serverPlans = [.pro]
        await model.accountFlowFinished()

        #expect(model.isPresented)
        #expect(model.isGuest == false)
        #expect(model.selectedId == "monthly")
        #expect(billing.calls.last == "purchase:monthly")
        #expect(model.phase == .success)
    }

    @Test func guestWhoDoesNotSignUpBuysNothing() async {
        server.userId = nil
        let model = await openModel()
        await model.subscribe()
        model.dismissed()
        model.accountFlowStarted()

        await model.accountFlowFinished() // closed sign-up without an account

        #expect(model.isPresented == false)
        #expect(billing.calls.contains { $0.hasPrefix("purchase") } == false)
    }

    @Test func unavailableBuildShowsTheNotice() async {
        let model = await openModel(FakeBillingService(isAvailable: false))

        #expect(model.phase == .unavailable)
        await model.subscribe()
        #expect(model.phase == .unavailable)
    }

    @Test func restoreSucceedsOnlyWhenTheServerSaysPro() async {
        let model = await openModel()

        await model.restore()
        #expect(model.error == PaywallModel.nothingToRestoreMessage)
        #expect(model.failed == .restore)
        #expect(server.refreshes == PaywallModel.configAttempts)

        server.serverPlans = [.pro]
        await model.restore()
        #expect(model.phase == .success)
        #expect(billing.calls.contains("restore"))
    }
}
