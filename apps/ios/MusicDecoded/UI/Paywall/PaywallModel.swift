import Foundation
import Observation

/// A plan card: the store's localized price, and the yearly saving against 12 months.
struct PlanCard: Equatable, Sendable {
    var option: PlanOption
    var title: String
    var price: String
    var saving: String?
}

/// The Pro paywall (T-602i, DESIGN.md §3.11), one for the whole app so it survives the sign-up
/// detour. The app never decides Pro: after a purchase or restore it re-fetches `/config`
/// (`refreshConfig`), which the API updates from RevenueCat's webhook, and every screen reads
/// `plan` from there. Ports PaywallViewModel.kt.
@MainActor
@Observable
final class PaywallModel {
    enum Phase: Equatable, Sendable {
        case loading, ready, purchasing, success, unavailable
    }

    /// What Retry re-runs after an error.
    enum Action: Equatable, Sendable {
        case load, purchase, restore
    }

    static let configAttempts = 3
    static let noPlansMessage = "No plans are available right now. Try again later."
    static let nothingToRestoreMessage = "We couldn't find a Pro subscription for this account."

    var isPresented = false
    private(set) var phase = Phase.loading
    private(set) var plans: [PlanCard] = []
    var selectedId: String?
    /// The free plan's daily limit for the comparison row; `nil` when unknown.
    private(set) var freeDailyJobs: Int?
    private(set) var error: String?
    private(set) var failed: Action?
    /// A guest tapped Subscribe: the app opens sign-up (one-shot, reset by `accountFlowStarted`).
    private(set) var needsAccount = false

    /// The sheet can't be dismissed while the store is busy.
    var isDismissible: Bool { phase != .purchasing }

    var isGuest: Bool { accountUserId() == nil }

    private let billing: any BillingService
    /// Our userId when signed in with an account, otherwise `nil`.
    private let accountUserId: @MainActor () -> String?
    private let config: @MainActor () -> ClientConfig?
    private let refreshConfig: @MainActor () async -> ClientConfig?
    /// The webhook may land a moment after the store confirms: re-read `/config` a few times.
    private let configRetryDelay: Duration

    /// The plan a guest picked before signing up.
    private var pendingPurchase: String?

    init(
        billing: any BillingService,
        accountUserId: @escaping @MainActor () -> String?,
        config: @escaping @MainActor () -> ClientConfig?,
        refreshConfig: @escaping @MainActor () async -> ClientConfig?,
        configRetryDelay: Duration = .seconds(2)
    ) {
        self.billing = billing
        self.accountUserId = accountUserId
        self.config = config
        self.refreshConfig = refreshConfig
        self.configRetryDelay = configRetryDelay
    }

    func open() {
        pendingPurchase = nil
        if let dailyJobs = config()?.limits.dailyJobs {
            freeDailyJobs = dailyJobs
        }
        error = nil
        failed = nil
        isPresented = true
    }

    /// Run by the sheet when it appears: fetches the plans the first time (or after a failure).
    func loadIfNeeded() async {
        if plans.isEmpty {
            await load()
        }
    }

    /// Called when the sheet goes away (Done, swipe down). After a success the app re-reads `/config`.
    func dismissed() {
        if phase == .success {
            Task { await refreshConfig() }
        }
        isPresented = false
        error = nil
        failed = nil
        phase = plans.isEmpty ? .loading : .ready
    }

    func select(_ id: String) {
        selectedId = id
    }

    func load() async {
        guard billing.isAvailable else {
            phase = .unavailable
            plans = []
            return
        }
        phase = .loading
        error = nil
        failed = nil
        do {
            let cards = try await Self.planCards(billing.offerings())
            plans = cards
            phase = .ready
            // Keep the user's choice; otherwise yearly is preselected when the store offers it.
            let current = selectedId
            if cards.contains(where: { $0.option.id == current }) == false {
                selectedId = (cards.first { $0.option.period == .annual } ?? cards.first)?.option.id
            }
            error = cards.isEmpty ? Self.noPlansMessage : nil
            failed = cards.isEmpty ? .load : nil
        } catch is CancellationError {
            return
        } catch {
            fail(.load, error)
        }
    }

    /// Guests sign up first (`needsAccount`); the purchase resumes after (`accountFlowFinished`).
    func subscribe() async {
        guard phase == .ready,
              let option = plans.first(where: { $0.option.id == selectedId })?.option
        else { return }
        guard let userId = accountUserId() else {
            pendingPurchase = option.id
            isPresented = false
            needsAccount = true
            return
        }
        phase = .purchasing
        error = nil
        failed = nil
        do {
            try await billing.logIn(userId: userId) // the store identity must be this account before buying
            switch try await billing.purchase(option) {
            case .purchased:
                _ = await awaitPro()
                phase = .success
            case .cancelled:
                phase = .ready
            }
        } catch {
            fail(.purchase, error)
        }
    }

    /// Restore succeeds only when `/config` then says Pro.
    func restore() async {
        guard phase == .ready else { return }
        guard let userId = accountUserId() else {
            isPresented = false
            needsAccount = true
            return
        }
        phase = .purchasing
        error = nil
        failed = nil
        do {
            try await billing.logIn(userId: userId)
            try await billing.restore()
            if await awaitPro() {
                phase = .success
            } else {
                phase = .ready
                error = Self.nothingToRestoreMessage
                failed = .restore
            }
        } catch {
            fail(.restore, error)
        }
    }

    /// Re-runs the action that failed.
    func retry() async {
        switch failed {
        case .purchase: await subscribe()
        case .restore: await restore()
        case .load, nil: await load()
        }
    }

    func accountFlowStarted() {
        needsAccount = false
    }

    /// The sign-up / log-in flow closed. A guest who now has an account continues the purchase they
    /// started; otherwise the pending purchase is dropped.
    func accountFlowFinished() async {
        guard let pending = pendingPurchase else { return }
        pendingPurchase = nil
        guard accountUserId() != nil else { return }
        selectedId = pending
        isPresented = true
        await subscribe()
    }

    /// Re-reads `/config` until it says Pro (or gives up); returns whether it did.
    private func awaitPro() async -> Bool {
        for attempt in 0..<Self.configAttempts {
            if await refreshConfig()?.plan == .pro { return true }
            if attempt < Self.configAttempts - 1 {
                try? await Task.sleep(for: configRetryDelay)
            }
        }
        return false
    }

    private func fail(_ action: Action, _ error: any Error) {
        self.error = (error as? BillingError)?.message ?? BillingError.generic.message
        failed = action
        phase = .ready
    }

    /// Monthly, then yearly with its saving against 12 months of the monthly price.
    static func planCards(_ options: [PlanOption]) -> [PlanCard] {
        let monthly = options.first { $0.period == .monthly }
        return options.sorted { $0.period < $1.period }.map { option -> PlanCard in
            switch option.period {
            case .monthly:
                PlanCard(option: option, title: "Monthly", price: "\(option.price) / month", saving: nil)
            case .annual:
                PlanCard(
                    option: option,
                    title: "Yearly",
                    price: "\(option.price) / year",
                    saving: monthly.flatMap { savingLabel(monthly: $0, annual: option) }
                )
            }
        }
    }

    private static func savingLabel(monthly: PlanOption, annual: PlanOption) -> String? {
        let twelveMonths = NSDecimalNumber(decimal: monthly.amount * 12).doubleValue
        guard twelveMonths > 0 else { return nil }
        let annualPrice = NSDecimalNumber(decimal: annual.amount).doubleValue
        let percent = Int(((1 - annualPrice / twelveMonths) * 100).rounded())
        return percent > 0 ? "Save \(percent)%" : nil
    }
}

#if DEBUG
extension PaywallModel {
    /// For previews: no RevenueCat key, so it shows "Purchases aren't available in this build".
    static func preview() -> PaywallModel {
        PaywallModel(
            billing: RevenueCatBillingService(apiKey: ""),
            accountUserId: { nil },
            config: { nil },
            refreshConfig: { nil }
        )
    }
}
#endif
