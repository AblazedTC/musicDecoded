import Foundation
@testable import MusicDecoded

/// In-memory `BillingService` for tests: scripted offerings and purchase outcomes, recorded calls.
/// Ports FakeBillingService.kt.
@MainActor
final class FakeBillingService: BillingService {
    nonisolated static let monthly = PlanOption(id: "monthly", period: .monthly, price: "$4.99", amount: Decimal(string: "4.99")!)
    nonisolated static let annual = PlanOption(id: "annual", period: .annual, price: "$39.99", amount: Decimal(string: "39.99")!)

    let isAvailable: Bool
    var plans = [FakeBillingService.monthly, FakeBillingService.annual]
    var offeringsError: (any Error)?
    var purchaseResult: @MainActor () throws -> PurchaseOutcome = { .purchased }
    var restoreError: (any Error)?
    /// When set, `purchase` awaits it first (to observe the in-progress state).
    var purchaseGate: (@MainActor () async -> Void)?
    private(set) var calls: [String] = []

    init(isAvailable: Bool = true) {
        self.isAvailable = isAvailable
    }

    func logIn(userId: String) async throws {
        calls.append("logIn:\(userId)")
    }

    func logOut() async throws {
        calls.append("logOut")
    }

    func offerings() async throws -> [PlanOption] {
        calls.append("offerings")
        if let offeringsError { throw offeringsError }
        return plans
    }

    func purchase(_ option: PlanOption) async throws -> PurchaseOutcome {
        calls.append("purchase:\(option.id)")
        if let purchaseGate { await purchaseGate() }
        return try purchaseResult()
    }

    func restore() async throws {
        calls.append("restore")
        if let restoreError { throw restoreError }
    }
}
