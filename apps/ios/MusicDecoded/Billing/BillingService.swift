import Foundation

enum PlanPeriod: Int, Comparable, Sendable {
    case monthly, annual

    static func < (lhs: PlanPeriod, rhs: PlanPeriod) -> Bool { lhs.rawValue < rhs.rawValue }
}

/// A store package with the store's localized price (never a hard-coded price, DESIGN.md §3.11).
struct PlanOption: Equatable, Sendable {
    var id: String
    var period: PlanPeriod
    /// The store's formatted price, e.g. "$4.99".
    var price: String
    /// The same price as a number, only to compute the yearly saving.
    var amount: Decimal
}

enum PurchaseOutcome: Equatable, Sendable {
    case purchased, cancelled
}

/// A store failure with a message that can be shown as is.
struct BillingError: LocalizedError, Equatable, Sendable {
    let message: String

    var errorDescription: String? { message }

    static let unavailable = BillingError(message: "Purchases aren't available in this build.")
    static let generic = BillingError(message: "The purchase didn't go through. Try again.")
    static let offline = BillingError(message: "Couldn't reach the store. Check your connection and try again.")
    static let notAllowed = BillingError(message: "Purchases aren't allowed on this device.")
}

/// The store (PLAN.md §9): RevenueCat in the app, a fake in tests. It never decides Pro: after a
/// purchase or restore the app re-reads `/config`, which the API updates from RevenueCat's webhook.
/// Ports BillingService.kt.
@MainActor
protocol BillingService: AnyObject, Sendable {
    /// `false` when this build has no RevenueCat key.
    var isAvailable: Bool { get }

    /// The store identity follows our account: `appUserID` = our `userId`.
    func logIn(userId: String) async throws

    /// Back to an anonymous store identity (guest).
    func logOut() async throws

    /// The current offering's monthly and annual packages, monthly first.
    func offerings() async throws -> [PlanOption]

    func purchase(_ option: PlanOption) async throws -> PurchaseOutcome

    func restore() async throws
}
