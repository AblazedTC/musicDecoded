import Foundation
@preconcurrency import RevenueCat

/// `BillingService` on RevenueCat (StoreKit 2). An empty key means purchases are unavailable in
/// this build. The SDK is configured on first use, with `appUserID` = our userId for accounts.
/// Ports RevenueCatBillingService.kt.
@MainActor
final class RevenueCatBillingService: BillingService {
    let isAvailable: Bool
    private let apiKey: String

    /// The packages behind the last `offerings()`, by id (a `PlanOption` carries no SDK types).
    private var packages: [String: Package] = [:]

    init(apiKey: String) {
        self.apiKey = apiKey.trimmingCharacters(in: .whitespacesAndNewlines)
        isAvailable = self.apiKey.isEmpty == false
    }

    /// Reads `RevenueCatAPIKey` from Info.plist (`REVENUECAT_API_KEY` in Config/*.xcconfig).
    static func configured(bundle: Bundle = .main) -> RevenueCatBillingService {
        RevenueCatBillingService(apiKey: bundle.object(forInfoDictionaryKey: "RevenueCatAPIKey") as? String ?? "")
    }

    func logIn(userId: String) async throws {
        try check()
        if configure(appUserID: userId) { return }
        guard Purchases.shared.appUserID != userId else { return }
        do {
            _ = try await Purchases.shared.logIn(userId)
        } catch {
            throw Self.billingError(for: error)
        }
    }

    func logOut() async throws {
        try check()
        if configure(appUserID: nil) { return }
        guard Purchases.shared.isAnonymous == false else { return }
        do {
            _ = try await Purchases.shared.logOut()
        } catch {
            throw Self.billingError(for: error)
        }
    }

    func offerings() async throws -> [PlanOption] {
        try check()
        configure(appUserID: nil)
        let offerings: Offerings
        do {
            offerings = try await Purchases.shared.offerings()
        } catch {
            throw Self.billingError(for: error)
        }
        guard let current = offerings.current else { return [] }
        let found = [current.monthly, current.annual].compactMap { $0 }
        packages = Dictionary(found.map { ($0.identifier, $0) }, uniquingKeysWith: { first, _ in first })
        return found.map { package in
            PlanOption(
                id: package.identifier,
                period: package.packageType == .annual ? .annual : .monthly,
                price: package.storeProduct.localizedPriceString,
                amount: package.storeProduct.price
            )
        }
    }

    func purchase(_ option: PlanOption) async throws -> PurchaseOutcome {
        try check()
        guard let package = packages[option.id] else { throw BillingError.generic }
        do {
            let result = try await Purchases.shared.purchase(package: package)
            return result.userCancelled ? .cancelled : .purchased
        } catch {
            if let code = error as? RevenueCat.ErrorCode, code == .purchaseCancelledError { return .cancelled }
            throw Self.billingError(for: error)
        }
    }

    func restore() async throws {
        try check()
        configure(appUserID: nil)
        do {
            _ = try await Purchases.shared.restorePurchases()
        } catch {
            throw Self.billingError(for: error)
        }
    }

    private func check() throws {
        guard isAvailable else { throw BillingError.unavailable }
    }

    /// Configures the SDK once; returns `true` if it just did (already as `appUserID`).
    @discardableResult
    private func configure(appUserID: String?) -> Bool {
        guard Purchases.isConfigured == false else { return false }
        #if DEBUG
        Purchases.logLevel = .debug
        #else
        Purchases.logLevel = .warn
        #endif
        Purchases.configure(withAPIKey: apiKey, appUserID: appUserID)
        return true
    }

    static func billingError(for error: any Error) -> BillingError {
        if let error = error as? BillingError { return error }
        if let code = error as? RevenueCat.ErrorCode {
            switch code {
            case .networkError, .offlineConnectionError: return .offline
            case .purchaseNotAllowedError: return .notAllowed
            default: return .generic
            }
        }
        if error is URLError { return .offline }
        return .generic
    }
}
