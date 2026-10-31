package com.musicdecoded.mobile.billing

import android.app.Activity
import android.content.Context
import com.revenuecat.purchases.LogLevel
import com.revenuecat.purchases.Package
import com.revenuecat.purchases.PackageType
import com.revenuecat.purchases.PurchaseParams
import com.revenuecat.purchases.Purchases
import com.revenuecat.purchases.PurchasesConfiguration
import com.revenuecat.purchases.PurchasesErrorCode
import com.revenuecat.purchases.PurchasesException
import com.revenuecat.purchases.PurchasesTransactionException
import com.revenuecat.purchases.awaitLogIn
import com.revenuecat.purchases.awaitLogOut
import com.revenuecat.purchases.awaitOfferings
import com.revenuecat.purchases.awaitPurchase
import com.revenuecat.purchases.awaitRestore
import kotlinx.coroutines.sync.Mutex
import kotlinx.coroutines.sync.withLock

/**
 * [BillingService] on RevenueCat (Play Billing). An empty [apiKey] means purchases are unavailable
 * in this build. The SDK is configured on first use, with `appUserID` = our userId for accounts.
 */
class RevenueCatBillingService(
    private val context: Context,
    private val apiKey: String,
    private val debugLogs: Boolean,
) : BillingService {
    override val isAvailable: Boolean = apiKey.isNotBlank()

    private val mutex = Mutex()
    private val sdk: Purchases get() = Purchases.sharedInstance

    /** The packages behind the last [offerings], by id (a [PlanOption] carries no SDK types). */
    private var packages: Map<String, Package> = emptyMap()

    override suspend fun logIn(userId: String) = store {
        if (!configure(userId) && sdk.appUserID != userId) sdk.awaitLogIn(userId)
    }

    override suspend fun logOut() = store {
        if (!configure(null) && !sdk.isAnonymous) sdk.awaitLogOut()
    }

    override suspend fun offerings(): List<PlanOption> = store {
        configure(null)
        val offering = sdk.awaitOfferings().current ?: return@store emptyList()
        val found = listOfNotNull(offering.monthly, offering.annual)
        packages = found.associateBy(Package::identifier)
        found.map { pkg ->
            PlanOption(
                id = pkg.identifier,
                period = if (pkg.packageType == PackageType.ANNUAL) PlanPeriod.ANNUAL else PlanPeriod.MONTHLY,
                price = pkg.product.price.formatted,
                priceMicros = pkg.product.price.amountMicros,
            )
        }
    }

    override suspend fun purchase(activity: Activity, option: PlanOption): PurchaseOutcome = store {
        val pkg = packages[option.id] ?: throw BillingException(GENERIC)
        try {
            sdk.awaitPurchase(PurchaseParams.Builder(activity, pkg).build())
            PurchaseOutcome.PURCHASED
        } catch (e: PurchasesTransactionException) {
            if (e.userCancelled) PurchaseOutcome.CANCELLED else throw e
        }
    }

    override suspend fun restore() = store {
        configure(null)
        sdk.awaitRestore()
        Unit
    }

    /** Runs [block] on the configured SDK, turning store errors into [BillingException]s. */
    private suspend fun <T> store(block: suspend () -> T): T {
        if (!isAvailable) throw BillingException(UNAVAILABLE)
        return mutex.withLock {
            try {
                block()
            } catch (e: PurchasesException) {
                throw BillingException(messageFor(e.code), e)
            }
        }
    }

    /** Configures the SDK once; returns `true` if it just did (already as [appUserId]). */
    private fun configure(appUserId: String?): Boolean {
        if (Purchases.isConfigured) return false
        Purchases.logLevel = if (debugLogs) LogLevel.DEBUG else LogLevel.WARN
        Purchases.configure(PurchasesConfiguration.Builder(context, apiKey).appUserID(appUserId).build())
        return true
    }

    companion object {
        const val UNAVAILABLE = "Purchases aren't available in this build."
        const val GENERIC = "The purchase didn't go through. Try again."
        const val OFFLINE = "Couldn't reach the store. Check your connection and try again."
        const val NOT_ALLOWED = "Purchases aren't allowed on this device."

        fun messageFor(code: PurchasesErrorCode): String = when (code) {
            PurchasesErrorCode.NetworkError -> OFFLINE
            PurchasesErrorCode.PurchaseNotAllowedError, PurchasesErrorCode.InsufficientPermissionsError -> NOT_ALLOWED
            else -> GENERIC
        }
    }
}
