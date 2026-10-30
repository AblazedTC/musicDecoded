package com.musicdecoded.mobile.billing

import android.app.Activity

enum class PlanPeriod { MONTHLY, ANNUAL }

/** A store package with the store's localized price (never a hard-coded price, DESIGN.md §3.11). */
data class PlanOption(
    val id: String,
    val period: PlanPeriod,
    /** The store's formatted price, e.g. "$4.99". */
    val price: String,
    /** The same price in micros, only to compute the yearly saving. */
    val priceMicros: Long,
)

enum class PurchaseOutcome { PURCHASED, CANCELLED }

/** A store failure with a message that can be shown as is. */
class BillingException(message: String, cause: Throwable? = null) : Exception(message, cause)

/**
 * The store (PLAN.md §9): RevenueCat in the app, a fake in tests. It never decides Pro: after a
 * purchase or restore the app re-reads `/config`, which the API updates from RevenueCat's webhook.
 */
interface BillingService {
    /** `false` when this build has no RevenueCat key. */
    val isAvailable: Boolean

    /** The store identity follows our account: `appUserID` = our `userId`. */
    suspend fun logIn(userId: String)

    /** Back to an anonymous store identity (guest). */
    suspend fun logOut()

    /** The current offering's monthly and annual packages, monthly first. */
    suspend fun offerings(): List<PlanOption>

    suspend fun purchase(activity: Activity, option: PlanOption): PurchaseOutcome

    suspend fun restore()
}
