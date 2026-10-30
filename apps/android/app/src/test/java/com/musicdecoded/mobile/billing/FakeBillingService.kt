package com.musicdecoded.mobile.billing

import android.app.Activity
import kotlinx.coroutines.CompletableDeferred

/** In-memory [BillingService] for tests: scripted offerings and purchase outcomes, recorded calls. */
class FakeBillingService(override val isAvailable: Boolean = true) : BillingService {
    var plans: List<PlanOption> = listOf(MONTHLY, ANNUAL)
    var offeringsError: Exception? = null
    var purchaseResult: () -> PurchaseOutcome = { PurchaseOutcome.PURCHASED }
    var restoreError: Exception? = null

    /** When set, [purchase] waits for it (to observe the in-progress state). */
    var purchaseGate: CompletableDeferred<Unit>? = null
    val calls = mutableListOf<String>()

    override suspend fun logIn(userId: String) {
        calls += "logIn:$userId"
    }

    override suspend fun logOut() {
        calls += "logOut"
    }

    override suspend fun offerings(): List<PlanOption> {
        calls += "offerings"
        offeringsError?.let { throw it }
        return plans
    }

    override suspend fun purchase(activity: Activity, option: PlanOption): PurchaseOutcome {
        calls += "purchase:${option.id}"
        purchaseGate?.await()
        return purchaseResult()
    }

    override suspend fun restore() {
        calls += "restore"
        restoreError?.let { throw it }
    }

    companion object {
        val MONTHLY = PlanOption("monthly", PlanPeriod.MONTHLY, "$4.99", 4_990_000)
        val ANNUAL = PlanOption("annual", PlanPeriod.ANNUAL, "$39.99", 39_990_000)
    }
}
