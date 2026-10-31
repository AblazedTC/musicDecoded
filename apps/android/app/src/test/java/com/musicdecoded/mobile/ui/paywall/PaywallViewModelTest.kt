package com.musicdecoded.mobile.ui.paywall

import android.app.Activity
import android.app.Application
import com.musicdecoded.mobile.auth.Session
import com.musicdecoded.mobile.billing.BillingException
import com.musicdecoded.mobile.billing.FakeBillingService
import com.musicdecoded.mobile.billing.PlanPeriod
import com.musicdecoded.mobile.billing.PurchaseOutcome
import com.musicdecoded.mobile.billing.RevenueCatBillingService
import com.musicdecoded.mobile.network.ClientConfig
import com.musicdecoded.mobile.network.Kind
import com.musicdecoded.mobile.network.Plan
import kotlinx.coroutines.CompletableDeferred
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.ExperimentalCoroutinesApi
import kotlinx.coroutines.flow.MutableStateFlow
import kotlinx.coroutines.test.UnconfinedTestDispatcher
import kotlinx.coroutines.test.resetMain
import kotlinx.coroutines.test.setMain
import org.junit.After
import org.junit.Assert.assertEquals
import org.junit.Assert.assertFalse
import org.junit.Assert.assertNull
import org.junit.Assert.assertTrue
import org.junit.Before
import org.junit.Test
import org.junit.runner.RunWith
import org.robolectric.Robolectric
import org.robolectric.RobolectricTestRunner
import org.robolectric.annotation.Config

@OptIn(ExperimentalCoroutinesApi::class)
@RunWith(RobolectricTestRunner::class)
@Config(application = Application::class)
class PaywallViewModelTest {
    private val guest = Session("g", "guest-1", Kind.GUEST, Plan.FREE)
    private val account = Session("a", "user-1", Kind.ACCOUNT, Plan.FREE, email = "a@b.co")
    private val session = MutableStateFlow<Session?>(account)
    private val config = MutableStateFlow<ClientConfig?>(config(Plan.FREE))
    private val billing = FakeBillingService()

    /** What the next `/config` answers (the API flips it after RevenueCat's webhook). */
    private var serverPlan = Plan.FREE
    private var configRefreshes = 0
    private lateinit var activity: Activity

    @Before fun setUp() {
        Dispatchers.setMain(UnconfinedTestDispatcher())
        activity = Robolectric.buildActivity(Activity::class.java).get()
    }

    @After fun tearDown() = Dispatchers.resetMain()

    private fun config(plan: Plan) = ClientConfig(
        offlineStreamAudio = false,
        maxUploadBytes = 1,
        plan = plan,
        limits = ClientConfig.Limits(dailyJobs = if (plan == Plan.PRO) null else 50, jobsUsedToday = 50),
    )

    private fun viewModel(service: FakeBillingService = billing) = PaywallViewModel(
        billing = service,
        session = session,
        config = config,
        refreshConfig = { configRefreshes++; config(serverPlan).also { config.value = it } },
        configRetryDelayMs = 0,
    )

    @Test fun offeringsBecomeCardsWithYearlySavingPreselected() {
        val vm = viewModel().apply { open() }
        val state = vm.state.value
        assertTrue(state.visible)
        assertEquals(PaywallPhase.READY, state.phase)
        assertEquals(listOf("Monthly", "Yearly"), state.plans.map { it.title })
        assertEquals("$4.99 / month", state.plans[0].price)
        assertEquals("$39.99 / year", state.plans[1].price)
        assertNull(state.plans[0].saving)
        assertEquals("Save 33%", state.plans[1].saving) // 39.99 vs 12 × 4.99 = 59.88
        assertEquals("annual", state.selectedId)
        assertEquals(50, state.freeDailyJobs)
    }

    @Test fun noMonthlyMeansNoSavingLabel() {
        val cards = PaywallViewModel.planCards(listOf(FakeBillingService.ANNUAL))
        assertEquals(PlanPeriod.ANNUAL, cards.single().option.period)
        assertNull(cards.single().saving)
    }

    @Test fun purchaseSuccessRefreshesConfigAndShowsSuccess() {
        val vm = viewModel().apply { open() }
        vm.select("monthly")
        serverPlan = Plan.PRO
        vm.subscribe(activity)
        assertEquals(listOf("offerings", "logIn:user-1", "purchase:monthly"), billing.calls)
        assertEquals(PaywallPhase.SUCCESS, vm.state.value.phase)
        assertEquals(1, configRefreshes)
        assertEquals(Plan.PRO, config.value?.plan) // every screen reads the plan from /config
        vm.dismiss()
        assertFalse(vm.state.value.visible)
    }

    @Test fun purchaseRetriesConfigUntilTheWebhookLands() {
        var answers = listOf(Plan.FREE, Plan.FREE, Plan.PRO)
        val vm = PaywallViewModel(billing, session, config, refreshConfig = {
            configRefreshes++
            config(answers.first()).also { answers = answers.drop(1) }
        }, configRetryDelayMs = 0).apply { open() }
        vm.subscribe(activity)
        assertEquals(3, configRefreshes)
        assertEquals(PaywallPhase.SUCCESS, vm.state.value.phase)
    }

    @Test fun cancelledPurchaseIsSilent() {
        billing.purchaseResult = { PurchaseOutcome.CANCELLED }
        val vm = viewModel().apply { open() }
        vm.subscribe(activity)
        val state = vm.state.value
        assertEquals(PaywallPhase.READY, state.phase)
        assertNull(state.error)
        assertEquals(0, configRefreshes)
    }

    @Test fun purchaseErrorShowsMessageAndRetryWorks() {
        billing.purchaseResult = { throw BillingException(RevenueCatBillingService.OFFLINE) }
        val vm = viewModel().apply { open() }
        vm.subscribe(activity)
        assertEquals(RevenueCatBillingService.OFFLINE, vm.state.value.error)
        assertEquals(PaywallAction.PURCHASE, vm.state.value.failed)
        assertEquals(PaywallPhase.READY, vm.state.value.phase)

        billing.purchaseResult = { PurchaseOutcome.PURCHASED }
        serverPlan = Plan.PRO
        vm.subscribe(activity) // Retry re-runs the failed action
        assertEquals(PaywallPhase.SUCCESS, vm.state.value.phase)
        assertNull(vm.state.value.error)
    }

    @Test fun offeringsErrorRetriesLoad() {
        billing.offeringsError = BillingException(RevenueCatBillingService.OFFLINE)
        val vm = viewModel().apply { open() }
        assertEquals(PaywallAction.LOAD, vm.state.value.failed)
        assertTrue(vm.state.value.plans.isEmpty())
        billing.offeringsError = null
        vm.load()
        assertNull(vm.state.value.error)
        assertEquals(2, vm.state.value.plans.size)
    }

    @Test fun sheetCantBeDismissedWhilePurchasing() {
        val gate = CompletableDeferred<Unit>()
        billing.purchaseGate = gate
        val vm = viewModel().apply { open() }
        vm.subscribe(activity)
        assertEquals(PaywallPhase.PURCHASING, vm.state.value.phase)
        vm.dismiss()
        assertTrue(vm.state.value.visible)
        gate.complete(Unit)
        assertEquals(PaywallPhase.SUCCESS, vm.state.value.phase)
    }

    @Test fun guestSignsUpFirstThenThePurchaseContinues() {
        session.value = guest
        val vm = viewModel().apply { open() }
        assertTrue(vm.state.value.isGuest)
        vm.select("monthly")
        vm.subscribe(activity)
        assertTrue(vm.state.value.needsAccount)
        assertFalse(vm.state.value.visible)
        assertFalse(billing.calls.any { it.startsWith("purchase") }) // guests can't buy (PLAN.md §9)
        vm.accountFlowStarted()
        assertFalse(vm.state.value.needsAccount)

        session.value = account // signed up
        val state = vm.state.value
        assertTrue(state.visible)
        assertTrue(state.resumePurchase)
        assertFalse(state.isGuest)
        assertEquals("monthly", state.selectedId)
        vm.resumeHandled()
        vm.subscribe(activity)
        assertEquals("purchase:monthly", billing.calls.last())
    }

    @Test fun unavailableBuildShowsTheNotice() {
        val vm = viewModel(FakeBillingService(isAvailable = false)).apply { open() }
        assertEquals(PaywallPhase.UNAVAILABLE, vm.state.value.phase)
        vm.subscribe(activity)
        assertEquals(PaywallPhase.UNAVAILABLE, vm.state.value.phase)
    }

    @Test fun restoreSucceedsOnlyWhenTheServerSaysPro() {
        val vm = viewModel().apply { open() }
        vm.restore()
        assertEquals(PaywallViewModel.NOTHING_TO_RESTORE, vm.state.value.error)
        assertEquals(PaywallAction.RESTORE, vm.state.value.failed)

        serverPlan = Plan.PRO
        vm.restore()
        assertEquals(PaywallPhase.SUCCESS, vm.state.value.phase)
        assertTrue("restore" in billing.calls)
    }
}
