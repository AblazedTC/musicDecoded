package com.musicdecoded.mobile.ui.paywall

import android.app.Activity
import androidx.lifecycle.ViewModel
import androidx.lifecycle.ViewModelProvider
import androidx.lifecycle.viewModelScope
import androidx.lifecycle.viewmodel.initializer
import androidx.lifecycle.viewmodel.viewModelFactory
import com.musicdecoded.mobile.MusicDecodedApplication
import com.musicdecoded.mobile.auth.Session
import com.musicdecoded.mobile.billing.BillingException
import com.musicdecoded.mobile.billing.BillingService
import com.musicdecoded.mobile.billing.PlanOption
import com.musicdecoded.mobile.billing.PlanPeriod
import com.musicdecoded.mobile.billing.PurchaseOutcome
import com.musicdecoded.mobile.billing.RevenueCatBillingService
import com.musicdecoded.mobile.network.ClientConfig
import com.musicdecoded.mobile.network.Kind
import com.musicdecoded.mobile.network.Plan
import kotlinx.coroutines.CancellationException
import kotlinx.coroutines.delay
import kotlinx.coroutines.flow.MutableStateFlow
import kotlinx.coroutines.flow.StateFlow
import kotlinx.coroutines.flow.asStateFlow
import kotlinx.coroutines.flow.update
import kotlinx.coroutines.launch
import kotlin.math.roundToInt

enum class PaywallPhase { LOADING, READY, PURCHASING, SUCCESS, UNAVAILABLE }

/** What Retry re-runs after an error. */
enum class PaywallAction { LOAD, PURCHASE, RESTORE }

/** A plan card: the store's localized price, and the yearly saving against 12 months. */
data class PlanCard(val option: PlanOption, val title: String, val price: String, val saving: String?)

data class PaywallUiState(
    val visible: Boolean = false,
    val phase: PaywallPhase = PaywallPhase.LOADING,
    val plans: List<PlanCard> = emptyList(),
    val selectedId: String? = null,
    val isGuest: Boolean = true,
    /** The free plan's daily limit for the comparison row; `null` when unknown. */
    val freeDailyJobs: Int? = null,
    val error: String? = null,
    val failed: PaywallAction? = null,
    /** One-shot: a guest tapped Subscribe, open sign-up. */
    val needsAccount: Boolean = false,
    /** One-shot: the guest signed up, continue the purchase they started. */
    val resumePurchase: Boolean = false,
) {
    /** The sheet can't be dismissed while the store is busy. */
    val dismissible: Boolean get() = phase != PaywallPhase.PURCHASING
}

/**
 * The Pro paywall (T-602a, DESIGN.md §3.11), one per activity so it survives the sign-up detour.
 * The app never decides Pro: after a purchase or restore it re-fetches `/config` ([refreshConfig]),
 * which the API updates from RevenueCat's webhook, and every screen reads `plan` from there.
 */
class PaywallViewModel(
    private val billing: BillingService,
    private val session: StateFlow<Session?>,
    private val config: StateFlow<ClientConfig?>,
    private val refreshConfig: suspend () -> ClientConfig?,
    /** The webhook may land a moment after the store confirms: re-read `/config` a few times. */
    private val configRetryDelayMs: Long = 2_000L,
) : ViewModel() {
    private val _state = MutableStateFlow(PaywallUiState())
    val state: StateFlow<PaywallUiState> = _state.asStateFlow()

    /** The plan a guest picked before signing up. */
    private var pendingPurchase: String? = null

    init {
        viewModelScope.launch {
            session.collect { s ->
                val guest = s?.kind != Kind.ACCOUNT
                _state.update { it.copy(isGuest = guest) }
                if (!guest && pendingPurchase != null) {
                    _state.update { it.copy(visible = true, selectedId = pendingPurchase ?: it.selectedId, resumePurchase = true) }
                    pendingPurchase = null
                }
            }
        }
    }

    fun open() {
        pendingPurchase = null
        val dailyJobs = config.value?.limits?.dailyJobs
        _state.update { it.copy(visible = true, error = null, failed = null, freeDailyJobs = dailyJobs ?: it.freeDailyJobs) }
        if (_state.value.plans.isEmpty() || _state.value.phase == PaywallPhase.SUCCESS) load()
    }

    fun dismiss() {
        val current = _state.value
        if (!current.dismissible) return
        if (current.phase == PaywallPhase.SUCCESS) viewModelScope.launch { refreshConfig() }
        _state.update {
            it.copy(visible = false, error = null, failed = null, phase = if (it.plans.isEmpty()) PaywallPhase.LOADING else PaywallPhase.READY)
        }
    }

    fun select(id: String) = _state.update { it.copy(selectedId = id) }

    fun load() {
        if (!billing.isAvailable) {
            _state.update { it.copy(phase = PaywallPhase.UNAVAILABLE, plans = emptyList()) }
            return
        }
        _state.update { it.copy(phase = PaywallPhase.LOADING, error = null, failed = null) }
        viewModelScope.launch {
            try {
                val cards = planCards(billing.offerings())
                _state.update {
                    it.copy(
                        phase = PaywallPhase.READY,
                        plans = cards,
                        // Yearly is preselected when the store offers it.
                        selectedId = it.selectedId?.takeIf { id -> cards.any { c -> c.option.id == id } }
                            ?: (cards.firstOrNull { c -> c.option.period == PlanPeriod.ANNUAL } ?: cards.firstOrNull())?.option?.id,
                        error = if (cards.isEmpty()) NO_PLANS else null,
                        failed = if (cards.isEmpty()) PaywallAction.LOAD else null,
                    )
                }
            } catch (e: CancellationException) {
                throw e
            } catch (e: Exception) {
                fail(PaywallAction.LOAD, e, phase = PaywallPhase.READY)
            }
        }
    }

    /** Guests sign up first ([PaywallUiState.needsAccount]); the purchase resumes after. */
    fun subscribe(activity: Activity) {
        val current = _state.value
        if (current.phase != PaywallPhase.READY) return
        val option = current.plans.firstOrNull { it.option.id == current.selectedId }?.option ?: return
        val account = session.value?.takeIf { it.kind == Kind.ACCOUNT }
        if (account == null) {
            pendingPurchase = option.id
            _state.update { it.copy(visible = false, needsAccount = true) }
            return
        }
        _state.update { it.copy(phase = PaywallPhase.PURCHASING, error = null, failed = null) }
        viewModelScope.launch {
            try {
                billing.logIn(account.userId) // the store identity must be this account before buying
                when (billing.purchase(activity, option)) {
                    PurchaseOutcome.PURCHASED -> {
                        awaitPro()
                        _state.update { it.copy(phase = PaywallPhase.SUCCESS) }
                    }
                    PurchaseOutcome.CANCELLED -> _state.update { it.copy(phase = PaywallPhase.READY) }
                }
            } catch (e: CancellationException) {
                throw e
            } catch (e: Exception) {
                fail(PaywallAction.PURCHASE, e)
            }
        }
    }

    fun restore() {
        val current = _state.value
        if (current.phase != PaywallPhase.READY) return
        val account = session.value?.takeIf { it.kind == Kind.ACCOUNT }
        if (account == null) {
            _state.update { it.copy(visible = false, needsAccount = true) }
            return
        }
        _state.update { it.copy(phase = PaywallPhase.PURCHASING, error = null, failed = null) }
        viewModelScope.launch {
            try {
                billing.logIn(account.userId)
                billing.restore()
                if (awaitPro()) {
                    _state.update { it.copy(phase = PaywallPhase.SUCCESS) }
                } else {
                    _state.update { it.copy(phase = PaywallPhase.READY, error = NOTHING_TO_RESTORE, failed = PaywallAction.RESTORE) }
                }
            } catch (e: CancellationException) {
                throw e
            } catch (e: Exception) {
                fail(PaywallAction.RESTORE, e)
            }
        }
    }

    fun accountFlowStarted() = _state.update { it.copy(needsAccount = false) }

    fun resumeHandled() = _state.update { it.copy(resumePurchase = false) }

    /** Re-reads `/config` until it says Pro (or gives up); returns whether it did. */
    private suspend fun awaitPro(): Boolean {
        repeat(CONFIG_ATTEMPTS) { attempt ->
            if (refreshConfig()?.plan == Plan.PRO) return true
            if (attempt < CONFIG_ATTEMPTS - 1) delay(configRetryDelayMs)
        }
        return false
    }

    private fun fail(action: PaywallAction, e: Exception, phase: PaywallPhase = PaywallPhase.READY) {
        val message = (e as? BillingException)?.message ?: RevenueCatBillingService.GENERIC
        _state.update { it.copy(phase = phase, error = message, failed = action) }
    }

    companion object {
        const val CONFIG_ATTEMPTS = 3
        const val NO_PLANS = "No plans are available right now. Try again later."
        const val NOTHING_TO_RESTORE = "We couldn't find a Pro subscription for this account."

        /** Monthly, then yearly with its saving against 12 months of the monthly price. */
        fun planCards(options: List<PlanOption>): List<PlanCard> {
            val monthly = options.firstOrNull { it.period == PlanPeriod.MONTHLY }
            return options.sortedBy { it.period }.map { option ->
                when (option.period) {
                    PlanPeriod.MONTHLY -> PlanCard(option, "Monthly", "${option.price} / month", saving = null)
                    PlanPeriod.ANNUAL -> PlanCard(option, "Yearly", "${option.price} / year", saving = monthly?.let { savingLabel(it, option) })
                }
            }
        }

        private fun savingLabel(monthly: PlanOption, annual: PlanOption): String? {
            val twelveMonths = monthly.priceMicros * 12
            if (twelveMonths <= 0) return null
            val percent = ((1 - annual.priceMicros.toDouble() / twelveMonths) * 100).roundToInt()
            return if (percent > 0) "Save $percent%" else null
        }

        fun factory(app: MusicDecodedApplication): ViewModelProvider.Factory = viewModelFactory {
            initializer {
                PaywallViewModel(
                    billing = app.billingService,
                    session = app.authRepository.session,
                    config = app.authRepository.config,
                    refreshConfig = app.authRepository::refreshConfig,
                )
            }
        }
    }
}
