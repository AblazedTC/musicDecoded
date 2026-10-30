package com.musicdecoded.mobile.ui.paywall

import android.app.Application
import androidx.compose.runtime.Composable
import androidx.compose.runtime.CompositionLocalProvider
import androidx.compose.ui.test.assertIsNotEnabled
import androidx.compose.ui.test.junit4.createComposeRule
import androidx.compose.ui.test.onNodeWithText
import androidx.compose.ui.test.performClick
import androidx.compose.ui.test.performScrollTo
import com.musicdecoded.mobile.billing.FakeBillingService
import com.musicdecoded.mobile.billing.RevenueCatBillingService
import com.musicdecoded.mobile.library.TranscriptionRepository
import com.musicdecoded.mobile.network.ApiError
import com.musicdecoded.mobile.network.ApiException
import com.musicdecoded.mobile.sources.youtube.SearchError
import com.musicdecoded.mobile.ui.components.ErrorState
import com.musicdecoded.mobile.ui.theme.MusicDecodedTheme
import org.junit.Assert.assertEquals
import org.junit.Assert.assertFalse
import org.junit.Assert.assertTrue
import org.junit.Rule
import org.junit.Test
import org.junit.runner.RunWith
import org.robolectric.RobolectricTestRunner
import org.robolectric.annotation.Config

/** Where the paywall opens from (DESIGN.md §3.11) and the sheet's guest/unavailable copy. */
@RunWith(RobolectricTestRunner::class)
@Config(application = Application::class)
class PaywallEntryTest {
    @get:Rule val compose = createComposeRule()

    private var opened = 0

    private fun setContent(openPaywall: (() -> Unit)? = { opened++ }, content: @Composable () -> Unit) = compose.setContent {
        MusicDecodedTheme { CompositionLocalProvider(LocalOpenPaywall provides openPaywall, content = content) }
    }

    private val dailyLimit = ApiException(ApiError("rate_limited", "Daily limit reached"), 429)

    @Test fun dailyLimitFromJobsOffersGoPro() {
        val message = TranscriptionRepository.userMessage(dailyLimit)
        assertEquals(TranscriptionRepository.RATE_LIMITED, message)
        assertTrue(opensPaywall(message))
        setContent { ErrorState(message) }
        compose.onNodeWithText("Daily limit reached").assertExists()
        compose.onNodeWithText("Go Pro").performClick()
        assertEquals(1, opened)
    }

    @Test fun youTubeSearchQuotaNeverOpensThePaywall() {
        val error = SearchError.from(dailyLimit)
        assertEquals(SearchError.QUOTA, error)
        assertFalse(opensPaywall(error.message))
        setContent { ErrorState(error.message, title = error.title, onRetry = {}) }
        compose.onNodeWithText("Go Pro").assertDoesNotExist()
    }

    @Test fun proUsersSeeNoUpsell() {
        setContent(openPaywall = null) { ErrorState(TranscriptionRepository.RATE_LIMITED) }
        compose.onNodeWithText("Go Pro").assertDoesNotExist()
    }

    @Test fun otherErrorsHaveNoGoPro() {
        setContent { ErrorState(TranscriptionRepository.OFFLINE) }
        compose.onNodeWithText("Go Pro").assertDoesNotExist()
    }

    @Test fun guestsAreAskedToCreateAnAccount() {
        val cards = PaywallViewModel.planCards(FakeBillingService().plans)
        setContent {
            PaywallContent(
                PaywallUiState(visible = true, phase = PaywallPhase.READY, plans = cards, selectedId = "annual", isGuest = true),
                onSelect = {}, onSubscribe = { opened++ }, onRestore = {}, onRetry = {}, onDone = {},
            )
        }
        compose.onNodeWithText("Save 33%").assertExists()
        compose.onNodeWithText("Create an account to subscribe").performScrollTo().performClick()
        assertEquals(1, opened)
    }

    @Test fun unavailableBuildSaysSo() {
        setContent {
            PaywallContent(
                PaywallUiState(visible = true, phase = PaywallPhase.UNAVAILABLE, isGuest = false),
                onSelect = {}, onSubscribe = {}, onRestore = {}, onRetry = {}, onDone = {},
            )
        }
        compose.onNodeWithText(RevenueCatBillingService.UNAVAILABLE).assertExists()
        compose.onNodeWithText("Subscribe").assertDoesNotExist()
    }

    @Test fun loadingDisablesSubscribe() {
        setContent {
            PaywallContent(
                PaywallUiState(visible = true, phase = PaywallPhase.LOADING, isGuest = false),
                onSelect = {}, onSubscribe = {}, onRestore = {}, onRetry = {}, onDone = {},
            )
        }
        compose.onNodeWithText("Subscribe").assertIsNotEnabled()
    }
}
