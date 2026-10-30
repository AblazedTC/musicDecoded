package com.musicdecoded.mobile.ui.paywall

import android.app.Activity
import android.content.Context
import android.content.ContextWrapper
import androidx.compose.foundation.background
import androidx.compose.foundation.border
import androidx.compose.foundation.layout.Arrangement
import androidx.compose.foundation.layout.Box
import androidx.compose.foundation.layout.Column
import androidx.compose.foundation.layout.Row
import androidx.compose.foundation.layout.fillMaxWidth
import androidx.compose.foundation.layout.height
import androidx.compose.foundation.layout.navigationBarsPadding
import androidx.compose.foundation.layout.padding
import androidx.compose.foundation.layout.size
import androidx.compose.foundation.rememberScrollState
import androidx.compose.foundation.selection.selectable
import androidx.compose.foundation.selection.selectableGroup
import androidx.compose.foundation.shape.CircleShape
import androidx.compose.foundation.verticalScroll
import androidx.compose.material.icons.Icons
import androidx.compose.material.icons.rounded.AutoAwesome
import androidx.compose.material.icons.rounded.CheckCircle
import androidx.compose.material.icons.rounded.ErrorOutline
import androidx.compose.material.icons.rounded.Refresh
import androidx.compose.material3.ExperimentalMaterial3Api
import androidx.compose.material3.Icon
import androidx.compose.material3.MaterialTheme
import androidx.compose.material3.ModalBottomSheet
import androidx.compose.material3.ModalBottomSheetProperties
import androidx.compose.material3.SheetValue
import androidx.compose.material3.Text
import androidx.compose.material3.TextButton
import androidx.compose.material3.rememberModalBottomSheetState
import androidx.compose.runtime.Composable
import androidx.compose.runtime.LaunchedEffect
import androidx.compose.runtime.getValue
import androidx.compose.runtime.rememberUpdatedState
import androidx.compose.runtime.staticCompositionLocalOf
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.platform.LocalContext
import androidx.compose.ui.semantics.LiveRegionMode
import androidx.compose.ui.semantics.Role
import androidx.compose.ui.semantics.heading
import androidx.compose.ui.semantics.liveRegion
import androidx.compose.ui.semantics.semantics
import androidx.compose.ui.text.style.TextAlign
import androidx.compose.ui.tooling.preview.Preview
import androidx.lifecycle.compose.collectAsStateWithLifecycle
import com.musicdecoded.mobile.billing.PlanOption
import com.musicdecoded.mobile.library.TranscriptionRepository
import com.musicdecoded.mobile.billing.PlanPeriod
import com.musicdecoded.mobile.billing.RevenueCatBillingService
import com.musicdecoded.mobile.ui.components.PreviewSurface
import com.musicdecoded.mobile.ui.components.PrimaryButton
import com.musicdecoded.mobile.ui.components.SecondaryButton
import com.musicdecoded.mobile.ui.theme.MusicDecodedColors
import com.musicdecoded.mobile.ui.theme.Shapes
import com.musicdecoded.mobile.ui.theme.Sizes
import com.musicdecoded.mobile.ui.theme.Spacing
import com.musicdecoded.mobile.ui.theme.TextStyles

/**
 * Opens the paywall from anywhere below MainScreen; `null` when the user is already Pro (no upsell).
 */
val LocalOpenPaywall = staticCompositionLocalOf<(() -> Unit)?> { null }

/**
 * Whether an error message is the daily plan limit (429 `rate_limited` from `POST /jobs`), which
 * offers "Go Pro". The YouTube search quota has its own message and never does (DESIGN.md §3.11).
 */
fun opensPaywall(message: String?): Boolean = message == TranscriptionRepository.RATE_LIMITED

/** The Pro sheet (DESIGN.md §3.11), shown over the whole app while [PaywallUiState.visible]. */
@OptIn(ExperimentalMaterial3Api::class)
@Composable
fun PaywallSheet(viewModel: PaywallViewModel) {
    val state by viewModel.state.collectAsStateWithLifecycle()
    if (!state.visible) return
    val activity = LocalContext.current.findActivity()
    val dismissible by rememberUpdatedState(state.dismissible)
    val sheetState = rememberModalBottomSheetState(
        skipPartiallyExpanded = true,
        confirmValueChange = { it != SheetValue.Hidden || dismissible },
    )
    // A guest who just signed up continues the purchase they started.
    LaunchedEffect(state.resumePurchase) {
        if (state.resumePurchase) {
            viewModel.resumeHandled()
            activity?.let(viewModel::subscribe)
        }
    }
    ModalBottomSheet(
        onDismissRequest = viewModel::dismiss,
        sheetState = sheetState,
        shape = Shapes.sheet,
        containerColor = MusicDecodedColors.TransportBg,
        properties = ModalBottomSheetProperties(shouldDismissOnBackPress = state.dismissible),
    ) {
        PaywallContent(
            state,
            onSelect = viewModel::select,
            onSubscribe = { activity?.let(viewModel::subscribe) },
            onRestore = viewModel::restore,
            onRetry = {
                when (state.failed) {
                    PaywallAction.PURCHASE -> activity?.let(viewModel::subscribe)
                    PaywallAction.RESTORE -> viewModel.restore()
                    PaywallAction.LOAD, null -> viewModel.load()
                }
            },
            onDone = viewModel::dismiss,
        )
    }
}

/** The sheet's content for every state; stateless so it can be previewed and tested. */
@Composable
fun PaywallContent(
    state: PaywallUiState,
    onSelect: (String) -> Unit,
    onSubscribe: () -> Unit,
    onRestore: () -> Unit,
    onRetry: () -> Unit,
    onDone: () -> Unit,
) {
    Column(
        Modifier
            .fillMaxWidth()
            .verticalScroll(rememberScrollState())
            .padding(horizontal = Spacing.gutter)
            .padding(bottom = Spacing.lg)
            .navigationBarsPadding(),
        verticalArrangement = Arrangement.spacedBy(Spacing.md),
        horizontalAlignment = Alignment.CenterHorizontally,
    ) {
        if (state.phase == PaywallPhase.SUCCESS) {
            Success(onDone)
            return@Column
        }
        Header()
        Comparison(state.freeDailyJobs)
        when (state.phase) {
            PaywallPhase.UNAVAILABLE -> Note(RevenueCatBillingService.UNAVAILABLE)
            PaywallPhase.LOADING -> Column(verticalArrangement = Arrangement.spacedBy(Spacing.xs)) {
                repeat(2) { SkeletonCard() }
            }
            else -> Column(Modifier.selectableGroup(), verticalArrangement = Arrangement.spacedBy(Spacing.xs)) {
                state.plans.forEach { card ->
                    PlanCardRow(
                        card,
                        selected = card.option.id == state.selectedId,
                        enabled = state.phase == PaywallPhase.READY,
                        onClick = { onSelect(card.option.id) },
                    )
                }
            }
        }
        state.error?.let { ErrorNote(it, onRetry = onRetry.takeIf { state.phase == PaywallPhase.READY }) }
        if (state.phase != PaywallPhase.UNAVAILABLE) {
            val ready = state.phase == PaywallPhase.READY && state.plans.isNotEmpty()
            PrimaryButton(
                if (state.isGuest) "Create an account to subscribe" else "Subscribe",
                onClick = onSubscribe,
                enabled = ready && state.selectedId != null,
                loading = state.phase == PaywallPhase.PURCHASING,
            )
            TextButton(onClick = onRestore, enabled = state.phase == PaywallPhase.READY, modifier = Modifier.fillMaxWidth()) {
                Text("Restore purchases", style = MaterialTheme.typography.labelLarge, color = MusicDecodedColors.Primary)
            }
        }
        Text(FINE_PRINT, style = MaterialTheme.typography.bodySmall, color = MusicDecodedColors.TextMuted, textAlign = TextAlign.Center)
    }
}

@Composable
private fun Header() {
    Column(horizontalAlignment = Alignment.CenterHorizontally, verticalArrangement = Arrangement.spacedBy(Spacing.xs)) {
        Box(Modifier.size(Sizes.songArt).background(MusicDecodedColors.PrimarySoft, CircleShape), contentAlignment = Alignment.Center) {
            Icon(Icons.Rounded.AutoAwesome, contentDescription = null, tint = MusicDecodedColors.Primary)
        }
        Text(
            "MusicDecoded Pro",
            style = MaterialTheme.typography.titleLarge,
            color = MusicDecodedColors.Text,
            modifier = Modifier.semantics { heading() },
        )
        Text("Decode more, practice deeper.", style = MaterialTheme.typography.bodyMedium, color = MusicDecodedColors.TextSecondary)
    }
}

/** Free vs Pro (PLAN.md §9). */
@Composable
private fun Comparison(freeDailyJobs: Int?) {
    Column(
        Modifier
            .fillMaxWidth()
            .background(MusicDecodedColors.Surface, Shapes.card)
            .border(Sizes.hairline, MusicDecodedColors.Line, Shapes.card)
            .padding(Spacing.md),
        verticalArrangement = Arrangement.spacedBy(Spacing.sm),
    ) {
        ComparisonRow("", "Free", "Pro", header = true)
        ComparisonRow("Daily transcriptions", freeDailyJobs?.let { "$it / day" } ?: "Limited", "Unlimited*")
        ComparisonRow("Analysis models", "Standard", "Premium")
        ComparisonRow("Sync your own audio files", "—", "Included")
    }
}

@Composable
private fun ComparisonRow(label: String, free: String, pro: String, header: Boolean = false) {
    val valueStyle = if (header) TextStyles.overline else MaterialTheme.typography.bodySmall
    Row(verticalAlignment = Alignment.CenterVertically, horizontalArrangement = Arrangement.spacedBy(Spacing.xs)) {
        Text(label, style = MaterialTheme.typography.bodySmall, color = MusicDecodedColors.Text, modifier = Modifier.weight(2f))
        Text(free, style = valueStyle, color = MusicDecodedColors.TextSecondary, modifier = Modifier.weight(1f), textAlign = TextAlign.Center)
        Text(
            pro,
            style = valueStyle,
            color = if (header) MusicDecodedColors.Primary else MusicDecodedColors.Text,
            modifier = Modifier.weight(1f),
            textAlign = TextAlign.Center,
        )
    }
}

@Composable
private fun PlanCardRow(card: PlanCard, selected: Boolean, enabled: Boolean, onClick: () -> Unit) {
    Row(
        Modifier
            .fillMaxWidth()
            .height(Sizes.planCard)
            .background(if (selected) MusicDecodedColors.PrimarySoft else MusicDecodedColors.Surface, Shapes.card)
            .border(Sizes.hairline, if (selected) MusicDecodedColors.Primary else MusicDecodedColors.Line, Shapes.card)
            .selectable(selected = selected, enabled = enabled, role = Role.RadioButton, onClick = onClick)
            .padding(horizontal = Spacing.md),
        verticalAlignment = Alignment.CenterVertically,
        horizontalArrangement = Arrangement.spacedBy(Spacing.xs),
    ) {
        Column(Modifier.weight(1f)) {
            Text(card.title, style = MaterialTheme.typography.titleSmall, color = MusicDecodedColors.Text)
            Text(card.price, style = MaterialTheme.typography.bodySmall, color = MusicDecodedColors.TextSecondary)
        }
        card.saving?.let {
            Text(
                it,
                style = MaterialTheme.typography.labelMedium,
                color = MusicDecodedColors.OnChord,
                modifier = Modifier.background(MusicDecodedColors.Chord, Shapes.chip).padding(horizontal = Spacing.xs, vertical = Spacing.xxs),
            )
        }
        if (selected) Icon(Icons.Rounded.CheckCircle, contentDescription = null, tint = MusicDecodedColors.Primary)
    }
}

@Composable
private fun SkeletonCard() {
    Box(
        Modifier
            .fillMaxWidth()
            .height(Sizes.planCard)
            .background(MusicDecodedColors.SurfaceSoft, Shapes.card),
    )
}

@Composable
private fun Note(text: String) {
    Text(
        text,
        style = MaterialTheme.typography.bodyMedium,
        color = MusicDecodedColors.TextSecondary,
        textAlign = TextAlign.Center,
        modifier = Modifier
            .fillMaxWidth()
            .background(MusicDecodedColors.Surface, Shapes.control)
            .border(Sizes.hairline, MusicDecodedColors.Line, Shapes.control)
            .padding(Spacing.sm),
    )
}

/** PolicyNote-style inline error with Retry (DESIGN.md §3.11). */
@Composable
private fun ErrorNote(message: String, onRetry: (() -> Unit)?) {
    Column(
        Modifier
            .fillMaxWidth()
            .background(MusicDecodedColors.Surface, Shapes.control)
            .border(Sizes.hairline, MusicDecodedColors.Line, Shapes.control)
            .padding(Spacing.sm)
            .semantics { liveRegion = LiveRegionMode.Polite },
        verticalArrangement = Arrangement.spacedBy(Spacing.xs),
    ) {
        Row(horizontalArrangement = Arrangement.spacedBy(Spacing.xs)) {
            Icon(Icons.Rounded.ErrorOutline, contentDescription = null, tint = MusicDecodedColors.Record, modifier = Modifier.size(Sizes.smallIcon))
            Text(message, style = MaterialTheme.typography.bodySmall, color = MusicDecodedColors.TextSecondary)
        }
        if (onRetry != null) SecondaryButton("Retry", onClick = onRetry, icon = Icons.Rounded.Refresh)
    }
}

@Composable
private fun Success(onDone: () -> Unit) {
    Column(
        Modifier.padding(vertical = Spacing.xl),
        horizontalAlignment = Alignment.CenterHorizontally,
        verticalArrangement = Arrangement.spacedBy(Spacing.md),
    ) {
        Box(Modifier.size(Sizes.songArt).background(MusicDecodedColors.PrimarySoft, CircleShape), contentAlignment = Alignment.Center) {
            Icon(Icons.Rounded.CheckCircle, contentDescription = null, tint = MusicDecodedColors.Primary)
        }
        Text(
            "You're Pro",
            style = MaterialTheme.typography.titleLarge,
            color = MusicDecodedColors.Text,
            modifier = Modifier.semantics { heading(); liveRegion = LiveRegionMode.Polite },
        )
        Text("Thanks for supporting MusicDecoded.", style = MaterialTheme.typography.bodyMedium, color = MusicDecodedColors.TextSecondary)
        PrimaryButton("Done", onClick = onDone)
    }
}

private tailrec fun Context.findActivity(): Activity? = when (this) {
    is Activity -> this
    is ContextWrapper -> baseContext.findActivity()
    else -> null
}

const val FINE_PRINT =
    "Subscriptions renew automatically until cancelled. Cancel anytime in Google Play at least 24 hours " +
        "before the end of the period. * Fair-use limit applies."

@Preview
@Composable
private fun PaywallPreview() = PreviewSurface {
    PaywallContent(
        PaywallUiState(
            visible = true,
            phase = PaywallPhase.READY,
            plans = PaywallViewModel.planCards(
                listOf(
                    PlanOption("monthly", PlanPeriod.MONTHLY, "$4.99", 4_990_000),
                    PlanOption("annual", PlanPeriod.ANNUAL, "$39.99", 39_990_000),
                ),
            ),
            selectedId = "annual",
            isGuest = false,
            freeDailyJobs = 50,
        ),
        onSelect = {}, onSubscribe = {}, onRestore = {}, onRetry = {}, onDone = {},
    )
}
