package com.musicdecoded.mobile.ui.auth

import androidx.compose.foundation.background
import androidx.compose.foundation.layout.Arrangement
import androidx.compose.foundation.layout.Column
import androidx.compose.foundation.layout.Row
import androidx.compose.foundation.layout.fillMaxSize
import androidx.compose.foundation.layout.fillMaxWidth
import androidx.compose.foundation.layout.imePadding
import androidx.compose.foundation.layout.padding
import androidx.compose.foundation.rememberScrollState
import androidx.compose.foundation.text.KeyboardActions
import androidx.compose.foundation.text.KeyboardOptions
import androidx.compose.foundation.verticalScroll
import androidx.compose.material.icons.Icons
import androidx.compose.material.icons.automirrored.rounded.ArrowBack
import androidx.compose.material3.MaterialTheme
import androidx.compose.material3.OutlinedTextField
import androidx.compose.material3.OutlinedTextFieldDefaults
import androidx.compose.material3.Text
import androidx.compose.runtime.Composable
import androidx.compose.runtime.LaunchedEffect
import androidx.compose.runtime.getValue
import androidx.compose.ui.Modifier
import androidx.compose.ui.platform.LocalContext
import androidx.compose.ui.semantics.LiveRegionMode
import androidx.compose.ui.semantics.liveRegion
import androidx.compose.ui.semantics.semantics
import androidx.compose.ui.text.input.ImeAction
import androidx.compose.ui.text.input.KeyboardType
import androidx.compose.ui.text.input.PasswordVisualTransformation
import androidx.compose.ui.text.input.VisualTransformation
import androidx.lifecycle.compose.collectAsStateWithLifecycle
import androidx.lifecycle.viewmodel.compose.viewModel
import androidx.navigation.NavGraphBuilder
import androidx.navigation.NavHostController
import androidx.navigation.compose.composable
import com.musicdecoded.mobile.MusicDecodedApplication
import com.musicdecoded.mobile.ui.components.MdIconButton
import com.musicdecoded.mobile.ui.components.PageHeader
import com.musicdecoded.mobile.ui.components.PrimaryButton
import com.musicdecoded.mobile.ui.components.SegmentedToggle
import com.musicdecoded.mobile.ui.theme.MusicDecodedColors
import com.musicdecoded.mobile.ui.theme.Shapes
import com.musicdecoded.mobile.ui.theme.Spacing

/** Pushed account routes (DESIGN.md §2: no tab bar). */
object AccountRoutes {
    const val AUTH = "auth/{mode}"
    const val MIGRATE = "account/migrate"
    fun auth(mode: AuthMode) = "auth/${mode.name}"
}

/** Sign up / log in → (guest data) migration → back to Settings. */
fun NavGraphBuilder.accountRoutes(navController: NavHostController) {
    composable(AccountRoutes.AUTH) { entry ->
        val mode = entry.arguments?.getString("mode")?.let { runCatching { AuthMode.valueOf(it) }.getOrNull() } ?: AuthMode.SIGN_UP
        val app = LocalContext.current.applicationContext as MusicDecodedApplication
        AuthScreen(
            viewModel(factory = AuthViewModel.factory(app, mode)),
            onBack = { navController.popBackStack() },
            onMigrate = {
                navController.navigate(AccountRoutes.MIGRATE) { popUpTo(AccountRoutes.AUTH) { inclusive = true } }
            },
            onDone = { navController.popBackStack() },
        )
    }
    composable(AccountRoutes.MIGRATE) {
        val app = LocalContext.current.applicationContext as MusicDecodedApplication
        MigrationScreen(viewModel(factory = MigrationViewModel.factory(app)), onDone = { navController.popBackStack() })
    }
}

@Composable
fun AuthScreen(viewModel: AuthViewModel, onBack: () -> Unit, onMigrate: () -> Unit, onDone: () -> Unit) {
    val state by viewModel.state.collectAsStateWithLifecycle()
    LaunchedEffect(state.outcome) {
        when (state.outcome) {
            AuthOutcome.MIGRATE -> onMigrate()
            AuthOutcome.DONE -> onDone()
            null -> Unit
        }
    }
    val signUp = state.mode == AuthMode.SIGN_UP
    Column(
        Modifier
            .fillMaxSize()
            .background(MusicDecodedColors.Canvas)
            .imePadding()
            .verticalScroll(rememberScrollState()),
    ) {
        Row(Modifier.padding(horizontal = Spacing.gutter, vertical = Spacing.xs)) {
            MdIconButton(Icons.AutoMirrored.Rounded.ArrowBack, contentDescription = "Back", onClick = onBack)
        }
        Column(
            Modifier.padding(horizontal = Spacing.gutter),
            verticalArrangement = Arrangement.spacedBy(Spacing.md),
        ) {
            PageHeader(if (signUp) "Create account" else "Log in", overline = "Account")
            Text(
                "Sync your chord timelines and your own audio across your devices. Music already on this device comes with you.",
                style = MaterialTheme.typography.bodySmall,
                color = MusicDecodedColors.TextSecondary,
            )
            SegmentedToggle(
                options = listOf("Sign up", "Log in"),
                selectedIndex = if (signUp) 0 else 1,
                onSelect = { viewModel.setMode(if (it == 0) AuthMode.SIGN_UP else AuthMode.LOG_IN) },
                modifier = Modifier.fillMaxWidth(),
            )
            AuthField(
                value = state.email,
                onValueChange = viewModel::setEmail,
                label = "Email",
                keyboardOptions = KeyboardOptions(keyboardType = KeyboardType.Email, imeAction = ImeAction.Next),
            )
            AuthField(
                value = state.password,
                onValueChange = viewModel::setPassword,
                label = "Password",
                password = true,
                keyboardOptions = KeyboardOptions(keyboardType = KeyboardType.Password, imeAction = ImeAction.Done),
                keyboardActions = KeyboardActions(onDone = { viewModel.submit() }),
            )
            if (signUp) {
                Text("At least 8 characters.", style = MaterialTheme.typography.labelSmall, color = MusicDecodedColors.TextMuted)
            }
            state.error?.let {
                Text(
                    it,
                    style = MaterialTheme.typography.bodySmall,
                    color = MusicDecodedColors.Record,
                    modifier = Modifier.semantics { liveRegion = LiveRegionMode.Polite },
                )
            }
            PrimaryButton(
                text = when {
                    state.busy -> if (signUp) "Creating account…" else "Logging in…"
                    signUp -> "Create account"
                    else -> "Log in"
                },
                onClick = viewModel::submit,
                enabled = !state.busy,
            )
        }
    }
}

@Composable
private fun AuthField(
    value: String,
    onValueChange: (String) -> Unit,
    label: String,
    keyboardOptions: KeyboardOptions,
    password: Boolean = false,
    keyboardActions: KeyboardActions = KeyboardActions.Default,
) {
    OutlinedTextField(
        value = value,
        onValueChange = onValueChange,
        modifier = Modifier.fillMaxWidth(),
        singleLine = true,
        label = { Text(label) },
        visualTransformation = if (password) PasswordVisualTransformation() else VisualTransformation.None,
        keyboardOptions = keyboardOptions,
        keyboardActions = keyboardActions,
        shape = Shapes.control,
        colors = OutlinedTextFieldDefaults.colors(
            focusedBorderColor = MusicDecodedColors.Primary,
            unfocusedBorderColor = MusicDecodedColors.BorderOnInk,
            focusedContainerColor = MusicDecodedColors.Surface,
            unfocusedContainerColor = MusicDecodedColors.Surface,
            focusedTextColor = MusicDecodedColors.Text,
            unfocusedTextColor = MusicDecodedColors.Text,
            focusedLabelColor = MusicDecodedColors.Primary,
            unfocusedLabelColor = MusicDecodedColors.TextSecondary,
            cursorColor = MusicDecodedColors.Primary,
        ),
    )
}
