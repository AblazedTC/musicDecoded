package com.musicdecoded.mobile.ui.auth

import androidx.lifecycle.ViewModel
import androidx.lifecycle.ViewModelProvider
import androidx.lifecycle.viewModelScope
import androidx.lifecycle.viewmodel.initializer
import androidx.lifecycle.viewmodel.viewModelFactory
import com.musicdecoded.mobile.MusicDecodedApplication
import com.musicdecoded.mobile.network.ApiException
import kotlinx.coroutines.CancellationException
import kotlinx.coroutines.flow.MutableStateFlow
import kotlinx.coroutines.flow.StateFlow
import kotlinx.coroutines.flow.asStateFlow
import kotlinx.coroutines.flow.update
import kotlinx.coroutines.launch
import java.io.IOException

enum class AuthMode { SIGN_UP, LOG_IN }

/** Where to go after signing in: the migration screen when this device has guest data. */
enum class AuthOutcome { MIGRATE, DONE }

data class AuthUiState(
    val mode: AuthMode = AuthMode.SIGN_UP,
    val email: String = "",
    val password: String = "",
    val busy: Boolean = false,
    val error: String? = null,
    val outcome: AuthOutcome? = null,
)

/**
 * Sign up / log in (T-403a). The guest token is sent with the request, so the server moves the
 * guest's jobs and uploads to the account (AuthRepository). Then: guest data → migration screen;
 * otherwise sync right away (PLAN.md §5.2 "after login").
 */
class AuthViewModel(
    initialMode: AuthMode,
    private val signUp: suspend (email: String, password: String) -> Unit,
    private val logIn: suspend (email: String, password: String) -> Unit,
    private val hasGuestData: suspend () -> Boolean,
    private val requestSync: () -> Unit,
) : ViewModel() {
    private val _state = MutableStateFlow(AuthUiState(mode = initialMode))
    val state: StateFlow<AuthUiState> = _state.asStateFlow()

    fun setEmail(value: String) = _state.update { it.copy(email = value, error = null) }
    fun setPassword(value: String) = _state.update { it.copy(password = value, error = null) }
    fun setMode(mode: AuthMode) = _state.update { it.copy(mode = mode, error = null) }

    fun submit() {
        val current = _state.value
        if (current.busy) return
        val email = current.email.trim()
        validateCredentials(email, current.password)?.let { message ->
            _state.update { it.copy(error = message) }
            return
        }
        _state.update { it.copy(busy = true, error = null) }
        viewModelScope.launch {
            try {
                if (current.mode == AuthMode.SIGN_UP) signUp(email, current.password) else logIn(email, current.password)
                val outcome = if (hasGuestData()) AuthOutcome.MIGRATE else AuthOutcome.DONE.also { requestSync() }
                _state.update { it.copy(busy = false, outcome = outcome) }
            } catch (e: CancellationException) {
                throw e
            } catch (e: Exception) {
                _state.update { it.copy(busy = false, error = errorMessage(e)) }
            }
        }
    }

    companion object {
        const val MIN_PASSWORD_CHARS = 8
        const val MAX_PASSWORD_BYTES = 72

        /** Client-side checks mirroring `Credentials` in openapi; `null` when valid. */
        fun validateCredentials(email: String, password: String): String? = when {
            !EMAIL.matches(email) -> "Enter a valid email address."
            password.codePointCount(0, password.length) < MIN_PASSWORD_CHARS -> "Use at least 8 characters for your password."
            password.toByteArray(Charsets.UTF_8).size > MAX_PASSWORD_BYTES -> "This password is too long (72 bytes max; some characters count more than once)."
            else -> null
        }

        /** The API's own message (e.g. "Invalid email or password"), or a connection hint. */
        fun errorMessage(e: Throwable): String = when (e) {
            is ApiException -> e.message ?: "Something went wrong. Try again."
            is IOException -> "Couldn't reach MusicDecoded. Check your connection and try again."
            else -> "Something went wrong. Try again."
        }

        private val EMAIL = Regex("[^@\\s]+@[^@\\s]+\\.[^@\\s]+")

        fun factory(app: MusicDecodedApplication, mode: AuthMode): ViewModelProvider.Factory = viewModelFactory {
            initializer {
                AuthViewModel(
                    initialMode = mode,
                    signUp = app.authRepository::signup,
                    logIn = app.authRepository::login,
                    hasGuestData = { app.libraryDatabase.libraryDao().unsynced().isNotEmpty() },
                    requestSync = app::requestSync,
                )
            }
        }
    }
}
