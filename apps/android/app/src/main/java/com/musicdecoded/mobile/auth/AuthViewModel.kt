package com.musicdecoded.mobile.auth

import androidx.lifecycle.ViewModel
import androidx.lifecycle.ViewModelProvider
import androidx.lifecycle.viewModelScope
import com.musicdecoded.mobile.MusicDecodedApplication
import kotlinx.coroutines.flow.MutableStateFlow
import kotlinx.coroutines.flow.SharingStarted
import kotlinx.coroutines.flow.StateFlow
import kotlinx.coroutines.flow.asStateFlow
import kotlinx.coroutines.flow.map
import kotlinx.coroutines.flow.stateIn
import kotlinx.coroutines.launch

sealed interface AuthState {
    data object Loading : AuthState
    data object LoggedIn : AuthState
    data object LoggedOut : AuthState
}

sealed interface AuthUiState {
    data object Idle : AuthUiState
    data object Loading : AuthUiState
    data class Error(val message: String) : AuthUiState
}

class AuthViewModel(private val repository: AuthRepository) : ViewModel() {

    val authState: StateFlow<AuthState> = repository.tokenFlow
        .map { if (it != null) AuthState.LoggedIn else AuthState.LoggedOut }
        .stateIn(viewModelScope, SharingStarted.WhileSubscribed(5_000), AuthState.Loading)

    private val _uiState = MutableStateFlow<AuthUiState>(AuthUiState.Idle)
    val uiState: StateFlow<AuthUiState> = _uiState.asStateFlow()

    fun login(email: String, password: String) = perform { repository.login(email, password) }
    fun signup(email: String, password: String) = perform { repository.signup(email, password) }

    fun logout() {
        viewModelScope.launch { repository.logout() }
    }

    fun clearError() {
        _uiState.value = AuthUiState.Idle
    }

    private fun perform(block: suspend () -> Unit) {
        _uiState.value = AuthUiState.Loading
        viewModelScope.launch {
            try {
                block()
            } catch (e: Exception) {
                _uiState.value = AuthUiState.Error(e.message ?: "An error occurred")
            }
        }
    }
}

class AuthViewModelFactory(private val app: MusicDecodedApplication) : ViewModelProvider.Factory {
    @Suppress("UNCHECKED_CAST")
    override fun <T : ViewModel> create(modelClass: Class<T>): T = AuthViewModel(app.authRepository) as T
}
