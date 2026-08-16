package com.musicdecoded.mobile.auth

import com.musicdecoded.mobile.network.AuthApiService
import kotlinx.coroutines.flow.Flow

class AuthRepository(private val tokenStore: TokenStore) {
    val tokenFlow: Flow<String?> = tokenStore.tokenFlow

    suspend fun login(email: String, password: String) {
        val token = AuthApiService.login(email, password)
        tokenStore.saveToken(token)
    }

    suspend fun signup(email: String, password: String) {
        AuthApiService.signup(email, password)
        login(email, password)
    }

    suspend fun logout() = tokenStore.clearToken()
}
