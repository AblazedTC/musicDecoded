package com.musicdecoded.mobile.network

import io.ktor.client.call.*
import io.ktor.client.request.*
import io.ktor.http.*
import kotlinx.serialization.Serializable

@Serializable
private data class AuthRequest(val email: String, val password: String)

/** `AuthResponse` in contracts/openapi.yaml. */
@Serializable
data class AuthResponse(val token: String, val userId: String, val kind: String, val plan: String)

private const val BASE_URL = "http://10.0.2.2:8080"

object AuthApiService {
    suspend fun login(email: String, password: String): AuthResponse =
        httpClient.post("$BASE_URL/auth/login") {
            contentType(ContentType.Application.Json)
            setBody(AuthRequest(email, password))
        }.body()

    suspend fun signup(email: String, password: String): AuthResponse =
        httpClient.post("$BASE_URL/auth/signup") {
            contentType(ContentType.Application.Json)
            setBody(AuthRequest(email, password))
        }.body()
}
