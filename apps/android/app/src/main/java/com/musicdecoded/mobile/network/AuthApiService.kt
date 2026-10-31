package com.musicdecoded.mobile.network

import io.ktor.client.call.*
import io.ktor.client.plugins.expectSuccess
import io.ktor.client.request.*
import io.ktor.client.statement.*
import io.ktor.http.*
import kotlinx.serialization.Serializable

@Serializable
private data class AuthRequest(val email: String, val password: String)

/** `AuthResponse` in contracts/openapi.yaml. */
@Serializable
data class AuthResponse(val token: String, val userId: String, val kind: String, val plan: String)

/** `Error` in contracts/openapi.yaml. */
@Serializable
data class ApiError(val code: String, val message: String)

/** A non-2xx API response; [message] is the server's user-facing text (e.g. "Invalid email or password"). */
class ApiException(val error: ApiError, val status: Int) : Exception(error.message)

private const val BASE_URL = "http://10.0.2.2:8080"

object AuthApiService {
    suspend fun login(email: String, password: String): AuthResponse = postAuth("login", email, password)

    suspend fun signup(email: String, password: String): AuthResponse = postAuth("signup", email, password)

    private suspend fun postAuth(path: String, email: String, password: String): AuthResponse {
        val response = httpClient.post("$BASE_URL/auth/$path") {
            expectSuccess = false // decode the contract error body instead of throwing a generic exception
            contentType(ContentType.Application.Json)
            setBody(AuthRequest(email, password))
        }
        if (!response.status.isSuccess()) {
            val error = runCatching { response.body<ApiError>() }
                .getOrElse { ApiError("internal", "Something went wrong (${response.status.value})") }
            throw ApiException(error, response.status.value)
        }
        return response.body()
    }
}
