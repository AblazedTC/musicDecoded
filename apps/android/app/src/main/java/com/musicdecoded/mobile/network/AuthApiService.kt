package com.musicdecoded.mobile.network

import io.ktor.client.HttpClient
import io.ktor.client.request.bearerAuth
import io.ktor.client.request.post
import io.ktor.client.request.setBody
import io.ktor.http.ContentType
import io.ktor.http.contentType
import kotlinx.serialization.SerialName
import kotlinx.serialization.Serializable

/** Body of `POST /auth/anonymous`. */
@Serializable
data class AnonymousRequest(val deviceId: String, val platform: String)

/** `Credentials` in contracts/openapi.yaml. */
@Serializable
data class Credentials(val email: String, val password: String)

/** `AuthResponse` in contracts/openapi.yaml. */
@Serializable
data class AuthResponse(val token: String, val userId: String, val kind: Kind, val plan: Plan)

/** JWT `kind` claim. */
@Serializable
enum class Kind {
    @SerialName("guest") GUEST,
    @SerialName("account") ACCOUNT,
}

/** `Plan` in contracts/openapi.yaml. */
@Serializable
enum class Plan {
    @SerialName("free") FREE,
    @SerialName("pro") PRO,
}

/** `Error` in contracts/openapi.yaml. [code] is kept as text so new codes don't break decoding. */
@Serializable
data class ApiError(val code: String, val message: String)

/** A non-2xx API response; [message] is the server's user-facing text (e.g. "Invalid email or password"). */
class ApiException(val error: ApiError, val status: Int) : Exception(error.message)

/** The `/auth/...` endpoints. None of them retry on 401: a 401 here means bad credentials. */
class AuthApiService(private val http: HttpClient, private val baseUrl: String) {
    suspend fun anonymous(deviceId: String): AuthResponse =
        http.post("$baseUrl/auth/anonymous") {
            contentType(ContentType.Application.Json)
            setBody(AnonymousRequest(deviceId, platform = "android"))
        }.bodyOrThrow()

    /** [guestToken] links that guest's data to the account (optional bearer in the contract). */
    suspend fun login(email: String, password: String, guestToken: String?): AuthResponse =
        postCredentials("login", email, password, guestToken)

    suspend fun signup(email: String, password: String, guestToken: String?): AuthResponse =
        postCredentials("signup", email, password, guestToken)

    private suspend fun postCredentials(path: String, email: String, password: String, guestToken: String?): AuthResponse =
        http.post("$baseUrl/auth/$path") {
            guestToken?.let { bearerAuth(it) }
            contentType(ContentType.Application.Json)
            setBody(Credentials(email, password))
        }.bodyOrThrow()
}
