package com.musicdecoded.mobile.network

import io.ktor.client.request.*
import io.ktor.client.statement.*
import io.ktor.http.*
import kotlinx.serialization.Serializable

@Serializable
private data class AuthRequest(val email: String, val password: String)

private const val BASE_URL = "http://10.0.2.2:8080"

object AuthApiService {
    suspend fun login(email: String, password: String): String =
        httpClient.post("$BASE_URL/auth/login") {
            contentType(ContentType.Application.Json)
            setBody(AuthRequest(email, password))
        }.bodyAsText()

    // POST /auth/signup returns a plain success message; caller must call login to get a token
    suspend fun signup(email: String, password: String) {
        httpClient.post("$BASE_URL/auth/signup") {
            contentType(ContentType.Application.Json)
            setBody(AuthRequest(email, password))
        }
    }
}
