package com.musicdecoded.mobile

import android.app.Application
import com.musicdecoded.mobile.auth.AuthRepository
import com.musicdecoded.mobile.auth.TokenStore

class MusicDecodedApplication : Application() {
    val tokenStore by lazy { TokenStore(this) }
    val authRepository by lazy { AuthRepository(tokenStore) }
}
