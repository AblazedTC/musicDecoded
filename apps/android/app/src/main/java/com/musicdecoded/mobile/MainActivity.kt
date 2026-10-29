package com.musicdecoded.mobile

import android.os.Bundle
import androidx.activity.ComponentActivity
import androidx.activity.compose.setContent
import androidx.activity.enableEdgeToEdge
import androidx.compose.foundation.layout.Box
import androidx.compose.foundation.layout.fillMaxSize
import androidx.compose.runtime.collectAsState
import androidx.compose.runtime.getValue
import androidx.compose.ui.Modifier
import androidx.lifecycle.viewmodel.compose.viewModel
import com.musicdecoded.mobile.auth.AuthState
import com.musicdecoded.mobile.auth.AuthViewModel
import com.musicdecoded.mobile.auth.AuthViewModelFactory
import com.musicdecoded.mobile.ui.auth.AuthScreen
import com.musicdecoded.mobile.ui.main.MainScreen
import com.musicdecoded.mobile.ui.theme.MusicDecodedTheme

class MainActivity : ComponentActivity() {
    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        enableEdgeToEdge()
        setContent {
            MusicDecodedTheme {
                val authViewModel: AuthViewModel = viewModel(
                    factory = AuthViewModelFactory(application as MusicDecodedApplication)
                )
                val authState by authViewModel.authState.collectAsState()

                when (authState) {
                    AuthState.Loading -> Box(Modifier.fillMaxSize())
                    AuthState.LoggedOut -> AuthScreen(authViewModel)
                    AuthState.LoggedIn -> MainScreen(authViewModel)
                }
            }
        }
    }
}