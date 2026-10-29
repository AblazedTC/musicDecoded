package com.musicdecoded.mobile.auth

import android.content.Context
import android.security.keystore.KeyGenParameterSpec
import android.security.keystore.KeyProperties
import android.util.Base64
import androidx.datastore.core.DataStore
import androidx.datastore.preferences.core.Preferences
import androidx.datastore.preferences.core.edit
import androidx.datastore.preferences.core.stringPreferencesKey
import androidx.datastore.preferences.preferencesDataStore
import com.musicdecoded.mobile.network.ApiJson
import com.musicdecoded.mobile.network.ClientConfig
import com.musicdecoded.mobile.network.Kind
import com.musicdecoded.mobile.network.Plan
import kotlinx.coroutines.flow.first
import java.security.GeneralSecurityException
import java.security.KeyStore
import java.util.UUID
import javax.crypto.Cipher
import javax.crypto.KeyGenerator
import javax.crypto.SecretKey
import javax.crypto.spec.GCMParameterSpec

/** The signed-in identity: a guest (device) or an account JWT. */
data class Session(val token: String, val userId: String, val kind: Kind, val plan: Plan)

/** Persistent session state. [TokenStore] in the app; an in-memory fake in tests. */
interface SessionStore {
    suspend fun session(): Session?
    suspend fun saveSession(session: Session)
    suspend fun clearSession()

    /** A random UUID generated on first use and kept for the life of the install. */
    suspend fun deviceId(): String

    suspend fun cachedConfig(): ClientConfig?
    suspend fun saveConfig(config: ClientConfig)
}

private val Context.dataStore: DataStore<Preferences> by preferencesDataStore("auth")

/**
 * DataStore-backed [SessionStore]. The token and deviceId are encrypted with an AES-GCM key that
 * lives in the Android Keystore, so they're never on disk in plaintext. If the key is gone (e.g.
 * the data came back from a backup onto a new device), the values read as missing and the app
 * starts a fresh guest session.
 */
class TokenStore(private val context: Context) : SessionStore {
    private val tokenKey = stringPreferencesKey("token_enc")
    private val userIdKey = stringPreferencesKey("user_id")
    private val kindKey = stringPreferencesKey("kind")
    private val planKey = stringPreferencesKey("plan")
    private val deviceIdKey = stringPreferencesKey("device_id_enc")
    private val configKey = stringPreferencesKey("client_config")
    private val legacyPlaintextTokenKey = stringPreferencesKey("jwt")

    override suspend fun session(): Session? {
        val prefs = context.dataStore.data.first()
        val token = prefs[tokenKey]?.let(KeystoreCipher::decrypt) ?: return null
        return Session(
            token = token,
            userId = prefs[userIdKey] ?: return null,
            kind = prefs[kindKey]?.let { runCatching { Kind.valueOf(it) }.getOrNull() } ?: return null,
            plan = prefs[planKey]?.let { runCatching { Plan.valueOf(it) }.getOrNull() } ?: Plan.FREE,
        )
    }

    override suspend fun saveSession(session: Session) {
        val encrypted = KeystoreCipher.encrypt(session.token)
        context.dataStore.edit {
            it[tokenKey] = encrypted
            it[userIdKey] = session.userId
            it[kindKey] = session.kind.name
            it[planKey] = session.plan.name
            it.remove(legacyPlaintextTokenKey)
        }
    }

    override suspend fun clearSession() {
        context.dataStore.edit {
            it.remove(tokenKey)
            it.remove(userIdKey)
            it.remove(kindKey)
            it.remove(planKey)
            it.remove(legacyPlaintextTokenKey)
        }
    }

    override suspend fun deviceId(): String {
        var id: String? = null
        context.dataStore.edit { prefs ->
            id = prefs[deviceIdKey]?.let(KeystoreCipher::decrypt)
                ?: UUID.randomUUID().toString().also { prefs[deviceIdKey] = KeystoreCipher.encrypt(it) }
        }
        return checkNotNull(id)
    }

    override suspend fun cachedConfig(): ClientConfig? =
        context.dataStore.data.first()[configKey]?.let {
            runCatching { ApiJson.decodeFromString(ClientConfig.serializer(), it) }.getOrNull()
        }

    override suspend fun saveConfig(config: ClientConfig) {
        context.dataStore.edit { it[configKey] = ApiJson.encodeToString(ClientConfig.serializer(), config) }
    }
}

/** AES-256-GCM with a non-exportable Keystore key. Output = base64(iv ‖ ciphertext). */
private object KeystoreCipher {
    private const val ALIAS = "musicdecoded.session"
    private const val TRANSFORMATION = "AES/GCM/NoPadding"
    private const val IV_BYTES = 12
    private const val TAG_BITS = 128

    private fun key(): SecretKey {
        val keyStore = KeyStore.getInstance("AndroidKeyStore").apply { load(null) }
        (keyStore.getKey(ALIAS, null) as? SecretKey)?.let { return it }
        val generator = KeyGenerator.getInstance(KeyProperties.KEY_ALGORITHM_AES, "AndroidKeyStore")
        generator.init(
            KeyGenParameterSpec.Builder(ALIAS, KeyProperties.PURPOSE_ENCRYPT or KeyProperties.PURPOSE_DECRYPT)
                .setBlockModes(KeyProperties.BLOCK_MODE_GCM)
                .setEncryptionPaddings(KeyProperties.ENCRYPTION_PADDING_NONE)
                .setKeySize(256)
                .build(),
        )
        return generator.generateKey()
    }

    fun encrypt(plain: String): String {
        val cipher = Cipher.getInstance(TRANSFORMATION).apply { init(Cipher.ENCRYPT_MODE, key()) }
        return Base64.encodeToString(cipher.iv + cipher.doFinal(plain.toByteArray()), Base64.NO_WRAP)
    }

    /** `null` when the value can't be decrypted (missing key, corrupt data). */
    fun decrypt(encoded: String): String? = try {
        val bytes = Base64.decode(encoded, Base64.NO_WRAP)
        val cipher = Cipher.getInstance(TRANSFORMATION).apply {
            init(Cipher.DECRYPT_MODE, key(), GCMParameterSpec(TAG_BITS, bytes, 0, IV_BYTES))
        }
        String(cipher.doFinal(bytes, IV_BYTES, bytes.size - IV_BYTES))
    } catch (_: GeneralSecurityException) {
        null
    } catch (_: IllegalArgumentException) {
        null
    }
}
