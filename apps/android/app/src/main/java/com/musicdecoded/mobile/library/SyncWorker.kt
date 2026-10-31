package com.musicdecoded.mobile.library

import android.content.Context
import android.util.Log
import androidx.datastore.core.DataStore
import androidx.datastore.preferences.core.Preferences
import androidx.datastore.preferences.core.edit
import androidx.datastore.preferences.core.stringPreferencesKey
import androidx.datastore.preferences.preferencesDataStore
import androidx.work.BackoffPolicy
import androidx.work.Constraints
import androidx.work.CoroutineWorker
import androidx.work.ExistingWorkPolicy
import androidx.work.NetworkType
import androidx.work.OneTimeWorkRequestBuilder
import androidx.work.WorkManager
import androidx.work.WorkerParameters
import com.musicdecoded.mobile.MusicDecodedApplication
import kotlinx.coroutines.CancellationException
import kotlinx.coroutines.flow.first
import java.util.concurrent.TimeUnit

/** Runs [SyncRepository.sync] in the background (T-404a). Network errors retry with backoff. */
class SyncWorker(context: Context, params: WorkerParameters) : CoroutineWorker(context, params) {
    override suspend fun doWork(): Result {
        val repository = (applicationContext as MusicDecodedApplication).syncRepository
        return try {
            repository.sync()
            Result.success()
        } catch (e: CancellationException) {
            throw e
        } catch (e: Exception) {
            Log.w(TAG, "Sync failed (attempt $runAttemptCount)", e)
            if (runAttemptCount + 1 >= MAX_ATTEMPTS) Result.failure() else Result.retry()
        }
    }

    companion object {
        private const val TAG = "SyncWorker"
        private const val MAX_ATTEMPTS = 5

        /**
         * Requests a sync: app foreground, after a job completes, after login (PLAN.md §5.2).
         * Unique work; the latest request replaces a pending one. Guests return immediately.
         */
        fun schedule(context: Context) {
            val request = OneTimeWorkRequestBuilder<SyncWorker>()
                .setConstraints(Constraints.Builder().setRequiredNetworkType(NetworkType.CONNECTED).build())
                .setBackoffCriteria(BackoffPolicy.EXPONENTIAL, 30, TimeUnit.SECONDS)
                .build()
            WorkManager.getInstance(context).enqueueUniqueWork("library-sync", ExistingWorkPolicy.REPLACE, request)
        }
    }
}

private val Context.syncDataStore: DataStore<Preferences> by preferencesDataStore("sync")

/** DataStore-backed [SyncStateStore]. */
class DataStoreSyncState(private val context: Context) : SyncStateStore {
    private val userIdKey = stringPreferencesKey("user_id")
    private val cursorKey = stringPreferencesKey("cursor")

    override suspend fun userId(): String? = context.syncDataStore.data.first()[userIdKey]
    override suspend fun cursor(): String? = context.syncDataStore.data.first()[cursorKey]

    override suspend fun save(userId: String, cursor: String?) {
        context.syncDataStore.edit {
            it[userIdKey] = userId
            if (cursor != null) it[cursorKey] = cursor else it.remove(cursorKey)
        }
    }
}
