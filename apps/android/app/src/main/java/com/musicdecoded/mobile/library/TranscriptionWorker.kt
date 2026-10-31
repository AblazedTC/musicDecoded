package com.musicdecoded.mobile.library

import android.content.Context
import android.util.Log
import androidx.work.BackoffPolicy
import androidx.work.Constraints
import androidx.work.CoroutineWorker
import androidx.work.ExistingWorkPolicy
import androidx.work.NetworkType
import androidx.work.OneTimeWorkRequestBuilder
import androidx.work.WorkManager
import androidx.work.WorkerParameters
import androidx.work.workDataOf
import com.musicdecoded.mobile.MusicDecodedApplication
import kotlinx.coroutines.CancellationException
import java.util.concurrent.TimeUnit

/**
 * Polls one track's job in the background so decoding survives the app leaving the screen or the
 * process dying (T-204a). Unique work per track; a network error retries with backoff and the
 * next attempt resumes polling the same job.
 */
class TranscriptionWorker(context: Context, params: WorkerParameters) : CoroutineWorker(context, params) {
    override suspend fun doWork(): Result {
        val trackId = inputData.getString(KEY_TRACK_ID) ?: return Result.failure()
        val repository = (applicationContext as MusicDecodedApplication).transcriptionRepository
        return try {
            repository.poll(trackId)
            Result.success()
        } catch (e: CancellationException) {
            throw e
        } catch (e: Exception) {
            Log.w(TAG, "Polling $trackId failed (attempt $runAttemptCount)", e)
            if (runAttemptCount + 1 >= MAX_ATTEMPTS) {
                repository.markFailed(trackId, TranscriptionRepository.userMessage(e))
                Result.failure()
            } else {
                Result.retry()
            }
        }
    }

    /** [PollScheduler] backed by WorkManager. */
    class Scheduler(private val context: Context) : PollScheduler {
        override fun schedule(trackId: String, replace: Boolean) {
            val request = OneTimeWorkRequestBuilder<TranscriptionWorker>()
                .setInputData(workDataOf(KEY_TRACK_ID to trackId))
                .setConstraints(Constraints.Builder().setRequiredNetworkType(NetworkType.CONNECTED).build())
                .setBackoffCriteria(BackoffPolicy.EXPONENTIAL, 10, TimeUnit.SECONDS)
                .build()
            WorkManager.getInstance(context).enqueueUniqueWork(
                "transcription-$trackId",
                if (replace) ExistingWorkPolicy.REPLACE else ExistingWorkPolicy.KEEP,
                request,
            )
        }
    }

    private companion object {
        const val TAG = "TranscriptionWorker"
        const val KEY_TRACK_ID = "trackId"
        const val MAX_ATTEMPTS = 8
    }
}
