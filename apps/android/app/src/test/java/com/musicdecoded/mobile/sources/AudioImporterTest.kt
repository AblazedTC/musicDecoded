package com.musicdecoded.mobile.sources

import android.app.Application
import android.net.Uri
import com.musicdecoded.mobile.auth.AuthRepository
import com.musicdecoded.mobile.library.AnalysisStore
import com.musicdecoded.mobile.library.LibraryDao
import com.musicdecoded.mobile.library.TranscriptionException
import com.musicdecoded.mobile.library.TranscriptionRepository
import com.musicdecoded.mobile.network.AuthApiService
import com.musicdecoded.mobile.network.ConfigApiService
import com.musicdecoded.mobile.network.TranscriptionApiService
import com.musicdecoded.mobile.network.createHttpClient
import io.ktor.client.engine.mock.MockEngine
import kotlinx.coroutines.test.runTest
import org.junit.Assert.assertEquals
import org.junit.Assert.assertNull
import org.junit.Assert.assertTrue
import org.junit.Assert.fail
import org.junit.Rule
import org.junit.Test
import org.junit.rules.TemporaryFolder
import org.junit.runner.RunWith
import org.robolectric.RobolectricTestRunner
import org.robolectric.RuntimeEnvironment
import org.robolectric.annotation.Config
import java.io.File
import java.lang.reflect.Proxy

@RunWith(RobolectricTestRunner::class)
@Config(application = Application::class)
class AudioImporterTest {
    @get:Rule val folder = TemporaryFolder()

    private fun importer(filesDir: File): AudioImporter {
        val http = createHttpClient(MockEngine { error("no network in this test") }, debugLogging = false)
        // The importer only uses the repository for newPendingFile(); the DAO is never touched.
        val dao = Proxy.newProxyInstance(javaClass.classLoader, arrayOf(LibraryDao::class.java)) { _, _, _ ->
            error("unused")
        } as LibraryDao
        val repository = TranscriptionRepository(
            TranscriptionApiService(http, "http://api.test"),
            AuthRepository(AuthApiService(http, ""), ConfigApiService(http, ""), FakeStore),
            dao,
            AnalysisStore(File(filesDir, "analysis")),
            filesDir,
            { _, _ -> },
        )
        return AudioImporter(RuntimeEnvironment.getApplication(), repository)
    }

    @Test fun copiesIntoAudioDirAndFallsBackToTheFileNameForTitle() = runTest {
        val filesDir = folder.newFolder("files")
        val picked = folder.newFile("Morning Jam.mp3").apply { writeBytes(ByteArray(2048) { 1 }) }

        val draft = importer(filesDir).import(Uri.fromFile(picked))

        assertEquals(File(filesDir, "audio"), draft.file.parentFile)
        assertTrue(draft.file.name.startsWith("pending-") && draft.file.name.endsWith(".mp3"))
        assertEquals(2048L, draft.file.length())
        assertEquals("Morning Jam.mp3", draft.fileName)
        assertEquals("Morning Jam", draft.title)
        assertEquals("MP3", draft.typeLabel)
    }

    @Test fun unsupportedFilesAreRejectedWithoutCopying() = runTest {
        val filesDir = folder.newFolder("files")
        val picked = folder.newFile("notes.txt").apply { writeText("hello") }
        try {
            importer(filesDir).import(Uri.fromFile(picked))
            fail("Expected a TranscriptionException")
        } catch (e: TranscriptionException) {
            assertEquals(AudioImporter.UNSUPPORTED, e.message)
        }
        assertTrue(File(filesDir, "audio").listFiles().isNullOrEmpty())
    }

    @Test fun extensionComesFromTheNameOrTheMimeType() {
        assertEquals("m4a", AudioImporter.extensionFor("song.M4A", null))
        assertEquals("wav", AudioImporter.extensionFor("recording", "audio/x-wav"))
        assertEquals("mp3", AudioImporter.extensionFor("track", "audio/mpeg"))
        assertNull(AudioImporter.extensionFor("clip.mid", "audio/midi"))
    }

    @Test fun sizesAreHumanReadable() {
        assertEquals("8.4 MB", AudioImporter.formatSize(8_400_000))
        assertEquals("640 KB", AudioImporter.formatSize(640_000))
        assertEquals("1 KB", AudioImporter.formatSize(10))
    }

    private object FakeStore : com.musicdecoded.mobile.auth.SessionStore {
        override suspend fun session() = null
        override suspend fun saveSession(session: com.musicdecoded.mobile.auth.Session) = Unit
        override suspend fun clearSession() = Unit
        override suspend fun deviceId() = "d"
        override suspend fun cachedConfig() = null
        override suspend fun saveConfig(config: com.musicdecoded.mobile.network.ClientConfig) = Unit
    }
}
