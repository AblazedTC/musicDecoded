package com.musicdecoded.mobile.library

import android.content.Context
import androidx.room.Dao
import androidx.room.Database
import androidx.room.Query
import androidx.room.Room
import androidx.room.RoomDatabase
import androidx.room.TypeConverter
import androidx.room.TypeConverters
import androidx.room.Upsert
import androidx.room.migration.Migration
import androidx.sqlite.db.SupportSQLiteDatabase
import kotlinx.coroutines.flow.Flow
import java.time.Instant

@Dao
interface LibraryDao {
    /** Create or replace a row (by trackId). Callers set `updatedAt`. */
    @Upsert
    suspend fun upsert(item: LibraryItem)

    /** Any row, including tombstones (sync needs them). */
    @Query("SELECT * FROM library_items WHERE trackId = :trackId")
    suspend fun get(trackId: String): LibraryItem?

    /** One row as it changes (Processing screen). */
    @Query("SELECT * FROM library_items WHERE trackId = :trackId")
    fun observe(trackId: String): Flow<LibraryItem?>

    /** The visible row for a streaming source (e.g. a YouTube videoId), as it changes (video detail). */
    @Query("SELECT * FROM library_items WHERE source = :source AND sourceRef = :sourceRef AND deleted = 0 ORDER BY createdAt DESC LIMIT 1")
    fun observeBySourceRef(source: String, sourceRef: String): Flow<LibraryItem?>

    /** The visible history, newest first. */
    @Query("SELECT * FROM library_items WHERE deleted = 0 ORDER BY createdAt DESC, trackId")
    fun newestFirst(): Flow<List<LibraryItem>>

    /** The [limit] newest visible rows (Home "Recently Decoded"). */
    @Query("SELECT * FROM library_items WHERE deleted = 0 ORDER BY createdAt DESC, trackId LIMIT :limit")
    fun recent(limit: Int): Flow<List<LibraryItem>>

    /**
     * Library search (DESIGN.md §3.2), newest first. [pattern] is matched with `LIKE` against title,
     * artist and key (`''` = everything; escape `%`, `_` and `!` with `!`, see [likePattern]). [source] is a [com.musicdecoded.mobile.core.Source] name.
     */
    @Query(
        """
        SELECT * FROM library_items WHERE deleted = 0
          AND (:pattern = ''
               OR title LIKE '%' || :pattern || '%' ESCAPE '!'
               OR artist LIKE '%' || :pattern || '%' ESCAPE '!'
               OR musicKey LIKE '%' || :pattern || '%' ESCAPE '!')
          AND (:favoritesOnly = 0 OR favorite = 1)
          AND (:offlineOnly = 0 OR localAudioPath IS NOT NULL)
          AND (:source IS NULL OR source = :source)
        ORDER BY createdAt DESC, trackId
        """,
    )
    fun search(pattern: String, favoritesOnly: Boolean, offlineOnly: Boolean, source: String?): Flow<List<LibraryItem>>

    /** Library summary: decoded (ready) and available-offline (audio on this device) counts. */
    @Query(
        """
        SELECT COALESCE(SUM(status = 'READY'), 0) AS decoded,
               COALESCE(SUM(localAudioPath IS NOT NULL), 0) AS offline
        FROM library_items WHERE deleted = 0
        """,
    )
    fun counts(): Flow<LibraryCounts>

    /** Tombstone delete: the row stays with `deleted = 1` so the delete can sync. */
    @Query("UPDATE library_items SET deleted = 1, updatedAt = :now WHERE trackId = :trackId")
    suspend fun markDeleted(trackId: String, now: Instant)

    /** Rows whose job is still running (polling resumes for these at launch). */
    @Query("SELECT * FROM library_items WHERE status = 'ANALYZING' AND jobId IS NOT NULL AND deleted = 0")
    suspend fun analyzing(): List<LibraryItem>

    @Query("UPDATE library_items SET favorite = :favorite, updatedAt = :now WHERE trackId = :trackId")
    suspend fun setFavorite(trackId: String, favorite: Boolean, now: Instant)

    /**
     * Sync push candidates (T-404a): changed since their last sync, oldest change first. Only
     * finished rows and tombstones; a row still analyzing is pushed once its job completes.
     */
    @Query(
        """
        SELECT * FROM library_items
        WHERE (syncedAt IS NULL OR updatedAt > syncedAt) AND (status = 'READY' OR deleted = 1)
        ORDER BY updatedAt, trackId LIMIT :limit
        """,
    )
    suspend fun pendingPush(limit: Int): List<LibraryItem>

    /** Marks a pushed row synced, unless it changed again while the push was in flight. */
    @Query("UPDATE library_items SET syncedAt = :updatedAt WHERE trackId = :trackId AND updatedAt = :updatedAt")
    suspend fun markSynced(trackId: String, updatedAt: Instant)

    /** Visible rows that went through sync (they may be missing their analysis or audio here). */
    @Query("SELECT * FROM library_items WHERE syncedAt IS NOT NULL AND deleted = 0 AND status != 'ANALYZING'")
    suspend fun syncedVisible(): List<LibraryItem>

    /** Finished local rows never synced: the guest data the migration screen lists (DESIGN.md §3.10). */
    @Query("SELECT * FROM library_items WHERE syncedAt IS NULL AND deleted = 0 AND status = 'READY'")
    suspend fun unsynced(): List<LibraryItem>
}

data class LibraryCounts(val decoded: Int, val offline: Int)

/** Escapes user text for [LibraryDao.search]'s `LIKE … ESCAPE '!'`. */
fun likePattern(text: String): String =
    text.replace("!", "!!").replace("%", "!%").replace("_", "!_")

class InstantConverters {
    @TypeConverter fun toEpochMillis(value: Instant?): Long? = value?.toEpochMilli()
    @TypeConverter fun fromEpochMillis(value: Long?): Instant? = value?.let(Instant::ofEpochMilli)
}

@Database(entities = [LibraryItem::class], version = 2, exportSchema = false)
@TypeConverters(InstantConverters::class)
abstract class LibraryDatabase : RoomDatabase() {
    abstract fun libraryDao(): LibraryDao

    companion object {
        /** The `db` entry of the on-device layout (PLAN.md §5.1). */
        fun create(context: Context): LibraryDatabase =
            Room.databaseBuilder(context, LibraryDatabase::class.java, "db").addMigrations(MIGRATION_1_2).build()

        /** v2 adds the local-only `error` column (T-204a). */
        val MIGRATION_1_2 = object : Migration(1, 2) {
            override fun migrate(db: SupportSQLiteDatabase) {
                db.execSQL("ALTER TABLE library_items ADD COLUMN error TEXT")
            }
        }
    }
}
