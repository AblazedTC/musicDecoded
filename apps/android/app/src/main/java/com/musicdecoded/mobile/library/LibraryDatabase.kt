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

    /** The visible history, newest first. */
    @Query("SELECT * FROM library_items WHERE deleted = 0 ORDER BY createdAt DESC, trackId")
    fun newestFirst(): Flow<List<LibraryItem>>

    /** Tombstone delete: the row stays with `deleted = 1` so the delete can sync. */
    @Query("UPDATE library_items SET deleted = 1, updatedAt = :now WHERE trackId = :trackId")
    suspend fun markDeleted(trackId: String, now: Instant)

    /** Rows whose job is still running (polling resumes for these at launch). */
    @Query("SELECT * FROM library_items WHERE status = 'ANALYZING' AND jobId IS NOT NULL AND deleted = 0")
    suspend fun analyzing(): List<LibraryItem>

    @Query("UPDATE library_items SET favorite = :favorite, updatedAt = :now WHERE trackId = :trackId")
    suspend fun setFavorite(trackId: String, favorite: Boolean, now: Instant)
}

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
