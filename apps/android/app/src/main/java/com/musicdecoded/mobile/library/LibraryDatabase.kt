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

    @Query("UPDATE library_items SET favorite = :favorite, updatedAt = :now WHERE trackId = :trackId")
    suspend fun setFavorite(trackId: String, favorite: Boolean, now: Instant)
}

class InstantConverters {
    @TypeConverter fun toEpochMillis(value: Instant?): Long? = value?.toEpochMilli()
    @TypeConverter fun fromEpochMillis(value: Long?): Instant? = value?.let(Instant::ofEpochMilli)
}

@Database(entities = [LibraryItem::class], version = 1, exportSchema = false)
@TypeConverters(InstantConverters::class)
abstract class LibraryDatabase : RoomDatabase() {
    abstract fun libraryDao(): LibraryDao

    companion object {
        /** The `db` entry of the on-device layout (PLAN.md §5.1). */
        fun create(context: Context): LibraryDatabase =
            Room.databaseBuilder(context, LibraryDatabase::class.java, "db").build()
    }
}
