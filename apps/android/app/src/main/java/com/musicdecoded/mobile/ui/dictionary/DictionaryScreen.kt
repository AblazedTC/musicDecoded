package com.musicdecoded.mobile.ui.dictionary

import androidx.compose.foundation.layout.Column
import androidx.compose.foundation.layout.fillMaxSize
import androidx.compose.foundation.layout.fillMaxWidth
import androidx.compose.foundation.layout.padding
import androidx.compose.foundation.lazy.LazyColumn
import androidx.compose.foundation.lazy.items
import androidx.compose.material.icons.Icons
import androidx.compose.material.icons.filled.Search
import androidx.compose.material3.HorizontalDivider
import androidx.compose.material3.Icon
import androidx.compose.material3.ListItem
import androidx.compose.material3.OutlinedTextField
import androidx.compose.material3.Text
import androidx.compose.runtime.Composable
import androidx.compose.runtime.getValue
import androidx.compose.runtime.mutableStateOf
import androidx.compose.runtime.remember
import androidx.compose.runtime.saveable.rememberSaveable
import androidx.compose.runtime.setValue
import androidx.compose.ui.Modifier
import androidx.compose.ui.unit.dp

private data class ChordEntry(val name: String, val notes: String)

private val chords = listOf(
    ChordEntry("C Major", "C – E – G"),
    ChordEntry("D Major", "D – F# – A"),
    ChordEntry("E Major", "E – G# – B"),
    ChordEntry("F Major", "F – A – C"),
    ChordEntry("G Major", "G – B – D"),
    ChordEntry("A Major", "A – C# – E"),
    ChordEntry("B Major", "B – D# – F#"),
    ChordEntry("A Minor", "A – C – E"),
    ChordEntry("B Minor", "B – D – F#"),
    ChordEntry("C Minor", "C – Eb – G"),
    ChordEntry("D Minor", "D – F – A"),
    ChordEntry("E Minor", "E – G – B"),
    ChordEntry("F Minor", "F – Ab – C"),
    ChordEntry("G Minor", "G – Bb – D"),
    ChordEntry("Cmaj7", "C – E – G – B"),
    ChordEntry("Gmaj7", "G – B – D – F#"),
    ChordEntry("Amaj7", "A – C# – E – G#"),
    ChordEntry("Dm7", "D – F – A – C"),
    ChordEntry("Em7", "E – G – B – D"),
    ChordEntry("Am7", "A – C – E – G"),
    ChordEntry("G7", "G – B – D – F"),
    ChordEntry("C7", "C – E – G – Bb"),
    ChordEntry("D7", "D – F# – A – C"),
    ChordEntry("Csus2", "C – D – G"),
    ChordEntry("Gsus4", "G – C – D"),
    ChordEntry("Cadd9", "C – E – G – D"),
)

@Composable
fun DictionaryScreen() {
    var query by rememberSaveable { mutableStateOf("") }
    val filtered = remember(query) {
        if (query.isBlank()) chords
        else chords.filter { it.name.contains(query, ignoreCase = true) }
    }

    Column(Modifier.fillMaxSize()) {
        OutlinedTextField(
            value = query,
            onValueChange = { query = it },
            placeholder = { Text("Search chords…") },
            leadingIcon = { Icon(Icons.Default.Search, contentDescription = null) },
            singleLine = true,
            modifier = Modifier
                .fillMaxWidth()
                .padding(horizontal = 16.dp, vertical = 8.dp)
        )
        LazyColumn {
            items(filtered, key = { it.name }) { chord ->
                ListItem(
                    headlineContent = { Text(chord.name) },
                    supportingContent = { Text(chord.notes) }
                )
                HorizontalDivider()
            }
        }
    }
}
