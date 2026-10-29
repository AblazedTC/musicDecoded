// Player logic from DESIGN.md §4. All times are seconds; all lookups are binary searches
// over the sorted `chords` / `beats` arrays.

/// "Previous chord" restarts the current chord when the playhead is more than this far into it.
public let previousChordRestartThreshold: Double = 1

/// Current chord, beat and bar at time `t`.
public func currentState(at t: Double, in analysis: Analysis) -> PlaybackState {
    let chords = analysis.chords
    var chordIndex: Int? = upperBound(chords, t, \.start) - 1
    if let index = chordIndex, index < 0 || t >= chords[index].end {
        chordIndex = nil
    }

    let beatCount = upperBound(analysis.beats, t, \.t)
    let downbeats = analysis.beats[..<beatCount].reduce(0) { $0 + ($1.downbeat ? 1 : 0) }
    return PlaybackState(
        chordIndex: chordIndex,
        beatIndex: beatCount > 0 ? beatCount - 1 : nil,
        bar: downbeats > 0 ? downbeats - 1 : nil
    )
}

/// The next segment after the one at `t` whose label isn't `N`/`X`, and the beats until it.
public func upNext(at t: Double, in analysis: Analysis) -> UpNext? {
    let chords = analysis.chords
    let after = upperBound(chords, t, \.start)
    guard let index = chords[after...].firstIndex(where: { isNoChord($0.label) == false }) else {
        return nil
    }
    let beats = analysis.beats
    let inBeats = upperBound(beats, chords[index].start, \.t) - upperBound(beats, t, \.t)
    return UpNext(chordIndex: index, inBeats: inBeats)
}

/// Seek target for "previous chord": the current chord's start if more than 1 s into it,
/// otherwise the start of the previous non-`N` segment. `nil` when there is none.
public func previousChordStart(at t: Double, in analysis: Analysis) -> Double? {
    let chords = analysis.chords
    let current = upperBound(chords, t, \.start) - 1
    guard current >= 0 else { return nil }
    let chord = chords[current]
    if isNoChord(chord.label) == false, t - chord.start > previousChordRestartThreshold {
        return chord.start
    }
    return chords[..<current].last(where: { isNoChord($0.label) == false })?.start
}

/// Seek target for "next chord": the start of the next non-`N` segment, or `nil`.
public func nextChordStart(at t: Double, in analysis: Analysis) -> Double? {
    let chords = analysis.chords
    let after = upperBound(chords, t, \.start)
    return chords[after...].first(where: { isNoChord($0.label) == false })?.start
}

/// The bar containing `t` (downbeat to next downbeat), for "loop current bar".
/// Before the first downbeat it is the pickup `0..<firstDownbeat`; after the last one it ends
/// at `meta.durationSec`. `nil` if the analysis has no downbeats or `t` is outside `0..<durationSec`.
public func barRange(at t: Double, in analysis: Analysis) -> Range<Double>? {
    guard t >= 0, t < analysis.meta.durationSec else { return nil }
    let downbeats = analysis.beats.filter(\.downbeat).map(\.t)
    guard downbeats.isEmpty == false else { return nil }
    let after = upperBound(downbeats, t, \.self)
    let start = after == 0 ? 0 : downbeats[after - 1]
    let end = after < downbeats.count ? downbeats[after] : analysis.meta.durationSec
    return start < end ? start..<end : nil
}

/// Index of the first element whose `key` is greater than `t` (elements must be sorted by `key`).
func upperBound<Element>(_ elements: [Element], _ t: Double, _ key: (Element) -> Double) -> Int {
    var low = 0
    var high = elements.count
    while low < high {
        let mid = (low + high) / 2
        if key(elements[mid]) <= t {
            low = mid + 1
        } else {
            high = mid
        }
    }
    return low
}
