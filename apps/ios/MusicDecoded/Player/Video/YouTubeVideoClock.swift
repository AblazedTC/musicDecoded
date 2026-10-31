import Foundation

/// The video's position between bridge reports. The page reports `currentTime` about every
/// 250 ms; in between, while playing, the time is extrapolated from the last report × rate so the
/// 60 Hz timeline moves smoothly. Pure value type: `now` is passed in (seconds, monotonic).
struct YouTubeVideoClock: Sendable {
    /// While playing, a report up to this far behind the extrapolated time holds the playhead
    /// instead of jumping it back (bridge latency jitter).
    static let maxHold = 0.25
    /// After a seek, reports further than this from the target are stale (sent before the seek).
    static let seekTolerance = 1.0
    /// ...for at most this long.
    static let seekTimeout = 2.0

    private(set) var state = YouTubePlayerState.unstarted
    private(set) var rate = 1.0

    private var anchorTime = 0.0
    private var anchorStamp = 0.0
    /// The last position returned while playing; the playhead never moves back by less than `maxHold`.
    private var shown = 0.0
    private var seekTarget: Double?
    private var seekStamp = 0.0

    mutating func report(time: Double, at now: Double) {
        if let seekTarget {
            if abs(time - seekTarget) > Self.seekTolerance, now - seekStamp < Self.seekTimeout {
                return
            }
            self.seekTarget = nil
        }
        anchorTime = max(time, 0)
        anchorStamp = now
    }

    mutating func report(state: YouTubePlayerState, time: Double?, at now: Double) {
        // Freeze the extrapolated time at the change, then take the reported one if it's fresh.
        anchorTime = position(at: now)
        anchorStamp = now
        self.state = state
        if let time {
            report(time: time, at: now)
        }
        if state != .playing {
            shown = anchorTime
        }
    }

    mutating func setRate(_ rate: Double, at now: Double) {
        anchorTime = position(at: now)
        anchorStamp = now
        self.rate = rate
    }

    /// A seek was requested: show the target right away and ignore older reports.
    mutating func seek(to time: Double, at now: Double) {
        anchorTime = max(time, 0)
        anchorStamp = now
        shown = anchorTime
        seekTarget = anchorTime
        seekStamp = now
    }

    /// Seconds into the video at `now`.
    mutating func position(at now: Double) -> Double {
        var t = anchorTime
        if state == .playing {
            t += max(now - anchorStamp, 0) * rate
            if t < shown, shown - t < Self.maxHold {
                return shown
            }
        }
        shown = t
        return t
    }
}
