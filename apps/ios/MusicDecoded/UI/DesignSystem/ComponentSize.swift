import CoreGraphics

/// Fixed component sizes from DESIGN.md §1. These are sizes, not spacing.
enum ComponentSize {
    /// Smallest tappable area on iOS. Pad smaller visuals up to it.
    static let minTouchTarget: CGFloat = 44
    static let iconButton: CGFloat = 40
    static let smallIconButton: CGFloat = 32
    static let playButton: CGFloat = 56
    static let recordButton: CGFloat = 84
    static let songArt: CGFloat = 56
    static let sourceIconTile: CGFloat = 44
    static let primaryButtonHeight: CGFloat = 48
    static let searchFieldHeight: CGFloat = 44
    /// Recorder: live waveform drawing height and input-meter segment height.
    static let recorderWaveformHeight: CGFloat = 112
    static let meterSegmentHeight: CGFloat = 20
    /// Piano keyboard: 2 octaves in the player's visualizer card, edge to edge in the learning view.
    static let pianoCompactHeight: CGFloat = 96
    static let pianoLargeHeight: CGFloat = 190
    /// The big play button in the Piano learning view.
    static let learningPlayButton: CGFloat = 72
}
