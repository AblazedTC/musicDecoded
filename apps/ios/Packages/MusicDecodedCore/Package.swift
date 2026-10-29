// swift-tools-version: 6.0
import PackageDescription

// Pure logic shared by the app: the analysis model, chord voicings and the
// player's music-theory helpers. Foundation only, so `swift test` runs on
// macOS, Linux and Windows.
let package = Package(
    name: "MusicDecodedCore",
    platforms: [.iOS(.v17), .macOS(.v14)],
    products: [
        .library(name: "MusicDecodedCore", targets: ["MusicDecodedCore"]),
    ],
    targets: [
        .target(name: "MusicDecodedCore"),
        .testTarget(name: "MusicDecodedCoreTests", dependencies: ["MusicDecodedCore"]),
    ]
)
