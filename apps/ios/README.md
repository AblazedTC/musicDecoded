# MusicDecoded iOS

SwiftUI app for iOS 17+, Swift 6.

```
apps/ios/
  project.yml                  XcodeGen spec (the .xcodeproj is generated, not committed)
  Config/                      Debug/Release xcconfig (API_BASE_URL)
  MusicDecoded/                app sources (App/, UI/…)
  MusicDecodedTests/           app unit tests (Swift Testing)
  Packages/MusicDecodedCore/   pure logic: analysis model, chord voicings, theory helpers
```

## Generate and build the app (macOS)

```sh
brew install xcodegen
cd apps/ios
xcodegen generate            # writes MusicDecoded.xcodeproj
open MusicDecoded.xcodeproj
```

Run `xcodegen generate` again after pulling changes to `project.yml` or adding/removing files.
From the command line:

```sh
xcodebuild -scheme MusicDecoded -destination 'platform=iOS Simulator,name=iPhone 16' build test
```

Debug builds talk to `http://localhost:8080` (set in `Config/Debug.xcconfig`, with an ATS
exception for localhost in Debug only).

## Test the core package (macOS, Linux, Windows)

`MusicDecodedCore` uses only Foundation, so it builds anywhere the Swift toolchain runs,
including Windows and Linux machines without Xcode:

```sh
cd apps/ios/Packages/MusicDecodedCore
swift build
swift test
```

Get the toolchain (Swift 6.0 or later) from [swift.org/install](https://www.swift.org/install/).
On Windows: `winget install --id Swift.Toolchain -e`, which also needs the Visual Studio
Build Tools (C++ workload and Windows SDK).

The tests read `contracts/fixtures/sample.analysis.json` and `contracts/chord-voicings.json`
straight from the repo, so contract changes are tested immediately.

## CI

`.github/workflows/ios.yml` runs on every push and PR that touches `apps/ios/**`,
`contracts/**` or the workflow itself:

- **core-linux**: `swift test` for `MusicDecodedCore` in the `swift:6.0` container.
- **app-macos**: generates the project with XcodeGen, then builds the app and runs its tests on
  an iPhone simulator. This is how the app gets built when no Mac is around.
