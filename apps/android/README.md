# MusicDecoded Android

Kotlin + Jetpack Compose. Build and test: `./gradlew assembleDebug testDebugUnitTest` (see
[docs/running-locally.md](../../docs/running-locally.md)). Debug builds talk to `http://10.0.2.2:8080`, the host
machine as seen from the emulator.

## Release builds for Google Play

Release builds are signed with the Play **upload key** when it's configured, and unsigned otherwise. The key
never goes in the repo. Put these in `~/.gradle/gradle.properties` (or the matching `MUSICDECODED_UPLOAD_*`
environment variables, e.g. `MUSICDECODED_UPLOAD_STORE_FILE`):

```properties
musicdecodedUploadStoreFile=C:/path/to/musicdecoded-upload.jks
musicdecodedUploadStorePassword=...
musicdecodedUploadKeyAlias=upload
musicdecodedUploadKeyPassword=...
```

Then build the bundle. Every Play upload needs a higher `versionCode`:

```bash
./gradlew bundleRelease -PapiBaseUrl=https://api.example.com -PversionCode=2
# -> app/build/outputs/bundle/release/app-release.aab
```

`-PlocalRelease` instead builds a debug-signed `.local` APK for trying an optimized build next to the debug app.
