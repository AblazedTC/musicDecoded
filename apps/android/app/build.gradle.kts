plugins {
    alias(libs.plugins.android.application)
    alias(libs.plugins.kotlin.compose)
    alias(libs.plugins.kotlin.serialization)
    alias(libs.plugins.ksp)
}

/** `-PlocalRelease`: sign the release with the debug key as `….local` (see the release build type). */
val localRelease = project.hasProperty("localRelease")
/** Backend for `-PlocalRelease`: the emulator's host by default, or a LAN address via `-PapiBaseUrl`. */
val localApiUrl = (project.findProperty("apiBaseUrl") as String?) ?: "http://10.0.2.2:8080"

android {
    namespace = "com.musicdecoded.mobile"
    compileSdk {
        version = release(36) {
            minorApiLevel = 1
        }
    }

    defaultConfig {
        applicationId = "com.musicdecoded.mobile"
        minSdk = 26
        targetSdk = 36
        versionCode = 1
        versionName = "1.0"

        testInstrumentationRunner = "androidx.test.runner.AndroidJUnitRunner"

        // Release base URL: override with -PapiBaseUrl=... until the production host is decided.
        val releaseApiBaseUrl = (project.findProperty("apiBaseUrl") as String?) ?: "https://api.musicdecoded.app"
        buildConfigField("String", "API_BASE_URL", "\"$releaseApiBaseUrl\"")

        // RevenueCat public SDK key (PLAN.md §9): -PrevenuecatApiKey=... or REVENUECAT_ANDROID_API_KEY.
        // Never commit a key. Empty = purchases unavailable in this build (the paywall says so).
        val revenueCatApiKey = (project.findProperty("revenuecatApiKey") as String?)
            ?: providers.environmentVariable("REVENUECAT_ANDROID_API_KEY").orNull ?: ""
        buildConfigField("String", "REVENUECAT_API_KEY", "\"$revenueCatApiKey\"")
    }

    buildTypes {
        debug {
            // The host machine as seen from the Android emulator; cleartext is allowed only for it.
            buildConfigField("String", "API_BASE_URL", "\"http://10.0.2.2:8080\"")
        }
        release {
            isMinifyEnabled = true
            isShrinkResources = true
            proguardFiles(getDefaultProguardFile("proguard-android-optimize.txt"), "proguard-rules.pro")
            if (localRelease) {
                // -PlocalRelease: an optimized build to try on a device/emulator next to the debug app.
                // Debug-keystore signed and pointed at the local backend; never a store build.
                applicationIdSuffix = ".local"
                versionNameSuffix = "-local"
                signingConfig = signingConfigs.getByName("debug")
                buildConfigField("String", "API_BASE_URL", "\"$localApiUrl\"")
                resValue("string", "app_name", "MusicDecoded R")
            }
        }
    }
    compileOptions {
        sourceCompatibility = JavaVersion.VERSION_11
        targetCompatibility = JavaVersion.VERSION_11
    }
    buildFeatures {
        compose = true
        buildConfig = true
        resValues = true
    }
    // Robolectric Compose tests need the merged manifest (ui-test-manifest's ComponentActivity).
    testOptions.unitTests.isIncludeAndroidResources = true
    sourceSets {
        // Unit tests read the shared contracts (fixture, voicings) straight from the repo, never a copy.
        getByName("test").resources.srcDir("../../../contracts")
        // The app reads chord-voicings.json from assets, straight from the repo (no copy). Debug builds
        // also get the fixtures, so the player can be tried without the backend.
        getByName("main").assets.srcDir("../../../contracts")
        // The local release talks to the dev backend over HTTP, like debug.
        if (localRelease) {
            // Cleartext only for the configured backend host (and presigned storage URLs on it).
            val generated = layout.buildDirectory.dir("generated/localRelease/res").get().asFile
            generated.resolve("xml").mkdirs()
            generated.resolve("xml/network_security_config.xml").writeText(
                """<?xml version="1.0" encoding="utf-8"?>
                |<network-security-config>
                |    <domain-config cleartextTrafficPermitted="true">
                |        <domain includeSubdomains="false">${localApiUrl.substringAfter("//").substringBefore(":").substringBefore("/")}</domain>
                |    </domain-config>
                |</network-security-config>
                |""".trimMargin(),
            )
            getByName("release").res.srcDir(generated)
        }
    }
}

androidComponents {
    onVariants { variant ->
        // Only the JSON the app reads is packaged from contracts/; fixtures are debug-only.
        variant.androidResources.ignoreAssetsPatterns.addAll("*.py", "*.yaml", "*.schema.json")
        if (!variant.debuggable) variant.androidResources.ignoreAssetsPatterns.add("<dir>fixtures")
    }
}

dependencies {
    implementation(platform(libs.androidx.compose.bom))
    implementation(libs.androidx.activity.compose)
    implementation(libs.androidx.compose.material3)
    implementation(libs.androidx.compose.material3.adaptive.navigation.suite)
    implementation(libs.androidx.compose.ui)
    implementation(libs.androidx.compose.ui.graphics)
    implementation(libs.androidx.compose.ui.tooling.preview)
    implementation(libs.androidx.core.ktx)
    implementation(libs.androidx.lifecycle.runtime.ktx)
    implementation(libs.androidx.material.icons.extended)
    implementation(libs.ktor.client.android)
    implementation(libs.ktor.client.content.negotiation)
    implementation(libs.ktor.serialization.kotlinx.json)
    implementation(libs.ktor.client.logging)
    implementation(libs.navigation.compose)
    implementation(libs.datastore.preferences)
    implementation(libs.lifecycle.viewmodel.compose)
    implementation(libs.lifecycle.runtime.compose)
    implementation(libs.media3.exoplayer)
    implementation(libs.room.runtime)
    implementation(libs.work.runtime.ktx)
    // Play Billing through RevenueCat (PLAN.md §9); the API decides Pro from RevenueCat's webhook.
    implementation(libs.revenuecat.purchases)
    ksp(libs.room.compiler)
    testImplementation(libs.junit)
    testImplementation(libs.ktor.client.mock)
    testImplementation(libs.kotlinx.coroutines.test)
    testImplementation(libs.robolectric)
    // Compose semantics tests on Robolectric (T-503a); same catalog entry as the instrumented tests.
    testImplementation(platform(libs.androidx.compose.bom))
    testImplementation(libs.androidx.compose.ui.test.junit4)
    androidTestImplementation(platform(libs.androidx.compose.bom))
    androidTestImplementation(libs.androidx.compose.ui.test.junit4)
    androidTestImplementation(libs.androidx.espresso.core)
    androidTestImplementation(libs.androidx.junit)
    debugImplementation(libs.androidx.compose.ui.test.manifest)
    debugImplementation(libs.androidx.compose.ui.tooling)
}