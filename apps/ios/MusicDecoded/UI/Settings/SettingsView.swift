import Foundation
import MusicDecodedCore
import SwiftData
import SwiftUI

/// DESIGN.md §3.9 (v1 rows only): profile card, guest sync card, preferences, storage, about.
/// Ports SettingsScreen.kt.
struct SettingsView: View {
    @Environment(SessionStore.self) private var session
    @Environment(SettingsStore.self) private var settings
    @Environment(SyncService.self) private var sync
    @Environment(\.openPaywall) private var openPaywall
    @Query(LibraryItem.newestFirst) private var visible: [LibraryItem]
    @Query(LibraryItem.offlineItems) private var offline: [LibraryItem]
    @State private var path: [AccountRoute] = []
    @State private var storageBytes: Int64 = 0
    @State private var info: InfoTopic?

    var body: some View {
        let prefs = settings.preferences
        NavigationStack(path: $path) {
            ScrollView {
                VStack(alignment: .leading, spacing: Spacing.lg) {
                    VStack(alignment: .leading, spacing: Spacing.xxs) {
                        Text("Preferences & account")
                            .overlineStyle()
                        Text("Settings")
                            .font(.pageTitle)
                            .foregroundStyle(Palette.text)
                            .accessibilityAddTraits(.isHeader)
                    }

                    ProfileCard(
                        isAccount: session.isAccount,
                        isPro: session.isPro,
                        email: session.email,
                        onAuth: { path.append(.auth($0)) },
                        onLogOut: logOut
                    )
                    if session.isAccount == false {
                        SyncCard {
                            path.append(.auth(.signUp))
                        }
                    }
                    // Free accounts only (guests sign up from the profile card); `openPaywall` is nil for Pro.
                    if session.isAccount, let openPaywall {
                        SettingsGroup(title: "MusicDecoded Pro") {
                            Button {
                                openPaywall()
                            } label: {
                                SettingsRowLabel(label: "Go Pro", value: "Decode more", showsChevron: true)
                            }
                            .buttonStyle(.plain)
                        }
                    }

                    SettingsGroup(title: "Playback") {
                        ChoiceRow(
                            label: "Default speed",
                            options: PlayerPreferences.speeds,
                            selected: prefs.defaultSpeed,
                            title: PlayerFormat.speed
                        ) { value in
                            settings.update { $0.defaultSpeed = value }
                        }
                    }
                    SettingsGroup(title: "Music analysis") {
                        ChoiceRow(
                            label: "Chord notation",
                            options: [ChordsMode.full, ChordsMode.basic],
                            selected: prefs.chordsMode,
                            title: { $0.title }
                        ) { value in
                            settings.update { $0.chordsMode = value }
                        }
                        RowDivider()
                        ChoiceRow(
                            label: "Accidentals",
                            options: Accidentals.allCases,
                            selected: prefs.accidentals,
                            title: Self.accidentalsTitle
                        ) { value in
                            settings.update { $0.accidentals = value }
                        }
                        RowDivider()
                        ChoiceRow(
                            label: "Default visualization",
                            options: Visualizer.allCases,
                            selected: prefs.visualizer,
                            title: { $0.title }
                        ) { value in
                            settings.update { $0.visualizer = value }
                        }
                    }
                    SettingsGroup(title: "Storage") {
                        SettingsRowLabel(label: "Local storage", value: AccountFormat.bytes(storageBytes), showsChevron: false)
                        RowDivider()
                        SettingsRowLabel(label: "Offline files", value: "\(offline.count)", showsChevron: false)
                    }
                    SettingsGroup(title: "Application") {
                        Button {
                            info = .privacy
                        } label: {
                            SettingsRowLabel(label: "Privacy & data", value: nil, showsChevron: true)
                        }
                        .buttonStyle(.plain)
                        RowDivider()
                        Button {
                            info = .help
                        } label: {
                            SettingsRowLabel(label: "Help & support", value: nil, showsChevron: true)
                        }
                        .buttonStyle(.plain)
                    }
                }
                .padding(.horizontal, Spacing.md)
                .padding(.vertical, Spacing.lg)
            }
            .background(Palette.canvas)
            .toolbar(.hidden, for: .navigationBar)
            .navigationDestination(for: AccountRoute.self) { route in
                switch route {
                case .auth(let mode):
                    AuthView(
                        mode: mode,
                        onMigrate: { path = [.migrate] },
                        onDone: { path.removeAll() }
                    )
                case .migrate:
                    MigrationView(onDone: { path.removeAll() })
                }
            }
            .sheet(item: $info) { topic in
                InfoSheet(topic: topic)
            }
            // Storage is re-measured whenever the library changes.
            .task(id: "\(visible.count)-\(offline.count)") {
                storageBytes = Self.directorySize(sync.audioDirectory) + Self.directorySize(sync.analysisStore.directory)
            }
        }
    }

    /// Back to this device's guest session; the local library stays (T-403i).
    private func logOut() {
        Task {
            await session.logOut()
        }
    }

    static func accidentalsTitle(_ accidentals: Accidentals) -> String {
        switch accidentals {
        case .auto: "Auto"
        case .sharp: "♯"
        case .flat: "♭"
        }
    }

    /// Bytes of the regular files under `url`.
    static func directorySize(_ url: URL) -> Int64 {
        let keys: [URLResourceKey] = [.fileSizeKey, .isRegularFileKey]
        guard let enumerator = FileManager.default.enumerator(at: url, includingPropertiesForKeys: keys) else {
            return 0
        }
        var total: Int64 = 0
        while let file = enumerator.nextObject() as? URL {
            let values = try? file.resourceValues(forKeys: Set(keys))
            if values?.isRegularFile == true {
                total += Int64(values?.fileSize ?? 0)
            }
        }
        return total
    }
}

/// Privacy & data / Help & support, shown in-app until there's a policy URL (T-504).
enum InfoTopic: String, Identifiable {
    case privacy, help

    var id: String { rawValue }

    var title: String {
        switch self {
        case .privacy: "Privacy & data"
        case .help: "Help & support"
        }
    }

    var systemImage: String {
        switch self {
        case .privacy: "hand.raised"
        case .help: "questionmark.circle"
        }
    }

    var text: String {
        switch self {
        case .privacy:
            "Your library lives on this device. Files and recordings you decode are uploaded to our servers to be analyzed. "
                + "A copy stays on this device for offline playback. With an account, your library, chord timelines and your own "
                + "audio sync across your devices. YouTube and SoundCloud audio is never uploaded from this device."
        case .help:
            "Decode a song from Home: import a file, record, or pick a YouTube video. Open it to see the chord timeline. "
                + "Tap a chord to jump to it, drag to scrub, and tap loop to repeat the current bar. Speed changes keep the pitch."
        }
    }
}

private struct InfoSheet: View {
    let topic: InfoTopic

    @Environment(\.dismiss) private var dismiss

    var body: some View {
        VStack(spacing: Spacing.md) {
            Image(systemName: topic.systemImage)
                .font(.title)
                .foregroundStyle(Palette.secondary)
                .accessibilityHidden(true)
            Text(topic.title)
                .font(.sectionHeader)
                .foregroundStyle(Palette.text)
                .accessibilityAddTraits(.isHeader)
            Text(topic.text)
                .font(.detail)
                .foregroundStyle(Palette.textSecondary)
                .frame(maxWidth: .infinity, alignment: .leading)
            PrimaryButton(title: "OK") {
                dismiss()
            }
        }
        .padding(Spacing.lg)
        .frame(maxHeight: .infinity, alignment: .top)
        .background(Palette.transportBackground)
        .presentationDetents([.medium, .large])
        .presentationDragIndicator(.visible)
    }
}

private struct ProfileCard: View {
    let isAccount: Bool
    /// From `/config` (the server decides Pro): shows a "Pro" chip.
    let isPro: Bool
    let email: String?
    let onAuth: (AuthMode) -> Void
    let onLogOut: () -> Void

    var body: some View {
        VStack(spacing: Spacing.md) {
            HStack(spacing: Spacing.sm) {
                Image(systemName: "person.fill")
                    .foregroundStyle(Palette.secondary)
                    .frame(width: ComponentSize.songArt, height: ComponentSize.songArt)
                    .background(Palette.surfaceSoft, in: .circle)
                    .accessibilityHidden(true)
                VStack(alignment: .leading, spacing: Spacing.xxs) {
                    if isAccount {
                        Text(email.flatMap { $0.split(separator: "@").first.map(String.init) } ?? "Account")
                            .font(.rowTitle)
                            .foregroundStyle(Palette.text)
                        Text([email, "Synced"].compactMap { $0 }.joined(separator: " · "))
                            .font(.detail)
                            .foregroundStyle(Palette.textSecondary)
                    } else {
                        Text("Guest account")
                            .font(.rowTitle)
                            .foregroundStyle(Palette.text)
                        Text("Music is stored on this device")
                            .font(.detail)
                            .foregroundStyle(Palette.textSecondary)
                    }
                }
                .frame(maxWidth: .infinity, alignment: .leading)
                .accessibilityElement(children: .combine)
                if isPro {
                    Text("Pro")
                        .font(.caption.weight(.semibold))
                        .foregroundStyle(Palette.onPrimary)
                        .padding(.horizontal, Spacing.xs)
                        .padding(.vertical, Spacing.xxs)
                        .background(Palette.primary, in: .rect(cornerRadius: Radius.chip))
                        .accessibilityLabel("Pro plan")
                }
                if isAccount {
                    Image(systemName: "checkmark.circle.fill")
                        .foregroundStyle(Palette.primary)
                        .accessibilityLabel("Synced")
                }
            }
            if isAccount {
                SecondaryButton(title: "Log out", action: onLogOut)
            } else {
                PrimaryButton(title: "Create account") {
                    onAuth(.signUp)
                }
                Button {
                    onAuth(.logIn)
                } label: {
                    Text("I already have an account")
                        .font(.detail.weight(.semibold))
                        .foregroundStyle(Palette.primary)
                        .frame(maxWidth: .infinity, minHeight: ComponentSize.minTouchTarget)
                        .contentShape(.rect)
                }
                .buttonStyle(.plain)
            }
        }
        .padding(Spacing.md)
        .background(Palette.surface, in: .rect(cornerRadius: Radius.card))
        .overlay {
            RoundedRectangle(cornerRadius: Radius.card).strokeBorder(Palette.line, lineWidth: 1)
        }
    }
}

private struct SyncCard: View {
    let action: () -> Void

    var body: some View {
        Button(action: action) {
            HStack(spacing: Spacing.sm) {
                Image(systemName: "icloud")
                    .foregroundStyle(Palette.primary)
                    .accessibilityHidden(true)
                VStack(alignment: .leading, spacing: Spacing.xxs) {
                    Text("Sync your music")
                        .font(.rowTitle)
                        .foregroundStyle(Palette.text)
                    Text("Keep eligible chord timelines and files available across your devices.")
                        .font(.detail)
                        .foregroundStyle(Palette.textSecondary)
                }
                .frame(maxWidth: .infinity, alignment: .leading)
                Image(systemName: "arrow.right")
                    .foregroundStyle(Palette.primary)
                    .accessibilityHidden(true)
            }
            .padding(Spacing.md)
            .background(Palette.primarySoft, in: .rect(cornerRadius: Radius.card))
            .contentShape(.rect)
        }
        .buttonStyle(.plain)
    }
}

private struct SettingsGroup<Content: View>: View {
    let title: String
    @ViewBuilder let content: Content

    var body: some View {
        VStack(alignment: .leading, spacing: Spacing.xs) {
            Text(title)
                .overlineStyle()
                .accessibilityAddTraits(.isHeader)
            VStack(spacing: 0) {
                content
            }
            .background(Palette.surface, in: .rect(cornerRadius: Radius.card))
            .overlay {
                RoundedRectangle(cornerRadius: Radius.card).strokeBorder(Palette.line, lineWidth: 1)
            }
        }
    }
}

private struct RowDivider: View {
    var body: some View {
        Rectangle()
            .fill(Palette.line)
            .frame(height: 1)
            .padding(.leading, Spacing.md)
    }
}

/// Label · value · chevron. Read-only rows have no chevron.
private struct SettingsRowLabel: View {
    let label: String
    let value: String?
    let showsChevron: Bool

    var body: some View {
        HStack(spacing: Spacing.xs) {
            Text(label)
                .font(.rowTitle)
                .foregroundStyle(Palette.text)
                .frame(maxWidth: .infinity, alignment: .leading)
            if let value {
                Text(value)
                    .font(.detail)
                    .foregroundStyle(Palette.textSecondary)
            }
            if showsChevron {
                Image(systemName: "chevron.right")
                    .font(.detail.weight(.semibold))
                    .foregroundStyle(Palette.textMuted)
                    .accessibilityHidden(true)
            }
        }
        .padding(.horizontal, Spacing.md)
        .padding(.vertical, Spacing.sm)
        .frame(minHeight: ComponentSize.minTouchTarget)
        .contentShape(.rect)
        .accessibilityElement(children: .combine)
    }
}

/// A row that opens a menu of choices, with a checkmark on the current one.
private struct ChoiceRow<Option: Hashable>: View {
    let label: String
    let options: [Option]
    let selected: Option
    let title: (Option) -> String
    let onSelect: (Option) -> Void

    var body: some View {
        Menu {
            ForEach(options, id: \.self) { option in
                Button {
                    onSelect(option)
                } label: {
                    if option == selected {
                        Label(title(option), systemImage: "checkmark")
                    } else {
                        Text(title(option))
                    }
                }
            }
        } label: {
            SettingsRowLabel(label: label, value: title(selected), showsChevron: true)
        }
        .buttonStyle(.plain)
    }
}

#if DEBUG
#Preview {
    SettingsView()
        .environment(SessionStore.preview())
        .environment(SettingsStore())
        .environment(SyncService.preview())
        .modelContainer(for: LibraryItem.self, inMemory: true)
}
#endif
