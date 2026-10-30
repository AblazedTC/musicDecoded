import SwiftData
import SwiftUI

/// DESIGN.md §3.2: search (title, artist, key), filter chips, decoded/offline counts and the
/// newest-first list. Searching and filtering run in the store (`LibraryItem.search`).
/// DESIGN.md §1 color map: a lime header block (ink text, stripe texture), then the ink list.
struct LibraryView: View {
    @State private var path = NavigationPath()
    @State private var searchText = ""
    @State private var filter = LibraryFilter.all
    @Query(LibraryItem.decodedItems) private var decoded: [LibraryItem]
    @Query(LibraryItem.offlineItems) private var offline: [LibraryItem]

    var body: some View {
        NavigationStack(path: $path) {
            ScrollView {
                VStack(alignment: .leading, spacing: 0) {
                    header
                    LazyVStack(alignment: .leading, spacing: Spacing.sm) {
                        VStack(alignment: .leading, spacing: Spacing.lg) {
                            LibrarySummaryCard(decoded: decoded.count, offline: offline.count)
                            SectionHeader(title: "Recently added")
                        }
                        .padding(.bottom, Spacing.xxs)
                        LibraryResults(
                            query: LibraryQuery(text: searchText, filter: filter),
                            onOpenPlayer: { path.append(PlayerRoute(trackId: $0)) },
                            onOpenProcessing: { openProcessing($0) }
                        )
                    }
                    .padding(.horizontal, Spacing.md)
                    .padding(.vertical, Spacing.xl)
                }
            }
            .scrollDismissesKeyboard(.interactively)
            .background(Palette.canvas)
            .miniPlayerInset { path.append(PlayerRoute(trackId: $0)) }
            .toolbar(.hidden, for: .navigationBar)
            .appDestinations(path: $path)
        }
    }

    /// The lime block: violet "Your collection" sticker, the LIBRARY title, search and filters.
    private var header: some View {
        VStack(alignment: .leading, spacing: Spacing.lg) {
            VStack(alignment: .leading, spacing: Spacing.sm) {
                Sticker(text: "Your collection", fill: Palette.violet, foreground: Palette.white)
                DisplayHeadline(text: "Library", color: Palette.onColor)
            }
            .padding(.top, Spacing.xs)
            .padding(.horizontal, Spacing.md)

            LibrarySearchField(text: $searchText)
                .padding(.horizontal, Spacing.md)

            filterChips
        }
        .padding(.top, Spacing.lg)
        .padding(.bottom, Spacing.md)
        .frame(maxWidth: .infinity, alignment: .leading)
        .background(alignment: .topTrailing) {
            StripeTexture()
                .frame(width: 112, height: 112)
        }
        .background(Palette.lime)
    }

    /// Scrolls edge to edge inside the lime block: the first chip lines up with the gutter, the
    /// last one gets the same trailing inset, and a lime fade at the trailing edge shows the row
    /// continues instead of looking clipped.
    private var filterChips: some View {
        ScrollView(.horizontal) {
            HStack(spacing: Spacing.sm) {
                ForEach(LibraryFilter.allCases, id: \.self) { option in
                    FilterChip(title: option.title, isSelected: filter == option, onColorBlock: true) {
                        filter = option
                    }
                }
            }
            .padding(.leading, Spacing.md)
            .padding(.trailing, Spacing.xxl)
            // Room for the selected chip's hard shadow.
            .padding(.bottom, HardShadow.sm)
        }
        .scrollIndicators(.hidden)
        .overlay(alignment: .trailing) {
            LinearGradient(colors: [Palette.lime.opacity(0), Palette.lime], startPoint: .leading, endPoint: .trailing)
                .frame(width: Spacing.xl)
                .allowsHitTesting(false)
                .accessibilityHidden(true)
        }
    }

    private func openProcessing(_ trackId: String) {
        path.append(ProcessingRoute(trackId: trackId))
    }
}

/// The rows for one query. A separate view so its `@Query` is rebuilt when the query changes.
private struct LibraryResults: View {
    let isUnfiltered: Bool
    let onOpenPlayer: (String) -> Void
    let onOpenProcessing: (String) -> Void
    @Query private var items: [LibraryItem]

    init(query: LibraryQuery, onOpenPlayer: @escaping (String) -> Void, onOpenProcessing: @escaping (String) -> Void) {
        isUnfiltered = query.isUnfiltered
        self.onOpenPlayer = onOpenPlayer
        self.onOpenProcessing = onOpenProcessing
        _items = Query(LibraryItem.search(query))
    }

    var body: some View {
        if items.isEmpty {
            Text(isUnfiltered ? "Your library is empty. Decode a song from Home to get started." : "No songs match.")
                .font(.bodyText)
                .foregroundStyle(Palette.textSecondary)
        }
        ForEach(Array(items.enumerated()), id: \.element.id) { index, item in
            LibrarySongRow(
                item: item,
                onOpenPlayer: onOpenPlayer,
                onOpenProcessing: onOpenProcessing,
                numeral: Self.numeral(index)
            )
        }
    }

    /// "01", "02"… for the decorative number behind each row.
    static func numeral(_ index: Int) -> String {
        let number = index + 1
        return number < 10 ? "0\(number)" : "\(number)"
    }
}

/// "Songs, artists, keys..." (44 tall): a white input with a black border and hard shadow on the
/// lime header, with a clear button.
private struct LibrarySearchField: View {
    @Binding var text: String

    var body: some View {
        HStack(spacing: Spacing.xs) {
            Image(systemName: "magnifyingglass")
                .font(.body.weight(.bold))
                .foregroundStyle(Palette.onColor)
                .accessibilityHidden(true)
            TextField(
                "Songs, artists, keys...",
                text: $text,
                prompt: Text("Songs, artists, keys...").foregroundStyle(Palette.textSecondaryOnLime)
            )
            .font(.bodyText)
            .foregroundStyle(Palette.onColor)
            .tint(Palette.violet)
            .textInputAutocapitalization(.never)
            .autocorrectionDisabled()
            .submitLabel(.search)
            if text.isEmpty == false {
                Button("Clear search", systemImage: "xmark.circle.fill") {
                    text = ""
                }
                .labelStyle(.iconOnly)
                .foregroundStyle(Palette.onColor)
                .frame(minWidth: ComponentSize.minTouchTarget, minHeight: ComponentSize.minTouchTarget)
            }
        }
        .padding(.leading, Spacing.sm)
        .frame(minHeight: ComponentSize.searchFieldHeight)
        .block(Palette.white, shadow: HardShadow.md)
    }
}

/// Decoded · available-offline counts (the grid/list toggle is post-v1, so it's hidden).
private struct LibrarySummaryCard: View {
    let decoded: Int
    let offline: Int

    var body: some View {
        AdaptiveStack(spacing: Spacing.lg) {
            count(decoded, decoded == 1 ? "song decoded" : "songs decoded", systemImage: "music.note")
            count(offline, "available offline", systemImage: "arrow.down.to.line")
        }
        .frame(maxWidth: .infinity, alignment: .leading)
        .padding(Spacing.md)
        .inkCard(shadow: HardShadow.lg)
    }

    private func count(_ value: Int, _ label: String, systemImage: String) -> some View {
        HStack(alignment: .firstTextBaseline, spacing: Spacing.xs) {
            Image(systemName: systemImage)
                .font(.body.weight(.bold))
                .foregroundStyle(Palette.primary)
                .accessibilityHidden(true)
            Text(value, format: .number)
                .font(.display(28, relativeTo: .title2))
                .monospacedDigit()
                .foregroundStyle(Palette.primary)
            Text(label)
                .font(.detail)
                .foregroundStyle(Palette.textSecondary)
        }
        .accessibilityElement(children: .combine)
    }
}

#if DEBUG
#Preview {
    LibraryView()
        .environment(TranscriptionService.preview())
        .environment(NowPlaying())
        .modelContainer(for: LibraryItem.self, inMemory: true)
        .preferredColorScheme(.dark)
}
#endif
