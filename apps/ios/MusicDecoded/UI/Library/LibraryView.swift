import SwiftData
import SwiftUI

/// DESIGN.md §3.2: search (title, artist, key), filter chips, decoded/offline counts and the
/// newest-first list. Searching and filtering run in the store (`LibraryItem.search`).
struct LibraryView: View {
    @State private var path = NavigationPath()
    @State private var searchText = ""
    @State private var filter = LibraryFilter.all
    @Query(LibraryItem.decodedItems) private var decoded: [LibraryItem]
    @Query(LibraryItem.offlineItems) private var offline: [LibraryItem]

    var body: some View {
        NavigationStack(path: $path) {
            ScrollView {
                LazyVStack(alignment: .leading, spacing: Spacing.xs) {
                    VStack(alignment: .leading, spacing: Spacing.md) {
                        VStack(alignment: .leading, spacing: Spacing.xxs) {
                            Text("Your collection")
                                .overlineStyle()
                            Text("Library")
                                .font(.pageTitle)
                                .foregroundStyle(Palette.text)
                                .accessibilityAddTraits(.isHeader)
                        }
                        LibrarySearchField(text: $searchText)
                        filterChips
                        LibrarySummaryCard(decoded: decoded.count, offline: offline.count)
                        SectionHeader(title: "Recently added")
                    }
                    LibraryResults(
                        query: LibraryQuery(text: searchText, filter: filter),
                        onOpenPlayer: { path.append(PlayerRoute(trackId: $0)) },
                        onOpenProcessing: { openProcessing($0) }
                    )
                }
                .padding(.horizontal, Spacing.md)
                .padding(.vertical, Spacing.lg)
            }
            .scrollDismissesKeyboard(.interactively)
            .background(Palette.canvas)
            .miniPlayerInset { path.append(PlayerRoute(trackId: $0)) }
            .toolbar(.hidden, for: .navigationBar)
            .appDestinations(path: $path)
        }
    }

    private var filterChips: some View {
        ScrollView(.horizontal) {
            HStack(spacing: Spacing.xs) {
                ForEach(LibraryFilter.allCases, id: \.self) { option in
                    FilterChip(title: option.title, isSelected: filter == option) {
                        filter = option
                    }
                }
            }
        }
        .scrollIndicators(.hidden)
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
                .font(.detail)
                .foregroundStyle(Palette.textSecondary)
        }
        ForEach(items) { item in
            LibrarySongRow(item: item, onOpenPlayer: onOpenPlayer, onOpenProcessing: onOpenProcessing)
        }
    }
}

/// "Songs, artists, keys..." (44 tall, `surface` fill) with a clear button.
private struct LibrarySearchField: View {
    @Binding var text: String

    var body: some View {
        HStack(spacing: Spacing.xs) {
            Image(systemName: "magnifyingglass")
                .foregroundStyle(Palette.textMuted)
                .accessibilityHidden(true)
            TextField("Songs, artists, keys...", text: $text)
                .font(.subheadline)
                .foregroundStyle(Palette.text)
                .tint(Palette.primary)
                .textInputAutocapitalization(.never)
                .autocorrectionDisabled()
                .submitLabel(.search)
            if text.isEmpty == false {
                Button("Clear search", systemImage: "xmark.circle.fill") {
                    text = ""
                }
                .labelStyle(.iconOnly)
                .foregroundStyle(Palette.textSecondary)
                .frame(minWidth: ComponentSize.minTouchTarget, minHeight: ComponentSize.minTouchTarget)
            }
        }
        .padding(.leading, Spacing.sm)
        .frame(minHeight: ComponentSize.searchFieldHeight)
        .background(Palette.surface, in: .rect(cornerRadius: Radius.control))
        .overlay {
            RoundedRectangle(cornerRadius: Radius.control).strokeBorder(Palette.line, lineWidth: 1)
        }
    }
}

/// Decoded · available-offline counts (the grid/list toggle is post-v1, so it's hidden).
private struct LibrarySummaryCard: View {
    let decoded: Int
    let offline: Int

    var body: some View {
        HStack(spacing: Spacing.lg) {
            count(decoded, decoded == 1 ? "song decoded" : "songs decoded", systemImage: "music.note")
            count(offline, "available offline", systemImage: "arrow.down.circle")
        }
        .frame(maxWidth: .infinity, alignment: .leading)
        .padding(Spacing.md)
        .background(Palette.surface, in: .rect(cornerRadius: Radius.row))
        .overlay {
            RoundedRectangle(cornerRadius: Radius.row).strokeBorder(Palette.line, lineWidth: 1)
        }
    }

    private func count(_ value: Int, _ label: String, systemImage: String) -> some View {
        HStack(spacing: Spacing.xs) {
            Image(systemName: systemImage)
                .foregroundStyle(Palette.primary)
                .accessibilityHidden(true)
            Text(value, format: .number)
                .font(.rowTitle)
                .monospacedDigit()
                .foregroundStyle(Palette.text)
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
