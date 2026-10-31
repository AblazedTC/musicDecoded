import SwiftUI

/// DESIGN.md §3.4 search: an emphasized search field, recent-query chips, "Top results" with
/// infinite scroll, and the "Paste a video URL" card at the bottom.
struct YouTubeSearchScreen: View {
    /// Pushes the video detail (a result or a pasted link).
    let onOpenVideo: (VideoSummary) -> Void

    @Environment(SessionStore.self) private var session
    @State private var model: YouTubeSearchModel?

    var body: some View {
        Group {
            if let model {
                YouTubeSearchContent(model: model, onOpenVideo: onOpenVideo)
            } else {
                Palette.canvas
            }
        }
        .background(Palette.canvas)
        .navigationTitle("YouTube")
        .navigationBarTitleDisplayMode(.inline)
        .toolbarBackground(Palette.canvas, for: .navigationBar)
        .toolbar(.hidden, for: .tabBar)
        .onAppear {
            if model == nil {
                model = YouTubeSearchModel(session: session)
            }
        }
    }
}

private struct YouTubeSearchContent: View {
    @Bindable var model: YouTubeSearchModel
    let onOpenVideo: (VideoSummary) -> Void
    @FocusState private var searchFocused: Bool

    var body: some View {
        ScrollView {
            LazyVStack(alignment: .leading, spacing: Spacing.md) {
                YouTubeSearchField(text: $model.query, isFocused: $searchFocused) {
                    model.submit()
                }
                if model.recentQueries.isEmpty == false {
                    recentChips
                }
                results
                PasteVideoURLCard(onOpen: onOpenVideo)
                    .padding(.top, Spacing.xs)
            }
            .padding(.horizontal, Spacing.md)
            .padding(.vertical, Spacing.md)
        }
        .scrollDismissesKeyboard(.interactively)
        .background(Palette.canvas)
        .onAppear {
            if model.query.isEmpty {
                searchFocused = true
            }
        }
    }

    private var recentChips: some View {
        ScrollView(.horizontal) {
            HStack(spacing: Spacing.xs) {
                ForEach(model.recentQueries, id: \.self) { recent in
                    FilterChip(title: recent, isSelected: false) {
                        searchFocused = false
                        model.selectRecent(recent)
                    }
                }
            }
        }
        .scrollIndicators(.hidden)
        .accessibilityLabel("Recent searches")
    }

    @ViewBuilder
    private var results: some View {
        switch model.phase {
        case .idle:
            Text("Search for a song to decode its chords.")
                .font(.detail)
                .foregroundStyle(Palette.textSecondary)
        case .loading:
            ProgressView()
                .tint(Palette.primary)
                .frame(maxWidth: .infinity, minHeight: ComponentSize.videoHeroHeight)
        case .failed(let failure):
            SearchFailureCard(failure: failure) {
                model.retry()
            }
        case .loaded:
            if model.results.isEmpty {
                Text("No videos found. Try another search.")
                    .font(.detail)
                    .foregroundStyle(Palette.textSecondary)
            } else {
                SectionHeader(title: "Top results")
                ForEach(model.results) { video in
                    Button {
                        searchFocused = false
                        model.didOpen(video)
                        onOpenVideo(video)
                    } label: {
                        VideoResultRow(video: video)
                    }
                    .buttonStyle(.plain)
                    .onAppear {
                        model.loadMoreIfNeeded(after: video)
                    }
                }
                loadMoreFooter
            }
        }
    }

    @ViewBuilder
    private var loadMoreFooter: some View {
        if model.isLoadingMore {
            ProgressView()
                .tint(Palette.primary)
                .frame(maxWidth: .infinity, minHeight: ComponentSize.minTouchTarget)
        } else if model.loadMoreFailed {
            Button("Couldn't load more results. Retry") {
                model.loadMore()
            }
            .font(.labelSmall)
            .foregroundStyle(Palette.primary)
            .frame(maxWidth: .infinity, minHeight: ComponentSize.minTouchTarget)
        }
    }
}

/// The search field: a `surface` input with a lime border and violet hard shadow.
private struct YouTubeSearchField: View {
    @Binding var text: String
    var isFocused: FocusState<Bool>.Binding
    let onSubmit: () -> Void

    var body: some View {
        HStack(spacing: Spacing.xs) {
            Image(systemName: "magnifyingglass")
                .fontWeight(.bold)
                .foregroundStyle(Palette.primary)
                .accessibilityHidden(true)
            TextField("Search songs on YouTube", text: $text)
                .font(.bodyText)
                .foregroundStyle(Palette.text)
                .tint(Palette.primary)
                .textInputAutocapitalization(.never)
                .autocorrectionDisabled()
                .submitLabel(.search)
                .focused(isFocused)
                .onSubmit(onSubmit)
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
        .block(Palette.surface, shadow: HardShadow.sm, shadowColor: Palette.violet, border: Palette.primary)
    }
}

/// Unavailable / quota / offline / other. Never the plan upsell: a 429 here is the server's quota.
private struct SearchFailureCard: View {
    let failure: YouTubeSearchModel.Failure
    let onRetry: () -> Void

    var body: some View {
        ErrorState(message: failure.message, title: failure.title, onRetry: failure.canRetry ? onRetry : nil)
    }
}

#if DEBUG
#Preview {
    NavigationStack {
        YouTubeSearchScreen(onOpenVideo: { _ in })
    }
    .environment(SessionStore.preview())
    .preferredColorScheme(.dark)
}
#endif
