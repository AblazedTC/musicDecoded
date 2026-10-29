import SwiftUI

/// Sign up / log in (T-403i). After success: the migration screen when this device has guest
/// data, otherwise back to Settings. Ports AuthScreen.kt.
struct AuthView: View {
    let onMigrate: () -> Void
    let onDone: () -> Void

    @Environment(SessionStore.self) private var session
    @Environment(SyncService.self) private var sync
    @State private var model: AuthModel
    @FocusState private var focusedField: Field?

    private enum Field: Hashable {
        case email, password
    }

    init(mode: AuthMode, onMigrate: @escaping () -> Void, onDone: @escaping () -> Void) {
        self.onMigrate = onMigrate
        self.onDone = onDone
        _model = State(initialValue: AuthModel(mode: mode))
    }

    var body: some View {
        let isSignUp = model.mode == .signUp
        ScrollView {
            VStack(alignment: .leading, spacing: Spacing.md) {
                VStack(alignment: .leading, spacing: Spacing.xxs) {
                    Text("Account")
                        .overlineStyle()
                    Text(isSignUp ? "Create account" : "Log in")
                        .font(.pageTitle)
                        .foregroundStyle(Palette.text)
                        .accessibilityAddTraits(.isHeader)
                }
                Text("Sync your chord timelines and your own audio across your devices. Music already on this device comes with you.")
                    .font(.detail)
                    .foregroundStyle(Palette.textSecondary)

                SegmentedToggle(options: AuthMode.allCases, selection: $model.mode) { $0.title }

                AuthField(title: "Email") {
                    TextField("Email", text: $model.email, prompt: Text("you@example.com"))
                        .textContentType(.emailAddress)
                        .keyboardType(.emailAddress)
                        .textInputAutocapitalization(.never)
                        .autocorrectionDisabled()
                        .submitLabel(.next)
                        .focused($focusedField, equals: .email)
                        .onSubmit { focusedField = .password }
                }
                AuthField(title: "Password") {
                    SecureField("Password", text: $model.password)
                        .textContentType(isSignUp ? UITextContentType.newPassword : UITextContentType.password)
                        .submitLabel(.done)
                        .focused($focusedField, equals: .password)
                        .onSubmit(submit)
                }
                if isSignUp {
                    Text("At least 8 characters.")
                        .font(.overline)
                        .foregroundStyle(Palette.textMuted)
                }
                if let error = model.error {
                    Text(error)
                        .font(.detail)
                        .foregroundStyle(Palette.record)
                        .accessibilityAddTraits(.updatesFrequently)
                }
                PrimaryButton(title: buttonTitle, action: submit)
                    .disabled(model.isBusy)
            }
            .padding(.horizontal, Spacing.md)
            .padding(.vertical, Spacing.lg)
        }
        .scrollDismissesKeyboard(.interactively)
        .background(Palette.canvas)
        .navigationBarTitleDisplayMode(.inline)
        .toolbarBackground(Palette.canvas, for: .navigationBar)
        .toolbar(.hidden, for: .tabBar)
        .onChange(of: model.email) { model.error = nil }
        .onChange(of: model.password) { model.error = nil }
        .onChange(of: model.mode) { model.error = nil }
        .onChange(of: model.outcome) { _, outcome in
            switch outcome {
            case .some(.migrate): onMigrate()
            case .some(.done): onDone()
            case .none: break
            }
        }
    }

    private var buttonTitle: String {
        switch (model.mode, model.isBusy) {
        case (.signUp, true): "Creating account…"
        case (.logIn, true): "Logging in…"
        case (.signUp, false): "Create account"
        case (.logIn, false): "Log in"
        }
    }

    private func submit() {
        let session = session
        let sync = sync
        let actions = AuthActions(
            signUp: { email, password in try await session.signUp(email: email, password: password) },
            logIn: { email, password in try await session.logIn(email: email, password: password) },
            hasGuestData: { sync.unsyncedItems().isEmpty == false },
            requestSync: { sync.requestSync() }
        )
        Task {
            await model.submit(using: actions)
        }
    }
}

/// A labeled text field in the app's style.
private struct AuthField<Content: View>: View {
    let title: String
    @ViewBuilder let field: Content

    var body: some View {
        VStack(alignment: .leading, spacing: Spacing.xxs) {
            Text(title)
                .font(.detail.weight(.semibold))
                .foregroundStyle(Palette.textSecondary)
            field
                .font(.body)
                .foregroundStyle(Palette.text)
                .padding(.horizontal, Spacing.sm)
                .frame(minHeight: ComponentSize.searchFieldHeight)
                .background(Palette.surface, in: .rect(cornerRadius: Radius.control))
                .overlay {
                    RoundedRectangle(cornerRadius: Radius.control).strokeBorder(Palette.line, lineWidth: 1)
                }
        }
    }
}

#if DEBUG
#Preview {
    NavigationStack {
        AuthView(mode: .signUp, onMigrate: {}, onDone: {})
    }
    .environment(SessionStore.preview())
    .environment(SyncService.preview())
    .preferredColorScheme(.dark)
}
#endif
