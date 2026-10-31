import Foundation

/// A non-2xx response. Decodes the `Error` schema (`{code, message}`) from contracts/openapi.yaml;
/// named `APIError` because `Error` would shadow `Swift.Error`.
struct APIError: Error, Decodable, Equatable, Sendable {
    enum Code: String, Decodable, Sendable {
        case badRequest = "bad_request"
        case unauthorized
        case forbidden
        case notFound = "not_found"
        case conflict
        case tooLarge = "too_large"
        case rateLimited = "rate_limited"
        case featureDisabled = "feature_disabled"
        case accountRequired = "account_required"
        case proRequired = "pro_required"
        case internalError = "internal"
        /// A code this build doesn't know, or a body that isn't an `Error`.
        case unknown

        init(from decoder: any Decoder) throws {
            let rawValue = try decoder.singleValueContainer().decode(String.self)
            self = Code(rawValue: rawValue) ?? .unknown
        }
    }

    var code: Code
    var message: String
    /// The HTTP status code. Not part of the body.
    var status = 0

    private enum CodingKeys: String, CodingKey {
        case code, message
    }
}

extension APIError {
    /// Builds the error for a response with `status` and `body`, falling back to the status text
    /// when the body isn't an `Error`.
    init(status: Int, body: Data) {
        let decoded = try? JSONDecoder().decode(APIError.self, from: body)
        self.init(
            code: decoded?.code ?? .unknown,
            message: decoded?.message ?? HTTPURLResponse.localizedString(forStatusCode: status),
            status: status
        )
    }
}

extension APIError: LocalizedError {
    var errorDescription: String? { message }
}
