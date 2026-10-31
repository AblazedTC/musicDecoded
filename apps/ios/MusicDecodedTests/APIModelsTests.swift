import Foundation
import Testing
@testable import MusicDecoded

/// The DTOs decode bodies shaped exactly like contracts/openapi.yaml.
struct APIModelsTests {
    @Test func authResponseDecodes() throws {
        let response = try JSONDecoder().decode(AuthResponse.self, from: SampleJSON.authResponse)

        #expect(response == AuthResponse(token: "guest-token-1", userId: "user-1", kind: .guest, plan: .free))
    }

    @Test func clientConfigDecodes() throws {
        let config = try JSONDecoder().decode(ClientConfig.self, from: SampleJSON.clientConfig)

        #expect(config.offlineStreamAudio == false)
        #expect(config.maxUploadBytes == 104_857_600)
        #expect(config.soundcloudEnabled == false)
        #expect(config.plan == .free)
        #expect(config.limits == ClientConfig.Limits(dailyJobs: 10, jobsUsedToday: 2))
    }

    @Test func clientConfigAllowsUnlimitedJobsAndNoSoundcloudFlag() throws {
        let json = Data("""
        {"offlineStreamAudio": true, "maxUploadBytes": 1, "plan": "pro",
         "limits": {"dailyJobs": null, "jobsUsedToday": 0}}
        """.utf8)

        let config = try JSONDecoder().decode(ClientConfig.self, from: json)

        #expect(config.plan == .pro)
        #expect(config.limits.dailyJobs == nil)
        #expect(config.soundcloudEnabled == nil)
    }

    @Test func anonymousAuthRequestEncodesPlatform() throws {
        let data = try JSONEncoder().encode(AnonymousAuthRequest(deviceId: "device-1"))
        let object = try #require(JSONSerialization.jsonObject(with: data) as? [String: String])

        #expect(object == ["deviceId": "device-1", "platform": "ios"])
    }

    @Test func errorBodyDecodesWithStatus() {
        let error = APIError(status: 429, body: Data(#"{"code": "rate_limited", "message": "Daily limit reached"}"#.utf8))

        #expect(error == APIError(code: .rateLimited, message: "Daily limit reached", status: 429))
    }

    @Test func unknownErrorCodeDecodesAsUnknown() {
        let error = APIError(status: 400, body: Data(#"{"code": "brand_new", "message": "Something new"}"#.utf8))

        #expect(error.code == .unknown)
        #expect(error.message == "Something new")
    }

    @Test func nonJSONErrorBodyFallsBackToStatusText() {
        let error = APIError(status: 502, body: Data("Bad Gateway".utf8))

        #expect(error.code == .unknown)
        #expect(error.status == 502)
        #expect(error.message.isEmpty == false)
    }
}
