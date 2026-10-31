import Foundation

/// Response bodies shaped like contracts/openapi.yaml.
enum SampleJSON {
    static let authResponse = Data("""
    {"token": "guest-token-1", "userId": "user-1", "kind": "guest", "plan": "free"}
    """.utf8)

    static let clientConfig = Data("""
    {
      "offlineStreamAudio": false,
      "maxUploadBytes": 104857600,
      "soundcloudEnabled": false,
      "plan": "free",
      "limits": {"dailyJobs": 10, "jobsUsedToday": 2}
    }
    """.utf8)

    static let unauthorized = Data("""
    {"code": "unauthorized", "message": "Token expired"}
    """.utf8)

    static let notFound = Data("""
    {"code": "not_found", "message": "No such route"}
    """.utf8)
}
