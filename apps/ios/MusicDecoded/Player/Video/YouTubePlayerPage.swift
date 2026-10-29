import Foundation

/// The HTML page that hosts the IFrame Player API for `YouTubePlayerController`.
/// It posts `{event, ...}` objects to `window.webkit.messageHandlers.ytBridge`
/// (parsed by `YouTubeBridgeMessage`) and exposes the `player` global for commands.
enum YouTubePlayerPage {
    /// The script message handler name.
    static let messageName = "ytBridge"
    /// How often the page reports `currentTime`, in milliseconds.
    static let timeReportInterval = 250

    /// `videoId` must be a valid id (`YouTubeVideoID.isValid`): it's written into the script.
    static func html(videoId: String) -> String {
        let origin = YouTubeEmbed.origin.absoluteString
        return """
        <!DOCTYPE html>
        <html><head>
        <meta name="viewport" content="width=device-width, initial-scale=1, maximum-scale=1, user-scalable=no">
        <style>html,body{margin:0;padding:0;height:100%;background:#000;overflow:hidden}#player{position:absolute;inset:0;width:100%;height:100%}</style>
        </head><body>
        <div id="player"></div>
        <script>
        function post(message) { window.webkit.messageHandlers.\(messageName).postMessage(message); }
        var player;
        function onYouTubeIframeAPIReady() {
          player = new YT.Player('player', {
            videoId: '\(videoId)',
            width: '100%',
            height: '100%',
            playerVars: { playsinline: 1, rel: 0, enablejsapi: 1, origin: '\(origin)' },
            events: {
              onReady: function () {
                post({ event: 'ready' });
                setInterval(function () { post({ event: 'time', time: player.getCurrentTime() }); }, \(timeReportInterval));
              },
              onStateChange: function (e) { post({ event: 'state', state: e.data, time: player.getCurrentTime() }); },
              onPlaybackRateChange: function (e) { post({ event: 'rate', rate: e.data }); },
              onError: function (e) { post({ event: 'error', code: e.data }); }
            }
          });
        }
        </script>
        <script src="https://www.youtube.com/iframe_api" onerror="post({ event: 'error', code: -1 })"></script>
        </body></html>
        """
    }
}
