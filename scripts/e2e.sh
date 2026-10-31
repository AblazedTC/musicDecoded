#!/usr/bin/env bash
# End-to-end pipeline check (T-114): guest auth -> upload mp3 -> job -> poll -> download analysis -> validate.
#
# Needs a running API + worker (docs/running-locally.md), curl, and a Python with jsonschema.
#   scripts/e2e.sh
#   API_URL=http://127.0.0.1:8080 PYTHON=apps/worker/.venv/Scripts/python scripts/e2e.sh
# Env: API_URL (default http://127.0.0.1:8080), PYTHON (default python3, else python),
#      AUDIO (default apps/worker/data/sunnydays.mp3), E2E_TIMEOUT seconds (default 300).
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
API="${API_URL:-http://127.0.0.1:8080}"
AUDIO="${AUDIO:-$ROOT/apps/worker/data/sunnydays.mp3}"
TIMEOUT="${E2E_TIMEOUT:-300}"
PY="${PYTHON:-$(command -v python3 >/dev/null 2>&1 && echo python3 || echo python)}"
TMP="$(mktemp -d)"
trap 'rm -rf "$TMP"' EXIT

step() { printf '\n==> %s\n' "$*"; }
fail() { printf 'FAIL: %s\n' "$*" >&2; exit 1; }
# Windows Python (Git Bash) can't open /c/... or /tmp/... paths; give it C:/... ones.
native() { if command -v cygpath >/dev/null 2>&1; then cygpath -m "$1"; else printf '%s' "$1"; fi; }
# json_get FILE FIELD -> value ("" for null/missing)
json_get() {
  "$PY" -c 'import json,sys; v=json.load(open(sys.argv[1])).get(sys.argv[2]); print("" if v is None else v)' \
    "$(native "$1")" "$2"
}
# api METHOD PATH [JSON_BODY] -> response body in $TMP/resp.json; fails on a non-2xx status
api() {
  local method="$1" path="$2" body="${3:-}" code
  local args=(-sS -o "$TMP/resp.json" -w '%{http_code}' -X "$method" "$API$path" -H 'Accept: application/json')
  [[ -n "${TOKEN:-}" ]] && args+=(-H "Authorization: Bearer $TOKEN")
  [[ -n "$body" ]] && args+=(-H 'Content-Type: application/json' --data "$body")
  code="$(curl "${args[@]}")" || fail "$method $path: request failed"
  [[ "$code" == 2* ]] || fail "$method $path -> HTTP $code: $(cat "$TMP/resp.json")"
  echo "    $method $path -> $code"
}

[[ -f "$AUDIO" ]] || fail "audio file not found: $AUDIO"
command -v curl >/dev/null || fail "curl is required"
"$PY" -c 'import jsonschema' 2>/dev/null || fail "$PY needs jsonschema (pip install jsonschema), or set PYTHON"

step "API health ($API)"
api GET /health

step "Guest auth"
DEVICE_ID="$("$PY" -c 'import uuid; print(uuid.uuid4())')"
api POST /auth/anonymous "{\"deviceId\":\"e2e-$DEVICE_ID\",\"platform\":\"android\"}"
TOKEN="$(json_get "$TMP/resp.json" token)"
[[ -n "$TOKEN" ]] || fail "no token in /auth/anonymous response"
echo "    guest user $(json_get "$TMP/resp.json" userId)"

step "Request upload for $(basename "$AUDIO")"
SIZE="$(wc -c <"$AUDIO" | tr -d ' ')"
SHA="$("$PY" -c 'import hashlib,sys; print(hashlib.sha256(open(sys.argv[1],"rb").read()).hexdigest())' "$(native "$AUDIO")")"
api POST /uploads "{\"contentType\":\"audio/mpeg\",\"sizeBytes\":$SIZE,\"sha256\":\"$SHA\"}"
UPLOAD_ID="$(json_get "$TMP/resp.json" uploadId)"
PUT_URL="$(json_get "$TMP/resp.json" putUrl)"
echo "    upload $UPLOAD_ID ($SIZE bytes, sha256 ${SHA:0:12}...)"

step "PUT bytes to storage"
code="$(curl -sS -o "$TMP/put.txt" -w '%{http_code}' -X PUT -H 'Content-Type: audio/mpeg' \
  --data-binary "@$AUDIO" "$PUT_URL")" || fail "PUT to storage failed"
[[ "$code" == 2* ]] || fail "PUT to storage -> HTTP $code: $(cat "$TMP/put.txt")"
echo "    PUT -> $code"

step "Create job"
api POST /jobs "{\"source\":\"file\",\"uploadId\":\"$UPLOAD_ID\",\"title\":\"e2e $(basename "$AUDIO")\"}"
JOB_ID="$(json_get "$TMP/resp.json" jobId)"
TRACK_ID="$(json_get "$TMP/resp.json" trackId)"
echo "    job $JOB_ID, track $TRACK_ID, status $(json_get "$TMP/resp.json" status)"

step "Poll job (timeout ${TIMEOUT}s)"
start=$SECONDS last=""
while :; do
  api GET "/jobs/$JOB_ID" >/dev/null
  status="$(json_get "$TMP/resp.json" status)"
  line="$status $(json_get "$TMP/resp.json" stage) $(json_get "$TMP/resp.json" progress)%"
  [[ "$line" != "$last" ]] && echo "    [$((SECONDS - start))s] $line" && last="$line"
  case "$status" in
    complete) break ;;
    failed) fail "job failed: $(json_get "$TMP/resp.json" error)" ;;
  esac
  (( SECONDS - start < TIMEOUT )) || fail "job still $status after ${TIMEOUT}s"
  sleep 2
done

step "Get analysis URL"
api GET "/tracks/$TRACK_ID/analysis"
URL="$(json_get "$TMP/resp.json" url)"
echo "    schemaVersion $(json_get "$TMP/resp.json" schemaVersion)"

step "Download analysis"
OUT="$TMP/$TRACK_ID.analysis.json"
code="$(curl -sS -o "$OUT" -w '%{http_code}' "$URL")" || fail "download failed"
[[ "$code" == 2* ]] || fail "download -> HTTP $code"
echo "    $(wc -c <"$OUT" | tr -d ' ') bytes"

step "Validate against contracts/analysis.schema.json"
"$PY" "$(native "$ROOT/contracts/validate.py")" "$(native "$OUT")" || fail "analysis does not validate"
"$PY" -c 'import json,sys; d=json.load(open(sys.argv[1])); k=d["key"] or {}
print("    %.1fs, key %s %s, %s BPM, %d beats, %d chords" % (d["meta"]["durationSec"], k.get("tonic"), k.get("mode"),
      d["bpm"], len(d["beats"]), len(d["chords"])))' "$(native "$OUT")"

printf '\nPASS: end-to-end pipeline OK in %ss\n' "$SECONDS"
