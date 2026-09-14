# =============================================================================
# Jarvis dashboard watchdog
# =============================================================================
# Polls http://127.0.0.1:8765/health until Jarvis is ready (or timeout), then
# opens the dashboard in the default browser. Detached: launched by the
# visible launcher (run_jarvis_visible.vbs) in a hidden window.
# =============================================================================

$url      = "http://127.0.0.1:8765"
$health   = "$url/health"
$maxWait  = 90        # seconds to wait for Jarvis to become healthy
$interval = 2         # poll every 2s
$elapsed  = 0

while ($elapsed -lt $maxWait) {
    try {
        $r = Invoke-WebRequest -Uri $health -UseBasicParsing -TimeoutSec 2
        if ($r.StatusCode -eq 200) {
            Start-Process $url
            exit 0
        }
    } catch {
        # not up yet — keep waiting
    }
    Start-Sleep -Seconds $interval
    $elapsed += $interval
}

# Give up silently — Jarvis may be starting slowly or already running elsewhere.
exit 1