# KAIROS tunnel keeper — heals a stuck Cloudflare quick tunnel and publishes its URL.
# A quick tunnel process can sit alive for hours printing
# "Unauthorized: Tunnel not found" while serving nothing, so liveness is decided by
# asking the public URL, never by looking for the process.
$ErrorActionPreference = "SilentlyContinue"

$repo    = Split-Path -Parent $PSScriptRoot
$local   = "http://127.0.0.1:8600"
$exe     = Join-Path $PSScriptRoot "cloudflared.exe"
$out     = Join-Path $repo "data\tunnel.log"
$err     = Join-Path $repo "data\tunnel.err.log"
$urlFile = Join-Path $repo "data\tunnel_url.txt"

if (-not (Test-Path $exe)) { exit 0 }          # tunnel binary absent: nothing to keep

function Read-PublicUrl {
    # cloudflared's own error text mentions https://api.trycloudflare.com/tunnel:
    # the control API, not an assigned hostname. A failed launch therefore used
    # to be published as a live tunnel, so only a subdomain may match.
    foreach ($f in @($err, $out)) {
        if (Test-Path $f) {
            $m = Select-String -Path $f -Pattern "https://(?!api\.)[a-z0-9-]+\.trycloudflare\.com" |
                 Select-Object -Last 1
            if ($m) { return $m.Matches[0].Value }
        }
    }
    return $null
}

# nothing to expose until the tower itself answers locally
try { Invoke-WebRequest "$local/api/health" -TimeoutSec 5 -UseBasicParsing | Out-Null }
catch { exit 0 }

$url = $null
if (Test-Path $urlFile) { $url = (Get-Content $urlFile -TotalCount 1).Trim() }

function Test-Public {
    param($u)
    try {
        $r = Invoke-WebRequest "$u/api/health" -TimeoutSec 8 -UseBasicParsing
        # A 200 is not enough: a dead quick tunnel can answer with an error page
        # from the edge. The body has to be the tower's own health JSON.
        return ($r.StatusCode -eq 200 -and $r.Content -match '"ok"\s*:\s*true')
    } catch { return $false }
}

if ($url) {
    # One timeout is not proof the tunnel is dead. A quick tunnel's hostname is
    # new to DNS and the box is shared with whatever else is running, so a probe
    # can fail while the edge link is fine. Killing on a single miss is what
    # rotated the published hostname every couple of minutes, so each one is
    # asked three times before it is believed.
    for ($k = 0; $k -lt 3; $k++) {
        if (Test-Public $url) {
            # Re-stamp the file: the desk refuses a URL whose keeper has gone
            # quiet, so "healthy" has to leave fresh evidence of the check.
            Set-Content -Path $urlFile -Value $url -Encoding ascii
            exit 0
        }
        if ($k -lt 2) { Start-Sleep -Seconds 5 }
    }
}

# No usable stamp, but a tunnel may already be up: adopt the hostname its log
# carries when it answers, rather than killing a working process to relearn it.
$found = Read-PublicUrl
if ($found -and (Test-Public $found)) {
    Set-Content -Path $urlFile -Value $found -Encoding ascii
    Write-Output "$(Get-Date -Format s) TUNNEL_ADOPTED $found"
    exit 0
}

# A second keeper launching against the same repo must not kill the tunnel the
# first one just brought up: under 90 s old, the process is still the answer.
$live = Get-Process cloudflared -ErrorAction SilentlyContinue | Select-Object -First 1
if ($live -and ((Get-Date) - $live.StartTime).TotalSeconds -lt 90) { exit 0 }

# heal — kill the stuck registrar, relaunch, wait for its fresh hostname
Get-Process cloudflared | Stop-Process -Force
Start-Sleep -Seconds 3
Set-Content -Path $urlFile -Value "" -Encoding ascii
Remove-Item $out, $err -Force

Start-Process -FilePath $exe `
    -ArgumentList @("tunnel", "--no-autoupdate", "--url", $local) `
    -WindowStyle Hidden -RedirectStandardOutput $out -RedirectStandardError $err

for ($i = 0; $i -lt 24; $i++) {
    Start-Sleep -Seconds 3
    $fresh = Read-PublicUrl
    if ($fresh) {
        # An assigned hostname is not a working tunnel: a launch that timed out
        # against the control API still prints one. It counts only once it has
        # answered with the tower's own health JSON.
        for ($j = 0; $j -lt 10; $j++) {
            if (Test-Public $fresh) {
                Set-Content -Path $urlFile -Value $fresh -Encoding ascii
                Write-Output "$(Get-Date -Format s) TUNNEL_UP $fresh"
                exit 0
            }
            Start-Sleep -Seconds 4
        }
        Write-Output "$(Get-Date -Format s) TUNNEL_UNREACHABLE $fresh"
        exit 1
    }
}
Write-Output "$(Get-Date -Format s) TUNNEL_DOWN"
exit 1
