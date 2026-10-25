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
    $u = $null
    foreach ($f in @($out, $err)) {
        if (Test-Path $f) {
            $m = Select-String -Path $f -Pattern "https://[a-z0-9-]+\.trycloudflare\.com" |
                 Select-Object -Last 1
            if ($m) { $u = $m.Matches[0].Value }
        }
    }
    return $u
}

# nothing to expose until the tower itself answers locally
try { Invoke-WebRequest "$local/api/health" -TimeoutSec 5 -UseBasicParsing | Out-Null }
catch { exit 0 }

$url = $null
if (Test-Path $urlFile) { $url = (Get-Content $urlFile -TotalCount 1).Trim() }

if ($url) {
    try {
        $r = Invoke-WebRequest "$url/api/health" -TimeoutSec 12 -UseBasicParsing
        # A 200 is not enough: a dead quick tunnel can answer with an error page
        # from the edge. The body has to be the tower's own health JSON.
        if ($r.StatusCode -eq 200 -and $r.Content -match '"ok"\s*:\s*true') {
            # Re-stamp the file: the desk refuses a URL whose keeper has gone
            # quiet, so "healthy" has to leave fresh evidence of the check.
            Set-Content -Path $urlFile -Value $url -Encoding ascii
            exit 0
        }
    } catch { }
}

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
        Set-Content -Path $urlFile -Value $fresh -Encoding ascii
        Write-Output "TUNNEL_UP $fresh"
        exit 0
    }
}
Write-Output "TUNNEL_DOWN"
exit 1
