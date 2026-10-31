# WP3.0 detached re-runner: every 60 min run extract_seg_metrics.py (resumable) until all 427 tiles have both design days.
# Start with:  Start-Process powershell.exe -ArgumentList '-NoProfile','-ExecutionPolicy','Bypass','-File','scripts\street\analysis\run_extract_loop.ps1' -WindowStyle Hidden -PassThru
# Stops when WP3.0_tiles_done.csv holds 427 tiles x 2 days, or when both producer jobs (WP2.1 SOLWEIG, WP2.3 UTCI launcher)
# are dead and a final pass found nothing new.
$root = 'C:\Users\YE\PyCharmMiscProject\izmir_lcz_deep_morphology_mert_yavas'
Set-Location $root
$py   = 'C:\Users\YE\miniforge3\envs\climorfa-street\python.exe'
$log  = Join-Path $root 'outputs\street\analysis\wp30\rerunner.log'
$done = Join-Path $root 'data\street\analysis\WP3.0_tiles_done.csv'
$total = 427

function Get-DoneCount {
    if (-not (Test-Path $done)) { return 0 }
    $rows = Import-Csv $done
    if (-not $rows) { return 0 }
    return @($rows | Group-Object tile_id | Where-Object { $_.Count -ge 2 }).Count
}
function Test-Alive($pidFile) {
    if (-not (Test-Path $pidFile)) { return $false }
    $p = [int](((Get-Content $pidFile -Raw) -replace '[^0-9]', ''))
    return [bool](Get-Process -Id $p -ErrorAction SilentlyContinue)
}

$prev = -1
while ($true) {
    Add-Content $log ("{0} pass start (tiles done before: {1})" -f (Get-Date -Format s), (Get-DoneCount))
    cmd /c "`"$py`" scripts\street\analysis\extract_seg_metrics.py --workers 2 >> `"$log`" 2>&1"
    $n = Get-DoneCount
    if ($n -ge $total) { Add-Content $log ("{0} COMPLETE ({1} tiles); stopping" -f (Get-Date -Format s), $n); break }
    $alive = (Test-Alive (Join-Path $root 'logs\wp\WP2.1_full_run.pid')) -or (Test-Alive (Join-Path $root 'logs\wp\WP2.3_launcher.pid'))
    if ((-not $alive) -and ($n -eq $prev)) {
        Add-Content $log ("{0} producers dead, no progress; stopping with PARTIAL ({1} tiles)" -f (Get-Date -Format s), $n); break
    }
    $prev = $n
    Start-Sleep -Seconds 3600
}
