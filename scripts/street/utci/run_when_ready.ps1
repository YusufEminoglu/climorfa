# WP2.3 UTCI Periodic Launcher
# Re-runs compute_street_utci.py every 30 minutes until all 427 tiles are complete
# or until SOLWEIG exits and no remaining tiles can be processed.

$ErrorActionPreference = "Continue"

$ProjectRoot = "C:\Users\YE\PyCharmMiscProject\izmir_lcz_deep_morphology_mert_yavas"
$PythonExe = "C:\Users\YE\miniforge3\envs\climorfa-street\python.exe"
$ScriptPath = Join-Path $ProjectRoot "scripts\street\utci\compute_street_utci.py"
$LauncherPidFile = Join-Path $ProjectRoot "logs\wp\WP2.3_launcher.pid"
$SolweigPidFile = Join-Path $ProjectRoot "logs\wp\WP2.1_full_run.pid"
$ProgressCsv = Join-Path $ProjectRoot "logs\wp\WP2.3_progress.csv"
$LauncherLog = Join-Path $ProjectRoot "logs\wp\WP2.3_launcher.log"

Set-Location $ProjectRoot

# Save PID
[System.IO.File]::WriteAllText($LauncherPidFile, "$PID`n", [System.Text.Encoding]::ASCII)
$timestamp = (Get-Date).ToString("yyyy-MM-dd HH:mm:ss")
[System.IO.File]::AppendAllText($LauncherLog, "[$timestamp] Launcher started with PID $PID`n", [System.Text.Encoding]::UTF8)

$totalTargetTiles = 427

while ($true) {
    $now = (Get-Date).ToString("yyyy-MM-dd HH:mm:ss")
    [System.IO.File]::AppendAllText($LauncherLog, "[$now] Triggering compute_street_utci.py...`n", [System.Text.Encoding]::UTF8)

    # Run UTCI script synchronously
    $pinfo = New-Object System.Diagnostics.ProcessStartInfo
    $pinfo.FileName = $PythonExe
    $pinfo.Arguments = "`"$ScriptPath`""
    $pinfo.RedirectStandardOutput = $true
    $pinfo.RedirectStandardError = $true
    $pinfo.UseShellExecute = $false
    $pinfo.CreateNoWindow = $true
    $pinfo.WorkingDirectory = $ProjectRoot
    $proc = [System.Diagnostics.Process]::Start($pinfo)
    $stdout = $proc.StandardOutput.ReadToEnd()
    $stderr = $proc.StandardError.ReadToEnd()
    $proc.WaitForExit()

    [System.IO.File]::AppendAllText($LauncherLog, "$stdout`n$stderr`n", [System.Text.Encoding]::UTF8)

    # Count completed tiles
    $completedCount = 0
    if (Test-Path $ProgressCsv) {
        try {
            $lines = Import-Csv $ProgressCsv
            if ($lines) {
                $okTiles = @($lines | Where-Object { $_.status -eq "OK" } | Select-Object -ExpandProperty tile -Unique)
                $completedCount = $okTiles.Count
            }
        } catch {}
    }

    $now = (Get-Date).ToString("yyyy-MM-dd HH:mm:ss")
    [System.IO.File]::AppendAllText($LauncherLog, "[$now] Completed: $completedCount / $totalTargetTiles tiles.`n", [System.Text.Encoding]::UTF8)

    if ($completedCount -ge $totalTargetTiles) {
        [System.IO.File]::AppendAllText($LauncherLog, "[$now] All $totalTargetTiles completed! Exiting.`n", [System.Text.Encoding]::UTF8)
        break
    }

    # Check if SOLWEIG process is still alive
    $solweigAlive = $false
    if (Test-Path $SolweigPidFile) {
        $rawPid = [System.IO.File]::ReadAllText($SolweigPidFile)
        $m = [System.Text.RegularExpressions.Regex]::Match($rawPid, '\d+')
        if ($m.Success) {
            $sPid = [int]$m.Value
            $p = Get-Process -Id $sPid -ErrorAction SilentlyContinue
            if ($p) {
                $solweigAlive = $true
            }
        }
    }

    if (-not $solweigAlive) {
        [System.IO.File]::AppendAllText($LauncherLog, "[$now] SOLWEIG process exited. Exiting launcher.`n", [System.Text.Encoding]::UTF8)
        break
    }

    [System.IO.File]::AppendAllText($LauncherLog, "[$now] Sleeping for 30 minutes (1800s)...`n", [System.Text.Encoding]::UTF8)
    Start-Sleep -Seconds 1800
}
