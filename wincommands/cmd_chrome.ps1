#Requires -Version 7.0
#
# cmd_chrome.ps1 - Chrome Trace Format viewer launcher.
#
# SYNOPSIS
#   chrome                scan for trace JSON files, open the newest
#   chrome <file>         open a specific trace file
#   chrome list           scan and list without opening
#
# DESCRIPTION
#   The crate's ResourceTrace exports Chrome Trace Format JSON (a top-level
#   array of {"ph": ...} event objects). Neither chrome://tracing nor the
#   Perfetto UI accepts a file as a URL parameter (a deliberate browser
#   security boundary), so "opening" a trace is necessarily two-step: this
#   command launches https://ui.perfetto.dev in the default browser and opens
#   an Explorer window with the trace file preselected, ready to drag onto
#   the Perfetto drop target.
#
# DETECTION
#   Candidate files are *.json in the working directory whose first kilobyte
#   contains a "ph": key, which distinguishes trace exports from Cargo or
#   trace-session JSON. Detection is heuristic by design; passing an explicit
#   path bypasses it entirely.

param([Parameter(ValueFromRemainingArguments)][string[]]$RawArgs)

Get-ChildItem (Join-Path $PSScriptRoot "_*.ps1") | ForEach-Object { . $_.FullName }

function Test-ChromeTrace {
    param([string]$Path)
    try {
        $fs = [System.IO.File]::OpenRead($Path)
        try {
            $buf = [byte[]]::new(1024)
            $n = $fs.Read($buf, 0, 1024)
            $head = [System.Text.Encoding]::UTF8.GetString($buf, 0, $n)
            return ($head -match '"ph"\s*:')
        } finally { $fs.Dispose() }
    } catch { return $false }
}

$arg = if ($RawArgs -and $RawArgs.Count -ge 1) { $RawArgs[0] } else { '' }

$target = $null
if ($arg -and $arg -ne 'list') {
    if (Test-Path $arg) { $target = Get-Item $arg }
    else {
        Write-CmdHeader "chrome" $arg
        Write-Host "    file not found: $arg" -ForegroundColor Red
        return
    }
} else {
    $candidates = @(Get-ChildItem -Path '.' -Filter '*.json' -File -ErrorAction SilentlyContinue |
        Where-Object { Test-ChromeTrace $_.FullName } |
        Sort-Object LastWriteTime -Descending)

    if ($arg -eq 'list') {
        Write-CmdHeader "chrome" "$($candidates.Count) trace file(s)"
        foreach ($c in $candidates) {
            $sizeKb = [math]::Round($c.Length / 1024, 1)
            Write-Host ("    {0}  {1,8} KiB  " -f $c.LastWriteTime.ToString('yyyy-MM-dd HH:mm:ss'), $sizeKb) -NoNewline -ForegroundColor Gray
            Write-Host $c.Name -ForegroundColor White
        }
        if ($candidates.Count -eq 0) {
            Write-Host "    no Chrome Trace Format JSON in the working directory" -ForegroundColor DarkGray
            Write-Host "    export one via ResourceTrace::export_chrome_json" -ForegroundColor DarkGray
        }
        return
    }

    if ($candidates.Count -eq 0) {
        Write-CmdHeader "chrome" "scan"
        Write-Host "    no Chrome Trace Format JSON in the working directory" -ForegroundColor Yellow
        Write-Host "    export one via ResourceTrace::export_chrome_json, or pass a path" -ForegroundColor DarkGray
        return
    }
    $target = $candidates[0]
}

Write-CmdHeader "chrome" $target.Name
Write-Host "    opening Perfetto UI and revealing the file in Explorer" -ForegroundColor Gray
Write-Host "    drag $($target.Name) onto the Perfetto drop target" -ForegroundColor DarkGray
Start-Process 'https://ui.perfetto.dev'
Start-Process 'explorer.exe' -ArgumentList "/select,`"$($target.FullName)`""