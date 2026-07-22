#Requires -Version 7.0
#
# cmd_crash.ps1 - Crash report viewer.
#
# SYNOPSIS
#   crash                 list crash_report_*.md files, newest first
#   crash latest          render the newest report
#   crash <n>             render report number n from the listing
#
# DESCRIPTION
#   Renders the Markdown crash reports produced by the crate's CrashReporter
#   (crash_report_<timestamp>.md, written to the working directory or the OS
#   temp directory) directly in the terminal. The renderer handles the exact
#   subset the reporter emits: #/##/### headings, "- **Key:** value" bullet
#   metadata, pipe tables (drawn with aligned columns and highlighted PENDING
#   cells), fenced code blocks (dimmed verbatim), and plain paragraphs. It is
#   not a general Markdown engine and does not try to be.
#
#   Both the working directory and the OS temp directory are scanned, since
#   CrashReporter falls back to temp when the working directory is not
#   writable.

param([Parameter(ValueFromRemainingArguments)][string[]]$RawArgs)

Get-ChildItem (Join-Path $PSScriptRoot "_*.ps1") | ForEach-Object { . $_.FullName }

$reports = @(
    @(Get-ChildItem -Path '.' -Filter 'crash_report_*.md' -File -ErrorAction SilentlyContinue) +
    @(Get-ChildItem -Path ([System.IO.Path]::GetTempPath()) -Filter 'crash_report_*.md' -File -ErrorAction SilentlyContinue)
) | Sort-Object LastWriteTime -Descending

$action = if ($RawArgs -and $RawArgs.Count -ge 1) { $RawArgs[0] } else { 'list' }

if ($action -eq 'list') {
    Write-CmdHeader "crash" "$($reports.Count) report(s)"
    if ($reports.Count -eq 0) {
        Write-Host "    no crash_report_*.md found in the working or temp directory" -ForegroundColor DarkGray
        return
    }
    for ($i = 0; $i -lt $reports.Count; $i++) {
        $r = $reports[$i]
        $sizeKb = [math]::Round($r.Length / 1024, 1)
        Write-Host ("    {0,2}  {1}  {2,8} KiB  " -f ($i + 1), $r.LastWriteTime.ToString('yyyy-MM-dd HH:mm:ss'), $sizeKb) -NoNewline -ForegroundColor Gray
        Write-Host $r.Name -ForegroundColor White
    }
    Write-Host ""
    Write-Host "    use 'crash latest' or 'crash <n>' to view" -ForegroundColor DarkGray
    return
}

if ($reports.Count -eq 0) {
    Write-CmdHeader "crash" $action
    Write-Host "    no crash reports found" -ForegroundColor Yellow
    return
}

$target = $null
if ($action -eq 'latest') { $target = $reports[0] }
elseif ($action -match '^\d+$') {
    $idx = [int]$action - 1
    if ($idx -ge 0 -and $idx -lt $reports.Count) { $target = $reports[$idx] }
}
if ($null -eq $target) {
    Write-CmdHeader "crash" "unknown selector: $action"
    Write-Host "    use 'crash', 'crash latest', or 'crash <n>'" -ForegroundColor Yellow
    return
}

Write-CmdHeader "crash" $target.Name

$inCode = $false
foreach ($raw in (Get-Content $target.FullName)) {
    $line = $raw.TrimEnd()

    if ($line -match '^```') { $inCode = -not $inCode; continue }
    if ($inCode) { Write-Host "      $line" -ForegroundColor DarkGray; continue }

    if ($line -match '^# (.+)') {
        Write-Host ""
        Write-Host "    $($Matches[1])" -ForegroundColor Magenta
        Write-Host "    $('=' * [math]::Min(60, $Matches[1].Length))" -ForegroundColor DarkMagenta
    }
    elseif ($line -match '^## (.+)') {
        Write-Host ""
        Write-Host "    $($Matches[1])" -ForegroundColor Cyan
    }
    elseif ($line -match '^### (.+)') {
        Write-Host ""
        Write-Host "    $($Matches[1])" -ForegroundColor DarkCyan
    }
    elseif ($line -match '^\s*-\s+\*\*(.+?):?\*\*:?\s*(.*)$') {
        Write-Host ("    {0,-14} " -f $Matches[1]) -NoNewline -ForegroundColor White
        Write-Host $Matches[2] -ForegroundColor Gray
    }
    elseif ($line -match '^\|(.+)\|$') {
        $cells = $Matches[1] -split '\|' | ForEach-Object { $_.Trim() }
        if (($cells -join '') -match '^-+$') { continue }   # separator row
        $rendered = '    ' + (($cells | ForEach-Object { $_.PadRight(20) }) -join ' ')
        $col = if ($rendered -match 'PENDING') { 'Red' } else { 'Gray' }
        Write-Host $rendered.TrimEnd() -ForegroundColor $col
    }
    elseif ($line) {
        Write-Host "    $line" -ForegroundColor Gray
    }
}
Write-Host ""