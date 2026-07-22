#Requires -Version 7.0
#
# cmd_theme.ps1 - Select the workspace color theme.
#
# SYNOPSIS
#   theme               list available themes with color swatches
#   theme <name>        apply and persist a theme
#
# DESCRIPTION
#   Applies to the mux and live workspaces (the [Theme] statics consumed by
#   every pane). The selection is persisted to .ignis_trace\theme.txt and
#   restored automatically by the workspace launchers. The REPL prompt is not
#   themed; see _themes.ps1 for the rationale.
#
#   Swatches are rendered with 24-bit background escapes, so the listing
#   itself previews each palette regardless of the terminal scheme.

param([Parameter(ValueFromRemainingArguments)][string[]]$RawArgs)

Get-ChildItem (Join-Path $PSScriptRoot "_*.ps1") | ForEach-Object { . $_.FullName }

$table = Get-MuxThemeTable
$current = Restore-MuxTheme

if (-not $RawArgs -or $RawArgs.Count -eq 0) {
    Write-CmdHeader "theme" "current: $current"
    $e = [char]0x1b
    foreach ($name in $table.Keys) {
        $pal = $table[$name]
        $marker = if ($name -eq $current) { '>' } else { ' ' }
        $sw = ''
        foreach ($key in @('Bg', 'Panel', 'BorderFocus', 'Accent', 'StatusHi', 'Text')) {
            $rgb = $pal[$key]
            $sw += "$e[48;2;$($rgb[0]);$($rgb[1]);$($rgb[2])m  $e[0m"
        }
        Write-Host "    $marker " -NoNewline -ForegroundColor Cyan
        Write-Host ("{0,-12}" -f $name) -NoNewline -ForegroundColor White
        Write-Host " $sw"
    }
    Write-Host ""
    Write-Host "    use 'theme <name>' to apply; takes effect in mux and live" -ForegroundColor DarkGray
    return
}

$name = $RawArgs[0].ToLower()
if (Set-MuxTheme -Name $name -Persist) {
    Write-CmdHeader "theme" $name
    Write-Host "    applied and persisted; open 'mux' or 'live' to see it" -ForegroundColor Green
} else {
    Write-CmdHeader "theme" "unknown: $name"
    Write-Host "    available: $($table.Keys -join ', ')" -ForegroundColor Yellow
}