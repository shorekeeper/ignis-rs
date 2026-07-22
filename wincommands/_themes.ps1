#Requires -Version 7.0
#
# _themes.ps1 - Theme registry and persistence for the multiplexer workspaces.
#
# SCOPE
#   Themes recolor the mux and live workspaces by rewriting the static
#   properties of the [Theme] class defined in _mux_core.ps1. PowerShell class
#   statics are mutable at runtime, so applying a theme is a set of property
#   assignments with no engine changes; every pane reads [Theme]::* at render
#   time and picks up the change on the next frame. The REPL prompt itself
#   uses ConsoleColor names through Write-Host and is intentionally out of
#   scope: recoloring it would mean reserving ANSI palette slots globally,
#   which conflicts with the user's terminal scheme.
#
# PERSISTENCE
#   The selected theme name is stored in .ignis_trace\theme.txt. The mux and
#   live commands call Restore-MuxTheme after loading helpers, so the choice
#   survives across sessions. A missing or unknown persisted name silently
#   resolves to the default.
#
# LOAD ORDER
#   This file is dot-sourced alphabetically after _mux_core.ps1, so [Theme]
#   exists by the time these functions are defined. Type literals inside
#   plain functions bind at invocation, not at parse, so even out-of-order
#   loading would only fail at call time, never at dot-source time.
#
# ADDING A THEME
#   Append an entry to the table returned by Get-MuxThemeTable. Every key of
#   the default entry must be present; Set-MuxTheme assigns all of them
#   unconditionally, so a partial palette would leave stale colors.

function Get-MuxThemeTable {
    <#
    .SYNOPSIS
    Return the theme registry as an ordered dictionary of palettes.

    .DESCRIPTION
    Each palette maps every [Theme] static property name to an RGB triple.
    Triples rather than packed ints keep the table human-editable; packing
    happens once in Set-MuxTheme.
    #>
    $t = [ordered]@{}

    $t['dark'] = @{
        Bg = 24, 24, 28;        Panel = 32, 33, 40;      Border = 70, 72, 84
        BorderFocus = 90, 170, 255; Text = 210, 212, 220; TextDim = 130, 133, 145
        TextHead = 240, 242, 248; Accent = 120, 200, 140
        StatusBg = 45, 48, 60;  StatusFg = 200, 203, 214; StatusHi = 250, 220, 120
    }
    $t['nord'] = @{
        Bg = 46, 52, 64;        Panel = 59, 66, 82;      Border = 76, 86, 106
        BorderFocus = 136, 192, 208; Text = 216, 222, 233; TextDim = 130, 140, 160
        TextHead = 236, 239, 244; Accent = 163, 190, 140
        StatusBg = 67, 76, 94;  StatusFg = 216, 222, 233; StatusHi = 235, 203, 139
    }
    $t['dracula'] = @{
        Bg = 30, 31, 41;        Panel = 40, 42, 54;      Border = 68, 71, 90
        BorderFocus = 189, 147, 249; Text = 248, 248, 242; TextDim = 130, 132, 145
        TextHead = 255, 255, 255; Accent = 80, 250, 123
        StatusBg = 55, 58, 74;  StatusFg = 248, 248, 242; StatusHi = 241, 250, 140
    }
    $t['matrix'] = @{
        Bg = 8, 12, 8;          Panel = 14, 22, 14;      Border = 30, 70, 30
        BorderFocus = 80, 240, 80; Text = 140, 220, 140; TextDim = 70, 130, 70
        TextHead = 190, 255, 190; Accent = 80, 240, 80
        StatusBg = 18, 34, 18;  StatusFg = 140, 220, 140; StatusHi = 210, 255, 120
    }
    $t['hicontrast'] = @{
        Bg = 0, 0, 0;           Panel = 12, 12, 12;      Border = 160, 160, 160
        BorderFocus = 255, 255, 0; Text = 255, 255, 255; TextDim = 190, 190, 190
        TextHead = 255, 255, 255; Accent = 0, 255, 128
        StatusBg = 40, 40, 40;  StatusFg = 255, 255, 255; StatusHi = 255, 255, 0
    }

    return $t
}

function Get-MuxThemePath {
    Join-Path $PSScriptRoot '..\.ignis_trace\theme.txt'
}

function Set-MuxTheme {
    <#
    .SYNOPSIS
    Apply a named theme to the [Theme] statics and optionally persist it.

    .DESCRIPTION
    Returns $true on success, $false for an unknown name. Persist writes the
    name to the theme file for Restore-MuxTheme to pick up in later sessions.
    #>
    param([string]$Name, [switch]$Persist)
    $table = Get-MuxThemeTable
    if (-not $table.Contains($Name)) { return $false }
    $pal = $table[$Name]
    foreach ($key in $pal.Keys) {
        $rgb = $pal[$key]
        $packed = [MuxScreen]::Rgb($rgb[0], $rgb[1], $rgb[2])
        [Theme]::$key = $packed
    }
    if ($Persist) {
        try {
            $dir = Split-Path (Get-MuxThemePath)
            if (-not (Test-Path $dir)) { New-Item -ItemType Directory -Path $dir -Force | Out-Null }
            Set-Content -Path (Get-MuxThemePath) -Value $Name -Encoding UTF8
        } catch { }
    }
    return $true
}

function Restore-MuxTheme {
    <#
    .SYNOPSIS
    Reapply the persisted theme, defaulting to 'dark' when absent or invalid.
    #>
    $name = 'dark'
    $p = Get-MuxThemePath
    if (Test-Path $p) {
        $stored = (Get-Content $p -ErrorAction SilentlyContinue | Select-Object -First 1)
        if ($stored) { $name = $stored.Trim() }
    }
    if (-not (Set-MuxTheme -Name $name)) { $null = Set-MuxTheme -Name 'dark' }
    return $name
}