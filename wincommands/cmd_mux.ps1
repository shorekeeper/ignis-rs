#Requires -Version 7.0
#
# cmd_mux.ps1 - Launch the general terminal multiplexer.
#
# SYNOPSIS
#   mux [--reset]
#
# DESCRIPTION
#   Full screen, mouse-driven pane workspace with utility content (clock,
#   system information, help, session log). Splitting a pane opens a menu of
#   the registered content types. The layout is saved on quit to
#   .ignis_trace\mux_layout.json and restored on the next launch; --reset
#   starts from the default arrangement instead. Live data panes belong to
#   the live command, which registers them together with a link reader.
#
# PLATFORM
#   Windows only.

param([Parameter(ValueFromRemainingArguments)][string[]]$RawArgs)

Get-ChildItem (Join-Path $PSScriptRoot "_*.ps1") | ForEach-Object { . $_.FullName }

# Apply the persisted workspace theme before any pane renders.
$null = Restore-MuxTheme

$reset = ($RawArgs -contains '--reset')

Write-CmdHeader "mux" "terminal multiplexer"

Initialize-MuxNative

if (-not ([System.Management.Automation.PSTypeName]'IgnisConsole').Type) {
    Write-Host "    mux requires Windows console interop, which is unavailable here" -ForegroundColor Red
    return
}

$registry = [ordered]@{
    'Clock'  = @{ Make = { [ClockPane]::new() };   Type = 'ClockPane' }
    'System' = @{ Make = { [SysInfoPane]::new() }; Type = 'SysInfoPane' }
    'Help'   = @{ Make = { [HelpPane]::new() };    Type = 'HelpPane' }
    'Log'    = @{ Make = { [LogPane]::new() };     Type = 'LogPane' }
    'Menu'   = @{ Make = { [MenuPane]::new() };    Type = 'MenuPane' }
}

$layoutPath = Join-Path $PSScriptRoot '..\.ignis_trace\mux_layout.json'

$script:idc = 0
function New-MuxPaneNode {
    param([object]$PaneObject)
    $script:idc++
    return @{ Kind = 'pane'; Id = "p$script:idc"; Pane = $PaneObject }
}
function New-MuxSplitNode {
    param([string]$Dir, [double]$Ratio, [object]$A, [object]$B)
    return @{ Kind = 'split'; Dir = $Dir; Ratio = $Ratio; A = $A; B = $B }
}

$root = $null
if (-not $reset) {
    $root = Import-MuxLayout -Path $layoutPath -Registry $registry
}
if ($null -eq $root) {
    $clock = New-MuxPaneNode ([ClockPane]::new())
    $help  = New-MuxPaneNode ([HelpPane]::new())
    $sys   = New-MuxPaneNode ([SysInfoPane]::new())
    $log   = New-MuxPaneNode ([LogPane]::new())
    $rightCol = New-MuxSplitNode 'v' 0.55 $help $sys
    $topRow   = New-MuxSplitNode 'h' 0.42 $clock $rightCol
    $root     = New-MuxSplitNode 'v' 0.72 $topRow $log
}

$engine = [MuxEngine]::new()
$engine.Root = $root
$engine.Registry = $registry
$engine.FocusId = $engine.FirstPaneId($root)
$engine.LogMsg('ignis-mux started')

Write-Host "    entering multiplexer (Ctrl+Q to quit)" -ForegroundColor DarkGray

$prevEnc = [Console]::OutputEncoding
try {
    try { [Console]::OutputEncoding = [System.Text.UTF8Encoding]::new($false) } catch { }
    $engine.Run()
}
finally {
    try { [Console]::OutputEncoding = $prevEnc } catch { }
    Export-MuxLayout -Root $engine.Root -Path $layoutPath
}

Write-Host "    multiplexer closed (layout saved)" -ForegroundColor Green