#Requires -Version 7.0
#
# cmd_live.ps1 - Launch the live GPU link workspace.
#
# SYNOPSIS
#   live [name] [--reset]
#
# DESCRIPTION
#   Full screen, mouse-driven multiplexer workspace whose panes consume the
#   ignis live_link.rs shared-memory ring in real time. The workspace is user
#   composable: split any pane and choose replacement content from the menu,
#   which offers every registered pane type (all live panes plus the utility
#   panes). The resulting layout is saved on quit to
#   .ignis_trace\live_layout.json and restored on the next launch; --reset
#   discards the saved layout and starts from the default arrangement.
#
# PARAMETERS
#   name     Mapping name without the "Local\" prefix. Defaults to "ignis".
#   --reset  Ignore any saved layout for this session.
#
# INTERACTION
#   Beyond the standard multiplexer controls: the events pane filters by
#   category (keys 1..7 or clicking the legend) and scrolls with arrows,
#   paging keys, and the wheel; the validation pane selects rows with the
#   arrows and opens a detail overlay with Enter or a click, cross-referenced
#   against the offline VUID knowledge base; Escape closes the overlay. The
#   status bar shows the last raw input event received, which doubles as an
#   input-path diagnostic.
#
# PLATFORM
#   Windows only. Missing interop is reported cleanly without touching
#   terminal state.

param([Parameter(ValueFromRemainingArguments)][string[]]$RawArgs)

Get-ChildItem (Join-Path $PSScriptRoot "_*.ps1") | ForEach-Object { . $_.FullName }

# Apply the persisted workspace theme before any pane renders.
$null = Restore-MuxTheme

$mapName = 'ignis'
$reset = $false
foreach ($a in $RawArgs) {
    if ($a -eq '--reset') { $reset = $true }
    elseif ($a -and $a -notmatch '^-') { $mapName = $a }
}

Write-CmdHeader "live" ("GPU live link on Local\" + $mapName)

Initialize-MuxNative
Initialize-MuxLiveNative

if (-not ([System.Management.Automation.PSTypeName]'IgnisConsole').Type) {
    Write-Host "    live requires Windows console interop, which is unavailable here" -ForegroundColor Red
    return
}
if (-not ([System.Management.Automation.PSTypeName]'IgnisShm').Type) {
    Write-Host "    live requires Windows shared-memory interop, which is unavailable here" -ForegroundColor Red
    return
}

# Warm the VUID knowledge base cache so VuidDetailPane can read the global
# directly from class context.
$null = Get-VuidKnowledgeBase

$reader = [LiveLinkReader]::new($mapName)

# Content registry: everything the menu offers and the layout importer can
# rebuild. Factories close over $reader, whose scope stays alive for the
# duration of the engine run.
$registry = [ordered]@{
    'Events'      = @{ Make = { [LiveEventPane]::new($reader) };      Type = 'LiveEventPane' }
    'Status'      = @{ Make = { [LiveStatusPane]::new($reader) };     Type = 'LiveStatusPane' }
    'Memory'      = @{ Make = { [LiveMemPane]::new($reader) };        Type = 'LiveMemPane' }
    'Validation'  = @{ Make = { [LiveValidationPane]::new($reader) }; Type = 'LiveValidationPane' }
    'Sync'        = @{ Make = { [LiveSyncPane]::new($reader) };       Type = 'LiveSyncPane' }
    'GPU Scopes'  = @{ Make = { [LiveGpuPane]::new($reader) };        Type = 'LiveGpuPane' }
    'Alloc Sites' = @{ Make = { [LiveSitesPane]::new($reader) };      Type = 'LiveSitesPane' }
    'Hardened'    = @{ Make = { [LiveHardenedPane]::new($reader) };   Type = 'LiveHardenedPane' }
    'Printf'      = @{ Make = { [LivePrintfPane]::new($reader) };     Type = 'LivePrintfPane' }
    'Clock'       = @{ Make = { [ClockPane]::new() };                 Type = 'ClockPane' }
    'Help'        = @{ Make = { [HelpPane]::new() };                  Type = 'HelpPane' }
    'Log'         = @{ Make = { [LogPane]::new() };                   Type = 'LogPane' }
    'Menu'        = @{ Make = { [MenuPane]::new() };                  Type = 'MenuPane' }
}

$layoutPath = Join-Path $PSScriptRoot '..\.ignis_trace\live_layout.json'

$script:liveIdc = 0
function New-LiveLeaf {
    param([object]$PaneObject)
    $script:liveIdc++
    return @{ Kind = 'pane'; Id = "l$script:liveIdc"; Pane = $PaneObject }
}
function New-LiveSplit {
    param([string]$Dir, [double]$Ratio, [object]$A, [object]$B)
    return @{ Kind = 'split'; Dir = $Dir; Ratio = $Ratio; A = $A; B = $B }
}

$root = $null
if (-not $reset) {
    $root = Import-MuxLayout -Path $layoutPath -Registry $registry
}
if ($null -eq $root) {
    $events = New-LiveLeaf ([LiveEventPane]::new($reader))
    $status = New-LiveLeaf ([LiveStatusPane]::new($reader))
    $mem    = New-LiveLeaf ([LiveMemPane]::new($reader))
    $val    = New-LiveLeaf ([LiveValidationPane]::new($reader))
    $sync   = New-LiveLeaf ([LiveSyncPane]::new($reader))
    $rightCol = New-LiveSplit 'v' 0.45 $status $mem
    $topRow   = New-LiveSplit 'h' 0.52 $events $rightCol
    $botRow   = New-LiveSplit 'h' 0.5 $val $sync
    $root     = New-LiveSplit 'v' 0.68 $topRow $botRow
}

$engine = [MuxEngine]::new()
$engine.Root = $root
$engine.Registry = $registry
$engine.FocusId = $engine.FirstPaneId($root)
$engine.LogMsg('ignis-live started on Local\' + $mapName)

Write-Host ("    entering live workspace on Local\" + $mapName + " (Ctrl+Q to quit)") -ForegroundColor DarkGray

$prevEnc = [Console]::OutputEncoding
try {
    try { [Console]::OutputEncoding = [System.Text.UTF8Encoding]::new($false) } catch { }
    $engine.Run()
}
finally {
    try { [Console]::OutputEncoding = $prevEnc } catch { }
    try { ([type]'IgnisShm')::Close() } catch { }
    Export-MuxLayout -Root $engine.Root -Path $layoutPath
}

Write-Host "    live workspace closed (layout saved)" -ForegroundColor Green