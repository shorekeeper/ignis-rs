#Requires -Version 7.0
#
# cmd_gpu.ps1 - Vulkan device inspector.
#
# SYNOPSIS
#   gpu           parsed device summary
#   gpu raw       verbatim vulkaninfo --summary output
#
# DESCRIPTION
#   Runs vulkaninfo --summary and reshapes its GPU blocks into an aligned,
#   colored table: device name, type, Vulkan API version, and driver version
#   per physical device, with discrete GPUs highlighted. The parse is line
#   oriented and anchored on "GPU<n>:" block headers followed by "key = value"
#   properties, which is the stable summary layout of the LunarG tooling. Any
#   deviation degrades gracefully: unparsed lines are ignored and 'gpu raw'
#   remains available as ground truth.
#
#   The first device name is cached to .ignis_trace\gpu_cache.txt so the shell
#   boot banner can display the GPU without paying the vulkaninfo process
#   startup cost (several hundred milliseconds) on every launch.
#
# REQUIREMENTS
#   vulkaninfo must be on PATH (ships with the Vulkan SDK and most driver
#   packages). Absence is reported as a hint, not an error.

param([Parameter(ValueFromRemainingArguments)][string[]]$RawArgs)

Get-ChildItem (Join-Path $PSScriptRoot "_*.ps1") | ForEach-Object { . $_.FullName }

$mode = if ($RawArgs -and $RawArgs.Count -ge 1) { $RawArgs[0] } else { 'summary' }

Write-CmdHeader "gpu" $mode

$raw = $null
try { $raw = vulkaninfo --summary 2>&1 | Out-String } catch { }
if (-not $raw -or $raw -notmatch 'deviceName') {
    Write-Host "    vulkaninfo not found or no Vulkan ICD present" -ForegroundColor Yellow
    Write-Host "    install a Vulkan SDK or GPU driver to enable this command" -ForegroundColor DarkGray
    return
}

if ($mode -eq 'raw') {
    foreach ($ln in ($raw -split "`n")) { Write-Host "    $($ln.TrimEnd())" -ForegroundColor Gray }
    return
}

# Parse GPU blocks: a "GPU<n>:" header opens a block; "key = value" lines
# populate it until the next header.
$devices = [System.Collections.Generic.List[object]]::new()
$cur = $null
foreach ($ln in ($raw -split "`n")) {
    $line = $ln.Trim()
    if ($line -match '^GPU(\d+):') {
        $cur = @{ Index = [int]$Matches[1] }
        $devices.Add($cur)
        continue
    }
    if ($null -ne $cur -and $line -match '^([A-Za-z]+)\s*=\s*(.+)$') {
        $cur[$Matches[1]] = $Matches[2].Trim()
    }
}

# Loader and instance version, if the summary carried one.
if ($raw -match 'Vulkan Instance Version:\s*(\S+)') {
    Write-Host "    Instance : Vulkan $($Matches[1])" -ForegroundColor Gray
    Write-Host ""
}

if ($devices.Count -eq 0) {
    Write-Host "    summary produced no GPU blocks; falling back to raw" -ForegroundColor Yellow
    foreach ($ln in ($raw -split "`n")) { Write-Host "    $($ln.TrimEnd())" -ForegroundColor DarkGray }
    return
}

foreach ($d in $devices) {
    $name = [string]$d['deviceName']
    $type = [string]$d['deviceType']
    $api = [string]$d['apiVersion']
    $drv = [string]$d['driverVersion']
    $drvName = [string]$d['driverName']

    $isDiscrete = $type -match 'DISCRETE'
    $nameCol = if ($isDiscrete) { 'Green' } else { 'White' }
    $typeShort = ($type -replace 'PHYSICAL_DEVICE_TYPE_', '') -replace '_', ' '

    Write-Host "    GPU$($d['Index'])  " -NoNewline -ForegroundColor Cyan
    Write-Host $name -ForegroundColor $nameCol
    Write-Host "          type    : $typeShort" -ForegroundColor Gray
    if ($api) { Write-Host "          api     : $api" -ForegroundColor Gray }
    if ($drv) {
        $drvLine = $drv
        if ($drvName) { $drvLine += "  ($drvName)" }
        Write-Host "          driver  : $drvLine" -ForegroundColor Gray
    }
    Write-Host ""
}

# Cache the primary device name for the boot banner. Discrete wins over the
# first-listed device when both are present.
$primary = $devices | Where-Object { [string]$_['deviceType'] -match 'DISCRETE' } | Select-Object -First 1
if (-not $primary) { $primary = $devices[0] }
try {
    $cachePath = Join-Path $PSScriptRoot '..\.ignis_trace\gpu_cache.txt'
    $dir = Split-Path $cachePath
    if (-not (Test-Path $dir)) { New-Item -ItemType Directory -Path $dir -Force | Out-Null }
    Set-Content -Path $cachePath -Value ([string]$primary['deviceName']) -Encoding UTF8
} catch { }