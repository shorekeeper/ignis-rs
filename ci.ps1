#Requires -Version 5.1
# ci.ps1 - Ignis Windows CI orchestrator.
# Runs all test phases from wintests/, collects results, prints summary.
#
# Usage:
#   .\ci.ps1              # Run everything
#   .\ci.ps1 -Skip miri   # Skip miri phase
#   .\ci.ps1 -Only lint    # Run only lint phase

param(
    [string[]]$Only = @(),
    [string[]]$Skip = @()
)

$ErrorActionPreference = "Continue"
$global:CiStartTime = Get-Date
$global:CiResults = @()

# Helpers

function Write-Banner {
    param([string]$Text, [string]$Color = "Magenta")
    $line = "=" * 78
    Write-Host ""
    Write-Host "  $line" -ForegroundColor $Color
    $pad = [math]::Max(0, (78 - $Text.Length - 4) / 2)
    $left = " " * [math]::Floor($pad)
    $right = " " * [math]::Ceiling($pad)
    Write-Host "  ==$left$Text$right==" -ForegroundColor $Color
    Write-Host "  $line" -ForegroundColor $Color
    Write-Host ""
}

function Write-Phase {
    param([int]$Num, [int]$Total, [string]$Name, [string]$Desc)
    Write-Host ""
    Write-Host "  [$Num/$Total] " -NoNewline -ForegroundColor White
    Write-Host "$Name" -NoNewline -ForegroundColor Cyan
    Write-Host " - $Desc" -ForegroundColor DarkGray
    Write-Host "  $("-" * 70)" -ForegroundColor DarkGray
}

function Write-SystemInfo {
    Write-Host "  System Information" -ForegroundColor White
    Write-Host "  $("-" * 40)" -ForegroundColor DarkGray

    $os = [System.Environment]::OSVersion
    $cpu = (Get-CimInstance Win32_Processor -ErrorAction SilentlyContinue | Select-Object -First 1).Name
    if (-not $cpu) { $cpu = "unknown" }
    $ram = [math]::Round((Get-CimInstance Win32_ComputerSystem -ErrorAction SilentlyContinue).TotalPhysicalMemory / 1GB, 1)
    $rustVersion = (rustc --version 2>&1) -join ""
    $cargoVersion = (cargo --version 2>&1) -join ""

    Write-Host "    OS:      $($os.VersionString)" -ForegroundColor Gray
    Write-Host "    CPU:     $cpu" -ForegroundColor Gray
    Write-Host "    RAM:     ${ram} GB" -ForegroundColor Gray
    Write-Host "    Rust:    $rustVersion" -ForegroundColor Gray
    Write-Host "    Cargo:   $cargoVersion" -ForegroundColor Gray
    Write-Host "    PWD:     $(Get-Location)" -ForegroundColor Gray
    Write-Host "    Time:    $(Get-Date -Format 'yyyy-MM-dd HH:mm:ss')" -ForegroundColor Gray

    # Vulkan driver probe
    $vulkanInfo = $null
    try {
        $vulkanInfo = vulkaninfo --summary 2>&1 | Out-String
    } catch {}

    if ($vulkanInfo -and $vulkanInfo -match "deviceName") {
        $deviceLine = ($vulkanInfo -split "`n") | Where-Object { $_ -match "deviceName" } | Select-Object -First 1
        Write-Host "    Vulkan:  $($deviceLine.Trim())" -ForegroundColor Gray
    } else {
        Write-Host "    Vulkan:  no driver detected (smoke test will fail)" -ForegroundColor Yellow
    }
    Write-Host ""
}

function Record-Result {
    param(
        [string]$Phase,
        [string]$Status,  # PASS, FAIL, SKIP
        [int]$Passed = 0,
        [int]$Failed = 0,
        [int]$Skipped = 0,
        [double]$Seconds = 0
    )
    $global:CiResults += [PSCustomObject]@{
        Phase   = $Phase
        Status  = $Status
        Passed  = $Passed
        Failed  = $Failed
        Skipped = $Skipped
        Time    = $Seconds
    }
}

function Should-Run {
    param([string]$Phase)
    if ($Only.Count -gt 0 -and $Phase -notin $Only) { return $false }
    if ($Phase -in $Skip) { return $false }
    return $true
}

function Format-Duration {
    param([double]$Seconds)
    if ($Seconds -lt 1) { return "$([math]::Round($Seconds * 1000))ms" }
    if ($Seconds -lt 60) { return "$([math]::Round($Seconds, 1))s" }
    $m = [math]::Floor($Seconds / 60)
    $s = [math]::Round($Seconds % 60, 1)
    return "${m}m ${s}s"
}

function Write-Summary {
    $totalElapsed = ((Get-Date) - $global:CiStartTime).TotalSeconds

    Write-Banner "CI RESULTS"

    $totalPassed = ($global:CiResults | Measure-Object -Property Passed -Sum).Sum
    $totalFailed = ($global:CiResults | Measure-Object -Property Failed -Sum).Sum
    $totalSkipped = ($global:CiResults | Measure-Object -Property Skipped -Sum).Sum

    # Phase table
    Write-Host "  Phase                     Status   Passed  Failed  Skipped  Time" -ForegroundColor White
    Write-Host "  $("-" * 72)" -ForegroundColor DarkGray

    foreach ($r in $global:CiResults) {
        $statusColor = switch ($r.Status) {
            "PASS" { "Green" }
            "FAIL" { "Red" }
            "SKIP" { "Yellow" }
            default { "Gray" }
        }
        $line = "  {0,-27} {1,-8} {2,6}  {3,6}  {4,7}  {5,8}" -f `
            $r.Phase, $r.Status, $r.Passed, $r.Failed, $r.Skipped, (Format-Duration $r.Time)
        Write-Host $line -ForegroundColor $statusColor
    }

    Write-Host "  $("-" * 72)" -ForegroundColor DarkGray
    Write-Host ("  {0,-27} {1,-8} {2,6}  {3,6}  {4,7}  {5,8}" -f `
        "TOTAL", "", $totalPassed, $totalFailed, $totalSkipped, (Format-Duration $totalElapsed)) -ForegroundColor White

    Write-Host ""

    if ($totalFailed -gt 0) {
        Write-Host "  RESULT: FAILED ($totalFailed failure(s))" -ForegroundColor Red
        Write-Host ""
        Write-Host "  Failed phases:" -ForegroundColor Red
        foreach ($r in ($global:CiResults | Where-Object { $_.Status -eq "FAIL" })) {
            Write-Host "    - $($r.Phase) ($($r.Failed) failure(s))" -ForegroundColor Red
        }
    } elseif ($totalSkipped -gt 0) {
        Write-Host "  RESULT: PASSED with $totalSkipped skip(s)" -ForegroundColor Yellow
    } else {
        Write-Host "  RESULT: ALL PASSED" -ForegroundColor Green
    }

    Write-Host "  Total time: $(Format-Duration $totalElapsed)" -ForegroundColor DarkGray
    Write-Host ""

    return ($totalFailed -eq 0)
}

# Main

Write-Banner "IGNIS CI" "Magenta"
Write-SystemInfo

$phases = @(
    @{ Name = "features";    Script = "wintests\test_features.ps1";     Desc = "Feature matrix compilation" },
    @{ Name = "lint";        Script = "wintests\test_lint.ps1";         Desc = "Clippy, rustfmt, doc warnings" },
    @{ Name = "unit";        Script = "wintests\test_unit.ps1";         Desc = "Unit tests (lib)" },
    @{ Name = "audit";       Script = "wintests\test_audit.ps1";        Desc = "Cross-feature import audit" },
    @{ Name = "doc";         Script = "wintests\test_doc.ps1";          Desc = "Documentation coverage" },
    @{ Name = "smoke";       Script = "wintests\test_smoke.ps1";        Desc = "Smoke test (GPU required)" },
    @{ Name = "size";        Script = "wintests\test_size.ps1";         Desc = "Binary size analysis" },
    @{ Name = "miri";        Script = "wintests\test_miri.ps1";         Desc = "Miri UB detection (nightly)" }
)

$phaseNum = 0
foreach ($phase in $phases) {
    $phaseNum++
    if (-not (Should-Run $phase.Name)) {
        Write-Phase $phaseNum $phases.Count $phase.Name $phase.Desc
        Write-Host "    SKIPPED (filtered)" -ForegroundColor Yellow
        Record-Result -Phase $phase.Name -Status "SKIP" -Skipped 1
        continue
    }

    Write-Phase $phaseNum $phases.Count $phase.Name $phase.Desc

    if (-not (Test-Path $phase.Script)) {
        Write-Host "    SKIPPED (script not found: $($phase.Script))" -ForegroundColor Yellow
        Record-Result -Phase $phase.Name -Status "SKIP" -Skipped 1
        continue
    }

    $sw = [System.Diagnostics.Stopwatch]::StartNew()
    $result = & $phase.Script
    $sw.Stop()

    if ($result -is [PSCustomObject] -and $null -ne $result.Passed) {
        $status = if ($result.Failed -gt 0) { "FAIL" } else { "PASS" }
        Record-Result -Phase $phase.Name -Status $status `
            -Passed $result.Passed -Failed $result.Failed -Skipped $result.Skipped `
            -Seconds $sw.Elapsed.TotalSeconds
    } elseif ($LASTEXITCODE -ne 0) {
        Record-Result -Phase $phase.Name -Status "FAIL" -Failed 1 -Seconds $sw.Elapsed.TotalSeconds
    } else {
        Record-Result -Phase $phase.Name -Status "PASS" -Passed 1 -Seconds $sw.Elapsed.TotalSeconds
    }
}

$success = Write-Summary

if (-not $success) { exit 1 }