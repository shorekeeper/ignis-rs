#Requires -Version 5.1
# test_audit.ps1 - Cross-feature import audit.
# Finds use crate:: imports that cross feature boundaries without cfg.

$ErrorActionPreference = "Stop"

Write-Host "    Cross-feature import audit" -ForegroundColor Cyan
Write-Host "    Scanning for imports that cross feature boundaries without #[cfg]" -ForegroundColor DarkGray
Write-Host ""

$featureMap = @{
    "tracking/tracker.rs"      = "tracking"
    "tracking/deletion.rs"     = "tracking"
    "tracking/mipmap.rs"       = "tracking"
    "memory/slab.rs"           = "slab-allocator"
    "pipeline/descriptor.rs"   = "descriptors"
    "surface/"                 = "swapchain"
    "interop.rs"               = "interop"
    "debug/"                   = "debug-tools"
}

$importFeature = @{
    "tracking::tracker"       = "tracking"
    "tracking::deletion"      = "tracking"
    "tracking::mipmap"        = "tracking"
    "memory::slab"            = "slab-allocator"
    "pipeline::descriptor"    = "descriptors"
    "surface::"               = "swapchain"
    "interop::"               = "interop"
    "debug::"                 = "debug-tools"
}

$issues = @()
$filesScanned = 0
$importsChecked = 0

Get-ChildItem -Path "src" -Recurse -Filter "*.rs" |
    Where-Object { $_.Name -ne "lib.rs" -and $_.Name -ne "mod.rs" } |
    ForEach-Object {
        $path = $_.FullName
        $relative = ($path -replace '\\', '/') -replace '.*/src/', ''
        $content = Get-Content $path -Raw
        $filesScanned++

        # Determine this file's feature
        $myFeature = $null
        foreach ($pattern in $featureMap.Keys) {
            if ($relative -like "*$pattern*") {
                $myFeature = $featureMap[$pattern]
                break
            }
        }

        $lines = $content -split "`n"
        $lineNum = 0
        foreach ($line in $lines) {
            $lineNum++
            if ($line -notmatch "use crate::") { continue }
            $importsChecked++

            # Skip lines already cfg-gated
            if ($lineNum -gt 1 -and $lines[$lineNum - 2] -match '#\[cfg\(feature') { continue }

            foreach ($importPattern in $importFeature.Keys) {
                if ($line -match [regex]::Escape($importPattern)) {
                    $requiredFeature = $importFeature[$importPattern]
                    if ($myFeature -ne $requiredFeature) {
                        $issues += [PSCustomObject]@{
                            File    = $relative
                            Line    = $lineNum
                            Import  = $line.Trim()
                            Needs   = $requiredFeature
                            FileIn  = if ($myFeature) { $myFeature } else { "core" }
                        }
                    }
                }
            }
        }
    }

Write-Host "      Files scanned:  $filesScanned" -ForegroundColor DarkGray
Write-Host "      Imports checked: $importsChecked" -ForegroundColor DarkGray
Write-Host ""

$passed = 0
$failed = 0

if ($issues.Count -eq 0) {
    Write-Host "      No ungated cross-feature imports found" -ForegroundColor Green
    $passed = 1
} else {
    Write-Host "      Found $($issues.Count) potentially ungated cross-feature import(s):" -ForegroundColor Yellow
    Write-Host ""
    $failed = $issues.Count

    foreach ($issue in $issues) {
        Write-Host "      $($issue.File):$($issue.Line)" -ForegroundColor Red
        Write-Host "        $($issue.Import)" -ForegroundColor DarkRed
        Write-Host "        file is in [$($issue.FileIn)], import needs [$($issue.Needs)]" -ForegroundColor DarkYellow
        Write-Host "        add: #[cfg(feature = `"$($issue.Needs)`")]" -ForegroundColor DarkGray
        Write-Host ""
    }
}

# Also check for stale feature gate references
Write-Host "      Checking for stale feature references ..." -ForegroundColor DarkGray
$knownFeatures = @("tracking", "descriptors", "debug-tools", "slab-allocator", "swapchain", "interop", "full")
$staleCount = 0

Get-ChildItem -Path "src" -Recurse -Filter "*.rs" | ForEach-Object {
    $content = Get-Content $_.FullName -Raw
    $matches = [regex]::Matches($content, '#\[cfg\(feature\s*=\s*"([^"]+)"\)')
    foreach ($m in $matches) {
        $feat = $m.Groups[1].Value
        if ($feat -notin $knownFeatures) {
            $relative = ($_.FullName -replace '\\', '/') -replace '.*/src/', ''
            Write-Host "      STALE: $relative references unknown feature '$feat'" -ForegroundColor Yellow
            $staleCount++
        }
    }
}

if ($staleCount -eq 0) {
    Write-Host "      No stale feature references" -ForegroundColor Green
    $passed++
} else {
    Write-Host "      $staleCount stale feature reference(s) found" -ForegroundColor Yellow
}

Write-Host ""
Write-Host "    Audit: $passed passed, $failed issue(s)" -ForegroundColor $(if ($failed -gt 0) { "Yellow" } else { "Green" })
Write-Host ""

return [PSCustomObject]@{
    Passed  = $passed
    Failed  = $failed
    Skipped = 0
}