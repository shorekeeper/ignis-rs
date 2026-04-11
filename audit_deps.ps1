# audit_deps.ps1
# Finds use crate:: imports that cross feature boundaries without cfg.

$ErrorActionPreference = "Stop"

Write-Host "Auditing cross-feature imports..." -ForegroundColor Cyan

# Map: file pattern -> which feature gates it.
$featureMap = @{
    "tracking/tracker.rs"  = "tracking"
    "tracking/deletion.rs" = "tracking"
    "memory/slab.rs"       = "slab-allocator"
    "pipeline/descriptor.rs" = "descriptors"
    "surface/"             = "swapchain"
    "interop.rs"           = "interop"
    "debug/"               = "debug-tools"
}

# Map: import pattern -> which feature it belongs to.
$importFeature = @{
    "tracking::tracker"       = "tracking"
    "tracking::deletion"      = "tracking"
    "memory::slab"            = "slab-allocator"
    "pipeline::descriptor"    = "descriptors"
    "surface::"               = "swapchain"
    "interop::"               = "interop"
    "debug::"                 = "debug-tools"
}

$issues = @()

Get-ChildItem -Path "src" -Recurse -Filter "*.rs" |
    Where-Object { $_.Name -ne "lib.rs" -and $_.Name -ne "mod.rs" } |
    ForEach-Object {
        $path = $_.FullName
        $relative = ($path -replace '\\', '/') -replace '.*/src/', ''
        $content = Get-Content $path -Raw

        # Determine this file's feature (if any).
        $myFeature = $null
        foreach ($pattern in $featureMap.Keys) {
            if ($relative -like "*$pattern*") {
                $myFeature = $featureMap[$pattern]
                break
            }
        }

        # Scan imports.
        $lines = $content -split "`n"
        $lineNum = 0
        foreach ($line in $lines) {
            $lineNum++
            if ($line -notmatch "use crate::") { continue }
            # Skip lines already cfg-gated.
            if ($lineNum -gt 1 -and $lines[$lineNum - 2] -match '#\[cfg\(feature') { continue }

            foreach ($importPattern in $importFeature.Keys) {
                if ($line -match [regex]::Escape($importPattern)) {
                    $requiredFeature = $importFeature[$importPattern]

                    # If this file is NOT in the same feature, it is a cross-feature import.
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

if ($issues.Count -eq 0) {
    Write-Host "  No ungated cross-feature imports found" -ForegroundColor Green
} else {
    Write-Host "  Found $($issues.Count) potentially ungated cross-feature import(s):" -ForegroundColor Yellow
    $issues | ForEach-Object {
        Write-Host "    $($_.File):$($_.Line)" -ForegroundColor Red
        Write-Host "      $($_.Import)" -ForegroundColor DarkRed
        Write-Host "      file is in [$($_.FileIn)], import needs [$($_.Needs)]" -ForegroundColor DarkYellow
    }
}