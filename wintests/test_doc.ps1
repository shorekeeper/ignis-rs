#Requires -Version 5.1
# test_doc.ps1 - Documentation coverage and quality checks.

$ErrorActionPreference = "Continue"

Write-Host "    Documentation analysis" -ForegroundColor Cyan
Write-Host ""

$passed = 0
$failed = 0

# Missing doc check

Write-Host "      [1/3] Scanning for missing pub item docs ..." -ForegroundColor DarkGray

$missingDocs = @()
Get-ChildItem -Path "src" -Recurse -Filter "*.rs" | ForEach-Object {
    $path = $_.FullName
    $relative = ($path -replace '\\', '/') -replace '.*/src/', ''
    $lines = Get-Content $path

    for ($i = 0; $i -lt $lines.Count; $i++) {
        $line = $lines[$i].Trim()
        # Check pub items
        if ($line -match "^pub (fn|struct|enum|trait|type|mod|const|static) ") {
            # Look backwards for doc comment
            $hasDoc = $false
            for ($j = $i - 1; $j -ge [math]::Max(0, $i - 5); $j--) {
                $prev = $lines[$j].Trim()
                if ($prev -match "^///|^/\*\*|^#\[doc") {
                    $hasDoc = $true
                    break
                }
                if ($prev -ne "" -and $prev -notmatch "^#\[") { break }
            }
            if (-not $hasDoc) {
                $missingDocs += [PSCustomObject]@{
                    File = $relative
                    Line = $i + 1
                    Item = $line.Substring(0, [math]::Min(60, $line.Length))
                }
            }
        }
    }
}

if ($missingDocs.Count -eq 0) {
    Write-Host "        All pub items documented" -ForegroundColor Green
    $passed++
} else {
    Write-Host "        $($missingDocs.Count) pub item(s) missing documentation:" -ForegroundColor Yellow
    $missingDocs | Select-Object -First 20 | ForEach-Object {
        Write-Host "          $($_.File):$($_.Line)  $($_.Item)" -ForegroundColor DarkYellow
    }
    if ($missingDocs.Count -gt 20) {
        Write-Host "          ... $($missingDocs.Count - 20) more" -ForegroundColor DarkGray
    }
    # This is a warning, not a failure
    $passed++
}

# Module-level doc check 

Write-Host ""
Write-Host "      [2/3] Checking module-level //! docs ..." -ForegroundColor DarkGray

$modulesWithout = @()
Get-ChildItem -Path "src" -Recurse -Filter "*.rs" | ForEach-Object {
    $firstLine = (Get-Content $_.FullName | Select-Object -First 1).Trim()
    $relative = ($_.FullName -replace '\\', '/') -replace '.*/src/', ''
    if ($firstLine -notmatch "^//!" -and $_.Name -ne "mod.rs") {
        # Skip sub-test files etc
        if ($relative -notmatch "tests/") {
            $modulesWithout += $relative
        }
    }
}

if ($modulesWithout.Count -eq 0) {
    Write-Host "        All source files have module-level docs" -ForegroundColor Green
    $passed++
} else {
    Write-Host "        $($modulesWithout.Count) file(s) missing //! module docs:" -ForegroundColor Yellow
    $modulesWithout | Select-Object -First 15 | ForEach-Object {
        Write-Host "          $_" -ForegroundColor DarkYellow
    }
    $passed++ # Warning, not failure
}

# Doc build test

Write-Host ""
Write-Host -NoNewline "      [3/3] cargo doc --features full --no-deps ... "

$docOutput = cargo doc --features full --no-deps 2>&1
$docWarnings = @($docOutput | Where-Object { $_ -match "warning" })
$docErrors = @($docOutput | Where-Object { $_ -match "^error" })

if ($docErrors.Count -gt 0) {
    Write-Host "FAIL ($($docErrors.Count) error(s))" -ForegroundColor Red
    $failed++
    $docErrors | ForEach-Object { Write-Host "        $_" -ForegroundColor DarkRed }
} elseif ($docWarnings.Count -gt 0) {
    Write-Host "OK ($($docWarnings.Count) warning(s))" -ForegroundColor Yellow
    $passed++
} else {
    Write-Host "OK (clean)" -ForegroundColor Green
    $passed++
}

# Stats
Write-Host ""
$rsFiles = (Get-ChildItem -Path "src" -Recurse -Filter "*.rs").Count
$totalLines = (Get-ChildItem -Path "src" -Recurse -Filter "*.rs" | ForEach-Object { (Get-Content $_.FullName).Count } | Measure-Object -Sum).Sum
$docLines = (Get-ChildItem -Path "src" -Recurse -Filter "*.rs" | ForEach-Object {
    (Get-Content $_.FullName | Where-Object { $_ -match "^\s*///|^\s*//!" }).Count
} | Measure-Object -Sum).Sum

$docPct = if ($totalLines -gt 0) { [math]::Round($docLines / $totalLines * 100, 1) } else { 0 }

Write-Host "    Stats: $rsFiles source files, $totalLines total lines, $docLines doc lines ($docPct%)" -ForegroundColor DarkGray
Write-Host "    Doc: $passed passed, $failed failed" -ForegroundColor $(if ($failed -gt 0) { "Red" } else { "Green" })
Write-Host ""

return [PSCustomObject]@{
    Passed  = $passed
    Failed  = $failed
    Skipped = 0
}