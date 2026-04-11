# cmd_build.ps1 [full|release|minimal] [--features X,Y] [--release]
param([Parameter(ValueFromRemainingArguments)][string[]]$Args)


# Parse args
$features = ""
$release = $false
$target = "lib"

for ($i = 0; $i -lt $Args.Count; $i++) {
    switch ($Args[$i]) {
        "full"      { $features = "full" }
        "minimal"   { $features = "" }
        "release"   { $release = $true }
        "--release" { $release = $true }
        "--features" { $i++; if ($i -lt $Args.Count) { $features = $Args[$i] } }
        "--example" { $i++; $target = "example"; if ($i -lt $Args.Count) { $target = $Args[$i] } }
        default {
            if ($Args[$i] -notmatch "^-") { $features = $Args[$i] }
        }
    }
}

$label = if ($features) { $features } else { "no features" }
$mode = if ($release) { "release" } else { "dev" }

Write-CmdHeader "build" "[$label] [$mode]"

$cargoArgs = @("build", "--lib")
if ($features) { $cargoArgs += "--features"; $cargoArgs += $features }
if ($release) { $cargoArgs += "--release" }

$sw = [System.Diagnostics.Stopwatch]::StartNew()
$output = & cargo @cargoArgs 2>&1
$exitCode = $LASTEXITCODE
$sw.Stop()

$errors = @($output | Where-Object { "$_" -match "^error" })
$warnings = @($output | Where-Object { "$_" -match "^warning" })

if ($exitCode -eq 0) {
    Write-SubStep "compile" "OK" "($(Format-Duration $sw.Elapsed.TotalMilliseconds))"

    if ($warnings.Count -gt 0) {
        Write-Host ""
        Write-Host "    $($warnings.Count) warning(s):" -ForegroundColor Yellow
        $warnings | Select-Object -First 10 | ForEach-Object {
            $w = "$_"
            # Extract just the warning message
            if ($w -match "warning:\s*(.+)") {
                Write-Host "      $($Matches[1])" -ForegroundColor DarkYellow
            }
        }
        if ($warnings.Count -gt 10) {
            Write-Host "      ... $($warnings.Count - 10) more" -ForegroundColor DarkGray
        }
    }

    # Show binary size
    $targetDir = if ($release) { "target\release" } else { "target\debug" }
    $rlib = Get-ChildItem -Path $targetDir -Filter "libignis*.rlib" -ErrorAction SilentlyContinue | Select-Object -First 1
    if ($rlib) {
        $sizeKB = [math]::Round($rlib.Length / 1024, 1)
        Write-Host "    output: $($rlib.Name) (${sizeKB} KiB)" -ForegroundColor DarkGray
    }
} else {
    Write-SubStep "compile" "FAIL" ""

    # Parse and display errors nicely
    $parsedErrors = Format-CargoError ($output | ForEach-Object { "$_" })

    if ($parsedErrors.Count -gt 0) {
        Write-Host ""
        foreach ($err in $parsedErrors) {
            Write-Host "    $($err.Header)" -ForegroundColor Red
            if ($err.Location) {
                Write-Host "      at $($err.Location)" -ForegroundColor DarkGray
            }
            $err.Context | Select-Object -First 5 | ForEach-Object {
                Write-Host "      $_" -ForegroundColor DarkRed
            }
            Write-Host ""
        }
    } else {
        # Fallback: show raw errors
        $output | Where-Object { "$_" -match "error" } | Select-Object -First 20 | ForEach-Object {
            Write-Host "    $_" -ForegroundColor Red
        }
    }
}
