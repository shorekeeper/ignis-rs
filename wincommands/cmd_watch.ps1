#Requires -Version 7.0
#
# cmd_watch.ps1 - Continuous rebuild on source change.
#
# SYNOPSIS
#   watch [check|build|test|lint|smoke] [--features X]
#
# DESCRIPTION
#   Watches the Rust source tree and reruns the selected cargo operation
#   whenever a file changes, presenting each run with the standard progress
#   bar and an error and warning delta against the previous run. This turns
#   the shell into a background verifier during editing sessions.
#
# TARGETS
#   check   cargo check --lib (default)
#   build   cargo build --lib
#   test    cargo test --lib (streams test output)
#   lint    cargo clippy --all-targets
#   smoke   cargo run --example smoke_test (streams program output)
#   The feature set defaults to "full"; pass "--features X" to override, or
#   "--features minimal" for no features.
#
# CHANGE DETECTION
#   Polling, not FileSystemWatcher: a snapshot string of every watched
#   file's path, mtime ticks and length is rebuilt every 300 ms and compared
#   to the previous snapshot. Polling survives editors that replace files
#   (which breaks watcher handles) and costs microseconds at this tree size.
#   The watched set is src recursively (*.rs), Cargo.toml, and additionally
#   examples (*.rs) for the smoke target.
#
#   A change triggers a debounce loop that waits until the snapshot has been
#   stable for 350 ms before running, so editor save bursts and rustfmt
#   rewrites coalesce into a single run.
#
# CONTROLS
#   q or Esc  exit watch mode
#   r or Enter  force a rerun immediately
#   Ctrl+C     also exits (handled by the shell's interrupt path, which
#              kills any in-flight cargo process tree)
#
# OUTPUT
#   The screen is cleared per run. The header carries the run number, the
#   target, and the time; the footer shows the error and warning delta
#   relative to the previous run with green for improvement and red for
#   regression, then returns to the watching state line.

param([Parameter(ValueFromRemainingArguments)][string[]]$RawArgs)

Get-ChildItem (Join-Path $PSScriptRoot "_*.ps1") | ForEach-Object { . $_.FullName }

$target = 'check'
$features = 'full'
for ($i = 0; $i -lt $RawArgs.Count; $i++) {
    switch ($RawArgs[$i]) {
        '--features' { $i++; if ($i -lt $RawArgs.Count) { $features = $RawArgs[$i] } }
        default { if ($RawArgs[$i] -notmatch '^-') { $target = $RawArgs[$i] } }
    }
}
if ($features -eq 'minimal') { $features = '' }

# Resolve the cargo invocation for the selected target. ShowOutput selects
# streaming for targets whose per-line output is the product (tests, smoke).
$plan = switch ($target) {
    'check' { @{ Args = @('check', '--lib');                    Stream = $false } }
    'build' { @{ Args = @('build', '--lib');                    Stream = $false } }
    'test'  { @{ Args = @('test', '--lib');                     Stream = $true } }
    'lint'  { @{ Args = @('clippy', '--all-targets');           Stream = $false } }
    'smoke' { @{ Args = @('run', '--example', 'smoke_test');    Stream = $true } }
    default { $null }
}
if ($null -eq $plan) {
    Write-Host "    unknown watch target: $target (use check, build, test, lint, smoke)" -ForegroundColor Red
    return
}
$cargoArgs = $plan.Args
if ($features) { $cargoArgs += '--features'; $cargoArgs += $features }
if ($target -eq 'lint') { $cargoArgs += '--'; $cargoArgs += '-W'; $cargoArgs += 'clippy::all' }

function Get-WatchSnapshot {
    param([string[]]$Roots)
    $parts = [System.Collections.Generic.List[string]]::new()
    foreach ($r in $Roots) {
        if (-not (Test-Path $r)) { continue }
        $item = Get-Item $r
        if ($item -is [System.IO.DirectoryInfo]) {
            foreach ($f in (Get-ChildItem $r -Recurse -Filter '*.rs' -File -ErrorAction SilentlyContinue)) {
                $parts.Add($f.FullName + '|' + $f.LastWriteTimeUtc.Ticks + '|' + $f.Length)
            }
        } else {
            $parts.Add($item.FullName + '|' + $item.LastWriteTimeUtc.Ticks + '|' + $item.Length)
        }
    }
    return ($parts -join "`n")
}

$roots = @('src', 'Cargo.toml')
if ($target -eq 'smoke') { $roots += 'examples' }

$featLabel = if ($features) { $features } else { 'no features' }
$runNo = 0
$prevErr = $null
$prevWarn = $null

function Invoke-WatchRun {
    param([int]$N)
    Clear-Host
    Write-Host ""
    Write-Host "  watch run #$N" -ForegroundColor Cyan -NoNewline
    Write-Host "  $target [$featLabel]  $(Get-Date -Format 'HH:mm:ss')" -ForegroundColor DarkGray
    Write-Host "  $("-" * 60)" -ForegroundColor DarkGray
    return Invoke-CargoWithProgress `
        -Label "$target $featLabel" `
        -CargoArgs $cargoArgs `
        -ShowProgress $true `
        -ShowOutput $plan.Stream
}

function Show-WatchDelta {
    param($Result)
    $e = $Result.Errors.Count
    $w = $Result.Warnings.Count
    if ($null -ne $script:prevErr) {
        $de = $e - $script:prevErr
        $dw = $w - $script:prevWarn
        $eCol = if ($de -gt 0) { 'Red' } elseif ($de -lt 0) { 'Green' } else { 'DarkGray' }
        $wCol = if ($dw -gt 0) { 'Yellow' } elseif ($dw -lt 0) { 'Green' } else { 'DarkGray' }
        $eSign = if ($de -gt 0) { '+' } else { '' }
        $wSign = if ($dw -gt 0) { '+' } else { '' }
        Write-Host ""
        Write-Host "    errors $($script:prevErr) -> $e ($eSign$de)" -ForegroundColor $eCol -NoNewline
        Write-Host "   warnings $($script:prevWarn) -> $w ($wSign$dw)" -ForegroundColor $wCol
    }
    $script:prevErr = $e
    $script:prevWarn = $w
    Write-Host ""
    Write-Host "  watching $($roots -join ', ')  [q]uit  [r]erun" -ForegroundColor DarkGray
}

$snap = Get-WatchSnapshot -Roots $roots
$runNo++
$res = Invoke-WatchRun -N $runNo
Show-WatchDelta -Result $res

while ($true) {
    if ([Console]::KeyAvailable) {
        $k = [Console]::ReadKey($true)
        if ($k.Key -eq [ConsoleKey]::Q -or $k.Key -eq [ConsoleKey]::Escape) {
            Write-Host "  watch stopped after $runNo run(s)" -ForegroundColor DarkGray
            break
        }
        if ($k.Key -eq [ConsoleKey]::R -or $k.Key -eq [ConsoleKey]::Enter) {
            $runNo++
            $res = Invoke-WatchRun -N $runNo
            Show-WatchDelta -Result $res
            $snap = Get-WatchSnapshot -Roots $roots
            continue
        }
    }
    Start-Sleep -Milliseconds 300
    $new = Get-WatchSnapshot -Roots $roots
    if ($new -ne $snap) {
        # Debounce: wait for the tree to stop changing before running.
        do {
            $snap = $new
            Start-Sleep -Milliseconds 350
            $new = Get-WatchSnapshot -Roots $roots
        } while ($new -ne $snap)
        $runNo++
        $res = Invoke-WatchRun -N $runNo
        Show-WatchDelta -Result $res
        $snap = Get-WatchSnapshot -Roots $roots
    }
}