#Requires -Version 7.0
# cmd_stub.ps1 [--out FILE] [--filter substring] [--full] [--elide-lines N]
#
#   --out FILE       output path (default: api_stubs.md)
#   --filter STR     only files whose relative path contains STR
#   --full           disable exclusion map AND static elision (everything in)
#   --elide-lines N  const/static bodies longer than N lines are elided
#                    (default 30; 0 = never elide)
param([Parameter(ValueFromRemainingArguments)][string[]]$RawArgs)
Get-ChildItem (Join-Path $PSScriptRoot "_*.ps1") | ForEach-Object { . $_.FullName }

# ---- args ----------------------------------------------------------------

$outFile = "api_stubs.md"
$filter = ""
$full = $false
$elideLines = 30

for ($i = 0; $i -lt $RawArgs.Count; $i++) {
    switch ($RawArgs[$i]) {
        "--out"         { $i++; $outFile = $RawArgs[$i] }
        "--filter"      { $i++; $filter = $RawArgs[$i] }
        "--full"        { $full = $true }
        "--elide-lines" { $i++; $elideLines = [int]$RawArgs[$i] }
    }
}
if ($full) { $elideLines = 0 }

# ---- exclusion map ---------------------------------------------------------
# Files whose content is noise for an LLM digest. Each entry leaves a
# one-line breadcrumb in the output so the model knows the module exists.
# Globs match against the src/-relative path with forward slashes.

$excludeMap = @(
    @{ Glob = "src/debug_window/*"
       Why  = "self-contained CPU-rasterized debug window (feature debug-window); entry point: DebugWindow::builder(), see lib.rs re-exports" },
    @{ Glob = "src/debug/raster_common.rs"
       Why  = "internal CPU rasterizer + 8x8 font + BMP encoder used by sync_dag_viz; public surface: Framebuffer, palette, save_bmp" },
    @{ Glob = "src/live_link.rs"
       Why  = "IPC wire protocol producer; canonical protocol definition lives in ignis-viz/src/ipc.rs. Public surface: LiveLink::create + record_* methods + bridge_*_to_live_link functions" }
)

function Test-Excluded {
    param([string]$RelPath)
    foreach ($e in $excludeMap) {
        if ($RelPath -like $e.Glob) { return $e.Why }
    }
    return $null
}

# ---- post-pass: elide huge top-level static slice literals ----------------
# The lexer elides `{ }` initializer bodies, but slice statics like
# `static FOO: &[T] = &[ ...thousands of lines... ];` start with `[`.
# Catch them here: find `= &[` / `= [` at a line whose statement spans
# more than the threshold, and collapse the bracket body.

function Limit-HugeStatics {
    param([string]$Text, [int]$Threshold)
    if ($Threshold -le 0) { return $Text }

    $sb = [System.Text.StringBuilder]::new($Text.Length)
    $i = 0
    $n = $Text.Length
    while ($i -lt $n) {
        # cheap scan for the pattern start; full lexing not needed here
        # because this runs on ALREADY-stripped output (no fn bodies,
        # strings inside data are still possible - handled below).
        if ($Text[$i] -eq '=' ) {
            # lookahead: optional whitespace, optional &, then [
            $j = $i + 1
            while ($j -lt $n -and ($Text[$j] -eq ' ' -or $Text[$j] -eq "`n" -or $Text[$j] -eq "`r")) { $j++ }
            $amp = $false
            if ($j -lt $n -and $Text[$j] -eq '&') { $amp = $true; $j++ }
            if ($j -lt $n -and $Text[$j] -eq '[') {
                # measure the balanced bracket body, string-aware
                $k = $j
                $depth = 0
                $lines = 0
                $inStr = $false
                while ($k -lt $n) {
                    $c = $Text[$k]
                    if ($inStr) {
                        if ($c -eq '\') { $k += 2; continue }
                        if ($c -eq '"') { $inStr = $false }
                    } else {
                        if ($c -eq '"') { $inStr = $true }
                        elseif ($c -eq '[') { $depth++ }
                        elseif ($c -eq ']') {
                            $depth--
                            if ($depth -eq 0) { break }
                        }
                        elseif ($c -eq "`n") { $lines++ }
                    }
                    $k++
                }
                if ($lines -gt $Threshold) {
                    $prefix = if ($amp) { "= &[" } else { "= [" }
                    $null = $sb.Append($prefix)
                    $null = $sb.Append(" /* ~$lines lines of data elided */ ]")
                    $i = $k + 1
                    continue
                }
            }
        }
        $null = $sb.Append($Text[$i])
        $i++
    }
    return $sb.ToString()
}

# ---- crate name from Cargo.toml -------------------------------------------

$crateName = Split-Path (Get-Location) -Leaf
if (Test-Path "Cargo.toml") {
    $tomlName = Select-String -Path "Cargo.toml" -Pattern '^\s*name\s*=\s*"([^"]+)"' |
        Select-Object -First 1
    if ($tomlName) { $crateName = $tomlName.Matches[0].Groups[1].Value }
}

# ---- main ------------------------------------------------------------------

$modeStr = if ($full) { "full" } else { "digest" }
Write-CmdHeader "stub" "[$modeStr] -> $outFile$(if ($filter) { " (filter: $filter)" })"

$fence = [string]::new('`', 3)
$sb = [System.Text.StringBuilder]::new()
$null = $sb.AppendLine("# $crateName - API stubs (bodies stripped, not valid Rust)")
$null = $sb.AppendLine("")
$null = $sb.AppendLine("Conventions: fn bodies replaced with ``;``. Struct/enum fields kept.")
$null = $sb.AppendLine("``#[cfg(test)] mod tests`` stripped. Large const/static data elided.")

$totalIn = 0; $totalOut = 0; $count = 0; $excluded = 0; $failed = 0

$files = Get-ChildItem -Path "src" -Recurse -Filter "*.rs" | Sort-Object FullName
foreach ($f in $files) {
    $rel = ($f.FullName -replace '\\', '/') -replace '.*/src/', 'src/'
    if ($filter -and $rel -notlike "*$filter*") { continue }

    # exclusion map (bypassed by --full)
    if (-not $full) {
        $why = Test-Excluded $rel
        if ($why) {
            $null = $sb.AppendLine("").AppendLine("## $rel")
            $null = $sb.AppendLine("*(excluded: $why)*")
            Write-SubStep $rel "SKIP" "excluded"
            $excluded++
            continue
        }
    }

    $sw = [System.Diagnostics.Stopwatch]::StartNew()
    try {
        $stub = Convert-RustToStub -Path $f.FullName -DataElideThreshold $elideLines
        $stub = Limit-HugeStatics -Text $stub -Threshold $elideLines
    } catch {
        Write-SubStep $rel "FAIL" $_.Exception.Message
        $failed++
        continue
    }
    $sw.Stop()

    $inLen = (Get-Item $f.FullName).Length
    $totalIn += $inLen; $totalOut += $stub.Length; $count++

    $null = $sb.AppendLine("").AppendLine("## $rel").AppendLine("${fence}rust")
    $null = $sb.AppendLine($stub.TrimEnd()).AppendLine($fence)

    $pct = [math]::Round($stub.Length / [math]::Max(1, $inLen) * 100)
    Write-SubStep $rel "OK" "$pct% kept, $(Format-Duration $sw.Elapsed.TotalMilliseconds)"
}

[System.IO.File]::WriteAllText($outFile, $sb.ToString())

# ---- self-check: leaked fn bodies ------------------------------------------
# Invariant: a correct stub contains no lines starting with `let ` outside
# doc comments. Catches both misclassification bug classes.

$leaks = @(Select-String -Path $outFile -Pattern '^\s+let\s' |
    Where-Object { $_.Line -notmatch '^\s*//' })

# ---- summary ----------------------------------------------------------------

$tokIn = [math]::Round($totalIn / 3.5 / 1000)   # ~3.5 chars/token for code
$tokOut = [math]::Round($totalOut / 3.5 / 1000)
Write-Host ""
Write-Host "    $count stripped, $excluded excluded$(if ($failed) { ", $failed FAILED" })" -ForegroundColor $(if ($failed) { "Yellow" } else { "Green" })
Write-Host "    ~${tokIn}k tokens -> ~${tokOut}k tokens" -ForegroundColor Green
Write-Host "    written: $outFile" -ForegroundColor DarkGray

if ($leaks.Count -gt 0) {
    Write-Host ""
    Write-Host "    WARNING: $($leaks.Count) possible leaked fn bodies (run 'stub --full' unaffected):" -ForegroundColor Yellow
    $leaks | Select-Object -First 8 | ForEach-Object {
        $line = $_.Line.Trim()
        if ($line.Length -gt 70) { $line = $line.Substring(0, 67) + "..." }
        Write-Host "      line $($_.LineNumber): $line" -ForegroundColor DarkYellow
    }
    if ($leaks.Count -gt 8) {
        Write-Host "      ... +$($leaks.Count - 8) more" -ForegroundColor DarkGray
    }
}