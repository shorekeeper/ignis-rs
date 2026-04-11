# _trace_engine.ps1 - Structured diagnostic engine for trace analysis.
#
# Parses cargo/rustc output into structured error objects, correlates
# failures across commands, identifies root causes, and formats
# rich diagnostic reports.
#
# Loaded by shell.ps1 at startup. Used by cmd_trace.ps1 and
# automatically by Invoke-Command-Script for post-mortem analysis.

# ── Structured Error Types ───────────────────────────────────────────────────

class TraceError {
    [string]$Code          # E0308, E0599, etc. Empty for non-rustc errors.
    [string]$Level         # error, warning, note, help
    [string]$Message       # "mismatched types"
    [string]$File          # src/foo.rs
    [int]$Line             # 42
    [int]$Column           # 5
    [string[]]$Context     # source lines with | markers
    [string[]]$Notes       # "note: ..." lines
    [string[]]$Helps       # "help: ..." lines
    [string]$RawBlock      # full original text
    [string]$Category      # compile, lint, test, link, doc
}

class TraceWarning {
    [string]$Code
    [string]$Message
    [string]$File
    [int]$Line
    [string]$LintGroup     # clippy::xxx or rustc lint name
    [string]$Category
}

class TraceTestFailure {
    [string]$TestName      # module::test_name
    [string]$PanicMessage  # "assertion failed: ..."
    [string]$PanicLocation # src/foo.rs:42:5
    [string[]]$Backtrace
}

class TraceSession {
    [string]$SessionId
    [datetime]$StartTime
    [PSCustomObject[]]$Entries
    [TraceError[]]$AllErrors
    [TraceWarning[]]$AllWarnings
    [TraceTestFailure[]]$AllTestFailures
    [hashtable]$HotFiles        # file -> error count
    [hashtable]$ErrorCodes      # E0xxx -> count
    [string[]]$RootCauses       # deduced root cause descriptions
}

# ── Cargo/Rustc Output Parser ────────────────────────────────────────────────

function Parse-CargoOutput {
    <#
    .SYNOPSIS
    Parse raw cargo/rustc output into structured error and warning objects.

    .DESCRIPTION
    Handles multi-line error blocks with source context, notes, and help
    suggestions. Recognizes error codes (E0xxx), clippy lint names, and
    doc warnings. Groups continuation lines with their parent error.
    #>
    param([string[]]$Lines)

    $errors = [System.Collections.ArrayList]::new()
    $warnings = [System.Collections.ArrayList]::new()

    $i = 0
    while ($i -lt $Lines.Count) {
        $line = $Lines[$i]

        # ── Error block ──────────────────────────────────────────────────
        if ($line -match "^error(\[(E\d+)\])?:\s*(.+)") {
            $err = [TraceError]::new()
            $err.Code = if ($Matches[2]) { $Matches[2] } else { "" }
            $err.Message = $Matches[3]
            $err.Level = "error"
            $err.Context = @()
            $err.Notes = @()
            $err.Helps = @()
            $err.Category = "compile"
            $rawBlock = @($line)

            $i++

            # Scan continuation lines
            while ($i -lt $Lines.Count) {
                $cl = $Lines[$i]

                if ($cl -match "^\s+-->\s*(.+):(\d+):(\d+)") {
                    $err.File = $Matches[1]
                    $err.Line = [int]$Matches[2]
                    $err.Column = [int]$Matches[3]
                    $rawBlock += $cl
                    $i++
                }
                elseif ($cl -match "^\s+\|") {
                    $err.Context += $cl
                    $rawBlock += $cl
                    $i++
                }
                elseif ($cl -match "^\s+=\s*note:\s*(.+)") {
                    $err.Notes += $Matches[1]
                    $rawBlock += $cl
                    $i++
                }
                elseif ($cl -match "^\s+=\s*help:\s*(.+)") {
                    $err.Helps += $Matches[1]
                    $rawBlock += $cl
                    $i++
                }
                elseif ($cl -match "^\s+$" -or $cl -eq "") {
                    $rawBlock += $cl
                    $i++
                }
                else {
                    break
                }
            }

            $err.RawBlock = $rawBlock -join "`n"

            # Categorize
            if ($err.Message -match "aborting due to") { continue }
            if ($err.Message -match "could not compile") { continue }
            if ($line -match "error\[E") { $err.Category = "compile" }
            elseif ($err.Message -match "linker|undefined reference|unresolved") { $err.Category = "link" }
            elseif ($err.Message -match "doctest|rustdoc|documentation") { $err.Category = "doc" }

            $null = $errors.Add($err)
            continue
        }

        # ── Warning block ────────────────────────────────────────────────
        if ($line -match "^warning(\[([\w:]+)\])?:\s*(.+)") {
            $warn = [TraceWarning]::new()
            $warn.Code = if ($Matches[2]) { $Matches[2] } else { "" }
            $warn.Message = $Matches[3]
            $warn.Category = "compile"

            if ($warn.Code -match "^clippy::") { $warn.LintGroup = $warn.Code; $warn.Category = "lint" }
            elseif ($warn.Code -match "^unused") { $warn.LintGroup = $warn.Code }
            elseif ($warn.Code -match "^dead_code") { $warn.LintGroup = $warn.Code }
            else { $warn.LintGroup = "" }

            $i++

            # Grab location
            if ($i -lt $Lines.Count -and $Lines[$i] -match "^\s+-->\s*(.+):(\d+):(\d+)") {
                $warn.File = $Matches[1]
                $warn.Line = [int]$Matches[2]
                $i++
            }

            # Skip continuation
            while ($i -lt $Lines.Count -and ($Lines[$i] -match "^\s+[\|=]" -or $Lines[$i] -match "^\s*$")) {
                $i++
            }

            if ($warn.Message -notmatch "generated \d+ warning") {
                $null = $warnings.Add($warn)
            }
            continue
        }

        # ── Test failure ─────────────────────────────────────────────────
        # Handled separately in Parse-TestOutput

        $i++
    }

    return @{
        Errors   = [TraceError[]]$errors.ToArray()
        Warnings = [TraceWarning[]]$warnings.ToArray()
    }
}

function Parse-TestOutput {
    <#
    .SYNOPSIS
    Parse cargo test output into structured test failure objects.
    #>
    param([string[]]$Lines)

    $failures = [System.Collections.ArrayList]::new()
    $allOutput = $Lines -join "`n"

    # Find "thread 'test_name' panicked at" blocks
    for ($i = 0; $i -lt $Lines.Count; $i++) {
        $line = $Lines[$i]

        if ($line -match "thread '(.+)' panicked at (.+)") {
            $fail = [TraceTestFailure]::new()
            $fail.TestName = $Matches[1]
            $fail.PanicLocation = $Matches[2]
            $fail.Backtrace = @()

            # Next line is usually the panic message
            if ($i + 1 -lt $Lines.Count) {
                $msgLine = $Lines[$i + 1]
                if ($msgLine -notmatch "^note:" -and $msgLine -notmatch "^thread") {
                    $fail.PanicMessage = $msgLine.Trim()
                }
            }

            # Collect backtrace if present
            $j = $i + 1
            while ($j -lt $Lines.Count) {
                if ($Lines[$j] -match "^\s+\d+:") {
                    $fail.Backtrace += $Lines[$j].Trim()
                } elseif ($Lines[$j] -match "^thread|^test result:") {
                    break
                }
                $j++
            }

            $null = $failures.Add($fail)
        }
        elseif ($line -match "^test (.+) \.\.\. FAILED") {
            # Might not have a panic block (timeout, etc.)
            $testName = $Matches[1]
            $existing = $failures | Where-Object { $_.TestName -eq $testName }
            if (-not $existing) {
                $fail = [TraceTestFailure]::new()
                $fail.TestName = $testName
                $fail.PanicMessage = "(no panic info captured)"
                $null = $failures.Add($fail)
            }
        }
    }

    return [TraceTestFailure[]]$failures.ToArray()
}

# ── Analysis Engine ──────────────────────────────────────────────────────────

function Analyze-TraceSession {
    <#
    .SYNOPSIS
    Analyze a trace session and produce correlations, hot files, and root causes.
    #>
    param([TraceSession]$Session)

    # Hot files: which source files appear most in errors
    $hotFiles = @{}
    foreach ($err in $Session.AllErrors) {
        if ($err.File) {
            $key = $err.File
            if (-not $hotFiles.ContainsKey($key)) { $hotFiles[$key] = 0 }
            $hotFiles[$key]++
        }
    }
    $Session.HotFiles = $hotFiles

    # Error code frequency
    $codeCounts = @{}
    foreach ($err in $Session.AllErrors) {
        if ($err.Code) {
            if (-not $codeCounts.ContainsKey($err.Code)) { $codeCounts[$err.Code] = 0 }
            $codeCounts[$err.Code]++
        }
    }
    $Session.ErrorCodes = $codeCounts

    # Root cause deduction
    $causes = [System.Collections.ArrayList]::new()

    # Missing import/use -> usually one root cause cascades
    $missingItems = @($Session.AllErrors | Where-Object { $_.Code -eq "E0433" -or $_.Code -eq "E0412" })
    if ($missingItems.Count -gt 3) {
        $files = ($missingItems | ForEach-Object { $_.File } | Select-Object -Unique) -join ", "
        $null = $causes.Add("Multiple unresolved imports ($($missingItems.Count) errors) suggest a missing 'use' or 'mod' declaration. Check: $files")
    }

    # Type mismatch cascade
    $typeMismatch = @($Session.AllErrors | Where-Object { $_.Code -eq "E0308" })
    if ($typeMismatch.Count -gt 2) {
        $file = ($typeMismatch | Group-Object File | Sort-Object Count -Descending | Select-Object -First 1).Name
        $null = $causes.Add("$($typeMismatch.Count) type mismatches, concentrated in $file. The first one at line $($typeMismatch[0].Line) is likely the root cause.")
    }

    # Trait not implemented
    $traitErrors = @($Session.AllErrors | Where-Object { $_.Code -eq "E0277" })
    if ($traitErrors.Count -gt 0) {
        foreach ($te in ($traitErrors | Select-Object -First 3)) {
            if ($te.Message -match "the trait bound `(.+)` is not satisfied") {
                $null = $causes.Add("Missing trait impl: $($Matches[1]) at $($te.File):$($te.Line)")
            }
        }
    }

    # Borrow checker
    $borrowErrors = @($Session.AllErrors | Where-Object { $_.Code -match "E050[1-9]|E051[0-9]|E0382|E0383" })
    if ($borrowErrors.Count -gt 0) {
        $null = $causes.Add("$($borrowErrors.Count) borrow checker error(s). Start with the first one at $($borrowErrors[0].File):$($borrowErrors[0].Line) - later ones often resolve when the first is fixed.")
    }

    # Dead code warnings suggest unused feature-gated code
    $deadCode = @($Session.AllWarnings | Where-Object { $_.LintGroup -eq "dead_code" })
    if ($deadCode.Count -gt 5) {
        $null = $causes.Add("$($deadCode.Count) dead_code warnings. Likely feature-gated code not covered by current feature set. Run 'build full' to verify.")
    }

    # Linker errors
    $linkErrors = @($Session.AllErrors | Where-Object { $_.Category -eq "link" })
    if ($linkErrors.Count -gt 0) {
        $null = $causes.Add("Linker error(s) detected. Check that native libraries (Vulkan loader) are installed and in PATH.")
    }

    $Session.RootCauses = [string[]]$causes.ToArray()

    return $Session
}

# ── Formatting ───────────────────────────────────────────────────────────────

function Format-TraceError {
    <#
    .SYNOPSIS
    Format a single structured error with colors and context.
    #>
    param([TraceError]$Err, [int]$Index)

    $codeStr = if ($Err.Code) { "[$($Err.Code)] " } else { "" }

    Write-Host ""
    Write-Host "    ┌─ Error #$Index $codeStr" -ForegroundColor Red
    Write-Host "    │  $($Err.Message)" -ForegroundColor White

    if ($Err.File) {
        $location = "$($Err.File):$($Err.Line):$($Err.Column)"
        Write-Host "    │  at $location" -ForegroundColor DarkGray
    }

    if ($Err.Context.Count -gt 0) {
        Write-Host "    │" -ForegroundColor DarkGray
        foreach ($ctx in ($Err.Context | Select-Object -First 8)) {
            $ctxStr = "$ctx"
            # Highlight the error marker line
            if ($ctxStr -match "\^") {
                Write-Host "    │  $ctxStr" -ForegroundColor Red
            } else {
                Write-Host "    │  $ctxStr" -ForegroundColor DarkGray
            }
        }
        if ($Err.Context.Count -gt 8) {
            Write-Host "    │  ... $($Err.Context.Count - 8) more lines" -ForegroundColor DarkGray
        }
    }

    foreach ($note in $Err.Notes) {
        Write-Host "    │  note: $note" -ForegroundColor Cyan
    }
    foreach ($help in $Err.Helps) {
        Write-Host "    │  help: $help" -ForegroundColor Green
    }

    # Explain known error codes
    $explanation = Get-ErrorExplanation $Err.Code
    if ($explanation) {
        Write-Host "    │" -ForegroundColor DarkGray
        Write-Host "    │  $explanation" -ForegroundColor DarkYellow
    }

    Write-Host "    └─" -ForegroundColor DarkGray
}

function Format-TraceTestFailure {
    param([TraceTestFailure]$Fail, [int]$Index)

    Write-Host ""
    Write-Host "    ┌─ Test Failure #$Index" -ForegroundColor Red
    Write-Host "    │  $($Fail.TestName)" -ForegroundColor White

    if ($Fail.PanicLocation) {
        Write-Host "    │  at $($Fail.PanicLocation)" -ForegroundColor DarkGray
    }
    if ($Fail.PanicMessage) {
        Write-Host "    │  $($Fail.PanicMessage)" -ForegroundColor Yellow
    }

    if ($Fail.Backtrace.Count -gt 0) {
        Write-Host "    │" -ForegroundColor DarkGray
        Write-Host "    │  backtrace:" -ForegroundColor DarkGray
        foreach ($frame in ($Fail.Backtrace | Select-Object -First 8)) {
            $color = if ($frame -match "ignis::|smoke_test::") { "Cyan" } else { "DarkGray" }
            Write-Host "    │    $frame" -ForegroundColor $color
        }
    }

    Write-Host "    └─" -ForegroundColor DarkGray
}

function Format-TraceWarningGroup {
    <#
    .SYNOPSIS
    Format warnings grouped by lint/category, collapsed to avoid noise.
    #>
    param([TraceWarning[]]$Warnings)

    if ($Warnings.Count -eq 0) { return }

    # Group by lint
    $groups = @{}
    foreach ($w in $Warnings) {
        $key = if ($w.LintGroup) { $w.LintGroup } elseif ($w.Code) { $w.Code } else { "other" }
        if (-not $groups.ContainsKey($key)) { $groups[$key] = @() }
        $groups[$key] += $w
    }

    Write-Host ""
    Write-Host "    Warnings ($($Warnings.Count) total, $($groups.Count) categories):" -ForegroundColor Yellow
    Write-Host ""

    $sorted = $groups.GetEnumerator() | Sort-Object { $_.Value.Count } -Descending

    foreach ($g in $sorted) {
        $count = $g.Value.Count
        $name = $g.Key
        $first = $g.Value[0]

        $countStr = if ($count -gt 1) { " (x$count)" } else { "" }

        Write-Host -NoNewline "      " -ForegroundColor Yellow
        Write-Host -NoNewline "$name$countStr" -ForegroundColor DarkYellow
        Write-Host -NoNewline " - " -ForegroundColor DarkGray
        Write-Host "$($first.Message)" -ForegroundColor Gray

        # Show unique locations (collapsed)
        $uniqueFiles = $g.Value | Where-Object { $_.File } | ForEach-Object { "$($_.File):$($_.Line)" } | Select-Object -Unique
        if ($uniqueFiles.Count -gt 0) {
            $show = $uniqueFiles | Select-Object -First 3
            $more = if ($uniqueFiles.Count -gt 3) { " +$($uniqueFiles.Count - 3) more" } else { "" }
            Write-Host "        at $($show -join ", ")$more" -ForegroundColor DarkGray
        }
    }
}

function Format-HotFiles {
    param([hashtable]$HotFiles)

    if ($HotFiles.Count -eq 0) { return }

    $sorted = $HotFiles.GetEnumerator() | Sort-Object Value -Descending | Select-Object -First 10

    Write-Host ""
    Write-Host "    Hot Files (most errors):" -ForegroundColor White
    Write-Host ""

    $maxCount = ($sorted | Select-Object -First 1).Value
    foreach ($f in $sorted) {
        $bar = "█" * [math]::Min(30, [math]::Max(1, [math]::Round($f.Value / $maxCount * 30)))
        $color = if ($f.Value -ge 5) { "Red" } elseif ($f.Value -ge 3) { "Yellow" } else { "DarkGray" }
        Write-Host "      $($f.Value.ToString().PadLeft(3)) $bar $($f.Key)" -ForegroundColor $color
    }
}

function Format-RootCauses {
    param([string[]]$Causes)

    if ($Causes.Count -eq 0) { return }

    Write-Host ""
    Write-Host "    ╔══════════════════════════════════════════════════════╗" -ForegroundColor Yellow
    Write-Host "    ║  ROOT CAUSE ANALYSIS                                ║" -ForegroundColor Yellow
    Write-Host "    ╚══════════════════════════════════════════════════════╝" -ForegroundColor Yellow
    Write-Host ""

    for ($i = 0; $i -lt $Causes.Count; $i++) {
        Write-Host "    $($i + 1). " -NoNewline -ForegroundColor White
        Write-Host $Causes[$i] -ForegroundColor Yellow
        Write-Host ""
    }
}

function Format-SessionTimeline {
    <#
    .SYNOPSIS
    Show a visual timeline of commands in the session.
    #>
    param([PSCustomObject[]]$Entries)

    if ($Entries.Count -eq 0) { return }

    Write-Host ""
    Write-Host "    Session Timeline:" -ForegroundColor White
    Write-Host ""

    $baseTime = $null

    foreach ($e in $Entries) {
        if (-not $baseTime) {
            try { $baseTime = [datetime]::Parse($e.Timestamp) } catch { $baseTime = Get-Date }
        }

        try {
            $ts = [datetime]::Parse($e.Timestamp)
            $offset = ($ts - $baseTime).TotalSeconds
            $offsetStr = "T+$([math]::Round($offset, 1))s"
        } catch {
            $offsetStr = "T+?"
        }

        $errCount = if ($e.ErrorCount) { $e.ErrorCount } else { 0 }
        $warnCount = if ($e.WarnCount) { $e.WarnCount } else { 0 }
        $exitCode = if ($null -ne $e.ExitCode) { $e.ExitCode } else { 0 }

        $marker = if ($exitCode -ne 0 -or $errCount -gt 0) { "✗" }
                  elseif ($warnCount -gt 0) { "!" }
                  else { "✓" }

        $color = if ($exitCode -ne 0 -or $errCount -gt 0) { "Red" }
                 elseif ($warnCount -gt 0) { "Yellow" }
                 else { "Green" }

        $durStr = if ($e.DurationMs) { Format-Duration $e.DurationMs } else { "?" }

        $cmdDisplay = "$($e.Command) $($e.Args)".Trim()
        if ($cmdDisplay.Length -gt 35) { $cmdDisplay = $cmdDisplay.Substring(0, 32) + "..." }

        Write-Host -NoNewline "      $($offsetStr.PadRight(10))" -ForegroundColor DarkGray
        Write-Host -NoNewline " $marker " -ForegroundColor $color
        Write-Host -NoNewline "$($cmdDisplay.PadRight(36))" -ForegroundColor White
        Write-Host -NoNewline " $durStr" -ForegroundColor DarkGray

        if ($errCount -gt 0) {
            Write-Host -NoNewline "  ${errCount}err" -ForegroundColor Red
        }
        if ($warnCount -gt 0) {
            Write-Host -NoNewline "  ${warnCount}warn" -ForegroundColor Yellow
        }
        Write-Host ""
    }
}

function Format-SessionDiff {
    <#
    .SYNOPSIS
    Compare two trace sessions and show what changed.
    #>
    param(
        [PSCustomObject[]]$Old,
        [PSCustomObject[]]$New
    )

    $oldErrors = ($Old | Measure-Object -Property ErrorCount -Sum).Sum
    $newErrors = ($New | Measure-Object -Property ErrorCount -Sum).Sum
    $oldWarns = ($Old | Measure-Object -Property WarnCount -Sum).Sum
    $newWarns = ($New | Measure-Object -Property WarnCount -Sum).Sum

    if ($null -eq $oldErrors) { $oldErrors = 0 }
    if ($null -eq $newErrors) { $newErrors = 0 }
    if ($null -eq $oldWarns) { $oldWarns = 0 }
    if ($null -eq $newWarns) { $newWarns = 0 }

    Write-Host ""
    Write-Host "    Session Comparison:" -ForegroundColor White
    Write-Host ""

    $errDelta = $newErrors - $oldErrors
    $warnDelta = $newWarns - $oldWarns

    $errColor = if ($errDelta -gt 0) { "Red" } elseif ($errDelta -lt 0) { "Green" } else { "DarkGray" }
    $warnColor = if ($warnDelta -gt 0) { "Yellow" } elseif ($warnDelta -lt 0) { "Green" } else { "DarkGray" }

    $errSign = if ($errDelta -gt 0) { "+" } elseif ($errDelta -lt 0) { "" } else { "=" }
    $warnSign = if ($warnDelta -gt 0) { "+" } elseif ($warnDelta -lt 0) { "" } else { "=" }

    Write-Host "      Errors:   $oldErrors -> $newErrors ($errSign$errDelta)" -ForegroundColor $errColor
    Write-Host "      Warnings: $oldWarns -> $newWarns ($warnSign$warnDelta)" -ForegroundColor $warnColor
}

# ── Error Code Encyclopedia ──────────────────────────────────────────────────

function Get-ErrorExplanation {
    <#
    .SYNOPSIS
    Returns a short human-readable hint for common rustc error codes.
    #>
    param([string]$Code)

    switch ($Code) {
        "E0277" { return "A trait bound is not satisfied. Check that the type implements the required trait, or add a where clause." }
        "E0308" { return "Type mismatch. The expected and found types differ. Check return types, variable assignments, and function arguments." }
        "E0382" { return "Use of moved value. The value was consumed by a previous operation. Clone it, or restructure to avoid the move." }
        "E0412" { return "Unresolved type name. Check imports (use statements) and module visibility (pub)." }
        "E0425" { return "Unresolved name. The identifier is not in scope. Check imports and spelling." }
        "E0433" { return "Unresolved path. The module path doesn't exist. Check mod declarations and feature gates." }
        "E0499" { return "Multiple mutable borrows. Rust prevents aliased mutation. Split the borrows into separate scopes." }
        "E0502" { return "Conflicting borrow: immutable borrow exists while trying to mutably borrow. Reorder operations or clone." }
        "E0505" { return "Moved value still borrowed. The borrow must end before the value can be moved. Limit borrow scope." }
        "E0507" { return "Cannot move out of borrowed content. Use clone(), or restructure to work with references." }
        "E0599" { return "Method not found. The type doesn't have this method. Check trait imports (use Trait;) and type spelling." }
        "E0615" { return "Attempted field access on non-struct. Check that the expression has the expected type." }
        "E0658" { return "Unstable feature used. Either use nightly, or find a stable alternative." }
        default { return $null }
    }
}

# ── Full Trace Report ────────────────────────────────────────────────────────

function Format-FullTraceReport {
    <#
    .SYNOPSIS
    Generate a complete diagnostic report for a trace entry or session.
    Combines parsed errors, warnings, test failures, hot file analysis,
    and root cause deduction into a single rich display.
    #>
    param(
        [string]$Command,
        [string]$Args,
        [string[]]$RawOutput,
        [double]$DurationMs,
        [int]$ExitCode
    )

    $parsed = Parse-CargoOutput $RawOutput
    $testFails = Parse-TestOutput $RawOutput

    Write-Host ""
    Write-Host "  ╔════════════════════════════════════════════════════════════╗" -ForegroundColor $(if ($ExitCode -ne 0) { "Red" } else { "Green" })
    Write-Host "  ║  TRACE REPORT: $Command $Args" -ForegroundColor White
    Write-Host "  ╚════════════════════════════════════════════════════════════╝" -ForegroundColor $(if ($ExitCode -ne 0) { "Red" } else { "Green" })

    # Summary bar
    $errCount = $parsed.Errors.Count
    $warnCount = $parsed.Warnings.Count
    $testFailCount = $testFails.Count

    Write-Host ""
    Write-Host -NoNewline "    exit=$ExitCode" -ForegroundColor $(if ($ExitCode -ne 0) { "Red" } else { "Green" })
    Write-Host -NoNewline "  $errCount error(s)" -ForegroundColor $(if ($errCount -gt 0) { "Red" } else { "Green" })
    Write-Host -NoNewline "  $warnCount warning(s)" -ForegroundColor $(if ($warnCount -gt 0) { "Yellow" } else { "Green" })
    Write-Host -NoNewline "  $testFailCount test failure(s)" -ForegroundColor $(if ($testFailCount -gt 0) { "Red" } else { "Green" })
    Write-Host "  $(Format-Duration $DurationMs)" -ForegroundColor DarkGray

    # Errors
    if ($errCount -gt 0) {
        Write-Host ""
        Write-Host "    ── Errors ──────────────────────────────────────────" -ForegroundColor Red

        for ($i = 0; $i -lt [math]::Min($errCount, 15); $i++) {
            Format-TraceError $parsed.Errors[$i] ($i + 1)
        }
        if ($errCount -gt 15) {
            Write-Host ""
            Write-Host "    ... $($errCount - 15) more errors (use 'trace N' for full output)" -ForegroundColor DarkGray
        }
    }

    # Test failures
    if ($testFailCount -gt 0) {
        Write-Host ""
        Write-Host "    ── Test Failures ───────────────────────────────────" -ForegroundColor Red

        for ($i = 0; $i -lt $testFailCount; $i++) {
            Format-TraceTestFailure $testFails[$i] ($i + 1)
        }
    }

    # Warnings (collapsed)
    if ($warnCount -gt 0) {
        Write-Host ""
        Write-Host "    ── Warnings ────────────────────────────────────────" -ForegroundColor Yellow
        Format-TraceWarningGroup $parsed.Warnings
    }

    # Hot files
    $hotFiles = @{}
    foreach ($err in $parsed.Errors) {
        if ($err.File) {
            if (-not $hotFiles.ContainsKey($err.File)) { $hotFiles[$err.File] = 0 }
            $hotFiles[$err.File]++
        }
    }
    if ($hotFiles.Count -gt 1) {
        Format-HotFiles $hotFiles
    }

    # Root cause analysis
    $session = [TraceSession]::new()
    $session.AllErrors = $parsed.Errors
    $session.AllWarnings = $parsed.Warnings
    $session.AllTestFailures = $testFails
    $session = Analyze-TraceSession $session

    if ($session.RootCauses.Count -gt 0) {
        Format-RootCauses $session.RootCauses
    }

    # Quick fix suggestions
    if ($errCount -gt 0 -or $testFailCount -gt 0) {
        Write-Host ""
        Write-Host "    ── Next Steps ──────────────────────────────────────" -ForegroundColor Cyan
        Write-Host ""

        if ($parsed.Errors | Where-Object { $_.Category -eq "compile" }) {
            Write-Host "      1. Fix the first compile error (later errors often cascade)" -ForegroundColor Gray
        }
        if ($testFailCount -gt 0) {
            $first = $testFails[0]
            Write-Host "      1. Check $($first.PanicLocation)" -ForegroundColor Gray
        }
        Write-Host "      2. Run 'build full' to re-check" -ForegroundColor Gray
        Write-Host "      3. Run 'trace last' after fixing to compare" -ForegroundColor Gray
        Write-Host ""
    }
}

# ── Integration Hook ─────────────────────────────────────────────────────────

function Invoke-TraceAnalysis {
    <#
    .SYNOPSIS
    Called by shell.ps1 after a failed command to provide immediate analysis.
    Only activates when errors are detected.
    #>
    param(
        [string]$Command,
        [string]$Args,
        [string[]]$Output,
        [int]$ExitCode,
        [double]$DurationMs
    )

    if ($ExitCode -eq 0) { return }

    # Quick error count
    $errorLines = @($Output | Where-Object { "$_" -match "^error" })

    if ($errorLines.Count -eq 0 -and $ExitCode -ne 0) {
        # Might be a panic or other non-rustc failure
        $panicLines = @($Output | Where-Object { "$_" -match "panicked|FATAL" })
        if ($panicLines.Count -eq 0) { return }
    }

    Write-Host ""
    Write-Host "  ─── trace analysis ───────────────────────────────────" -ForegroundColor DarkCyan

    $parsed = Parse-CargoOutput $Output
    $testFails = Parse-TestOutput $Output

    $errCount = $parsed.Errors.Count
    $testFailCount = $testFails.Count

    # Show first error compactly
    if ($errCount -gt 0) {
        $first = $parsed.Errors[0]
        $codeStr = if ($first.Code) { "[$($first.Code)] " } else { "" }
        Write-Host "    first error: $codeStr$($first.Message)" -ForegroundColor Red
        if ($first.File) {
            Write-Host "      at $($first.File):$($first.Line)" -ForegroundColor DarkGray
        }
        $explanation = Get-ErrorExplanation $first.Code
        if ($explanation) {
            Write-Host "      hint: $explanation" -ForegroundColor DarkYellow
        }
        if ($errCount -gt 1) {
            Write-Host "      ... +$($errCount - 1) more. run 'trace last' for full report" -ForegroundColor DarkGray
        }
    }

    # Show first test failure compactly
    if ($testFailCount -gt 0) {
        $first = $testFails[0]
        Write-Host "    first test failure: $($first.TestName)" -ForegroundColor Red
        if ($first.PanicMessage) {
            Write-Host "      $($first.PanicMessage)" -ForegroundColor Yellow
        }
    }

    Write-Host "  ──────────────────────────────────────────────────────" -ForegroundColor DarkCyan
}