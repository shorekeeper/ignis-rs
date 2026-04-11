#Requires -Version 5.1
# shell.ps1 - Ignis interactive command shell.
#
# Usage:
#   .\shell.ps1          # Interactive REPL
#   .\shell.ps1 build    # Single command mode
#   .\shell.ps1 test smoke --step 22

param(
    [Parameter(ValueFromRemainingArguments)]
    [string[]]$DirectArgs
)

$ErrorActionPreference = "Continue"
$global:IgnisShellVersion = "1.0"
$global:TraceDir = ".ignis_trace"
$global:SessionId = (Get-Date -Format "yyyyMMdd_HHmmss")
$global:History = @()
$global:LastResult = $null
$global:CommandCount = 0
$global:ErrorCount = 0

# ── Bootstrap ────────────────────────────────────────────────────────────────

if (-not (Test-Path $global:TraceDir)) {
    New-Item -ItemType Directory -Path $global:TraceDir -Force | Out-Null
}

$commandDir = Join-Path $PSScriptRoot "wincommands"
$commonPath = Join-Path $commandDir "_common.ps1"
if (Test-Path $commonPath) { . $commonPath }
$traceEnginePath = Join-Path $commandDir "_trace_engine.ps1"
if (Test-Path $traceEnginePath) { . $traceEnginePath }

# ── Aliases ──────────────────────────────────────────────────────────────────

$global:Aliases = @{
    "b"     = "build"
    "t"     = "test"
    "c"     = "check"
    "l"     = "lint"
    "r"     = "run"
    "s"     = "status"
    "i"     = "info"
    "h"     = "help"
    "q"     = "exit"
    "tr"    = "trace"
    "cl"    = "clean"
    "p"     = "prof"
    "!!"    = "repeat"
}

# ── Helpers ──────────────────────────────────────────────────────────────────

function Write-Prompt {
    $errTag = if ($global:ErrorCount -gt 0) { " $($global:ErrorCount)err" } else { "" }
    $baseColor = if ($global:ErrorCount -gt 0) { "Red" } else { "Cyan" }

    # Git info
    $gitBranch = $null
    $gitDirty = $false
    $gitStaged = 0
    $gitUnstaged = 0
    $gitUntracked = 0
    $gitAhead = 0
    $gitBehind = 0
    $gitCommit = $null

    $hasGit = $false
    try {
        $gitDir = git rev-parse --git-dir 2>$null
        if ($LASTEXITCODE -eq 0 -and $gitDir) { $hasGit = $true }
    } catch {}

    if ($hasGit) {
        # Branch name
        $gitBranch = (git branch --show-current 2>$null)
        if (-not $gitBranch) {
            # Detached HEAD
            $gitBranch = (git rev-parse --short HEAD 2>$null)
            if ($gitBranch) { $gitBranch = ":$gitBranch" }
        }

        # Short commit hash
        $gitCommit = git rev-parse --short HEAD 2>$null

        # Status counts
        $statusLines = @(git status --porcelain 2>$null)
        foreach ($sl in $statusLines) {
            if ($sl.Length -lt 2) { continue }
            $idx = $sl[0]
            $wt = $sl[1]

            if ($idx -ne ' ' -and $idx -ne '?') { $gitStaged++ }
            if ($wt -ne ' ' -and $wt -ne '?') { $gitUnstaged++ }
            if ($idx -eq '?' -and $wt -eq '?') { $gitUntracked++ }
        }

        $gitDirty = ($gitStaged + $gitUnstaged + $gitUntracked) -gt 0

        # Ahead/behind
        try {
            $ab = git rev-list --left-right --count "@{u}...HEAD" 2>$null
            if ($LASTEXITCODE -eq 0 -and $ab -match "(\d+)\s+(\d+)") {
                $gitBehind = [int]$Matches[1]
                $gitAhead = [int]$Matches[2]
            }
        } catch {}
    }

    # Build prompt
    Write-Host -NoNewline "ignis" -ForegroundColor $baseColor

    if ($hasGit -and $gitBranch) {
        # Branch
        $branchColor = if ($gitDirty) { "Yellow" } else { "Green" }
        Write-Host -NoNewline " " -ForegroundColor DarkGray
        Write-Host -NoNewline "$gitBranch" -ForegroundColor $branchColor

        # Commit
        if ($gitCommit) {
            Write-Host -NoNewline " $gitCommit" -ForegroundColor DarkGray
        }

        # Status indicators
        $indicators = ""
        if ($gitStaged -gt 0)    { $indicators += " +$gitStaged" }
        if ($gitUnstaged -gt 0)  { $indicators += " ~$gitUnstaged" }
        if ($gitUntracked -gt 0) { $indicators += " ?$gitUntracked" }

        if ($indicators) {
            Write-Host -NoNewline $indicators -ForegroundColor Yellow
        }

        # Ahead/behind
        if ($gitAhead -gt 0)  { Write-Host -NoNewline " ↑$gitAhead" -ForegroundColor Green }
        if ($gitBehind -gt 0) { Write-Host -NoNewline " ↓$gitBehind" -ForegroundColor Red }

        # Clean indicator
        if (-not $gitDirty -and $gitAhead -eq 0 -and $gitBehind -eq 0) {
            Write-Host -NoNewline " ✓" -ForegroundColor Green
        }
    }

    if ($errTag) {
        Write-Host -NoNewline $errTag -ForegroundColor Red
    }

    Write-Host -NoNewline "> " -ForegroundColor DarkGray
}

function Resolve-Alias {
    param([string]$Name)
    if ($global:Aliases.ContainsKey($Name)) {
        return $global:Aliases[$Name]
    }
    return $Name
}

function Parse-Input {
    param([string]$Line)
    $Line = $Line.Trim()
    if (-not $Line) { return $null }

    # Split respecting quoted strings
    $tokens = @()
    $current = ""
    $inQuote = $false
    foreach ($ch in $Line.ToCharArray()) {
        if ($ch -eq '"') { $inQuote = !$inQuote; continue }
        if ($ch -eq ' ' -and -not $inQuote -and $current) {
            $tokens += $current
            $current = ""
            continue
        }
        $current += $ch
    }
    if ($current) { $tokens += $current }

    if ($tokens.Count -eq 0) { return $null }

    $cmd = Resolve-Alias $tokens[0].ToLower()
    $args = if ($tokens.Count -gt 1) { $tokens[1..($tokens.Count - 1)] } else { @() }

    return @{ Command = $cmd; Args = $args; Raw = $Line }
}

function Save-Trace {
    param(
        [string]$Command,
        [string[]]$Args,
        [int]$ExitCode,
        [double]$DurationMs,
        [string[]]$Output,
        [string[]]$Errors,
        [string[]]$Warnings
    )

    $entry = [PSCustomObject]@{
        Index     = $global:CommandCount
        Timestamp = (Get-Date -Format "yyyy-MM-dd HH:mm:ss.fff")
        Command   = $Command
        Args      = ($Args -join " ")
        ExitCode  = $ExitCode
        DurationMs = [math]::Round($DurationMs)
        ErrorCount = $Errors.Count
        WarnCount  = $Warnings.Count
        Errors    = $Errors
        Warnings  = $Warnings
        OutputTail = ($Output | Select-Object -Last 50)
    }

    $global:History += $entry

    $tracePath = Join-Path $global:TraceDir "$global:SessionId.json"
    $allEntries = if (Test-Path $tracePath) {
        (Get-Content $tracePath -Raw | ConvertFrom-Json)
    } else { @() }

    # ConvertFrom-Json returns a single object if only 1 entry, force array
    if ($allEntries -isnot [array]) { $allEntries = @($allEntries) }
    $allEntries += $entry

    $allEntries | ConvertTo-Json -Depth 5 | Set-Content $tracePath -Encoding UTF8
}

function Invoke-Command-Script {
    param([string]$Command, [string[]]$CmdArgs)

    $scriptPath = Join-Path $commandDir "cmd_$Command.ps1"

    if (-not (Test-Path $scriptPath)) {
        Write-Host "  unknown command: $Command" -ForegroundColor Red
        Write-Host "  type 'help' for available commands" -ForegroundColor DarkGray
        return
    }

    $global:CommandCount++

    $sw = [System.Diagnostics.Stopwatch]::StartNew()

    $allOutput = @()
    $allErrors = @()
    $allWarnings = @()
    $returnValue = $null

    try {
        # Run the command, capture all output
        $rawResult = & $scriptPath @CmdArgs 2>&1

        # Separate the return object from display output.
        # Command scripts return [PSCustomObject] with Passed/Failed/Skipped.
        # Everything else is display text (strings, error records, etc).
        foreach ($item in $rawResult) {
            if ($item -is [PSCustomObject] -and $null -ne $item.PSObject.Properties['Passed']) {
                # This is the result object, not display output
                $returnValue = $item
                continue
            }

            $str = "$item"
            $allOutput += $str

            if ($str -match "^error|FAIL|panicked") { $allErrors += $str }
            elseif ($str -match "^warning|WARN") { $allWarnings += $str }
        }
    } catch {
        $allErrors += $_.Exception.Message
        Write-Host "  command crashed: $($_.Exception.Message)" -ForegroundColor Red
    }

    $sw.Stop()
    $exitCode = $LASTEXITCODE
    if ($null -eq $exitCode) { $exitCode = 0 }

    # Display captured output (the scripts already formatted it)
    foreach ($line in $allOutput) {
        Write-Host $line
    }

    # If the script returned a result object, use it for status
    if ($returnValue) {
        $failed = [int]$returnValue.Failed
        $passed = [int]$returnValue.Passed
        $skipped = [int]$returnValue.Skipped

        if ($failed -gt 0) {
            $exitCode = 1
            $allErrors += "($failed sub-test(s) failed)"
        }
    }

    # Post-command status
    $elapsed = $sw.Elapsed.TotalMilliseconds
    $elapsedStr = if ($elapsed -lt 1000) { "$([math]::Round($elapsed))ms" }
                  elseif ($elapsed -lt 60000) { "$([math]::Round($elapsed / 1000, 1))s" }
                  else { "$([math]::Floor($elapsed / 60000))m $([math]::Round(($elapsed % 60000) / 1000, 1))s" }

    if ($allErrors.Count -gt 0) {
        $global:ErrorCount += $allErrors.Count
        Write-Host ""
        Write-Host "  $($allErrors.Count) error(s) | $elapsedStr | use 'trace last' for details" -ForegroundColor Red
    } elseif ($allWarnings.Count -gt 0) {
        Write-Host ""
        Write-Host "  $($allWarnings.Count) warning(s) | $elapsedStr" -ForegroundColor Yellow
    } else {
        Write-Host ""
        Write-Host "  ok | $elapsedStr" -ForegroundColor Green
    }

    # Save trace
    Save-Trace -Command $Command -Args $CmdArgs -ExitCode $exitCode `
        -DurationMs $elapsed -Output $allOutput -Errors $allErrors -Warnings $allWarnings

    # Auto-analysis on failure
    if ($exitCode -ne 0 -and (Get-Command Invoke-TraceAnalysis -ErrorAction SilentlyContinue)) {
        Invoke-TraceAnalysis `
            -Command $Command `
            -Args ($CmdArgs -join " ") `
            -Output $allOutput `
            -ExitCode $exitCode `
            -DurationMs $elapsed
    }

    $global:LastResult = @{
        Command = $Command; Args = $CmdArgs; ExitCode = $exitCode
        Errors = $allErrors; Warnings = $allWarnings; Result = $returnValue
    }
}

function Show-Banner {
    $ver = $global:IgnisShellVersion
    $title = "IGNIS COMMAND SHELL v$ver"
    $hint  = "type 'help' for commands, 'q' to quit"
    $inner = 42

    $tPad = $inner - $title.Length
    $tL = [math]::Floor($tPad / 2)
    $tR = $tPad - $tL

    $hPad = $inner - $hint.Length
    $hL = [math]::Floor($hPad / 2)
    $hR = $hPad - $hL

    Write-Host ""
    Write-Host "  ╔$("═" * $inner)╗" -ForegroundColor DarkCyan
    Write-Host "  ║$(" " * $tL)$title$(" " * $tR)║" -ForegroundColor Cyan
    Write-Host "  ║$(" " * $hL)$hint$(" " * $hR)║" -ForegroundColor DarkGray
    Write-Host "  ╚$("═" * $inner)╝" -ForegroundColor DarkCyan
    Write-Host ""
}

# ── Main ─────────────────────────────────────────────────────────────────────

# Single command mode
if ($DirectArgs -and $DirectArgs.Count -gt 0) {
    $parsed = Parse-Input ($DirectArgs -join " ")
    if ($parsed) {
        Invoke-Command-Script $parsed.Command $parsed.Args
    }
    exit $LASTEXITCODE
}

# Interactive REPL
Show-Banner

$lastLine = ""

while ($true) {
    Write-Prompt
    $line = Read-Host

    if (-not $line) { continue }

    $parsed = Parse-Input $line

    if (-not $parsed) { continue }

    switch ($parsed.Command) {
        "exit" {
            Write-Host "  session: $($global:CommandCount) commands, $($global:ErrorCount) errors" -ForegroundColor DarkGray
            Write-Host ""
            exit 0
        }
        "repeat" {
            if ($lastLine) {
                Write-Host "  repeating: $lastLine" -ForegroundColor DarkGray
                $parsed = Parse-Input $lastLine
                if ($parsed) {
                    Invoke-Command-Script $parsed.Command $parsed.Args
                }
            } else {
                Write-Host "  nothing to repeat" -ForegroundColor Yellow
            }
            continue
        }
        default {
            $lastLine = $line
            Invoke-Command-Script $parsed.Command $parsed.Args
        }
    }

    Write-Host ""
}