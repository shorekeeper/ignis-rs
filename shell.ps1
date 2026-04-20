#Requires -Version 7.0
# shell.ps1 - Ignis interactive command shell.
#
# Usage:
#   .\shell.ps1              # Interactive REPL
#   .\shell.ps1 build full   # Single command mode
#   .\shell.ps1 test smoke --step 22

param(
    [Parameter(ValueFromRemainingArguments)]
    [string[]]$DirectArgs
)

$ErrorActionPreference = "Continue"
$global:CiStartTime = Get-Date
$global:CommandCount = 0
$global:ErrorCount = 0
$global:LastResult = $null

# Load command modules 

$commandDir = Join-Path $PSScriptRoot "wincommands"
Get-ChildItem (Join-Path $commandDir "_*.ps1") -ErrorAction SilentlyContinue |
    ForEach-Object { . $_.FullName }

# Trace directory

$global:TraceDir = Join-Path $PSScriptRoot ".ignis_trace"
$global:SessionId = Get-Date -Format "yyyyMMdd_HHmmss"
if (-not (Test-Path $global:TraceDir)) {
    New-Item -ItemType Directory -Path $global:TraceDir -Force | Out-Null
}

# Aliases

$global:Aliases = @{
    "b"  = "build"
    "t"  = "test"
    "c"  = "check"
    "l"  = "lint"
    "r"  = "run"
    "s"  = "status"
    "i"  = "info"
    "h"  = "help"
    "q"  = "exit"
    "tr" = "trace"
    "cl" = "clean"
    "p"  = "prof"
    "!!" = "repeat"
    "ul" = "unlock"
}

# Helpers

function Resolve-CmdAlias {
    param([string]$Name)
    $clean = $Name.Trim().Trim([char]0)
    if ($global:Aliases.ContainsKey($clean)) { return $global:Aliases[$clean] }
    return $clean
}

function Parse-Input {
    param([string]$Line)
    $Line = $Line.Trim().Trim([char]0)
    if (-not $Line) { return $null }

    $tokens = @()
    $current = ""
    $inQuote = $false
    foreach ($ch in $Line.ToCharArray()) {
        if ($ch -eq [char]0) { continue }
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

    $cmd = Resolve-CmdAlias $tokens[0].ToLower()
    $cmdArgs = if ($tokens.Count -gt 1) { $tokens[1..($tokens.Count - 1)] } else { @() }

    return @{ Command = $cmd; Args = $cmdArgs; Raw = $Line }
}

function Write-Prompt {
    $errTag = ($global:ErrorCount -gt 0) ? " $($global:ErrorCount)err" : ""
    $baseColor = ($global:ErrorCount -gt 0) ? "Red" : "Cyan"

    # Git info
    $gitInfo = ""
    $hasGit = $false
    try {
        $gitDir = git rev-parse --git-dir 2>$null
        if ($LASTEXITCODE -eq 0 -and $gitDir) { $hasGit = $true }
    } catch {}

    if ($hasGit) {
        $branch = git branch --show-current 2>$null
        if (-not $branch) {
            $branch = git rev-parse --short HEAD 2>$null
            if ($branch) { $branch = ":$branch" }
        }

        $commit = git rev-parse --short HEAD 2>$null

        $staged = 0; $unstaged = 0; $untracked = 0
        $statusLines = @(git status --porcelain 2>$null)
        foreach ($sl in $statusLines) {
            if ($sl.Length -lt 2) { continue }
            $idx = $sl[0]; $wt = $sl[1]
            if ($idx -ne ' ' -and $idx -ne '?') { $staged++ }
            if ($wt -ne ' ' -and $wt -ne '?') { $unstaged++ }
            if ($idx -eq '?' -and $wt -eq '?') { $untracked++ }
        }
        $dirty = ($staged + $unstaged + $untracked) -gt 0

        $ahead = 0; $behind = 0
        try {
            $ab = git rev-list --left-right --count "@{u}...HEAD" 2>$null
            if ($LASTEXITCODE -eq 0 -and $ab -match "(\d+)\s+(\d+)") {
                $behind = [int]$Matches[1]; $ahead = [int]$Matches[2]
            }
        } catch {}

        $branchColor = $dirty ? "Yellow" : "Green"
        Write-Host -NoNewline "ignis" -ForegroundColor $baseColor
        Write-Host -NoNewline " " -ForegroundColor DarkGray
        Write-Host -NoNewline $branch -ForegroundColor $branchColor

        if ($commit) {
            Write-Host -NoNewline " $commit" -ForegroundColor DarkGray
        }

        if ($staged -gt 0)    { Write-Host -NoNewline " +$staged" -ForegroundColor Yellow }
        if ($unstaged -gt 0)  { Write-Host -NoNewline " ~$unstaged" -ForegroundColor Yellow }
        if ($untracked -gt 0) { Write-Host -NoNewline " ?$untracked" -ForegroundColor Yellow }
        if ($ahead -gt 0)     { Write-Host -NoNewline " ↑$ahead" -ForegroundColor Green }
        if ($behind -gt 0)    { Write-Host -NoNewline " ↓$behind" -ForegroundColor Red }

        if (-not $dirty -and $ahead -eq 0 -and $behind -eq 0) {
            Write-Host -NoNewline " ✓" -ForegroundColor Green
        }

        if ($errTag) { Write-Host -NoNewline $errTag -ForegroundColor Red }
        Write-Host -NoNewline "> " -ForegroundColor DarkGray
    } else {
        Write-Host -NoNewline "ignis" -ForegroundColor $baseColor
        if ($errTag) { Write-Host -NoNewline $errTag -ForegroundColor Red }
        Write-Host -NoNewline "> " -ForegroundColor DarkGray
    }
}

function Save-Trace {
    param(
        [string]$Command,
        [string[]]$CmdArgs,
        [int]$ExitCode,
        [double]$DurationMs,
        [string[]]$Output,
        [string[]]$Errors,
        [string[]]$Warnings
    )

    $entry = [PSCustomObject]@{
        Index      = $global:CommandCount
        Timestamp  = Get-Date -Format "yyyy-MM-dd HH:mm:ss.fff"
        Command    = $Command
        Args       = ($CmdArgs -join " ")
        ExitCode   = $ExitCode
        DurationMs = [math]::Round($DurationMs)
        ErrorCount = $Errors.Count
        WarnCount  = $Warnings.Count
        Errors     = $Errors
        Warnings   = $Warnings
        OutputTail = ($Output | Select-Object -Last 50)
    }

    $tracePath = Join-Path $global:TraceDir "$global:SessionId.json"
    $allEntries = if (Test-Path $tracePath) {
        try {
            $raw = Get-Content $tracePath -Raw | ConvertFrom-Json
            if ($raw -isnot [array]) { @($raw) } else { $raw }
        } catch { @() }
    } else { @() }

    $allEntries = @($allEntries) + @($entry)
    $allEntries | ConvertTo-Json -Depth 5 | Set-Content $tracePath -Encoding UTF8
}

function Invoke-Command-Script {
    param([string]$Command, [string[]]$CmdArgs)

    $Command = $Command.Trim().Trim([char]0)
    if (-not $Command) { return }

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
        $rawResult = & $scriptPath @CmdArgs 2>&1

        foreach ($item in $rawResult) {
            if ($item -is [PSCustomObject] -and $null -ne $item.PSObject.Properties['Passed']) {
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

    foreach ($line in $allOutput) {
        Write-Host $line
    }

    if ($returnValue) {
        $failed = [int]$returnValue.Failed
        if ($failed -gt 0) {
            $exitCode = 1
            $allErrors += "($failed sub-test(s) failed)"
        }
    }

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

    Save-Trace -Command $Command -CmdArgs $CmdArgs -ExitCode $exitCode `
        -DurationMs $elapsed -Output $allOutput -Errors $allErrors -Warnings $allWarnings

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
    $ver = "1.0"
    $title = "IGNIS COMMAND SHELL v$ver"
    $hint  = "type 'help' for commands, 'q' to quit"
    $inner = 44

    $tPad = $inner - $title.Length
    $tL = [math]::Floor($tPad / 2)
    $tR = $tPad - $tL

    $hPad = $inner - $hint.Length
    $hL = [math]::Floor($hPad / 2)
    $hR = $hPad - $hL

    Write-Host ""
    Write-Host "  $([char]0x2554)$("$([char]0x2550)" * $inner)$([char]0x2557)" -ForegroundColor DarkCyan
    Write-Host "  $([char]0x2551)$(" " * $tL)$title$(" " * $tR)$([char]0x2551)" -ForegroundColor Cyan
    Write-Host "  $([char]0x2551)$(" " * $hL)$hint$(" " * $hR)$([char]0x2551)" -ForegroundColor DarkGray
    Write-Host "  $([char]0x255A)$("$([char]0x2550)" * $inner)$([char]0x255D)" -ForegroundColor DarkCyan
    Write-Host ""
}

# Cleanup on exit

$null = Register-EngineEvent PowerShell.Exiting -Action {
    if ($script:ActiveCargoProcess -and -not $script:ActiveCargoProcess.HasExited) {
        Kill-CargoTree $script:ActiveCargoProcess
    }
}

# Main

# Single command mode
if ($DirectArgs -and $DirectArgs.Count -gt 0) {
    $parsed = Parse-Input ($DirectArgs -join " ")
    if ($parsed) {
        try {
            Invoke-Command-Script $parsed.Command $parsed.Args
        } catch {
            if ($script:ActiveCargoProcess -and -not $script:ActiveCargoProcess.HasExited) {
                Kill-CargoTree $script:ActiveCargoProcess
            }
        }
    }
    exit $LASTEXITCODE
}

# Interactive REPL
Show-Banner

$lastLine = ""

while ($true) {
    Write-Prompt

    $line = $null
    try {
        $line = Read-Host
    } catch {
        Write-Host ""
        continue
    }

    if (-not $line) { continue }
    $line = $line.Trim()
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
                    try {
                        Invoke-Command-Script $parsed.Command $parsed.Args
                    } catch [System.Management.Automation.PipelineStoppedException] {
                        if ($script:ActiveCargoProcess -and -not $script:ActiveCargoProcess.HasExited) {
                            Kill-CargoTree $script:ActiveCargoProcess
                            $script:ActiveCargoProcess = $null
                        }
                        Write-Host "`n  interrupted" -ForegroundColor Yellow
                    }
                }
            } else {
                Write-Host "  nothing to repeat" -ForegroundColor Yellow
            }
            continue
        }
        default {
            $lastLine = $line
            try {
                Invoke-Command-Script $parsed.Command $parsed.Args
            } catch [System.Management.Automation.PipelineStoppedException] {
                if ($script:ActiveCargoProcess -and -not $script:ActiveCargoProcess.HasExited) {
                    Kill-CargoTree $script:ActiveCargoProcess
                    $script:ActiveCargoProcess = $null
                }
                Show-Cursor
                Write-Host "`n  interrupted" -ForegroundColor Yellow
            } catch {
                Write-Host "`n  error: $($_.Exception.Message)" -ForegroundColor Red
            }
        }
    }

    Write-Host ""
}