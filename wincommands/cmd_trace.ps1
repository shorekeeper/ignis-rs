# cmd_trace.ps1 [last|list|errors|N]
param([Parameter(ValueFromRemainingArguments)][string[]]$Args)


# Load trace engine
$enginePath = Join-Path $PSScriptRoot "_trace_engine.ps1"
if (Test-Path $enginePath) { . $enginePath }

# Load common helpers
$commonPath = Join-Path $PSScriptRoot "_common.ps1"
if (Test-Path $commonPath) { . $commonPath }

$action = if ($Args.Count -gt 0) { $Args[0] } else { "last" }

$traceDir = Join-Path $PSScriptRoot "..\.ignis_trace"

function Get-Sessions {
    if (-not (Test-Path $traceDir)) { return @() }
    Get-ChildItem $traceDir -Filter "*.json" | Sort-Object Name -Descending
}

function Get-CurrentSession {
    $sessions = Get-Sessions
    if ($sessions.Count -eq 0) { return @() }
    $content = Get-Content $sessions[0].FullName -Raw -ErrorAction SilentlyContinue
    if (-not $content) { return @() }
    $entries = $content | ConvertFrom-Json
    if ($entries -isnot [array]) { $entries = @($entries) }
    return $entries
}

Write-CmdHeader "trace" "[$action]"

switch ($action) {
    "last" {
        $entries = Get-CurrentSession
        if ($entries.Count -eq 0) {
            Write-Host "    No history in current session" -ForegroundColor Yellow
            return
        }
        $entry = $entries[-1]

        $statusColor = if ($entry.ExitCode -ne 0 -or $entry.ErrorCount -gt 0) { "Red" } else { "Green" }

        Write-Host "    Command:  $($entry.Command) $($entry.Args)" -ForegroundColor White
        Write-Host "    Time:     $($entry.Timestamp)" -ForegroundColor DarkGray
        Write-Host "    Duration: $(Format-Duration $entry.DurationMs)" -ForegroundColor DarkGray
        Write-Host "    Exit:     $($entry.ExitCode)" -ForegroundColor $statusColor
        Write-Host "    Errors:   $($entry.ErrorCount)" -ForegroundColor $(if ($entry.ErrorCount -gt 0) { "Red" } else { "Green" })
        Write-Host "    Warnings: $($entry.WarnCount)" -ForegroundColor $(if ($entry.WarnCount -gt 0) { "Yellow" } else { "Green" })

        if ($entry.Errors -and $entry.Errors.Count -gt 0) {
            Write-Host ""
            Write-Host "    Errors:" -ForegroundColor Red
            foreach ($err in $entry.Errors) {
                Write-Host "      $err" -ForegroundColor DarkRed
            }
        }

        if ($entry.Warnings -and $entry.Warnings.Count -gt 0) {
            Write-Host ""
            Write-Host "    Warnings:" -ForegroundColor Yellow
            $entry.Warnings | Select-Object -First 10 | ForEach-Object {
                Write-Host "      $_" -ForegroundColor DarkYellow
            }
        }

        if ($entry.OutputTail -and $entry.OutputTail.Count -gt 0) {
            Write-Host ""
            Write-Host "    Last output:" -ForegroundColor DarkGray
            $entry.OutputTail | Select-Object -Last 15 | ForEach-Object {
                Write-Host "      $_" -ForegroundColor Gray
            }
        }
    }

    "list" {
        $entries = Get-CurrentSession
        if ($entries.Count -eq 0) {
            Write-Host "    No history" -ForegroundColor Yellow
            return
        }

        Write-Host "    #   Time                 Cmd              Exit  Err  Warn  Duration" -ForegroundColor White
        Write-Host "    $("-" * 75)" -ForegroundColor DarkGray

        foreach ($e in $entries) {
            $statusColor = if ($e.ExitCode -ne 0 -or $e.ErrorCount -gt 0) { "Red" }
                          elseif ($e.WarnCount -gt 0) { "Yellow" }
                          else { "Green" }

            $line = "    {0,-3} {1,-20} {2,-16} {3,-5} {4,-4} {5,-5} {6}" -f `
                $e.Index, $e.Timestamp, "$($e.Command) $($e.Args)".Substring(0, [math]::Min(15, "$($e.Command) $($e.Args)".Length)),
                $e.ExitCode, $e.ErrorCount, $e.WarnCount, (Format-Duration $e.DurationMs)

            Write-Host $line -ForegroundColor $statusColor
        }
    }

    "errors" {
        $entries = Get-CurrentSession
        $errorEntries = @($entries | Where-Object { $_.ErrorCount -gt 0 -or $_.ExitCode -ne 0 })

        if ($errorEntries.Count -eq 0) {
            Write-Host "    No errors in session" -ForegroundColor Green
            return
        }

        Write-Host "    $($errorEntries.Count) command(s) with errors:" -ForegroundColor Red
        Write-Host ""

        foreach ($e in $errorEntries) {
            Write-Host "    #$($e.Index) [$($e.Command) $($e.Args)] exit=$($e.ExitCode)" -ForegroundColor Red
            if ($e.Errors) {
                $e.Errors | Select-Object -First 3 | ForEach-Object {
                    Write-Host "      $_" -ForegroundColor DarkRed
                }
            }
            Write-Host ""
        }
    }

    default {
        # Try as index
        if ($action -match "^\d+$") {
            $entries = Get-CurrentSession
            $idx = [int]$action
            $entry = $entries | Where-Object { $_.Index -eq $idx }
            if ($entry) {
                Write-Host "    Command:  $($entry.Command) $($entry.Args)" -ForegroundColor White
                Write-Host "    Exit:     $($entry.ExitCode)" -ForegroundColor $(if ($entry.ExitCode -ne 0) { "Red" } else { "Green" })
                Write-Host "    Duration: $(Format-Duration $entry.DurationMs)" -ForegroundColor DarkGray

                if ($entry.Errors -and $entry.Errors.Count -gt 0) {
                    Write-Host ""
                    Write-Host "    Errors:" -ForegroundColor Red
                    $entry.Errors | ForEach-Object { Write-Host "      $_" -ForegroundColor DarkRed }
                }

                if ($entry.OutputTail) {
                    Write-Host ""
                    Write-Host "    Output:" -ForegroundColor DarkGray
                    $entry.OutputTail | ForEach-Object { Write-Host "      $_" -ForegroundColor Gray }
                }
            } else {
                Write-Host "    No entry #$idx found" -ForegroundColor Yellow
            }
        } else {
            Write-Host "    Unknown action: $action" -ForegroundColor Red
            Write-Host "    Use: last, list, errors, or a number" -ForegroundColor DarkGray
        }
    }
}
