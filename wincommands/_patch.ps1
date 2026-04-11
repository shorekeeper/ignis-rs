# _patch3.ps1 - Nuclear fix: rewrites _trace_engine.ps1 line by line.
# Removes class blocks, replaces ::new() calls, strips type annotations.
# Run once, then delete.

$file = Join-Path $PSScriptRoot "_trace_engine.ps1"

if (-not (Test-Path $file)) {
    Write-Host "  _trace_engine.ps1 not found" -ForegroundColor Red
    exit 1
}

$lines = Get-Content $file
$output = [System.Collections.ArrayList]::new()

$inClass = $false
$braceDepth = 0
$classesRemoved = 0
$replacements = 0

for ($i = 0; $i -lt $lines.Count; $i++) {
    $line = $lines[$i]

    # Detect class block start
    if (-not $inClass -and $line -match '^\s*class\s+(TraceError|TraceWarning|TraceTestFailure|TraceSession)\s*\{?\s*$') {
        $inClass = $true
        $braceDepth = 0
        # Count braces on this line
        foreach ($ch in $line.ToCharArray()) {
            if ($ch -eq '{') { $braceDepth++ }
            if ($ch -eq '}') { $braceDepth-- }
        }
        $null = $output.Add("# [class $($Matches[1]) removed by _patch3.ps1]")
        $classesRemoved++
        # If the opening brace wasn't on this line, it might be next
        if ($braceDepth -le 0 -and $line -notmatch '\{') {
            # Class might start brace on next line
            $braceDepth = 0
        }
        continue
    }

    # Inside a class block: skip lines until braces balance
    if ($inClass) {
        foreach ($ch in $line.ToCharArray()) {
            if ($ch -eq '{') { $braceDepth++ }
            if ($ch -eq '}') { $braceDepth-- }
        }
        if ($braceDepth -le 0) {
            $inClass = $false
        }
        continue
    }

    # Replace ::new() calls
    if ($line -match '\[TraceError\]::new\(\)') {
        $line = $line -replace '\[TraceError\]::new\(\)', 'New-TraceError'
        $replacements++
    }
    if ($line -match '\[TraceWarning\]::new\(\)') {
        $line = $line -replace '\[TraceWarning\]::new\(\)', 'New-TraceWarning'
        $replacements++
    }
    if ($line -match '\[TraceTestFailure\]::new\(\)') {
        $line = $line -replace '\[TraceTestFailure\]::new\(\)', 'New-TraceTestFailure'
        $replacements++
    }
    if ($line -match '\[TraceSession\]::new\(\)') {
        $line = $line -replace '\[TraceSession\]::new\(\)', 'New-TraceSession'
        $replacements++
    }

    # Strip type annotations from param lines
    $line = $line -replace '\[TraceError\]\$', '$'
    $line = $line -replace '\[TraceWarning\[\]\]\$', '$'
    $line = $line -replace '\[TraceWarning\]\$', '$'
    $line = $line -replace '\[TraceTestFailure\]\$', '$'
    $line = $line -replace '\[TraceSession\]\$', '$'
    $line = $line -replace '\[string\[\]\]\$Causes', '$Causes'

    # Strip return type casts
    $line = $line -replace '\[TraceError\[\]\](\$\w+\.ToArray\(\))', '$1'
    $line = $line -replace '\[TraceWarning\[\]\](\$\w+\.ToArray\(\))', '$1'
    $line = $line -replace '\[TraceTestFailure\[\]\](\$\w+\.ToArray\(\))', '$1'
    $line = $line -replace '\[string\[\]\](\$\w+\.ToArray\(\))', '$1'

    $null = $output.Add($line)
}

# Check if constructors already exist
$joined = $output -join "`n"
if ($joined -notmatch 'function New-TraceError') {
    # Find insertion point: before "# ── Cargo/Rustc Output Parser"
    $insertIdx = -1
    for ($i = 0; $i -lt $output.Count; $i++) {
        if ($output[$i] -match 'Cargo/Rustc Output Parser') {
            $insertIdx = $i
            break
        }
    }

    $constructorLines = @(
        ""
        "# ── Structured types (PSCustomObject constructors) ────────────────────────────"
        ""
        "function New-TraceError {"
        '    return [PSCustomObject]@{'
        '        Code     = ""; Level = "error"; Message = ""'
        '        File     = ""; Line = 0; Column = 0'
        '        Context  = @(); Notes = @(); Helps = @()'
        '        RawBlock = ""; Category = "compile"'
        '    }'
        "}"
        ""
        "function New-TraceWarning {"
        '    return [PSCustomObject]@{'
        '        Code = ""; Message = ""; File = ""; Line = 0'
        '        LintGroup = ""; Category = "compile"'
        '    }'
        "}"
        ""
        "function New-TraceTestFailure {"
        '    return [PSCustomObject]@{'
        '        TestName = ""; PanicMessage = ""'
        '        PanicLocation = ""; Backtrace = @()'
        '    }'
        "}"
        ""
        "function New-TraceSession {"
        '    return [PSCustomObject]@{'
        '        SessionId = ""; StartTime = (Get-Date)'
        '        Entries = @(); AllErrors = @(); AllWarnings = @()'
        '        AllTestFailures = @(); HotFiles = @{}'
        '        ErrorCodes = @{}; RootCauses = @()'
        '    }'
        "}"
        ""
    )

    if ($insertIdx -gt 0) {
        $output.InsertRange($insertIdx, $constructorLines)
        Write-Host "  inserted constructor functions at line $insertIdx" -ForegroundColor Green
    } else {
        foreach ($cl in $constructorLines) { $null = $output.Add($cl) }
        Write-Host "  appended constructor functions" -ForegroundColor Yellow
    }
} else {
    Write-Host "  constructors already present" -ForegroundColor DarkGray
}

Set-Content $file -Value ($output.ToArray()) -Encoding UTF8

Write-Host "  removed $classesRemoved class blocks" -ForegroundColor Green
Write-Host "  made $replacements ::new() replacements" -ForegroundColor Green

# ── Also patch cmd_trace.ps1 ─────────────────────────────────────────────────

$traceCmd = Join-Path $PSScriptRoot "cmd_trace.ps1"
if (Test-Path $traceCmd) {
    $tc = Get-Content $traceCmd -Raw
    $changed = $false

    @(
        @('\[TraceSession\]::new\(\)', 'New-TraceSession'),
        @('\[TraceError\]::new\(\)',   'New-TraceError'),
        @('\[TraceWarning\]::new\(\)', 'New-TraceWarning'),
        @('\[TraceTestFailure\]::new\(\)', 'New-TraceTestFailure')
    ) | ForEach-Object {
        if ($tc -match $_[0]) {
            $tc = $tc -replace $_[0], $_[1]
            $changed = $true
            Write-Host "  cmd_trace.ps1: replaced $($_[0])" -ForegroundColor Green
        }
    }

    if ($changed) {
        Set-Content $traceCmd -Value $tc -Encoding UTF8
    } else {
        Write-Host "  cmd_trace.ps1 already clean" -ForegroundColor DarkGray
    }
}

Write-Host ""
Write-Host "  Done. Delete this file: Remove-Item $($MyInvocation.MyCommand.Path)" -ForegroundColor Cyan