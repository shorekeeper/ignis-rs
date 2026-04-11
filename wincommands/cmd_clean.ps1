# cmd_clean.ps1 [all|target|traces]
param([string]$What = "target")


Write-CmdHeader "clean" "[$What]"

switch ($What) {
    "target" {
        Write-Host -NoNewline "    cargo clean ... "
        cargo clean 2>&1 | Out-Null
        Write-Host "OK" -ForegroundColor Green
    }
    "traces" {
        $traceDir = Join-Path $PSScriptRoot "..\.ignis_trace"
        if (Test-Path $traceDir) {
            $count = (Get-ChildItem $traceDir -File).Count
            Remove-Item "$traceDir\*" -Force
            Write-Host "    Removed $count trace file(s)" -ForegroundColor Green
        } else {
            Write-Host "    No trace directory" -ForegroundColor DarkGray
        }
    }
    "all" {
        Write-Host -NoNewline "    cargo clean ... "
        cargo clean 2>&1 | Out-Null
        Write-Host "OK" -ForegroundColor Green

        $traceDir = Join-Path $PSScriptRoot "..\.ignis_trace"
        if (Test-Path $traceDir) {
            Remove-Item "$traceDir\*" -Force
            Write-Host "    Traces cleared" -ForegroundColor Green
        }
    }
    default { Write-Host "    Unknown: $What (use target, traces, all)" -ForegroundColor Red }
}
