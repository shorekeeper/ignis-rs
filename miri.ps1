# miri.ps1
# Runs unit tests under Miri to detect undefined behavior.
# Requires: rustup +nightly component add miri

Write-Host "Running unit tests under Miri..." -ForegroundColor Cyan
Write-Host "(Miri cannot run Vulkan calls, only pure-Rust logic)" -ForegroundColor DarkGray

cargo +nightly miri test --features full --lib 2>&1 | ForEach-Object {
    if ($_ -match "Undefined Behavior|error") {
        Write-Host "  $_" -ForegroundColor Red
    } else {
        Write-Host "  $_"
    }
}