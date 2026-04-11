# lint.ps1

Write-Host "cargo check --features full" -ForegroundColor Cyan
cargo check --features full
if ($LASTEXITCODE -ne 0) { exit 1 }

Write-Host ""
Write-Host "cargo clippy --features full" -ForegroundColor Cyan
cargo clippy --features full -- -W clippy::all -W clippy::pedantic `
    -A clippy::module_name_repetitions `
    -A clippy::too_many_arguments `
    -A clippy::missing_errors_doc `
    -A clippy::must_use_candidate `
    -A clippy::return_self_not_must_use `
    -A clippy::cast_possible_truncation `
    -A clippy::cast_sign_loss `
    -A clippy::cast_precision_loss `
    -A clippy::missing_panics_doc
if ($LASTEXITCODE -ne 0) { exit 1 }

Write-Host ""
Write-Host "cargo doc --features full --no-deps" -ForegroundColor Cyan
cargo doc --features full --no-deps 2>&1 | ForEach-Object {
    if ($_ -match "warning|error") {
        Write-Host "  $_" -ForegroundColor Yellow
    }
}

Write-Host ""
Write-Host "cargo test --features full --lib" -ForegroundColor Cyan
cargo test --features full --lib