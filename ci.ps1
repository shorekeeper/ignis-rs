# ci.ps1 - Run everything.
Write-Host "=== IGNIS CI ===" -ForegroundColor Magenta
Write-Host ""

Write-Host "Phase 1: Feature matrix" -ForegroundColor Cyan
& .\test_features.ps1
Write-Host ""

Write-Host "Phase 2: Lint + doc + unit tests" -ForegroundColor Cyan
& .\lint.ps1
Write-Host ""

Write-Host "Phase 3: Cross-feature audit" -ForegroundColor Cyan
& .\audit_deps.ps1
Write-Host ""

Write-Host "Phase 4: Smoke test" -ForegroundColor Cyan
cargo run --example smoke_test --features full
Write-Host ""

Write-Host "=== CI COMPLETE ===" -ForegroundColor Magenta