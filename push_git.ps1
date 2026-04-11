#Requires -Version 5.1
# push_git.ps1 - Format, lint, and push to main in one shot.

$ErrorActionPreference = "Stop"

Write-Host ""
Write-Host "  IGNIS PUSH" -ForegroundColor Magenta
Write-Host ""

# ── Commit message ───────────────────────────────────────────────────────────

$msg = Read-Host "  Commit message"
if ([string]::IsNullOrWhiteSpace($msg)) {
    Write-Host "  Aborted: empty commit message" -ForegroundColor Red
    exit 1
}

# ── Check working tree ───────────────────────────────────────────────────────

$status = git status --porcelain 2>&1
if (-not $status) {
    Write-Host "  Nothing to commit (working tree clean)" -ForegroundColor Yellow
    exit 0
}

$fileCount = ($status | Measure-Object).Count
Write-Host "  $fileCount file(s) changed" -ForegroundColor DarkGray

# ── Format ───────────────────────────────────────────────────────────────────

Write-Host -NoNewline "  cargo fmt ... "
cargo fmt --all 2>&1 | Out-Null
if ($LASTEXITCODE -ne 0) {
    Write-Host "FAILED" -ForegroundColor Red
    exit 1
}
Write-Host "OK" -ForegroundColor Green

# ── Clippy ───────────────────────────────────────────────────────────────────

Write-Host -NoNewline "  cargo clippy --features full ... "
$clippyOut = cargo clippy --all-targets --features full -- `
    -W clippy::all -W clippy::pedantic `
    -A clippy::module_name_repetitions `
    -A clippy::too_many_arguments `
    -A clippy::missing_errors_doc `
    -A clippy::must_use_candidate `
    -A clippy::return_self_not_must_use `
    -A clippy::cast_possible_truncation `
    -A clippy::cast_sign_loss `
    -A clippy::cast_precision_loss `
    -A clippy::missing_panics_doc 2>&1

if ($LASTEXITCODE -ne 0) {
    Write-Host "FAILED" -ForegroundColor Red
    Write-Host ""
    $clippyOut | Where-Object { $_ -match "error|warning" } | Select-Object -First 30 | ForEach-Object {
        Write-Host "    $_" -ForegroundColor DarkRed
    }
    Write-Host ""
    Write-Host "  Fix clippy errors before pushing" -ForegroundColor Red
    exit 1
}

$warnings = @($clippyOut | Where-Object { $_ -match "^warning" }).Count
if ($warnings -gt 0) {
    Write-Host "OK ($warnings warning(s))" -ForegroundColor Yellow
} else {
    Write-Host "OK" -ForegroundColor Green
}

# ── Check ────────────────────────────────────────────────────────────────────

Write-Host -NoNewline "  cargo check --features full ... "
cargo check --features full 2>&1 | Out-Null
if ($LASTEXITCODE -ne 0) {
    Write-Host "FAILED" -ForegroundColor Red
    exit 1
}
Write-Host "OK" -ForegroundColor Green

# ── Stage + Commit + Push ────────────────────────────────────────────────────

Write-Host ""
Write-Host -NoNewline "  git add -A ... "
git add -A 2>&1 | Out-Null
Write-Host "OK" -ForegroundColor Green

Write-Host -NoNewline "  git commit ... "
git commit -m $msg 2>&1 | Out-Null
if ($LASTEXITCODE -ne 0) {
    Write-Host "FAILED" -ForegroundColor Red
    exit 1
}
Write-Host "OK" -ForegroundColor Green

Write-Host -NoNewline "  git push origin main ... "
$pushOut = git push origin main 2>&1
if ($LASTEXITCODE -ne 0) {
    Write-Host "FAILED" -ForegroundColor Red
    $pushOut | ForEach-Object { Write-Host "    $_" -ForegroundColor DarkRed }
    Write-Host ""
    Write-Host "  Commit is local. Fix and push manually." -ForegroundColor Yellow
    exit 1
}
Write-Host "OK" -ForegroundColor Green

# ── Done ─────────────────────────────────────────────────────────────────────

Write-Host ""
Write-Host "  Pushed to main: $msg" -ForegroundColor Green
Write-Host ""