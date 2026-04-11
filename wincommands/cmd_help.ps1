# cmd_help.ps1 [command]
param([string]$Topic = "")
Get-ChildItem (Join-Path $PSScriptRoot "_*.ps1") | ForEach-Object { . $_.FullName }

$commands = @(
    @{ Cmd = "build";  Alias = "b";  Desc = "Compile the project";
       Usage = "build [full|release|minimal] [--features X,Y] [--release]" },
    @{ Cmd = "check";  Alias = "c";  Desc = "Type-check without codegen";
       Usage = "check [full|minimal] [--features X,Y]" },
    @{ Cmd = "test";   Alias = "t";  Desc = "Run test suites";
       Usage = "test [all|unit|smoke|features|lint|audit|doc|size|miri] [--step N] [--filter X]" },
    @{ Cmd = "lint";   Alias = "l";  Desc = "Clippy, fmt, doc warnings";
       Usage = "lint [clippy|fmt|doc|all] [--fix]" },
    @{ Cmd = "run";    Alias = "r";  Desc = "Run an example";
       Usage = "run [example_name] [--features X] [--release]" },
    @{ Cmd = "trace";  Alias = "tr"; Desc = "Inspect failures and session history";
       Usage = "trace [last|list|errors|timeline|diff|report|N]" },
    @{ Cmd = "info";   Alias = "i";  Desc = "System, Vulkan, and project info";
       Usage = "info [system|vulkan|project|deps|all]" },
    @{ Cmd = "status"; Alias = "s";  Desc = "Git and build status";
       Usage = "status" },
    @{ Cmd = "clean";  Alias = "cl"; Desc = "Clean build artifacts";
       Usage = "clean [all|target|traces]" },
    @{ Cmd = "prof";   Alias = "p";  Desc = "Build/test timing profiler";
       Usage = "prof [build|test] [--features X]" },
    @{ Cmd = "unlock"; Alias = "ul"; Desc = "Kill stuck cargo/rustc and release locks";
       Usage = "unlock" },
    @{ Cmd = "help";   Alias = "h";  Desc = "This help";
       Usage = "help [command]" }
)

if ($Topic) {
    $found = $commands | Where-Object { $_.Cmd -eq $Topic -or $_.Alias -eq $Topic }
    if ($found) {
        Write-Host ""
        Write-Host "  $($found.Cmd)" -ForegroundColor Cyan -NoNewline
        Write-Host " ($($found.Alias))" -ForegroundColor DarkGray -NoNewline
        Write-Host " - $($found.Desc)" -ForegroundColor White
        Write-Host ""
        Write-Host "  usage: $($found.Usage)" -ForegroundColor Gray
        Write-Host ""
    } else {
        Write-Host "  unknown command: $Topic" -ForegroundColor Red
    }
    return
}

Write-Host ""
Write-Host "  Commands:" -ForegroundColor White
Write-Host ""

foreach ($c in $commands) {
    $alias = "($($c.Alias))".PadRight(5)
    Write-Host "    " -NoNewline
    Write-Host "$($c.Cmd.PadRight(8))" -NoNewline -ForegroundColor Cyan
    Write-Host " $alias " -NoNewline -ForegroundColor DarkGray
    Write-Host "$($c.Desc)" -ForegroundColor Gray
}

Write-Host ""
Write-Host "  Shortcuts:" -ForegroundColor White
Write-Host "    !!       repeat last command" -ForegroundColor Gray
Write-Host "    q        quit" -ForegroundColor Gray
Write-Host ""
Write-Host "  Examples:" -ForegroundColor White
Write-Host "    build full              compile with all features" -ForegroundColor DarkGray
Write-Host "    test smoke --step 22    run smoke test, highlight step 22" -ForegroundColor DarkGray
Write-Host "    lint clippy --fix       auto-fix clippy warnings" -ForegroundColor DarkGray
Write-Host "    trace last              inspect last failure" -ForegroundColor DarkGray
Write-Host ""
