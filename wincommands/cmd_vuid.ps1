#Requires -Version 7.0
#
# cmd_vuid.ps1 - Offline VUID knowledge browser.
#
# SYNOPSIS
#   vuid <suffix | full-VUID>     show one entry in full
#   vuid list [category]          tabular index, optionally filtered
#   vuid search <term...>         substring search across all entry fields
#   vuid categories               category names with entry counts
#
# DESCRIPTION
#   Serves the ignis crate's static VUID knowledge base (parsed from
#   src/debug/vuid_kb.rs by _vuid_kb.ps1) directly in the terminal. The full
#   entry view mirrors the crate's own diagnostic sectioning: title, category
#   and spec reference, then What Happened, Why Rejected, and Ignis Fix. All
#   multi-line field content, including code-style indented fix snippets, is
#   printed verbatim with color per section.
#
#   Lookup tokens are forgiving: "01213", "VUID-VkImageMemoryBarrier-
#   oldLayout-01213", and any exact custom suffix all resolve. On a miss the
#   command falls back to a fuzzy scan over suffix plus title and lists the
#   closest candidates rather than failing dry.
#
# DATA SOURCE
#   Static base only. Entries registered at runtime through the crate's
#   register_runtime_entry exist inside a Rust process and are not visible
#   here. Off-repository (no src/debug/vuid_kb.rs) the command reports the
#   missing source and exits cleanly.

param([Parameter(ValueFromRemainingArguments)][string[]]$RawArgs)

Get-ChildItem (Join-Path $PSScriptRoot "_*.ps1") | ForEach-Object { . $_.FullName }

$kb = Get-VuidKnowledgeBase
if ($kb.Count -eq 0) {
    Write-CmdHeader "vuid" "knowledge base"
    Write-Host "    src\debug\vuid_kb.rs not found or contains no entries" -ForegroundColor Red
    return
}

$action = if ($RawArgs -and $RawArgs.Count -ge 1) { $RawArgs[0] } else { 'list' }

function Show-VuidEntry {
    param($E)
    Write-Host ""
    Write-Host "    VUID suffix $($E.Suffix)" -ForegroundColor Cyan -NoNewline
    Write-Host "  [$($E.Category)]" -ForegroundColor DarkGray
    Write-Host "    $($E.Title)" -ForegroundColor White
    Write-Host "    spec: $($E.SpecSection)" -ForegroundColor DarkGray
    Write-Host ""
    Write-Host "    What happened" -ForegroundColor Yellow
    foreach ($ln in ($E.WhatHappened -split "`n")) {
        Write-Host "      $ln" -ForegroundColor Gray
    }
    Write-Host ""
    Write-Host "    Why Vulkan rejected it" -ForegroundColor Yellow
    foreach ($ln in ($E.WhyRejected -split "`n")) {
        Write-Host "      $ln" -ForegroundColor Gray
    }
    Write-Host ""
    Write-Host "    Ignis fix" -ForegroundColor Green
    foreach ($ln in ($E.IgnisFix -split "`n")) {
        Write-Host "      $ln" -ForegroundColor Gray
    }
    Write-Host ""
}

switch ($action) {
    'list' {
        $filter = if ($RawArgs.Count -ge 2) { $RawArgs[1] } else { '' }
        $items = $kb
        if ($filter) {
            $items = @($kb | Where-Object { $_.Category -like "*$filter*" })
        }
        Write-CmdHeader "vuid" "list ($($items.Count) of $($kb.Count) entries)"
        Write-Host "    Suffix   Category                 Title" -ForegroundColor White
        Write-Host "    $("-" * 72)" -ForegroundColor DarkGray
        foreach ($e in ($items | Sort-Object Suffix)) {
            $title = $e.Title
            if ($title.Length -gt 46) { $title = $title.Substring(0, 45) + '~' }
            Write-Host ("    {0,-8} {1,-24} " -f $e.Suffix, $e.Category) -NoNewline -ForegroundColor Gray
            Write-Host $title -ForegroundColor DarkGray
        }
        Write-Host ""
        Write-Host "    use 'vuid <suffix>' for a full entry" -ForegroundColor DarkGray
    }

    'categories' {
        Write-CmdHeader "vuid" "categories"
        $groups = $kb | Group-Object Category | Sort-Object Count -Descending
        foreach ($g in $groups) {
            Write-Host ("    {0,3}  {1}" -f $g.Count, $g.Name) -ForegroundColor Gray
        }
    }

    'search' {
        $terms = if ($RawArgs.Count -ge 2) { $RawArgs[1..($RawArgs.Count - 1)] } else { @() }
        if ($terms.Count -eq 0) {
            Write-Host "    usage: vuid search <term...>" -ForegroundColor Yellow
            return
        }
        $needle = ($terms -join ' ')
        Write-CmdHeader "vuid" "search '$needle'"
        $hits = @()
        foreach ($e in $kb) {
            $hay = "$($e.Suffix) $($e.Title) $($e.Category) $($e.WhatHappened) $($e.WhyRejected) $($e.IgnisFix)"
            if ($hay -like "*$needle*") { $hits += $e }
        }
        if ($hits.Count -eq 0) {
            Write-Host "    no matches" -ForegroundColor Yellow
            return
        }
        foreach ($e in $hits) {
            Write-Host ("    {0,-8} " -f $e.Suffix) -NoNewline -ForegroundColor Cyan
            Write-Host $e.Title -ForegroundColor Gray
        }
        Write-Host ""
        Write-Host "    $($hits.Count) match(es); use 'vuid <suffix>' for detail" -ForegroundColor DarkGray
    }

    default {
        $entry = Find-VuidEntry -Token $action
        if ($entry) {
            Write-CmdHeader "vuid" $action
            Show-VuidEntry $entry
            return
        }
        # Fuzzy fallback over suffix and title.
        Write-CmdHeader "vuid" "'$action' not found, closest matches"
        $scored = foreach ($e in $kb) {
            $m = Get-FuzzyMatch -Pattern $action -Text ($e.Suffix + ' ' + $e.Title)
            if ($null -ne $m) { [pscustomobject]@{ E = $e; S = $m.Score } }
        }
        $best = @($scored | Sort-Object S -Descending | Select-Object -First 8)
        if ($best.Count -eq 0) {
            Write-Host "    nothing similar; try 'vuid list'" -ForegroundColor Yellow
            return
        }
        foreach ($b in $best) {
            Write-Host ("    {0,-8} " -f $b.E.Suffix) -NoNewline -ForegroundColor Cyan
            Write-Host $b.E.Title -ForegroundColor Gray
        }
    }
}