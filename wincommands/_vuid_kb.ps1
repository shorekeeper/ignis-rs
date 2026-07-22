#Requires -Version 7.0
#
# _vuid_kb.ps1 - Parser and cache for the ignis crate's VUID knowledge base.
#
# PURPOSE
#   Extracts the static KnowledgeEntry slice from src/debug/vuid_kb.rs into
#   PowerShell objects so the shell can serve as an offline VUID reference.
#   Runtime-registered entries (register_runtime_entry) live only inside a
#   running Rust process and are out of scope by definition; this parser
#   covers the static base, which is the durable, versioned content.
#
# PARSING STRATEGY
#   The entries are Rust struct literals with a fixed field order matching
#   the KnowledgeEntry declaration: vuid_suffix, title, category,
#   what_happened, why_rejected, ignis_fix, spec_section. The parser scans
#   from the "static STATIC_BASE" marker onward, locates each
#   "KnowledgeEntry {" occurrence, and for every field seeks the field name
#   followed by a colon, then decodes the Rust string literal that follows.
#   The category field is the one non-string field and is captured by a
#   DiagnosticCategory::<Name> pattern instead.
#
#   The string decoder handles the escape forms actually present in the
#   source: \" and \\ and \n and \t, hex escapes of the form \x20, and the
#   Rust line-continuation (a backslash immediately before a physical line
#   break), which consumes the break and all leading whitespace of the next
#   line. Unknown escapes degrade to the escaped character verbatim rather
#   than failing the entry.
#
# CACHING
#   Results are cached in $global:IgnisVuidKb keyed by the source file's
#   LastWriteTimeUtc ticks. Command scripts dot-source helper files on every
#   invocation, so a script-scoped cache would be discarded each time; the
#   global survives for the REPL session and invalidates automatically when
#   the Rust file changes.
#
# PUBLIC SURFACE
#   Get-VuidKnowledgeBase  - full entry list (possibly empty off-repo).
#   Find-VuidEntry         - resolve one entry from a user-supplied token
#                            (bare suffix, full VUID identifier, or fuzzy).

function Convert-RustStringLiteral {
    <#
    .SYNOPSIS
    Decode a Rust string literal starting at a given index in a text blob.

    .DESCRIPTION
    $Start must be the index of the opening double quote. Returns a hashtable
    with Value (the decoded string) and End (the index one past the closing
    quote). On a truncated literal the scan stops at end of text and returns
    what was accumulated; callers treat the entry as best-effort.
    #>
    param([string]$Text, [int]$Start)
    $sb = [System.Text.StringBuilder]::new()
    $i = $Start + 1
    $n = $Text.Length
    while ($i -lt $n) {
        $c = $Text[$i]
        if ($c -eq '"') {
            return @{ Value = $sb.ToString(); End = $i + 1 }
        }
        if ($c -eq '\') {
            $i++
            if ($i -ge $n) { break }
            $e = $Text[$i]
            if ($e -eq 'n') { [void]$sb.Append("`n"); $i++ }
            elseif ($e -eq 't') { [void]$sb.Append("`t"); $i++ }
            elseif ($e -eq 'r') { $i++ }
            elseif ($e -eq '"') { [void]$sb.Append('"'); $i++ }
            elseif ($e -eq '\') { [void]$sb.Append('\'); $i++ }
            elseif ($e -eq 'x' -and ($i + 2) -lt $n) {
                $hex = $Text.Substring($i + 1, 2)
                try { [void]$sb.Append([char][Convert]::ToInt32($hex, 16)) } catch { }
                $i += 3
            }
            elseif ($e -eq "`r" -or $e -eq "`n") {
                # Rust line continuation: swallow the physical break and the
                # indentation of the continuation line.
                while ($i -lt $n -and ($Text[$i] -eq "`r" -or $Text[$i] -eq "`n")) { $i++ }
                while ($i -lt $n -and ($Text[$i] -eq ' ' -or $Text[$i] -eq "`t")) { $i++ }
            }
            else {
                [void]$sb.Append($e); $i++
            }
        }
        else {
            [void]$sb.Append($c)
            $i++
        }
    }
    return @{ Value = $sb.ToString(); End = $n }
}

function Get-VuidKnowledgeBase {
    <#
    .SYNOPSIS
    Parse (or return the cached parse of) the static VUID knowledge base.

    .DESCRIPTION
    Returns an array of PSCustomObject with the fields Suffix, Title,
    Category, WhatHappened, WhyRejected, IgnisFix, SpecSection. Returns an
    empty array when the source file cannot be located, which keeps every
    consumer (command, completion, palette) degradation-safe outside the
    repository.
    #>
    $kbFile = Join-Path $PSScriptRoot '..\src\debug\vuid_kb.rs'
    if (-not (Test-Path $kbFile)) { $kbFile = 'src\debug\vuid_kb.rs' }
    if (-not (Test-Path $kbFile)) { return @() }

    $ticks = (Get-Item $kbFile).LastWriteTimeUtc.Ticks
    if ($global:IgnisVuidKb -and $global:IgnisVuidKb.Ticks -eq $ticks) {
        return $global:IgnisVuidKb.Entries
    }

    $text = Get-Content $kbFile -Raw
    $anchor = $text.IndexOf('static STATIC_BASE')
    if ($anchor -lt 0) { $anchor = 0 }

    $entries = [System.Collections.Generic.List[object]]::new()
    $fields = @('vuid_suffix', 'title', 'category', 'what_happened',
                'why_rejected', 'ignis_fix', 'spec_section')
    $pos = $anchor

    while ($true) {
        $ei = $text.IndexOf('KnowledgeEntry {', $pos)
        if ($ei -lt 0) { break }
        $pos = $ei + 15
        $vals = @{}
        $ok = $true
        foreach ($f in $fields) {
            $fi = $text.IndexOf(($f + ':'), $pos)
            if ($fi -lt 0) { $ok = $false; break }
            $pos = $fi + $f.Length + 1
            if ($f -eq 'category') {
                $win = $text.Substring($pos, [math]::Min(120, $text.Length - $pos))
                $m = [regex]::Match($win, 'DiagnosticCategory::(\w+)')
                $vals[$f] = if ($m.Success) { $m.Groups[1].Value } else { 'Other' }
                if ($m.Success) { $pos += $m.Index + $m.Length }
            }
            else {
                $qi = $text.IndexOf('"', $pos)
                if ($qi -lt 0) { $ok = $false; break }
                $r = Convert-RustStringLiteral -Text $text -Start $qi
                $vals[$f] = $r.Value
                $pos = $r.End
            }
        }
        if ($ok) {
            $entries.Add([pscustomobject]@{
                Suffix       = $vals['vuid_suffix']
                Title        = $vals['title']
                Category     = $vals['category']
                WhatHappened = $vals['what_happened']
                WhyRejected  = $vals['why_rejected']
                IgnisFix     = $vals['ignis_fix']
                SpecSection  = $vals['spec_section']
            })
        }
    }

    $result = $entries.ToArray()
    $global:IgnisVuidKb = @{ Ticks = $ticks; Entries = $result }
    return $result
}

function Find-VuidEntry {
    <#
    .SYNOPSIS
    Resolve one knowledge base entry from a user-supplied token.

    .DESCRIPTION
    Accepts a bare numeric suffix ("01213"), a full VUID identifier (the
    trailing dash-separated segment is extracted), or an arbitrary token
    matched exactly against suffixes. Returns the entry or $null.
    #>
    param([string]$Token)
    $kb = Get-VuidKnowledgeBase
    if ($kb.Count -eq 0 -or -not $Token) { return $null }
    $needle = $Token
    if ($Token -match '^VUID-.*-([^-]+)$') { $needle = $Matches[1] }
    foreach ($e in $kb) {
        if ($e.Suffix -eq $needle) { return $e }
    }
    return $null
}