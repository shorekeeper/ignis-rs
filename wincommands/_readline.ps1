#Requires -Version 7.0
#
# _readline.ps1 - Custom line editor, Tab completion, history, and fuzzy
#                 command palette for the ignis shell REPL.
#
# WHY A CUSTOM EDITOR
#   The REPL reads input with Read-Host, which offers no completion, no
#   in-session history, and no key bindings; PSReadLine does not attach to
#   Read-Host. Read-IgnisLine replaces it with a raw key loop over
#   [Console]::ReadKey, giving the shell context-aware Tab completion,
#   persisted history, inline command validity coloring, and a Ctrl+P fuzzy
#   palette, while degrading to Read-Host automatically when input is
#   redirected (non-interactive host).
#
# KEY BINDINGS (Read-IgnisLine)
#   Enter            accept the line (appended to persistent history)
#   Tab / Shift+Tab  cycle completions for the token at end of line
#   Up / Down        history navigation (the in-progress edit is preserved)
#   Left / Right / Home / End / Backspace / Delete   editing
#   Escape           clear the line
#   Ctrl+U           delete to start of line
#   Ctrl+K           delete to end of line
#   Ctrl+W           delete the previous word
#   Ctrl+P           open the fuzzy palette; a selection runs immediately
#   Ctrl+C           cancel the line (prints ^C, returns empty)
#
# COMPLETION MODEL
#   Completion applies to the final whitespace-delimited token and only when
#   the cursor sits at the end of the buffer (the overwhelmingly dominant
#   case; mid-line completion is intentionally not attempted). The first
#   token completes against command names discovered from cmd_*.ps1 files
#   plus the live alias table. Subsequent tokens complete per command from a
#   static argument table, with two dynamic sources: feature names parsed
#   from Cargo.toml [features] (including comma-separated continuation, so
#   "tracking,desc<Tab>" completes the tail) and example names parsed from
#   Cargo.toml [[example]] blocks. Cargo metadata is cached globally, keyed
#   by Cargo.toml mtime.
#
# PALETTE
#   Invoke-IgnisPalette renders a full-screen finder in the alternate screen
#   buffer: type to filter, Up/Down to select, Enter to run, Escape to
#   cancel. Candidates are command names with descriptions, one "run <name>"
#   entry per cargo example, a set of curated quick actions, and recent
#   unique history. Matching is subsequence-based with bonuses for
#   consecutive runs and word starts, and matched characters are highlighted
#   in the list. The alternate buffer guarantees the prompt screen is
#   restored pixel-exact on exit.
#
# RENDERING NOTES
#   The editor records the cursor position where input began and redraws by
#   repositioning there, rewriting the colored buffer, and clearing to end
#   of screen. Wrapped lines are handled through linear index arithmetic
#   over the window width; if the write would exceed the buffer bottom the
#   recorded origin is shifted up to account for scrolling. The first token
#   is drawn green when it resolves to a known command or alias and red
#   otherwise, giving pre-execution validity feedback.
#
# STATE AND PERSISTENCE
#   History lives in .ignis_trace\shell_history.txt (last 500 entries loaded
#   at first use, appends deduplicated against the immediately previous
#   entry) and in $global:IgnisHistory for the session. Consecutive
#   duplicates are collapsed; empty lines are never recorded.

# --------------------------------------------------------------------------
# Cargo metadata (features, example names)
# --------------------------------------------------------------------------

function Get-IgnisCargoFeatures {
    $p = 'Cargo.toml'
    if (-not (Test-Path $p)) { return @() }
    $ticks = (Get-Item $p).LastWriteTimeUtc.Ticks
    if ($global:IgnisCargoMeta -and $global:IgnisCargoMeta.Ticks -eq $ticks) {
        return $global:IgnisCargoMeta.Features
    }
    $raw = Get-Content $p -Raw
    $features = @()
    if ($raw -match '(?s)\[features\](.*?)(\r?\n\[|$)') {
        $body = $Matches[1]
        foreach ($ln in ($body -split "`n")) {
            if ($ln -match '^\s*([A-Za-z0-9_-]+)\s*=') { $features += $Matches[1] }
        }
    }
    $examples = @([regex]::Matches($raw, '\[\[example\]\]\s+name\s*=\s*"([^"]+)"') |
        ForEach-Object { $_.Groups[1].Value })
    $global:IgnisCargoMeta = @{ Ticks = $ticks; Features = $features; Examples = $examples }
    return $features
}

function Get-IgnisCargoExamples {
    $null = Get-IgnisCargoFeatures
    if ($global:IgnisCargoMeta) { return $global:IgnisCargoMeta.Examples }
    return @()
}

# --------------------------------------------------------------------------
# Command discovery and history
# --------------------------------------------------------------------------

function Get-IgnisCommandNames {
    Get-ChildItem (Join-Path $PSScriptRoot 'cmd_*.ps1') -ErrorAction SilentlyContinue |
        ForEach-Object { $_.BaseName.Substring(4) }
}

function Get-IgnisHistoryPath {
    Join-Path $PSScriptRoot '..\.ignis_trace\shell_history.txt'
}

function Get-IgnisHistory {
    if (-not $global:IgnisHistory) {
        $global:IgnisHistory = [System.Collections.Generic.List[string]]::new()
        $p = Get-IgnisHistoryPath
        if (Test-Path $p) {
            foreach ($l in (Get-Content $p -ErrorAction SilentlyContinue | Select-Object -Last 500)) {
                if ($l) { $global:IgnisHistory.Add($l) }
            }
        }
    }
    # The unary comma wraps the List in a one-element array; the pipeline
    # unwraps exactly one level, so the caller receives the List object
    # itself rather than its enumerated elements. Without the comma, an
    # empty list enumerates to $null and a populated one to object[], and
    # both break callers that mutate the collection through .Add. This is
    # the standard PowerShell idiom for returning a live collection.
    return ,$global:IgnisHistory
}

function Add-IgnisHistory {
    param([string]$Line)
    if (-not $Line -or -not $Line.Trim()) { return }
    # Ensure initialization, then mutate the global List directly. Going
    # through a captured return value is exactly the enumeration hazard
    # documented in Get-IgnisHistory, so it is deliberately avoided here.
    $null = Get-IgnisHistory
    $h = $global:IgnisHistory
    if ($h.Count -gt 0 -and $h[$h.Count - 1] -eq $Line) { return }
    $h.Add($Line)
    try {
        $dir = Split-Path (Get-IgnisHistoryPath)
        if (-not (Test-Path $dir)) { New-Item -ItemType Directory -Path $dir -Force | Out-Null }
        Add-Content -Path (Get-IgnisHistoryPath) -Value $Line -Encoding UTF8
    } catch { }
}

# --------------------------------------------------------------------------
# Fuzzy matching
# --------------------------------------------------------------------------

function Get-FuzzyMatch {
    <#
    .SYNOPSIS
    Subsequence fuzzy match with a simple additive score.

    .DESCRIPTION
    Returns $null when Pattern is not a subsequence of Text (case
    insensitive). Otherwise returns a hashtable with Score and Indices (the
    matched character positions in Text, used for highlight rendering).
    Scoring: +1 per matched character, +3 for a character adjacent to the
    previous match, +2 for a match at a word start, minus a small length
    penalty so shorter candidates rank first among equals. An empty pattern
    matches everything with score 0, preserving caller ordering.
    #>
    param([string]$Pattern, [string]$Text)
    if (-not $Pattern) { return @{ Score = 0; Indices = @() } }
    if (-not $Text) { return $null }
    $p = $Pattern.ToLowerInvariant()
    $t = $Text.ToLowerInvariant()
    $indices = [System.Collections.Generic.List[int]]::new()
    $score = 0
    $ti = 0
    $prev = -2
    for ($pi = 0; $pi -lt $p.Length; $pi++) {
        $found = $t.IndexOf($p[$pi], $ti)
        if ($found -lt 0) { return $null }
        $score += 1
        if ($found -eq ($prev + 1)) { $score += 3 }
        if ($found -eq 0) { $score += 2 }
        elseif ($t[$found - 1] -eq ' ' -or $t[$found - 1] -eq '-' -or $t[$found - 1] -eq '_' -or $t[$found - 1] -eq '.') { $score += 2 }
        $indices.Add($found)
        $prev = $found
        $ti = $found + 1
    }
    $score -= [int]($t.Length / 10)
    return @{ Score = $score; Indices = $indices.ToArray() }
}

# --------------------------------------------------------------------------
# Completion
# --------------------------------------------------------------------------

function Get-IgnisCompletions {
    <#
    .SYNOPSIS
    Compute completion candidates for the final token of a command line.

    .DESCRIPTION
    Returns full replacement strings for the token under completion (the
    caller splices them at the token start). See the file header for the
    completion model, including the comma-continuation rule for feature
    lists.
    #>
    param([string]$Line)
    $endsSpace = $Line.EndsWith(' ')
    $tokens = @(($Line.Trim()) -split '\s+' | Where-Object { $_ })

    $prefix = ''
    $tokenIndex = 0
    if ($tokens.Count -gt 0) {
        if ($endsSpace) { $tokenIndex = $tokens.Count }
        else { $tokenIndex = $tokens.Count - 1; $prefix = $tokens[-1] }
    }

    # First token: command names plus aliases.
    if ($tokenIndex -eq 0) {
        $names = @(Get-IgnisCommandNames)
        if ($global:Aliases) { $names += @($global:Aliases.Keys) }
        return @($names | Where-Object { $_ -like "$prefix*" } | Sort-Object -Unique)
    }

    $cmdTok = $tokens[0].ToLower()
    $cmd = if ($global:Aliases -and $global:Aliases.ContainsKey($cmdTok)) { $global:Aliases[$cmdTok] } else { $cmdTok }

    $done = @()
    if ($endsSpace) { $done = $tokens }
    elseif ($tokens.Count -ge 2) { $done = $tokens[0..($tokens.Count - 2)] }
    $prev = if ($done.Count -gt 0) { $done[-1] } else { '' }

    # Feature list completion, including the tail after a comma.
    if ($prev -eq '--features') {
        $feats = @(Get-IgnisCargoFeatures)
        $head = ''
        $tail = $prefix
        if ($prefix -match '^(.*,)([^,]*)$') { $head = $Matches[1]; $tail = $Matches[2] }
        return @($feats | Where-Object { $_ -like "$tail*" } | ForEach-Object { $head + $_ })
    }

    $table = @{
        build  = @('full', 'minimal', 'release', '--features', '--release')
        check  = @('full', 'minimal', '--features')
        test   = @('all', 'unit', 'smoke', 'features', 'lint', 'audit', 'doc', 'size', 'miri', '--step', '--filter')
        lint   = @('all', 'clippy', 'fmt', 'doc', '--fix')
        run    = @(@(Get-IgnisCargoExamples) + @('--features', '--release'))
        trace  = @('last', 'list', 'errors', 'timeline', 'diff', 'report')
        info   = @('all', 'system', 'vulkan', 'project', 'deps')
        clean  = @('target', 'traces', 'all')
        prof   = @('build', 'test', '--features')
        watch  = @('check', 'build', 'test', 'lint', 'smoke', '--features')
        vuid   = @('list', 'search', 'categories')
        help   = @(Get-IgnisCommandNames)
        stub   = @('--out', '--filter', '--full', '--elide-lines')
        live   = @('ignis', 'ignis_demo')
        theme  = @((Get-MuxThemeTable).Keys)
        crash  = @('list', 'latest')
        chrome = @('list')
        gpu    = @('raw')
    }
    if ($table.ContainsKey($cmd)) {
        return @($table[$cmd] | Where-Object { $_ -like "$prefix*" })
    }
    return @()
}

# --------------------------------------------------------------------------
# Inline coloring
# --------------------------------------------------------------------------

function Format-IgnisLineColored {
    param([string]$Buffer, [string[]]$Known)
    if (-not $Buffer) { return '' }
    $sp = $Buffer.IndexOf(' ')
    $first = if ($sp -ge 0) { $Buffer.Substring(0, $sp) } else { $Buffer }
    $rest = if ($sp -ge 0) { $Buffer.Substring($sp) } else { '' }
    $ok = $Known -contains $first.ToLower()
    $col = if ($ok) { "`e[38;2;120;200;140m" } else { "`e[38;2;240;120;120m" }
    return $col + $first + "`e[0m" + $rest
}

# --------------------------------------------------------------------------
# Palette
# --------------------------------------------------------------------------

function Get-IgnisPaletteCandidates {
    $desc = @{
        build = 'compile the project';      check = 'type-check without codegen'
        test = 'run test suites';           lint = 'clippy, fmt, doc warnings'
        run = 'run an example';             trace = 'inspect failures'
        info = 'system and project info';   status = 'git and build status'
        clean = 'clean artifacts';          prof = 'timing profiler'
        unlock = 'kill stuck cargo';        stub = 'LLM API digest'
        mux = 'terminal multiplexer';       live = 'GPU live link workspace'
        watch = 'rerun on source change';   vuid = 'VUID knowledge browser'
        help = 'command reference'
        theme = 'workspace color theme';  gpu = 'Vulkan device inspector'
        crash = 'crash report viewer';    chrome = 'trace viewer launcher'
    }
    $items = [System.Collections.Generic.List[object]]::new()
    foreach ($n in (Get-IgnisCommandNames | Sort-Object)) {
        $items.Add(@{ Text = $n; Hint = [string]$desc[$n] })
    }
    foreach ($ex in (Get-IgnisCargoExamples)) {
        $items.Add(@{ Text = "run $ex"; Hint = 'cargo example' })
    }
    foreach ($q in @('build full', 'check full', 'test unit', 'test smoke',
                     'lint clippy --fix', 'watch check', 'live ignis_demo',
                     'vuid list', 'trace last')) {
        $items.Add(@{ Text = $q; Hint = 'quick action' })
    }
    # Recent unique history, newest first.
    $hist = Get-IgnisHistory
    $seen = @{}
    $added = 0
    for ($i = $hist.Count - 1; $i -ge 0 -and $added -lt 40; $i--) {
        $l = $hist[$i]
        if (-not $seen.ContainsKey($l)) {
            $seen[$l] = $true
            $items.Add(@{ Text = $l; Hint = 'history' })
            $added++
        }
    }
    return $items.ToArray()
}

function Invoke-IgnisPalette {
    <#
    .SYNOPSIS
    Full-screen fuzzy finder over commands, examples, quick actions, and
    recent history. Returns the selected command line or $null on cancel.
    #>
    $cands = Get-IgnisPaletteCandidates
    $query = ''
    $sel = 0
    $out = [Console]::Out
    $out.Write("`e[?1049h`e[?25l")
    try {
        while ($true) {
            $w = [math]::Max(20, [Console]::WindowWidth)
            $h = [math]::Max(8, [Console]::WindowHeight)
            $maxRows = $h - 5

            $scored = foreach ($c in $cands) {
                $m = Get-FuzzyMatch -Pattern $query -Text $c.Text
                if ($null -ne $m) {
                    [pscustomobject]@{ C = $c; S = $m.Score; I = $m.Indices }
                }
            }
            $list = @($scored | Sort-Object S -Descending | Select-Object -First $maxRows)
            if ($sel -ge $list.Count) { $sel = [math]::Max(0, $list.Count - 1) }

            $sb = [System.Text.StringBuilder]::new(8192)
            [void]$sb.Append("`e[H`e[2J`e[0m")
            [void]$sb.Append("`e[1;36m  ignis palette`e[0m")
            [void]$sb.Append("`e[2m   type to filter, enter run, esc cancel`e[0m`n")
            [void]$sb.Append("  > ").Append($query).Append("`e[7m `e[0m`n`n")

            for ($row = 0; $row -lt $list.Count; $row++) {
                $it = $list[$row]
                $text = $it.C.Text
                $hint = [string]$it.C.Hint
                $idxSet = [System.Collections.Generic.HashSet[int]]::new()
                foreach ($ix in $it.I) { [void]$idxSet.Add($ix) }
                $isSel = ($row -eq $sel)
                if ($isSel) { [void]$sb.Append("`e[48;2;45;48;60m") }
                [void]$sb.Append('  ')
                $limit = [math]::Min($text.Length, 40)
                for ($ci = 0; $ci -lt $limit; $ci++) {
                    if ($idxSet.Contains($ci)) {
                        [void]$sb.Append("`e[1;36m").Append($text[$ci]).Append("`e[22;39m")
                    } else {
                        [void]$sb.Append($text[$ci])
                    }
                }
                if ($text.Length -lt 42) { [void]$sb.Append(' ' * (42 - $limit)) }
                if ($hint) {
                    $hlimit = [math]::Max(0, $w - 46)
                    if ($hint.Length -gt $hlimit) { $hint = $hint.Substring(0, $hlimit) }
                    [void]$sb.Append("`e[2m").Append($hint).Append("`e[22m")
                }
                [void]$sb.Append("`e[0m`n")
            }
            if ($list.Count -eq 0) {
                [void]$sb.Append("`e[2m  (no matches)`e[0m`n")
            }
            $out.Write($sb.ToString())

            $k = [Console]::ReadKey($true)
            if ($k.Key -eq [ConsoleKey]::Escape) { return $null }
            elseif ($k.Key -eq [ConsoleKey]::Enter) {
                if ($list.Count -gt 0) { return $list[$sel].C.Text }
                return $null
            }
            elseif ($k.Key -eq [ConsoleKey]::UpArrow) { if ($sel -gt 0) { $sel-- } }
            elseif ($k.Key -eq [ConsoleKey]::DownArrow) { if ($sel -lt ($list.Count - 1)) { $sel++ } }
            elseif ($k.Key -eq [ConsoleKey]::Backspace) {
                if ($query.Length -gt 0) { $query = $query.Substring(0, $query.Length - 1); $sel = 0 }
            }
            else {
                $ch = $k.KeyChar
                if ($ch -and -not [char]::IsControl($ch)) { $query += $ch; $sel = 0 }
            }
        }
    }
    finally {
        $out.Write("`e[?25h`e[?1049l")
    }
}

# --------------------------------------------------------------------------
# Line editor
# --------------------------------------------------------------------------

function Read-IgnisLine {
    <#
    .SYNOPSIS
    Interactive line editor for the ignis REPL. See the file header for the
    complete binding table and behavior contract.

    .DESCRIPTION
    Falls back to Read-Host when console input is redirected. Editor state
    is held in a hashtable so the redraw scriptblock can mutate the recorded
    origin (dynamic scope in PowerShell would otherwise shadow assignments
    into a child scope).
    #>
    if ([Console]::IsInputRedirected) { return (Read-Host) }

    $st = @{
        Buffer = ''
        Cursor = 0
        Left   = [Console]::CursorLeft
        Top    = [Console]::CursorTop
    }
    $known = @(Get-IgnisCommandNames)
    if ($global:Aliases) { $known += @($global:Aliases.Keys) }
    $hist = Get-IgnisHistory
    $histIdx = $hist.Count
    $savedEdit = ''
    $tabCands = $null
    $tabIdx = -1
    $tabStart = 0

    $redraw = {
        param($st, $known)
        $w = [math]::Max(1, [Console]::WindowWidth)
        $bh = [Console]::BufferHeight
        $rows = [math]::Floor(($st.Left + $st.Buffer.Length) / $w)
        if (($st.Top + $rows) -ge $bh) { $st.Top = [math]::Max(0, $bh - 1 - $rows) }
        [Console]::SetCursorPosition($st.Left, $st.Top)
        [Console]::Out.Write((Format-IgnisLineColored -Buffer $st.Buffer -Known $known) + "`e[0m`e[0J")
        $cpos = $st.Left + $st.Cursor
        $cy = [math]::Min($bh - 1, $st.Top + [math]::Floor($cpos / $w))
        [Console]::SetCursorPosition(($cpos % $w), $cy)
    }

    $prevCC = [Console]::TreatControlCAsInput
    [Console]::TreatControlCAsInput = $true
    try {
        while ($true) {
            & $redraw $st $known
            $k = [Console]::ReadKey($true)
            $ctrl = ($k.Modifiers -band [ConsoleModifiers]::Control) -ne 0
            if ($k.Key -ne [ConsoleKey]::Tab) { $tabCands = $null }

            if ($k.Key -eq [ConsoleKey]::Enter) {
                [Console]::Out.Write("`n")
                Add-IgnisHistory $st.Buffer
                return $st.Buffer
            }
            elseif ($ctrl -and $k.Key -eq [ConsoleKey]::C) {
                [Console]::Out.Write("^C`n")
                return ''
            }
            elseif ($ctrl -and $k.Key -eq [ConsoleKey]::P) {
                $selText = Invoke-IgnisPalette
                if ($selText) {
                    $st.Buffer = $selText
                    $st.Cursor = $selText.Length
                    & $redraw $st $known
                    [Console]::Out.Write("`n")
                    Add-IgnisHistory $selText
                    return $selText
                }
            }
            elseif ($ctrl -and $k.Key -eq [ConsoleKey]::U) {
                $st.Buffer = $st.Buffer.Substring($st.Cursor)
                $st.Cursor = 0
            }
            elseif ($ctrl -and $k.Key -eq [ConsoleKey]::K) {
                $st.Buffer = $st.Buffer.Substring(0, $st.Cursor)
            }
            elseif ($ctrl -and $k.Key -eq [ConsoleKey]::W) {
                if ($st.Cursor -gt 0) {
                    $i = $st.Cursor - 1
                    while ($i -gt 0 -and $st.Buffer[$i] -eq ' ') { $i-- }
                    while ($i -gt 0 -and $st.Buffer[$i - 1] -ne ' ') { $i-- }
                    $st.Buffer = $st.Buffer.Remove($i, $st.Cursor - $i)
                    $st.Cursor = $i
                }
            }
            elseif ($k.Key -eq [ConsoleKey]::Backspace) {
                if ($st.Cursor -gt 0) {
                    $st.Buffer = $st.Buffer.Remove($st.Cursor - 1, 1)
                    $st.Cursor--
                }
            }
            elseif ($k.Key -eq [ConsoleKey]::Delete) {
                if ($st.Cursor -lt $st.Buffer.Length) {
                    $st.Buffer = $st.Buffer.Remove($st.Cursor, 1)
                }
            }
            elseif ($k.Key -eq [ConsoleKey]::LeftArrow) {
                if ($st.Cursor -gt 0) { $st.Cursor-- }
            }
            elseif ($k.Key -eq [ConsoleKey]::RightArrow) {
                if ($st.Cursor -lt $st.Buffer.Length) { $st.Cursor++ }
            }
            elseif ($k.Key -eq [ConsoleKey]::Home) { $st.Cursor = 0 }
            elseif ($k.Key -eq [ConsoleKey]::End) { $st.Cursor = $st.Buffer.Length }
            elseif ($k.Key -eq [ConsoleKey]::Escape) {
                $st.Buffer = ''
                $st.Cursor = 0
            }
            elseif ($k.Key -eq [ConsoleKey]::UpArrow) {
                if ($histIdx -gt 0) {
                    if ($histIdx -eq $hist.Count) { $savedEdit = $st.Buffer }
                    $histIdx--
                    $st.Buffer = $hist[$histIdx]
                    $st.Cursor = $st.Buffer.Length
                }
            }
            elseif ($k.Key -eq [ConsoleKey]::DownArrow) {
                if ($histIdx -lt $hist.Count) {
                    $histIdx++
                    if ($histIdx -eq $hist.Count) { $st.Buffer = $savedEdit }
                    else { $st.Buffer = $hist[$histIdx] }
                    $st.Cursor = $st.Buffer.Length
                }
            }
            elseif ($k.Key -eq [ConsoleKey]::Tab) {
                if ($st.Cursor -eq $st.Buffer.Length) {
                    if ($null -eq $tabCands) {
                        $tabStart = $st.Buffer.LastIndexOf(' ') + 1
                        $tabCands = @(Get-IgnisCompletions -Line $st.Buffer)
                        $tabIdx = -1
                    }
                    if ($tabCands.Count -gt 0) {
                        $back = ($k.Modifiers -band [ConsoleModifiers]::Shift) -ne 0
                        if ($back) { $tabIdx-- } else { $tabIdx++ }
                        if ($tabIdx -ge $tabCands.Count) { $tabIdx = 0 }
                        if ($tabIdx -lt 0) { $tabIdx = $tabCands.Count - 1 }
                        $st.Buffer = $st.Buffer.Substring(0, $tabStart) + $tabCands[$tabIdx]
                        $st.Cursor = $st.Buffer.Length
                    }
                }
            }
            else {
                $ch = $k.KeyChar
                if ($ch -and -not [char]::IsControl($ch)) {
                    $st.Buffer = $st.Buffer.Insert($st.Cursor, $ch)
                    $st.Cursor++
                    # Typing invalidates history navigation position.
                    $histIdx = $hist.Count
                }
            }
        }
    }
    finally {
        [Console]::TreatControlCAsInput = $prevCC
    }
}