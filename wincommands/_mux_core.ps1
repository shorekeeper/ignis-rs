#Requires -Version 7.0
#
# _mux_core.ps1 - Rendering engine, theme, pane model, and multiplexer engine
#                 for the ignis terminal multiplexer.
#
# OVERVIEW
#   This file defines the pure-PowerShell class hierarchy that turns a raw
#   console into a mouse-driven, splittable pane workspace. It has no external
#   dependencies beyond the [IgnisConsole] interop type from _mux_native.ps1,
#   which it references only at runtime (never at parse time), so class parse
#   order across the two files does not matter.
#
# CLASS INVENTORY (defined in dependency order within this file)
#   MuxScreen  - Double buffered cell grid with diff based VT flushing.
#   Theme      - Static color palette (24-bit packed RGB integers).
#   MuxPane    - Abstract content provider placed inside a layout leaf.
#   ClockPane, SysInfoPane, HelpPane, LogPane, MenuPane - Concrete demo panes.
#   MuxEngine  - Layout tree, focus model, input dispatch, and render loop.
#
# LAYOUT MODEL
#   The layout is a binary tree of plain hashtables (chosen over classes so the
#   tree can carry computed geometry fields without rigid schemas):
#     Leaf  : @{ Kind='pane';  Id=<string>; Pane=<MuxPane> }
#     Split : @{ Kind='split'; Dir='h'|'v'; Ratio=<0..1>; A=<node>; B=<node> }
#   'h' splits produce a one-column vertical divider (panes side by side).
#   'v' splits produce a one-row horizontal divider (panes stacked).
#   Each frame the tree is walked to assign absolute rectangles and to collect
#   pane rectangles and divider rectangles into flat lists used for hit testing.
#
# RENDERING MODEL
#   MuxScreen maintains a back buffer (drawn into each frame) and a front buffer
#   (last flushed state). Flush compares the two cell by cell and emits only the
#   minimal VT sequences (cursor moves and SGR color changes) required to
#   reconcile them, which eliminates flicker and keeps output volume low. The
#   grid stores three parallel primitive arrays (char, foreground, background)
#   rather than an array of cell objects, for allocation-free per-frame updates.
#
# COORDINATE SYSTEM
#   All engine and screen coordinates are 0-based, origin top-left. The bottom
#   row of the screen is reserved for the status bar; the layout tree is
#   computed into the rectangle [0, 0, Width, Height-1].
#
# COLOR ENCODING
#   Colors are 24-bit packed integers 0xRRGGBB. The sentinel value -1 denotes
#   the terminal default color (emitted as SGR 39 / 49).
#
# CLASS METHOD SCOPING NOTE
#   PowerShell class methods cannot see script-scope functions. Consequently all
#   shared helpers used from inside methods are exposed as static methods (for
#   example [MuxScreen]::Rgb), and layout recursion is implemented as instance
#   methods on MuxEngine rather than as free functions.

# --------------------------------------------------------------------------
# MuxScreen
# --------------------------------------------------------------------------

class MuxScreen {
    # Thin wrapper over the compiled IgnisScreen cell buffer. Every per-cell
    # hot loop (fills, text runs, box borders, the full-screen diff flush)
    # executes natively; this class only forwards calls and mirrors the
    # dimensions, preserving the API surface that panes and the engine were
    # written against. The native type is resolved by name at runtime rather
    # than as a type literal because type literals inside class method bodies
    # are bound when the class is defined, which precedes compilation of the
    # interop assembly by Initialize-MuxNative.
    [int]$Width
    [int]$Height
    [object]$Native

    MuxScreen([int]$w, [int]$h) {
        $t = [type]'IgnisScreen'
        $this.Native = [Activator]::CreateInstance($t, @([object]$w, [object]$h))
        $this.Width = $this.Native.Width
        $this.Height = $this.Native.Height
    }

    # Pack RGB components into a single 24-bit integer. Kept here because the
    # Theme class initializers reference it at type load time.
    static [int] Rgb([int]$r, [int]$g, [int]$b) {
        return ((($r -band 0xFF) -shl 16) -bor (($g -band 0xFF) -shl 8) -bor ($b -band 0xFF))
    }

    [void] Resize([int]$w, [int]$h) {
        $this.Native.Resize($w, $h)
        $this.Width = $this.Native.Width
        $this.Height = $this.Native.Height
    }

    [void] Clear([int]$background) {
        $this.Native.Clear($background)
    }

    [void] Set([int]$x, [int]$y, [char]$c, [int]$f, [int]$b) {
        $this.Native.Set($x, $y, $c, $f, $b)
    }

    [void] Text([int]$x, [int]$y, [string]$s, [int]$f, [int]$b) {
        $this.Native.Text($x, $y, $s, $f, $b, -1)
    }

    [void] TextMax([int]$x, [int]$y, [string]$s, [int]$f, [int]$b, [int]$max) {
        $this.Native.Text($x, $y, $s, $f, $b, $max)
    }

    [void] FillRect([int]$x, [int]$y, [int]$w, [int]$h, [char]$c, [int]$f, [int]$b) {
        $this.Native.FillRect($x, $y, $w, $h, $c, $f, $b)
    }

    [void] Box([int]$x, [int]$y, [int]$w, [int]$h, [int]$f, [int]$b, [string]$style) {
        $round = 1
        if ($style -eq 'single') { $round = 0 }
        $this.Native.Box($x, $y, $w, $h, $f, $b, $round)
    }

    [void] VLine([int]$x, [int]$y, [int]$h, [char]$c, [int]$f, [int]$b) {
        $this.Native.VLine($x, $y, $h, $c, $f, $b)
    }

    [void] HLine([int]$x, [int]$y, [int]$w, [char]$c, [int]$f, [int]$b) {
        $this.Native.HLine($x, $y, $w, $c, $f, $b)
    }

    [void] Flush() {
        $this.Native.Flush()
    }
}

# --------------------------------------------------------------------------
# Theme
# --------------------------------------------------------------------------
# Static palette. Property initializers run when the type is loaded, at which
# point MuxScreen is already defined so [MuxScreen]::Rgb is available.

class Theme {
    static [int] $Bg          = [MuxScreen]::Rgb(24, 24, 28)
    static [int] $Panel       = [MuxScreen]::Rgb(32, 33, 40)
    static [int] $Border      = [MuxScreen]::Rgb(70, 72, 84)
    static [int] $BorderFocus = [MuxScreen]::Rgb(90, 170, 255)
    static [int] $Text        = [MuxScreen]::Rgb(210, 212, 220)
    static [int] $TextDim     = [MuxScreen]::Rgb(130, 133, 145)
    static [int] $TextHead    = [MuxScreen]::Rgb(240, 242, 248)
    static [int] $Accent      = [MuxScreen]::Rgb(120, 200, 140)
    static [int] $StatusBg    = [MuxScreen]::Rgb(45, 48, 60)
    static [int] $StatusFg    = [MuxScreen]::Rgb(200, 203, 214)
    static [int] $StatusHi    = [MuxScreen]::Rgb(250, 220, 120)
}

# --------------------------------------------------------------------------
# MuxPane (abstract base)
# --------------------------------------------------------------------------
# A pane provides the content of one layout leaf. The engine sets Engine and
# Node after the pane is placed, so panes can mutate the tree (for example a
# menu pane replacing its own content) by calling back into the engine.
#
# Engine and Node are typed [object] deliberately: MuxEngine is defined after
# this class, and using [object] avoids a forward type reference at parse time.
#
# Render draws into the inner content rectangle (already inside the border).
# OnKey / OnMouse return $true when the event was consumed and should not be
# processed further by the engine's global bindings.

class MuxPane {
    [string]$Title = 'pane'
    [object]$Engine = $null
    [object]$Node = $null

    [void] Render([MuxScreen]$s, [int]$x, [int]$y, [int]$w, [int]$h, [bool]$focused) { }
    [bool] OnKey([object]$ev) { return $false }
    [bool] OnMouse([object]$ev, [int]$lx, [int]$ly) { return $false }
    [void] OnTick() { }
}

# --------------------------------------------------------------------------
# ClockPane
# --------------------------------------------------------------------------

class ClockPane : MuxPane {
    [datetime]$Start

    ClockPane() {
        $this.Title = 'clock'
        $this.Start = Get-Date
    }

    [void] Render([MuxScreen]$s, [int]$x, [int]$y, [int]$w, [int]$h, [bool]$focused) {
        $now = Get-Date
        $t = $now.ToString('HH:mm:ss')
        $d = $now.ToString('yyyy-MM-dd dddd')
        $up = (Get-Date) - $this.Start
        $upStr = 'up ' + [int]$up.TotalMinutes + 'm ' + $up.Seconds + 's'
        $cy = $y + [int]($h / 2) - 1
        $s.TextMax($x + [math]::Max(0, [int](($w - $t.Length) / 2)), $cy, $t, [Theme]::Accent, [Theme]::Panel, $w)
        $s.TextMax($x + [math]::Max(0, [int](($w - $d.Length) / 2)), $cy + 1, $d, [Theme]::Text, [Theme]::Panel, $w)
        $s.TextMax($x + [math]::Max(0, [int](($w - $upStr.Length) / 2)), $cy + 2, $upStr, [Theme]::TextDim, [Theme]::Panel, $w)
    }
}

# --------------------------------------------------------------------------
# SysInfoPane
# --------------------------------------------------------------------------
# Gathers static host information once, lazily, on the first tick. The CIM
# queries can take tens of milliseconds; doing them once avoids a per-frame
# stall. $global:PSVersionTable is used rather than the bare automatic variable
# because automatic variables are not reliably visible in class method scope.

class SysInfoPane : MuxPane {
    [string[]]$Lines
    [bool]$Loaded = $false

    SysInfoPane() {
        $this.Title = 'system'
        $this.Lines = @('(loading system information)')
    }

    [void] OnTick() {
        if ($this.Loaded) { return }
        $this.Loaded = $true
        $acc = [System.Collections.Generic.List[string]]::new()
        try {
            $acc.Add('OS   : ' + [System.Environment]::OSVersion.VersionString)
            $cpu = (Get-CimInstance Win32_Processor -ErrorAction SilentlyContinue | Select-Object -First 1).Name
            $acc.Add('CPU  : ' + $cpu)
            $ram = [math]::Round((Get-CimInstance Win32_ComputerSystem -ErrorAction SilentlyContinue).TotalPhysicalMemory / 1GB, 1)
            $acc.Add('RAM  : ' + $ram + ' GB')
            $acc.Add('PS   : ' + $global:PSVersionTable.PSVersion.ToString())
            $acc.Add('PID  : ' + [System.Diagnostics.Process]::GetCurrentProcess().Id)
        } catch {
            $acc.Add('error gathering system information')
        }
        $this.Lines = $acc.ToArray()
    }

    [void] Render([MuxScreen]$s, [int]$x, [int]$y, [int]$w, [int]$h, [bool]$focused) {
        for ($i = 0; $i -lt $this.Lines.Count; $i++) {
            if ($i -ge $h) { break }
            $s.TextMax($x, $y + $i, $this.Lines[$i], [Theme]::Text, [Theme]::Panel, $w)
        }
    }
}

# --------------------------------------------------------------------------
# HelpPane
# --------------------------------------------------------------------------

class HelpPane : MuxPane {
    HelpPane() { $this.Title = 'help' }

    [void] Render([MuxScreen]$s, [int]$x, [int]$y, [int]$w, [int]$h, [bool]$focused) {
        $lines = @(
            'Keyboard',
            '  Ctrl+Arrows   move focus by direction',
            '  Tab / Sh+Tab  cycle focus',
            '  Ctrl+S / D    split side by side / stacked',
            '  Ctrl+W        close pane   Ctrl+Z zoom',
            '  Ctrl+Q        quit',
            '  Esc           close overlay / unzoom',
            '',
            'Live panes',
            '  events: 1..7 toggle filters, arrows and',
            '          PgUp/PgDn/Home/End scroll',
            '  validation: arrows select, Enter or click',
            '          opens the VUID detail overlay',
            '',
            'Mouse',
            '  click focus, double click zoom, drag',
            '  dividers, wheel scrolls, status buttons',
            '',
            'New panes open a menu; pick content there.',
            'Layout is saved on quit and restored.'
        )
        for ($i = 0; $i -lt $lines.Count; $i++) {
            if ($i -ge $h) { break }
            $ln = $lines[$i]
            $fg = if ($ln -match '^\S' -and $ln -notmatch '^\s') { [Theme]::TextHead } else { [Theme]::TextDim }
            $s.TextMax($x, $y + $i, $ln, $fg, [Theme]::Panel, $w)
        }
    }
}

# --------------------------------------------------------------------------
# LogPane
# --------------------------------------------------------------------------
# Renders the tail of the engine's shared event log. This demonstrates the
# engine-to-pane data channel that later PowerShell Cache iterations reuse for live data feeds.

class LogPane : MuxPane {
    LogPane() { $this.Title = 'log' }

    [void] Render([MuxScreen]$s, [int]$x, [int]$y, [int]$w, [int]$h, [bool]$focused) {
        if ($null -eq $this.Engine) {
            $s.TextMax($x, $y, '(no engine)', [Theme]::TextDim, [Theme]::Panel, $w)
            return
        }
        $log = $this.Engine.Log
        $count = $log.Count
        if ($count -eq 0) {
            $s.TextMax($x, $y, '(log empty)', [Theme]::TextDim, [Theme]::Panel, $w)
            return
        }
        $begin = [math]::Max(0, $count - $h)
        for ($i = 0; $i -lt $h; $i++) {
            $idx = $begin + $i
            if ($idx -ge $count) { break }
            $s.TextMax($x, $y + $i, $log[$idx], [Theme]::TextDim, [Theme]::Panel, $w)
        }
    }
}

# --------------------------------------------------------------------------
# MenuPane
# --------------------------------------------------------------------------
# Default content for freshly created panes. Selecting an option replaces the
# pane's own content via the engine, so a split immediately becomes useful.

class MenuPane : MuxPane {
    # Default content of a freshly split pane. Options come from the engine
    # registry, so the menu automatically offers whatever content the
    # launching command registered, including live data panes in the live
    # workspace. Selecting an option replaces this pane's content in place.
    [int]$Sel = 0

    MenuPane() { $this.Title = 'menu' }

    [string[]] Names() {
        if ($this.Engine -and $this.Engine.Registry) {
            return @($this.Engine.Registry.Keys)
        }
        return @()
    }

    [void] Render([MuxScreen]$s, [int]$x, [int]$y, [int]$w, [int]$h, [bool]$focused) {
        $names = $this.Names()
        $s.TextMax($x, $y, 'Select content:', [Theme]::TextHead, [Theme]::Panel, $w)
        if ($names.Count -eq 0) {
            $s.TextMax($x, $y + 2, '(no registry configured)', [Theme]::TextDim, [Theme]::Panel, $w)
            return
        }
        if ($this.Sel -ge $names.Count) { $this.Sel = $names.Count - 1 }
        for ($i = 0; $i -lt $names.Count; $i++) {
            $ry = $y + 2 + $i
            if ($ry -ge $y + $h) { break }
            $mark = if ($i -eq $this.Sel) { '> ' } else { '  ' }
            $fg = if ($i -eq $this.Sel) { [Theme]::Accent } else { [Theme]::Text }
            $s.TextMax($x, $ry, $mark + $names[$i], $fg, [Theme]::Panel, $w)
        }
    }

    [bool] OnKey([object]$ev) {
        if ($ev.Ctrl -or -not $ev.KeyDown) { return $false }
        $names = $this.Names()
        switch ($ev.VKey) {
            0x26 { if ($this.Sel -gt 0) { $this.Sel-- }; return $true }
            0x28 { if ($this.Sel -lt $names.Count - 1) { $this.Sel++ }; return $true }
            0x0D { $this.Activate(); return $true }
        }
        return $false
    }

    [bool] OnMouse([object]$ev, [int]$lx, [int]$ly) {
        if ($ev.Left) {
            $i = $ly - 2
            if ($i -ge 0 -and $i -lt $this.Names().Count) {
                $this.Sel = $i
                $this.Activate()
            }
            return $true
        }
        return $false
    }

    [void] Activate() {
        if ($null -eq $this.Engine -or $null -eq $this.Engine.Registry) { return }
        $names = $this.Names()
        if ($this.Sel -lt 0 -or $this.Sel -ge $names.Count) { return }
        $entry = $this.Engine.Registry[$names[$this.Sel]]
        $pane = & ($entry['Make'])
        if ($null -ne $pane -and $null -ne $this.Engine) {
            $this.Engine.SwapContent($this.Node, $pane)
        }
    }
}

# --------------------------------------------------------------------------
# MuxEngine
# --------------------------------------------------------------------------
# Owns the screen, the layout tree, focus, drag state, and the render loop.
# All tree operations are instance methods so they can recurse. The engine
# reserves the bottom screen row for the status bar and lays out the tree into
# the remaining region.

class MuxEngine {
    [MuxScreen]$Screen
    $Root
    [string]$FocusId
    [System.Collections.Generic.List[object]]$PaneRects
    [System.Collections.Generic.List[object]]$DivRects
    [System.Collections.Generic.List[object]]$Buttons
    [System.Collections.Generic.List[string]]$Log
    [bool]$Running = $false
    [bool]$Zoom = $false
    [bool]$PrevLeft = $false
    $Drag = $null
    [int]$PaneCounter = 100

    # Content registry: ordered map of display name to a hashtable with keys
    # Make (a factory scriptblock producing a fresh pane) and Type (the pane
    # class name, used by layout persistence to map saved trees back to
    # factories). Populated by the launching command; MenuPane presents it.
    [System.Collections.Specialized.OrderedDictionary]$Registry

    # Modal overlay: a MuxPane rendered centered above the workspace. While
    # set, it captures all input except Ctrl+Q; Escape or a click outside
    # closes it. OverlayRect caches the content rectangle from the last
    # render for mouse hit testing.
    [object]$Overlay = $null
    $OverlayRect = $null

    # Last raw input event, shown in the status bar. Serves as a built-in
    # input diagnostic: if keystrokes never update this readout while clicks
    # do, key records are not reaching the engine at all, which localizes a
    # fault to the console input layer rather than to bindings.
    [string]$LastInput = ''

    MuxEngine() {
        $this.PaneRects = [System.Collections.Generic.List[object]]::new()
        $this.DivRects = [System.Collections.Generic.List[object]]::new()
        $this.Buttons = [System.Collections.Generic.List[object]]::new()
        $this.Log = [System.Collections.Generic.List[string]]::new()
        $this.Registry = [System.Collections.Specialized.OrderedDictionary]::new()
    }

    # -- logging ----------------------------------------------------------

    [void] LogMsg([string]$m) {
        $ts = (Get-Date).ToString('HH:mm:ss')
        $this.Log.Add("[$ts] $m")
        while ($this.Log.Count -gt 300) { $this.Log.RemoveAt(0) }
    }

    # -- tree helpers -----------------------------------------------------

    [object] FindById($node, [string]$id) {
        if ($node['Kind'] -eq 'pane') {
            if ($node['Id'] -eq $id) { return $node }
            return $null
        }
        $a = $this.FindById($node['A'], $id)
        if ($a) { return $a }
        return $this.FindById($node['B'], $id)
    }

    [object] FocusedNode() {
        return $this.FindById($this.Root, $this.FocusId)
    }

    [object] FindParent($cur, $target) {
        if ($cur['Kind'] -ne 'split') { return $null }
        if ([object]::ReferenceEquals($cur['A'], $target) -or [object]::ReferenceEquals($cur['B'], $target)) {
            return $cur
        }
        $p = $this.FindParent($cur['A'], $target)
        if ($p) { return $p }
        return $this.FindParent($cur['B'], $target)
    }

    [bool] ReplaceNode($cur, $target, $replacement) {
        if ($cur['Kind'] -ne 'split') { return $false }
        if ([object]::ReferenceEquals($cur['A'], $target)) { $cur['A'] = $replacement; return $true }
        if ([object]::ReferenceEquals($cur['B'], $target)) { $cur['B'] = $replacement; return $true }
        if ($this.ReplaceNode($cur['A'], $target, $replacement)) { return $true }
        return $this.ReplaceNode($cur['B'], $target, $replacement)
    }

    [string] FirstPaneId($node) {
        if ($node['Kind'] -eq 'pane') { return $node['Id'] }
        return $this.FirstPaneId($node['A'])
    }

    # Bind Engine and Node back-references on every pane in the tree.
    [void] AttachPanes() { $this.AttachWalk($this.Root) }
    [void] AttachWalk($node) {
        if ($node['Kind'] -eq 'pane') {
            $p = [MuxPane]$node['Pane']
            $p.Engine = $this
            $p.Node = $node
            return
        }
        $this.AttachWalk($node['A'])
        $this.AttachWalk($node['B'])
    }

    [void] TickAll() { $this.TickWalk($this.Root) }
    [void] TickWalk($node) {
        if ($node['Kind'] -eq 'pane') {
            ([MuxPane]$node['Pane']).OnTick()
            return
        }
        $this.TickWalk($node['A'])
        $this.TickWalk($node['B'])
    }

    # Replace a leaf's content in place, keeping its id and focus.
    [void] SwapContent($node, [MuxPane]$newPane) {
        $node['Pane'] = $newPane
        $newPane.Engine = $this
        $newPane.Node = $node
        $this.LogMsg("content set to '$($newPane.Title)'")
    }

    # -- geometry ---------------------------------------------------------

    [void] ComputeLayout() {
        $this.PaneRects.Clear()
        $this.DivRects.Clear()
        $bottom = $this.Screen.Height - 1
        if ($bottom -lt 1) { $bottom = 1 }
        if ($this.Zoom) {
            $fn = $this.FocusedNode()
            if ($fn) {
                $this.PaneRects.Add(@{ Node = $fn; X = 0; Y = 0; W = $this.Screen.Width; H = $bottom })
                return
            }
        }
        $this.ComputeNode($this.Root, 0, 0, $this.Screen.Width, $bottom)
    }

    [void] ComputeNode($node, [int]$x, [int]$y, [int]$w, [int]$h) {
        $node['X'] = $x; $node['Y'] = $y; $node['W'] = $w; $node['H'] = $h
        if ($node['Kind'] -eq 'pane') {
            $this.PaneRects.Add(@{ Node = $node; X = $x; Y = $y; W = $w; H = $h })
            return
        }
        if ($node['Dir'] -eq 'h') {
            $avail = $w - 1
            if ($avail -lt 2) { $this.ComputeNode($node['A'], $x, $y, $w, $h); return }
            $leftW = [int][math]::Floor($avail * $node['Ratio'])
            if ($leftW -lt 1) { $leftW = 1 }
            if ($leftW -gt $avail - 1) { $leftW = $avail - 1 }
            $rightW = $avail - $leftW
            $divX = $x + $leftW
            $this.ComputeNode($node['A'], $x, $y, $leftW, $h)
            $this.DivRects.Add(@{ Node = $node; X = $divX; Y = $y; W = 1; H = $h; Dir = 'h' })
            $this.ComputeNode($node['B'], $divX + 1, $y, $rightW, $h)
        } else {
            $avail = $h - 1
            if ($avail -lt 2) { $this.ComputeNode($node['A'], $x, $y, $w, $h); return }
            $topH = [int][math]::Floor($avail * $node['Ratio'])
            if ($topH -lt 1) { $topH = 1 }
            if ($topH -gt $avail - 1) { $topH = $avail - 1 }
            $botH = $avail - $topH
            $divY = $y + $topH
            $this.ComputeNode($node['A'], $x, $y, $w, $topH)
            $this.DivRects.Add(@{ Node = $node; X = $x; Y = $divY; W = $w; H = 1; Dir = 'v' })
            $this.ComputeNode($node['B'], $x, $divY + 1, $w, $botH)
        }
    }

    [object] HitPane([int]$x, [int]$y) {
        foreach ($p in $this.PaneRects) {
            if ($x -ge $p['X'] -and $x -lt ($p['X'] + $p['W']) -and $y -ge $p['Y'] -and $y -lt ($p['Y'] + $p['H'])) {
                return $p
            }
        }
        return $null
    }

    # -- focus and structure ---------------------------------------------

    [void] MoveFocus([string]$dir) {
        $this.Zoom = $false
        $this.ComputeLayout()
        $cur = $this.FocusedNode()
        if (-not $cur) { return }
        $crect = $null
        foreach ($p in $this.PaneRects) {
            if ([object]::ReferenceEquals($p['Node'], $cur)) { $crect = $p; break }
        }
        if (-not $crect) { return }
        $ccx = $crect['X'] + $crect['W'] / 2.0
        $ccy = $crect['Y'] + $crect['H'] / 2.0
        $best = $null
        $bestd = [double]::MaxValue
        foreach ($p in $this.PaneRects) {
            if ([object]::ReferenceEquals($p['Node'], $cur)) { continue }
            $pcx = $p['X'] + $p['W'] / 2.0
            $pcy = $p['Y'] + $p['H'] / 2.0
            $dx = $pcx - $ccx
            $dy = $pcy - $ccy
            $ok = $false
            switch ($dir) {
                'left'  { $ok = ($dx -lt -0.5 -and [math]::Abs($dy) -le [math]::Abs($dx)) }
                'right' { $ok = ($dx -gt 0.5 -and [math]::Abs($dy) -le [math]::Abs($dx)) }
                'up'    { $ok = ($dy -lt -0.5 -and [math]::Abs($dx) -le [math]::Abs($dy)) }
                'down'  { $ok = ($dy -gt 0.5 -and [math]::Abs($dx) -le [math]::Abs($dy)) }
            }
            if (-not $ok) { continue }
            $d = $dx * $dx + $dy * $dy
            if ($d -lt $bestd) { $bestd = $d; $best = $p['Node'] }
        }
        if ($best) { $this.FocusId = $best['Id']; $this.LogMsg("focus $dir -> $($best['Id'])") }
    }

    [void] CycleFocus([int]$delta) {
        $this.Zoom = $false
        $this.ComputeLayout()
        if ($this.PaneRects.Count -eq 0) { return }
        $ids = @()
        foreach ($p in $this.PaneRects) { $ids += $p['Node']['Id'] }
        $idx = [array]::IndexOf($ids, $this.FocusId)
        if ($idx -lt 0) { $idx = 0 }
        $idx = ($idx + $delta + $ids.Count) % $ids.Count
        $this.FocusId = $ids[$idx]
    }

    [void] SplitFocused([string]$dir) {
        $node = $this.FocusedNode()
        if (-not $node) { return }
        $this.PaneCounter++
        $newId = "p$($this.PaneCounter)"
        $newNode = @{ Kind = 'pane'; Id = $newId; Pane = [MenuPane]::new() }
        $split = @{ Kind = 'split'; Dir = $dir; Ratio = 0.5; A = $node; B = $newNode }
        if ([object]::ReferenceEquals($this.Root, $node)) {
            $this.Root = $split
        } else {
            $this.ReplaceNode($this.Root, $node, $split)
        }
        $this.AttachPanes()
        $this.FocusId = $newId
        $this.Zoom = $false
        $label = if ($dir -eq 'h') { 'side by side' } else { 'top and bottom' }
        $this.LogMsg("split $label -> $newId")
    }

    [void] CloseFocused() {
        $node = $this.FocusedNode()
        if (-not $node) { return }
        if ([object]::ReferenceEquals($this.Root, $node)) {
            $this.LogMsg('cannot close the last pane')
            return
        }
        $par = $this.FindParent($this.Root, $node)
        if (-not $par) { return }
        $sibling = if ([object]::ReferenceEquals($par['A'], $node)) { $par['B'] } else { $par['A'] }
        if ([object]::ReferenceEquals($this.Root, $par)) {
            $this.Root = $sibling
        } else {
            $this.ReplaceNode($this.Root, $par, $sibling)
        }
        $this.AttachPanes()
        $this.FocusId = $this.FirstPaneId($sibling)
        $this.Zoom = $false
        $this.LogMsg('closed pane')
    }

    [void] DoAction([string]$a) {
        switch ($a) {
            'splitH' { $this.SplitFocused('h') }
            'splitV' { $this.SplitFocused('v') }
            'zoom'   { $this.Zoom = -not $this.Zoom }
            'close'  { $this.CloseFocused() }
            'quit'   { $this.Running = $false }
        }
    }

    # -- overlay and input description -------------------------------------

    [void] ShowOverlay([object]$pane) {
        $pane.Engine = $this
        $this.Overlay = $pane
    }

    [void] CloseOverlay() {
        $this.Overlay = $null
        $this.OverlayRect = $null
    }

    # Human-readable name for a key event, used by the status bar readout.
    [string] DescribeKey([object]$ev) {
        $mods = ''
        if ($ev.Ctrl) { $mods += 'Ctrl+' }
        if ($ev.Alt) { $mods += 'Alt+' }
        if ($ev.Shift) { $mods += 'Shift+' }
        $vk = $ev.VKey
        $name = switch ($vk) {
            0x08 { 'Bksp' } 0x09 { 'Tab' } 0x0D { 'Enter' } 0x1B { 'Esc' }
            0x20 { 'Space' } 0x21 { 'PgUp' } 0x22 { 'PgDn' } 0x23 { 'End' }
            0x24 { 'Home' } 0x25 { 'Left' } 0x26 { 'Up' } 0x27 { 'Right' }
            0x28 { 'Down' } 0x2E { 'Del' }
            default {
                if ($vk -ge 0x30 -and $vk -le 0x5A) { [string][char]$vk }
                elseif ($vk -ge 0x70 -and $vk -le 0x7B) { 'F' + ($vk - 0x6F) }
                else { 'vk' + $vk }
            }
        }
        return $mods + $name
    }

    # -- input dispatch ---------------------------------------------------

    [void] HandleKey([object]$ev) {
        if (-not $ev.KeyDown) { return }
        $vk = $ev.VKey
        $this.LastInput = 'key ' + $this.DescribeKey($ev)

        # Global quit outranks everything, including a modal overlay.
        if ($ev.Ctrl -and $vk -eq 0x51) { $this.Running = $false; return }

        # A modal overlay captures the keyboard: Escape closes it, all other
        # keys are forwarded to it and consumed regardless of the result.
        if ($this.Overlay) {
            if ($vk -eq 0x1B) { $this.CloseOverlay(); return }
            $null = $this.Overlay.OnKey($ev)
            return
        }

        # Focused pane gets first refusal; a pane may consume the key.
        $fn = $this.FocusedNode()
        if ($fn) {
            $pane = [MuxPane]$fn['Pane']
            if ($pane.OnKey($ev)) { return }
        }

        if ($ev.Ctrl) {
            switch ($vk) {
                0x25 { $this.MoveFocus('left'); return }
                0x27 { $this.MoveFocus('right'); return }
                0x26 { $this.MoveFocus('up'); return }
                0x28 { $this.MoveFocus('down'); return }
                0x53 { $this.SplitFocused('h'); return }   # S
                0x44 { $this.SplitFocused('v'); return }   # D
                0x57 { $this.CloseFocused(); return }      # W
                0x5A { $this.Zoom = -not $this.Zoom; return } # Z
            }
        }
        if ($vk -eq 0x09) {
            if ($ev.Shift) { $this.CycleFocus(-1) } else { $this.CycleFocus(1) }
            return
        }
        if ($vk -eq 0x1B) {
            if ($this.Zoom) { $this.Zoom = $false }
            return
        }
    }

    [void] HandleMouse([object]$ev) {
        $x = $ev.X
        $y = $ev.Y
        $leftDown = $ev.Left
        $pressEdge = ($leftDown -and -not $this.PrevLeft)
        $releaseEdge = (-not $leftDown -and $this.PrevLeft)

        if ($pressEdge) { $this.LastInput = 'click ' + $x + ',' + $y }
        elseif ($ev.Wheel -ne 0) { $this.LastInput = 'wheel' }

        # Modal overlay. The wheel and inside clicks go to the overlay in
        # its local content coordinates. A click outside closes the overlay
        # and then falls through to normal handling, so one press both
        # dismisses the modal and acts on whatever was clicked (pane focus,
        # divider, status button) instead of requiring a second press.
        if ($this.Overlay) {
            $orct = $this.OverlayRect
            $handled = $true
            if ($null -ne $orct) {
                $inside = ($x -ge $orct['X'] -and $x -lt ($orct['X'] + $orct['W']) -and
                           $y -ge $orct['Y'] -and $y -lt ($orct['Y'] + $orct['H']))
                if ($ev.Wheel -ne 0) {
                    $null = $this.Overlay.OnMouse($ev, $x - $orct['X'] - 1, $y - $orct['Y'] - 1)
                } elseif ($pressEdge) {
                    if ($inside) {
                        $null = $this.Overlay.OnMouse($ev, $x - $orct['X'] - 1, $y - $orct['Y'] - 1)
                    } else {
                        $this.CloseOverlay()
                        $handled = $false
                    }
                }
            }
            if ($handled) {
                $this.PrevLeft = $leftDown
                return
            }
        }

        if ($ev.Wheel -ne 0) {
            $hit = $this.HitPane($x, $y)
            if ($hit) {
                ([MuxPane]$hit['Node']['Pane']).OnMouse($ev, $x - $hit['X'] - 1, $y - $hit['Y'] - 1)
            }
            $this.PrevLeft = $leftDown
            return
        }

        if ($pressEdge) {
            foreach ($b in $this.Buttons) {
                if ($y -eq $b['Y'] -and $x -ge $b['X'] -and $x -lt ($b['X'] + $b['W'])) {
                    $this.DoAction($b['Action'])
                    $this.PrevLeft = $leftDown
                    return
                }
            }
            foreach ($d in $this.DivRects) {
                if ($x -ge $d['X'] -and $x -lt ($d['X'] + $d['W']) -and $y -ge $d['Y'] -and $y -lt ($d['Y'] + $d['H'])) {
                    $this.Drag = $d
                    $this.PrevLeft = $leftDown
                    return
                }
            }
            $hit = $this.HitPane($x, $y)
            if ($hit) {
                $this.FocusId = $hit['Node']['Id']
                if ($ev.DoubleClick) {
                    $this.Zoom = -not $this.Zoom
                } else {
                    ([MuxPane]$hit['Node']['Pane']).OnMouse($ev, $x - $hit['X'] - 1, $y - $hit['Y'] - 1)
                }
            }
            $this.PrevLeft = $leftDown
            return
        }

        if ($leftDown -and $ev.Moved -and $this.Drag) {
            $node = $this.Drag['Node']
            if ($node['Dir'] -eq 'h') {
                $avail = $node['W'] - 1
                if ($avail -ge 2) {
                    $r = ($x - $node['X']) / [double]$avail
                    $node['Ratio'] = [math]::Max(0.1, [math]::Min(0.9, $r))
                }
            } else {
                $avail = $node['H'] - 1
                if ($avail -ge 2) {
                    $r = ($y - $node['Y']) / [double]$avail
                    $node['Ratio'] = [math]::Max(0.1, [math]::Min(0.9, $r))
                }
            }
            $this.PrevLeft = $leftDown
            return
        }

        if ($releaseEdge) { $this.Drag = $null }
        $this.PrevLeft = $leftDown
    }

    # -- rendering --------------------------------------------------------

    [void] Render() {
        $s = $this.Screen
        if (-not $this.FocusedNode()) { $this.FocusId = $this.FirstPaneId($this.Root) }
        $s.Clear([Theme]::Bg)
        $this.ComputeLayout()

        foreach ($pr in $this.PaneRects) {
            $node = $pr['Node']
            $focused = ($node['Id'] -eq $this.FocusId)
            $pane = [MuxPane]$node['Pane']
            $bx = $pr['X']; $by = $pr['Y']; $bw = $pr['W']; $bh = $pr['H']
            $s.FillRect($bx, $by, $bw, $bh, [char]' ', [Theme]::Text, [Theme]::Panel)
            $bc = if ($focused) { [Theme]::BorderFocus } else { [Theme]::Border }
            $s.Box($bx, $by, $bw, $bh, $bc, [Theme]::Panel, 'round')
            $title = if ($focused) { ' * ' + $pane.Title + ' ' } else { ' ' + $pane.Title + ' ' }
            $s.TextMax($bx + 2, $by, $title, $bc, [Theme]::Panel, [math]::Max(0, $bw - 4))
            $ix = $bx + 1; $iy = $by + 1; $iw = $bw - 2; $ih = $bh - 2
            if ($iw -gt 0 -and $ih -gt 0) {
                $pane.Render($s, $ix, $iy, $iw, $ih, $focused)
            }
        }

        foreach ($d in $this.DivRects) {
            $hovered = ($this.Drag -and [object]::ReferenceEquals($this.Drag['Node'], $d['Node']))
            $col = if ($hovered) { [Theme]::Accent } else { [Theme]::Border }
            if ($d['Dir'] -eq 'h') {
                $s.VLine($d['X'], $d['Y'], $d['H'], [char]0x2502, $col, [Theme]::Bg)
                $midy = $d['Y'] + [int]($d['H'] / 2)
                $s.Set($d['X'], $midy, [char]0x2016, $col, [Theme]::Bg)
            } else {
                $s.HLine($d['X'], $d['Y'], $d['W'], [char]0x2500, $col, [Theme]::Bg)
                $midx = $d['X'] + [int]($d['W'] / 2)
                $s.Set($midx, $d['Y'], [char]0x2550, $col, [Theme]::Bg)
            }
        }

        $this.RenderStatus()

        # Modal overlay, drawn last so it covers panes and status bar alike.
        # The pane's own render is guarded: an exception here previously
        # propagated out of the frame loop and killed the workspace, and a
        # silently failing pane produced an empty box. Both now surface as
        # red text inside the overlay itself.
        if ($this.Overlay) {
            $ow = [int]($s.Width * 0.74)
            if ($ow -lt 44) { $ow = [math]::Min($s.Width - 2, 44) }
            $oh = [int]($s.Height * 0.74)
            if ($oh -lt 10) { $oh = [math]::Min($s.Height - 2, 10) }
            $ox = [int](($s.Width - $ow) / 2)
            $oy = [int](($s.Height - $oh) / 2)
            $s.FillRect($ox, $oy, $ow, $oh, [char]' ', [Theme]::Text, [Theme]::Panel)
            $s.Box($ox, $oy, $ow, $oh, [Theme]::BorderFocus, [Theme]::Panel, 'round')
            $otitle = ' ' + ([MuxPane]$this.Overlay).Title + '  (Esc closes) '
            $s.TextMax($ox + 2, $oy, $otitle, [Theme]::BorderFocus, [Theme]::Panel, [math]::Max(0, $ow - 4))
            try {
                ([MuxPane]$this.Overlay).Render($s, $ox + 1, $oy + 1, $ow - 2, $oh - 2, $true)
            } catch {
                $s.TextMax($ox + 2, $oy + 2, ('overlay render error: ' + $_.Exception.Message), [MuxScreen]::Rgb(240, 100, 100), [Theme]::Panel, [math]::Max(0, $ow - 4))
            }
            $this.OverlayRect = @{ X = $ox; Y = $oy; W = $ow; H = $oh }
        } else {
            $this.OverlayRect = $null
        }

        $s.Flush()
    }

    [void] RenderStatus() {
        $s = $this.Screen
        $y = $s.Height - 1
        if ($y -lt 0) { return }
        $s.FillRect(0, $y, $s.Width, 1, [char]' ', [Theme]::StatusFg, [Theme]::StatusBg)
        $fn = $this.FocusedNode()
        $ft = if ($fn) { ([MuxPane]$fn['Pane']).Title } else { '-' }
        $mode = if ($this.Zoom) { '[ZOOM] ' } else { '' }
        $s.TextMax(1, $y, 'ignis-mux', [Theme]::StatusHi, [Theme]::StatusBg, $s.Width)
        $left = $mode + 'focus: ' + $ft
        if ($this.LastInput) { $left += '   in: ' + $this.LastInput }
        if ($this.Overlay) { $left = '[overlay open, Esc closes]   ' + $left }
        $s.TextMax(12, $y, $left, [Theme]::StatusFg, [Theme]::StatusBg, [math]::Max(0, $s.Width - 12))

        $this.Buttons.Clear()
        $defs = @(
            @{ L = ' [ | ] '; A = 'splitH' },
            @{ L = ' [ - ] '; A = 'splitV' },
            @{ L = ' [ z ] '; A = 'zoom' },
            @{ L = ' [ x ] '; A = 'close' },
            @{ L = ' [ q ] '; A = 'quit' }
        )
        $curx = $s.Width - 1
        for ($k = $defs.Count - 1; $k -ge 0; $k--) {
            $lbl = $defs[$k]['L']
            $wid = $lbl.Length
            $startx = $curx - $wid + 1
            if ($startx -lt 0) { break }
            $s.Text($startx, $y, $lbl, [Theme]::StatusHi, [Theme]::StatusBg)
            $this.Buttons.Add(@{ X = $startx; Y = $y; W = $wid; Action = $defs[$k]['A'] })
            $curx = $startx - 1
        }
    }

    # -- main loop --------------------------------------------------------
    #
    # The native interop type IgnisConsole is resolved by name at runtime and
    # its static methods are invoked through the $ic variable, rather than
    # written as the literal [IgnisConsole]. This is deliberate and required:
    # a type literal that appears inside a PowerShell class method body is
    # resolved when the class is defined (at dot-source time), not when the
    # method executes. At dot-source time IgnisConsole does not yet exist,
    # because it is compiled later by Initialize-MuxNative. Resolving the type
    # dynamically defers the lookup to runtime, by which point the caller has
    # already invoked Initialize-MuxNative, so the type is present.
    [void] Run() {
        $ic = [type]'IgnisConsole'
        $ic::Setup()
        $prevCC = [Console]::TreatControlCAsInput
        [Console]::TreatControlCAsInput = $true
        $out = [Console]::Out
        $out.Write("`e[?1049h")   # enter alternate screen buffer
        $out.Write("`e[?25l")     # hide cursor
        $out.Flush()
        try {
            $sz = $ic::GetSize()
            $this.Screen = [MuxScreen]::new($sz[0], $sz[1])
            $this.AttachPanes()
            $this.Running = $true
            while ($this.Running) {
                $sz = $ic::GetSize()
                if ($sz[0] -ne $this.Screen.Width -or $sz[1] -ne $this.Screen.Height) {
                    $this.Screen.Resize($sz[0], $sz[1])
                }
                $this.ComputeLayout()
                foreach ($ev in $ic::Poll()) {
                    if ($ev.Kind -eq 'key') { $this.HandleKey($ev) }
                    elseif ($ev.Kind -eq 'mouse') { $this.HandleMouse($ev) }
                    # resize events are handled by polling GetSize above
                }
                $this.TickAll()
                $this.Render()
                Start-Sleep -Milliseconds 33
            }
        } finally {
            $out.Write("`e[?25h")     # show cursor
            $out.Write("`e[?1049l")   # leave alternate screen buffer
            $out.Flush()
            [Console]::TreatControlCAsInput = $prevCC
            $ic::Restore()
        }
    }
}