#Requires -Version 7.0
#
# _mux_layout.ps1 - Layout persistence for multiplexer workspaces.
#
# MODEL
#   A layout file is a JSON rendering of the binary split tree with panes
#   reduced to their class type names: splits carry Dir and Ratio, leaves
#   carry Type. Geometry fields (X, Y, W, H) and pane state are not saved;
#   geometry is recomputed every frame and pane state is runtime data.
#
#   Reconstruction maps each Type back to a factory through the engine
#   registry (see MuxEngine.Registry), so a layout can only be restored by a
#   command that has registered a factory for every pane type it contains.
#   Any unmapped type aborts the import and the caller falls back to its
#   default layout; a stale file can therefore never produce a broken tree.
#
# FILES
#   Each workspace passes its own path (mux_layout.json, live_layout.json
#   under .ignis_trace), keeping the two workspaces' layouts independent.

function ConvertTo-MuxLayoutNode {
    <#
    .SYNOPSIS
    Reduce a live layout node to its serializable form.
    #>
    param($Node)
    if ($Node['Kind'] -eq 'pane') {
        return @{ Kind = 'pane'; Type = $Node['Pane'].GetType().Name }
    }
    return @{
        Kind = 'split'; Dir = $Node['Dir']; Ratio = [double]$Node['Ratio']
        A = (ConvertTo-MuxLayoutNode $Node['A'])
        B = (ConvertTo-MuxLayoutNode $Node['B'])
    }
}

function Export-MuxLayout {
    <#
    .SYNOPSIS
    Serialize an engine's layout tree to a JSON file. Failures are silent by
    design: losing a layout save must never disturb workspace teardown.
    #>
    param($Root, [string]$Path)
    try {
        $dir = Split-Path $Path
        if ($dir -and -not (Test-Path $dir)) {
            New-Item -ItemType Directory -Path $dir -Force | Out-Null
        }
        $tree = ConvertTo-MuxLayoutNode $Root
        $tree | ConvertTo-Json -Depth 30 | Set-Content -Path $Path -Encoding UTF8
    } catch { }
}

function ConvertFrom-MuxLayoutNode {
    <#
    .SYNOPSIS
    Rebuild one node from its serialized form, or return $null when any pane
    type has no registered factory.
    #>
    param($Json, $Registry, [ref]$Counter)
    if ($Json.Kind -eq 'pane') {
        foreach ($name in $Registry.Keys) {
            $entry = $Registry[$name]
            if ($entry['Type'] -eq $Json.Type) {
                $pane = & ($entry['Make'])
                if ($null -eq $pane) { return $null }
                $Counter.Value++
                return @{ Kind = 'pane'; Id = "r$($Counter.Value)"; Pane = $pane }
            }
        }
        return $null
    }
    $a = ConvertFrom-MuxLayoutNode -Json $Json.A -Registry $Registry -Counter $Counter
    if ($null -eq $a) { return $null }
    $b = ConvertFrom-MuxLayoutNode -Json $Json.B -Registry $Registry -Counter $Counter
    if ($null -eq $b) { return $null }
    $ratio = [double]$Json.Ratio
    if ($ratio -lt 0.1) { $ratio = 0.1 }
    if ($ratio -gt 0.9) { $ratio = 0.9 }
    return @{ Kind = 'split'; Dir = [string]$Json.Dir; Ratio = $ratio; A = $a; B = $b }
}

function Import-MuxLayout {
    <#
    .SYNOPSIS
    Load a layout file and rebuild the tree through the registry. Returns
    the root node, or $null on any failure (missing file, parse error,
    unmapped pane type), in which case the caller uses its default layout.
    #>
    param([string]$Path, $Registry)
    if (-not (Test-Path $Path)) { return $null }
    try {
        $json = Get-Content $Path -Raw | ConvertFrom-Json
        $counter = 0
        return ConvertFrom-MuxLayoutNode -Json $json -Registry $Registry -Counter ([ref]$counter)
    } catch {
        return $null
    }
}