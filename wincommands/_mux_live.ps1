#Requires -Version 7.0
#
# _mux_live.ps1 - Live link consumer and live data panes for the multiplexer.
#
# OVERVIEW
#   Defines the managed layer that turns the raw shared-memory ring exposed
#   by [IgnisShm] into decoded, aggregated state, and the family of MuxPane
#   subclasses that render that state inside the multiplexer. Depends only on
#   the classes in _mux_core.ps1 and the [IgnisShm] interop type, which is
#   referenced through runtime type resolution so parse order across files is
#   irrelevant.
#
# CLASS INVENTORY
#   LiveLinkReader     - connection lifecycle, lossy ring polling, decode,
#                        aggregation; one instance shared by all live panes.
#   LivePane           - base class; pumps the shared reader on tick.
#   LiveStatusPane     - connection, throughput, and drop diagnostics; also
#                        pumps the reader from Render as the tick fallback.
#   LiveEventPane      - filterable scrolling feed (keys 1..7 toggle
#                        categories, arrows and paging keys scroll).
#   LiveMemPane        - allocation totals and per-block bars.
#   LiveValidationPane - deduplicated VUID list; row selection by keyboard
#                        or click, Enter or click opens VuidDetailPane.
#   VuidDetailPane     - modal overlay: full layer message plus the matching
#                        entry from the offline knowledge base when present.
#   LiveGpuPane        - most recent GPU scope durations as ranked bars.
#   LiveSitesPane      - current allocation-site snapshot, rank ordered.
#   LiveHardenedPane   - latest hardened allocator counters.
#   LivePrintfPane     - dedicated shader printf feed.
#
# POLLING AND BACKPRESSURE
#   Poll is rate limited (25 ms floor) and lossy: when the backlog exceeds a
#   256-record budget the oldest records are dropped and the cursor jumps,
#   because a viewer must shed load rather than fall permanently behind.
#   Structural graph kinds are filtered before decode. Feed timestamps are
#   computed once per batch; per-event cmdlet calls were measured to dominate
#   decode cost at producer rates.
#
# KNOWLEDGE BASE CROSS-REFERENCE
#   VuidDetailPane reads $global:IgnisVuidKb (the cache maintained by
#   _vuid_kb.ps1) directly rather than calling its functions, because a
#   global variable is unconditionally visible inside class methods while
#   function visibility depends on the caller's scope chain. The launching
#   command warms the cache once at startup.

# --------------------------------------------------------------------------
# LiveLinkReader
# --------------------------------------------------------------------------

class LiveLinkReader {
    [string]$Name

    [bool]$Connected = $false
    [long]$ReadIdx = 0
    [int]$Capacity = 0
    [int]$Version = 0
    [uint32]$WriterPid = 0
    [long]$HeartbeatNs = 0

    [int]$ConnectAttempts = 0
    [string]$LastFail = ''
    [long]$PollCalls = 0
    [long]$TickCalls = 0
    [long]$Dropped = 0
    [string]$FeedStamp = ''

    [long]$TotalSeen = 0
    [long]$RateCounter = 0
    [double]$EventsPerSec = 0.0

    [System.Diagnostics.Stopwatch]$PollWatch
    [System.Diagnostics.Stopwatch]$RetryWatch
    [System.Diagnostics.Stopwatch]$RateWatch

    [long]$ActiveBytes = 0
    [long]$ActiveAllocs = 0
    [long]$AllocCount = 0
    [long]$FreeCount = 0
    [System.Collections.Generic.Dictionary[string, long]]$MemBlocks

    [long]$SubmissionCount = 0
    [long]$PassCount = 0
    [long]$ValidationCount = 0

    # Main feed entries: @{ T = time; C = color; K = category; X = text }.
    # Categories: gpu, mem, sub, vl, pf, sync, misc.
    [System.Collections.Generic.List[object]]$Feed

    # Dedicated printf ring: @{ T = time; X = text }.
    [System.Collections.Generic.List[object]]$PrintfFeed

    # Deduplicated VUIDs, insertion ordered; values are hashtables with keys
    # Hits, Sev, Msg, Func. Accessed exclusively through the indexer: on a
    # Hashtable, dot notation for the name Count resolves to the read-only
    # ICollection.Count property instead of a key, so the counter key is
    # deliberately named Hits and reads go through ['...'].
    [System.Collections.Specialized.OrderedDictionary]$Vuids

    # Most recent GPU scope duration per label: label -> @{ QF; QI; Dur }.
    [System.Collections.Specialized.OrderedDictionary]$GpuScopes

    [System.Collections.Generic.List[object]]$SyncMarks
    [hashtable]$Hardened = $null
    [long]$SiteEpoch = -1
    [System.Collections.Generic.List[object]]$Sites

    LiveLinkReader([string]$name) {
        $this.Name = $name
        $this.PollWatch = [System.Diagnostics.Stopwatch]::StartNew()
        $this.RetryWatch = [System.Diagnostics.Stopwatch]::StartNew()
        $this.RateWatch = [System.Diagnostics.Stopwatch]::StartNew()
        $this.MemBlocks = [System.Collections.Generic.Dictionary[string, long]]::new()
        $this.Feed = [System.Collections.Generic.List[object]]::new()
        $this.PrintfFeed = [System.Collections.Generic.List[object]]::new()
        $this.Vuids = [System.Collections.Specialized.OrderedDictionary]::new()
        $this.GpuScopes = [System.Collections.Specialized.OrderedDictionary]::new()
        $this.SyncMarks = [System.Collections.Generic.List[object]]::new()
        $this.Sites = [System.Collections.Generic.List[object]]::new()
    }

    # -- static byte helpers ---------------------------------------------

    static [string] Str([byte[]]$b, [int]$off, [int]$len) {
        if ($off -lt 0 -or $off -ge $b.Length) { return '' }
        if ($off + $len -gt $b.Length) { $len = $b.Length - $off }
        if ($len -le 0) { return '' }
        $end = $off
        $max = $off + $len
        while ($end -lt $max -and $b[$end] -ne 0) { $end++ }
        if ($end -eq $off) { return '' }
        return [System.Text.Encoding]::UTF8.GetString($b, $off, $end - $off)
    }

    static [uint32] U32([byte[]]$b, [int]$off) { return [BitConverter]::ToUInt32($b, $off) }
    static [uint64] U64([byte[]]$b, [int]$off) { return [BitConverter]::ToUInt64($b, $off) }
    static [long]   U64L([byte[]]$b, [int]$off) { return [long][BitConverter]::ToUInt64($b, $off) }

    static [string] FmtBytes([long]$n) {
        if ($n -lt 0) { $n = 0 }
        if ($n -ge 1073741824) { return ('{0:N1} GiB' -f ($n / 1073741824.0)) }
        if ($n -ge 1048576)    { return ('{0:N1} MiB' -f ($n / 1048576.0)) }
        if ($n -ge 1024)       { return ('{0:N1} KiB' -f ($n / 1024.0)) }
        return "$n B"
    }

    # -- connection lifecycle --------------------------------------------

    [void] ResetAggregates() {
        $this.TotalSeen = 0
        $this.RateCounter = 0
        $this.EventsPerSec = 0.0
        $this.ActiveBytes = 0
        $this.ActiveAllocs = 0
        $this.AllocCount = 0
        $this.FreeCount = 0
        $this.SubmissionCount = 0
        $this.PassCount = 0
        $this.ValidationCount = 0
        $this.MemBlocks.Clear()
        $this.Feed.Clear()
        $this.PrintfFeed.Clear()
        $this.Vuids.Clear()
        $this.GpuScopes.Clear()
        $this.SyncMarks.Clear()
        $this.Hardened = $null
        $this.SiteEpoch = -1
        $this.Sites.Clear()
    }

    [void] Disconnect() {
        $shm = [type]'IgnisShm'
        try { $shm::Close() } catch { }
        $this.Connected = $false
    }

    [void] TryConnect($shm) {
        $this.ConnectAttempts++
        if (-not $shm::Open($this.Name)) {
            $this.LastFail = ('{0} failed, Win32 error {1}' -f $shm::LastStage(), $shm::LastError())
            return
        }
        try {
            $magic = $shm::ReadI64(0)
            if ($magic -ne [int64]0x49474E5356495A30) {
                $this.LastFail = ('magic mismatch: got 0x{0:x16}' -f $magic)
                $shm::Close()
                return
            }
            $this.Version = $shm::ReadI32(8)
            $cap = $shm::ReadI32(16)
            if ($cap -le 0 -or (($cap -band ($cap - 1)) -ne 0)) {
                $this.LastFail = ('bad capacity {0} (not a positive power of two)' -f $cap)
                $shm::Close()
                return
            }
            $this.Capacity = $cap
            $wi = $shm::ReadI64(24)
            # Start a short distance behind the write tip; decoding a full
            # ring of backlog in interpreted PowerShell stalls the first
            # frame for seconds and provides no value.
            $this.ReadIdx = [math]::Max([long]0, $wi - [long]256)
            $this.ResetAggregates()
            $this.LastFail = ''
            $this.Connected = $true
            $this.AddFeed([MuxScreen]::Rgb(120, 200, 220), 'misc', 'connected to Local\' + $this.Name)
        } catch {
            $this.LastFail = ('exception during connect: ' + $_.Exception.Message)
            $shm::Close()
        }
    }

    # -- feed and marker maintenance -------------------------------------

    [void] AddFeed([int]$color, [string]$cat, [string]$text) {
        if (-not $this.FeedStamp) {
            $this.FeedStamp = [DateTime]::Now.ToString('HH:mm:ss')
        }
        $entry = @{ T = $this.FeedStamp; C = $color; K = $cat; X = $text }
        $this.Feed.Add($entry)
        while ($this.Feed.Count -gt 500) { $this.Feed.RemoveAt(0) }
    }

    [void] ExpireSync() {
        $now = [DateTime]::Now
        for ($i = $this.SyncMarks.Count - 1; $i -ge 0; $i--) {
            if ($this.SyncMarks[$i]['Expire'] -lt $now) {
                $this.SyncMarks.RemoveAt($i)
            }
        }
    }

    # -- polling ---------------------------------------------------------

    [void] Poll() {
        $this.PollCalls++
        if ($this.PollWatch.ElapsedMilliseconds -lt 25) { return }
        $this.PollWatch.Restart()

        try {
            $shm = [type]'IgnisShm'

            if (-not $this.Connected) {
                if ($this.RetryWatch.ElapsedMilliseconds -lt 800) { return }
                $this.RetryWatch.Restart()
                $this.TryConnect($shm)
                return
            }

            $magic = $shm::ReadI64(0)
            if ($magic -ne [int64]0x49474E5356495A30) {
                $this.LastFail = 'magic vanished while connected'
                $this.Disconnect()
                return
            }

            $this.HeartbeatNs = $shm::ReadI64(40)
            $this.WriterPid = [uint32]($shm::ReadI32(12))

            $writeIdx = $shm::ReadI64(24)
            if ($writeIdx -gt $this.ReadIdx) {
                $avail = $writeIdx - $this.ReadIdx

                # Lossy catch-up: shed everything beyond one poll's budget.
                $budget = [long]256
                if ($avail -gt $budget) {
                    $this.Dropped += ($avail - $budget)
                    $this.ReadIdx = $writeIdx - $budget
                    $avail = $budget
                }
                $count = [int]$avail
                $buf = $shm::ReadRange($this.ReadIdx, $count, $this.Capacity)
                $this.FeedStamp = [DateTime]::Now.ToString('HH:mm:ss')

                for ($i = 0; $i -lt $count; $i++) {
                    $recBase = $i * 256
                    $kind = [LiveLinkReader]::U32($buf, $recBase + 16)

                    # Structural graph kinds carry nothing the panes display;
                    # count and skip before any allocation or string work.
                    if ($kind -eq 9 -or $kind -eq 1 -or $kind -eq 2 -or
                        $kind -eq 8 -or $kind -eq 10 -or $kind -eq 13 -or
                        $kind -eq 20 -or $kind -eq 23) {
                        $this.TotalSeen++
                        $this.RateCounter++
                        continue
                    }
                    if ($kind -eq 4) {
                        $this.TotalSeen++
                        $this.RateCounter++
                        $this.PassCount++
                        continue
                    }

                    $idx = $this.ReadIdx + $i
                    $seq = [LiveLinkReader]::U32($buf, $recBase + 20)
                    $expected = [uint32]($idx -band 0xFFFFFFFF)
                    if ($seq -ne $expected) { continue }
                    $this.DecodeRecord($buf, $recBase)
                }
                $this.ReadIdx += $count
            }

            if ($this.RateWatch.ElapsedMilliseconds -ge 1000) {
                $ms = [double]$this.RateWatch.ElapsedMilliseconds
                $this.EventsPerSec = $this.RateCounter * 1000.0 / $ms
                $this.RateCounter = 0
                $this.RateWatch.Restart()
            }

            $this.ExpireSync()
        } catch {
            $this.LastFail = 'poll exception: ' + $_.Exception.Message
            $this.Disconnect()
        }
    }

    # -- record decode ----------------------------------------------------
    #
    # Payload-relative offsets mirror the #[repr(C)] payload structs of
    # live_link.rs, including compiler alignment padding before 8-byte
    # fields. $recBase + 24 is the payload origin.

    [void] DecodeRecord([byte[]]$buf, [int]$recBase) {
        $kind = [LiveLinkReader]::U32($buf, $recBase + 16)
        $p = $recBase + 24
        $this.TotalSeen++
        $this.RateCounter++

        $cGreen  = [MuxScreen]::Rgb(120, 200, 140)
        $cOrange = [MuxScreen]::Rgb(220, 150, 90)
        $cBlue   = [MuxScreen]::Rgb(110, 170, 240)
        $cYellow = [MuxScreen]::Rgb(240, 210, 110)
        $cRed    = [MuxScreen]::Rgb(240, 100, 100)
        $cCyan   = [MuxScreen]::Rgb(120, 200, 220)
        $cDim    = [MuxScreen]::Rgb(130, 133, 145)

        switch ($kind) {
            3 {
                $qf = [LiveLinkReader]::U32($buf, $p + 0)
                $qi = [LiveLinkReader]::U32($buf, $p + 4)
                $dur = [LiveLinkReader]::U64L($buf, $p + 16)
                $lbl = [LiveLinkReader]::Str($buf, $p + 24, 64)
                $this.SubmissionCount++
                $durMs = [math]::Round($dur / 1e6, 2)
                $this.AddFeed($cBlue, 'sub', "submit Q$qf/$qi ${durMs}ms  $lbl")
            }
            5 {
                $mem = [LiveLinkReader]::U64($buf, $p + 0)
                $size = [LiveLinkReader]::U64L($buf, $p + 16)
                $site = [LiveLinkReader]::Str($buf, $p + 24, 64)
                $key = '0x{0:x}' -f $mem
                $this.ActiveBytes += $size
                $this.ActiveAllocs++
                $this.AllocCount++
                if ($this.MemBlocks.ContainsKey($key)) {
                    $this.MemBlocks[$key] = $this.MemBlocks[$key] + $size
                } else {
                    $this.MemBlocks[$key] = $size
                }
                $this.AddFeed($cGreen, 'mem', "alloc $([LiveLinkReader]::FmtBytes($size))  $site")
            }
            6 {
                $mem = [LiveLinkReader]::U64($buf, $p + 0)
                $size = [LiveLinkReader]::U64L($buf, $p + 16)
                $key = '0x{0:x}' -f $mem
                $this.ActiveBytes -= $size
                if ($this.ActiveBytes -lt 0) { $this.ActiveBytes = 0 }
                $this.ActiveAllocs--
                if ($this.ActiveAllocs -lt 0) { $this.ActiveAllocs = 0 }
                $this.FreeCount++
                if ($this.MemBlocks.ContainsKey($key)) {
                    $v = $this.MemBlocks[$key] - $size
                    if ($v -le 0) { [void]$this.MemBlocks.Remove($key) }
                    else { $this.MemBlocks[$key] = $v }
                }
                $this.AddFeed($cOrange, 'mem', "free  $([LiveLinkReader]::FmtBytes($size))  $key")
            }
            11 {
                $sev = [LiveLinkReader]::U32($buf, $p + 0)
                $func = [LiveLinkReader]::Str($buf, $p + 24, 48)
                $vuid = [LiveLinkReader]::Str($buf, $p + 72, 64)
                $msg = [LiveLinkReader]::Str($buf, $p + 136, 96)
                $this.ValidationCount++
                $key = if ($vuid) { $vuid } else { '(no-vuid)' }
                if ($this.Vuids.Contains($key)) {
                    $e = $this.Vuids[$key]
                    $e['Hits'] = $e['Hits'] + 1
                    $e['Sev'] = $sev
                    $e['Msg'] = $msg
                } else {
                    $this.Vuids.Add($key, @{ Hits = 1; Sev = $sev; Msg = $msg; Func = $func })
                    while ($this.Vuids.Count -gt 200) { $this.Vuids.RemoveAt(0) }
                }
                $col = if ($sev -eq 2) { $cRed } elseif ($sev -eq 1) { $cYellow } else { $cCyan }
                $this.AddFeed($col, 'vl', "vl $key")
            }
            12 {
                $qf = [LiveLinkReader]::U32($buf, $p + 0)
                $qi = [LiveLinkReader]::U32($buf, $p + 4)
                $dur = [LiveLinkReader]::U64L($buf, $p + 24)
                $lbl = [LiveLinkReader]::Str($buf, $p + 32, 64)
                if ($this.GpuScopes.Contains($lbl) -or $this.GpuScopes.Count -lt 64) {
                    $this.GpuScopes[$lbl] = @{ QF = $qf; QI = $qi; Dur = $dur }
                }
                $durMs = [math]::Round($dur / 1e6, 3)
                $this.AddFeed($cCyan, 'gpu', "gpu Q$qf/$qi ${durMs}ms  $lbl")
            }
            14 {
                $heap = [LiveLinkReader]::U32($buf, $p + 0)
                $used = [LiveLinkReader]::U64L($buf, $p + 8)
                $budget = [LiveLinkReader]::U64L($buf, $p + 16)
                $this.AddFeed($cDim, 'misc', "budget heap$heap $([LiveLinkReader]::FmtBytes($used)) / $([LiveLinkReader]::FmtBytes($budget))")
            }
            15 {
                $qf = [LiveLinkReader]::U32($buf, $p + 0)
                $qi = [LiveLinkReader]::U32($buf, $p + 4)
                $sev = [LiveLinkReader]::U32($buf, $p + 8)
                $ttl = [LiveLinkReader]::U32($buf, $p + 12)
                $desc = [LiveLinkReader]::Str($buf, $p + 32, 96)
                $key = "$qf/$qi"
                $expire = [DateTime]::Now.AddMilliseconds([double]$ttl)
                $found = $false
                foreach ($m in $this.SyncMarks) {
                    if ($m['Key'] -eq $key) {
                        $m['Sev'] = $sev; $m['Desc'] = $desc; $m['Expire'] = $expire
                        $found = $true; break
                    }
                }
                if (-not $found) {
                    $this.SyncMarks.Add(@{ Key = $key; QF = $qf; QI = $qi; Sev = $sev; Desc = $desc; Expire = $expire })
                }
                $col = if ($sev -ge 2) { $cRed } elseif ($sev -eq 1) { $cYellow } else { $cCyan }
                $label = if ($sev -ge 2) { 'CYCLE' } elseif ($sev -eq 1) { 'ORPHAN' } else { 'sync' }
                $this.AddFeed($col, 'sync', "$label Q$key  $desc")
            }
            16 {
                $desc = [LiveLinkReader]::Str($buf, $p + 144, 80)
                $this.AddFeed($cRed, 'misc', "canary corruption  $desc")
            }
            17 {
                $this.Hardened = @{
                    TotalAllocs = [LiveLinkReader]::U64L($buf, $p + 0)
                    TotalFrees  = [LiveLinkReader]::U64L($buf, $p + 8)
                    ActiveAllocs = [LiveLinkReader]::U64L($buf, $p + 16)
                    ActiveBytes = [LiveLinkReader]::U64L($buf, $p + 24)
                    QuarantineEntries = [LiveLinkReader]::U64L($buf, $p + 32)
                    QuarantineBytes = [LiveLinkReader]::U64L($buf, $p + 40)
                    Corruptions = [LiveLinkReader]::U64L($buf, $p + 48)
                    PeakAllocs = [LiveLinkReader]::U64L($buf, $p + 56)
                    PeakBytes = [LiveLinkReader]::U64L($buf, $p + 64)
                }
            }
            21 {
                $msg = [LiveLinkReader]::Str($buf, $p + 76, 156)
                $this.AddFeed($cCyan, 'pf', "printf  $msg")
                $this.PrintfFeed.Add(@{ T = $this.FeedStamp; X = $msg })
                while ($this.PrintfFeed.Count -gt 300) { $this.PrintfFeed.RemoveAt(0) }
            }
            22 {
                $elapsed = [LiveLinkReader]::U64L($buf, $p + 8)
                $lbl = [LiveLinkReader]::Str($buf, $p + 32, 64)
                $secs = [math]::Round($elapsed / 1e9, 2)
                $this.AddFeed($cRed, 'misc', "HANG ${secs}s  $lbl")
            }
            24 {
                $desc = [LiveLinkReader]::Str($buf, $p + 0, 128)
                $this.AddFeed($cRed, 'misc', "DEVICE FAULT  $desc")
            }
            30 {
                $epoch = [LiveLinkReader]::U64L($buf, $p + 0)
                if ($epoch -ne $this.SiteEpoch) {
                    $this.SiteEpoch = $epoch
                    $this.Sites.Clear()
                }
                $idxv = [LiveLinkReader]::U32($buf, $p + 60)
                $line = [LiveLinkReader]::U32($buf, $p + 56)
                $func = [LiveLinkReader]::Str($buf, $p + 64, 64)
                $file = [LiveLinkReader]::Str($buf, $p + 128, 88)
                $ab = [LiveLinkReader]::U64L($buf, $p + 32)
                $aa = [LiveLinkReader]::U64L($buf, $p + 24)
                $this.Sites.Add(@{ Idx = $idxv; Func = $func; File = $file; Line = $line; Bytes = $ab; Allocs = $aa })
            }
            default {
                $this.AddFeed($cDim, 'misc', "event kind $kind")
            }
        }
    }

    [double] HeartbeatAgeSec() {
        if ($this.HeartbeatNs -le 0) { return -1 }
        $nowMs = [DateTimeOffset]::UtcNow.ToUnixTimeMilliseconds()
        $hbMs = [long]($this.HeartbeatNs / 1000000)
        return [math]::Max(0.0, ($nowMs - $hbMs) / 1000.0)
    }
}

# --------------------------------------------------------------------------
# LivePane (abstract base)
# --------------------------------------------------------------------------

class LivePane : MuxPane {
    [object]$Reader

    LivePane([object]$reader) {
        $this.Reader = $reader
    }

    [void] OnTick() {
        if ($this.Reader) {
            $this.Reader.TickCalls++
            $this.Reader.Poll()
        }
    }
}

# --------------------------------------------------------------------------
# LiveStatusPane
# --------------------------------------------------------------------------

class LiveStatusPane : LivePane {
    LiveStatusPane([object]$reader) : base($reader) {
        $this.Title = 'live status'
    }

    [void] Render([MuxScreen]$s, [int]$x, [int]$y, [int]$w, [int]$h, [bool]$focused) {
        $r = [LiveLinkReader]$this.Reader
        # Render-driven polling: rendering provably executes every frame, so
        # the reader is pumped here in addition to OnTick; its internal rate
        # limit collapses the duplicates.
        $r.Poll()

        $tx = [Theme]::Text
        $td = [Theme]::TextDim
        $th = [Theme]::TextHead
        $line = 0

        $s.TextMax($x, $y, 'Producer : Local\' + $r.Name, $th, [Theme]::Panel, $w); $line++

        if ($r.Connected) {
            $s.TextMax($x, $y + $line, 'State    : CONNECTED', [Theme]::Accent, [Theme]::Panel, $w); $line++
            $s.TextMax($x, $y + $line, ('Writer   : pid ' + $r.WriterPid), $tx, [Theme]::Panel, $w); $line++
            $age = $r.HeartbeatAgeSec()
            $ageStr = if ($age -lt 0) { 'unknown' } else { ('{0:N1}s ago' -f $age) }
            $ageCol = if ($age -ge 3.0) { [MuxScreen]::Rgb(240, 100, 100) } else { $tx }
            $s.TextMax($x, $y + $line, ('Heartbeat: ' + $ageStr), $ageCol, [Theme]::Panel, $w); $line++
            $s.TextMax($x, $y + $line, ('Ring cap : ' + $r.Capacity), $td, [Theme]::Panel, $w); $line++
            $eps = '{0:N0}/s' -f $r.EventsPerSec
            $s.TextMax($x, $y + $line, ('Events   : ' + $r.TotalSeen + '  (' + $eps + ')'), $tx, [Theme]::Panel, $w); $line++
            if ($r.Dropped -gt 0) {
                $s.TextMax($x, $y + $line, ('Dropped  : ' + $r.Dropped + ' (backlog shed)'), $td, [Theme]::Panel, $w); $line++
            }
            $line++
            $s.TextMax($x, $y + $line, ('Active   : ' + [LiveLinkReader]::FmtBytes($r.ActiveBytes) + ' / ' + $r.ActiveAllocs + ' allocs'), $tx, [Theme]::Panel, $w); $line++
            $s.TextMax($x, $y + $line, ('Alloc/Free : ' + $r.AllocCount + ' / ' + $r.FreeCount), $td, [Theme]::Panel, $w); $line++
            $s.TextMax($x, $y + $line, ('Submit/Pass: ' + $r.SubmissionCount + ' / ' + $r.PassCount), $td, [Theme]::Panel, $w); $line++
            $s.TextMax($x, $y + $line, ('Validation : ' + $r.ValidationCount), $td, [Theme]::Panel, $w); $line++
            if ($r.Hardened) {
                $c = $r.Hardened.Corruptions
                $col = if ($c -gt 0) { [MuxScreen]::Rgb(240, 100, 100) } else { [Theme]::Accent }
                $s.TextMax($x, $y + $line, ('Corruptions: ' + $c), $col, [Theme]::Panel, $w); $line++
            }
        } else {
            $s.TextMax($x, $y + $line, 'State    : waiting for producer...', [MuxScreen]::Rgb(240, 210, 110), [Theme]::Panel, $w); $line++
            $s.TextMax($x, $y + $line, ('Attempts : ' + $r.ConnectAttempts), $td, [Theme]::Panel, $w); $line++
            $s.TextMax($x, $y + $line, ('Ticks    : ' + $r.TickCalls + '   Polls: ' + $r.PollCalls), $td, [Theme]::Panel, $w); $line++
            if ($r.LastFail) {
                $s.TextMax($x, $y + $line, ('Reason   : ' + $r.LastFail), [MuxScreen]::Rgb(240, 100, 100), [Theme]::Panel, $w); $line++
                if ($r.LastFail -match 'error 2') {
                    $line++
                    $s.TextMax($x, $y + $line, 'Error 2: mapping name not found.', $td, [Theme]::Panel, $w); $line++
                    $s.TextMax($x, $y + $line, 'Producer not running, name mismatch,', $td, [Theme]::Panel, $w); $line++
                    $s.TextMax($x, $y + $line, 'or a different logon session.', $td, [Theme]::Panel, $w); $line++
                } elseif ($r.LastFail -match 'error 5') {
                    $line++
                    $s.TextMax($x, $y + $line, 'Error 5: access denied. Run producer', $td, [Theme]::Panel, $w); $line++
                    $s.TextMax($x, $y + $line, 'and shell at the same elevation.', $td, [Theme]::Panel, $w); $line++
                }
            }
            $line++
            $s.TextMax($x, $y + $line, 'Expecting a producer that calls', $td, [Theme]::Panel, $w); $line++
            $s.TextMax($x, $y + $line, 'LiveLink::create("' + $r.Name + '", ...)', $td, [Theme]::Panel, $w); $line++
        }
    }
}

# --------------------------------------------------------------------------
# LiveEventPane
# --------------------------------------------------------------------------
# Filterable scrolling feed. The first content row is a filter legend; keys
# 1..7 toggle the corresponding category, arrows and paging keys scroll, End
# snaps to the tail, the wheel scrolls as before. Filtering happens at render
# time over the shared feed, so multiple event panes can carry different
# filter sets over the same data.

class LiveEventPane : LivePane {
    [int]$Scroll = 0
    [hashtable]$F

    static [string[]] $Cats = @('gpu', 'mem', 'sub', 'vl', 'pf', 'sync', 'misc')

    LiveEventPane([object]$reader) : base($reader) {
        $this.Title = 'events'
        $this.F = @{}
        foreach ($c in [LiveEventPane]::Cats) { $this.F[$c] = $true }
    }

    [void] Render([MuxScreen]$s, [int]$x, [int]$y, [int]$w, [int]$h, [bool]$focused) {
        $r = [LiveLinkReader]$this.Reader

        # Filter legend row.
        $lx = $x
        for ($i = 0; $i -lt [LiveEventPane]::Cats.Count; $i++) {
            $cat = [LiveEventPane]::Cats[$i]
            $on = $this.F[$cat]
            $tag = ('' + ($i + 1) + ':' + $cat + ' ')
            $col = if ($on) { [Theme]::Accent } else { [Theme]::TextDim }
            $s.TextMax($lx, $y, $tag, $col, [Theme]::Panel, [math]::Max(0, $x + $w - $lx))
            $lx += $tag.Length
        }

        $rows = $h - 1
        if ($rows -le 0) { return }
        $top = $y + 1

        $vis = [System.Collections.Generic.List[object]]::new()
        foreach ($e in $r.Feed) {
            $k = [string]$e['K']
            if (-not $this.F.ContainsKey($k) -or $this.F[$k]) { $vis.Add($e) }
        }
        $count = $vis.Count
        if ($count -eq 0) {
            $s.TextMax($x, $top, '(no events match the filters)', [Theme]::TextDim, [Theme]::Panel, $w)
            return
        }
        $maxScroll = [math]::Max(0, $count - $rows)
        if ($this.Scroll -gt $maxScroll) { $this.Scroll = $maxScroll }
        if ($this.Scroll -lt 0) { $this.Scroll = 0 }
        $begin = [math]::Max(0, $count - $rows - $this.Scroll)
        for ($i = 0; $i -lt $rows; $i++) {
            $idx = $begin + $i
            if ($idx -ge $count) { break }
            $e = $vis[$idx]
            $prefix = $e['T'] + ' '
            $s.Text($x, $top + $i, $prefix, [Theme]::TextDim, [Theme]::Panel)
            $tw = $w - $prefix.Length
            if ($tw -gt 0) {
                $s.TextMax($x + $prefix.Length, $top + $i, $e['X'], $e['C'], [Theme]::Panel, $tw)
            }
        }
    }

    [bool] OnKey([object]$ev) {
        if ($ev.Ctrl -or -not $ev.KeyDown) { return $false }
        $vk = $ev.VKey
        if ($vk -ge 0x31 -and $vk -le 0x37) {
            $cat = [LiveEventPane]::Cats[$vk - 0x31]
            $this.F[$cat] = -not $this.F[$cat]
            return $true
        }
        switch ($vk) {
            0x26 { $this.Scroll += 1; return $true }                        # Up
            0x28 { $this.Scroll -= 1; if ($this.Scroll -lt 0) { $this.Scroll = 0 }; return $true } # Down
            0x21 { $this.Scroll += 10; return $true }                       # PgUp
            0x22 { $this.Scroll -= 10; if ($this.Scroll -lt 0) { $this.Scroll = 0 }; return $true } # PgDn
            0x24 { $this.Scroll = 1000000; return $true }                   # Home: oldest
            0x23 { $this.Scroll = 0; return $true }                         # End: tail
        }
        return $false
    }

    [bool] OnMouse([object]$ev, [int]$lx, [int]$ly) {
        if ($ev.Wheel -gt 0) { $this.Scroll += 3; return $true }
        if ($ev.Wheel -lt 0) { $this.Scroll -= 3; if ($this.Scroll -lt 0) { $this.Scroll = 0 }; return $true }
        if ($ev.Left -and $ly -eq 0) {
            # Click on the legend row toggles the category under the cursor.
            $lx2 = 0
            for ($i = 0; $i -lt [LiveEventPane]::Cats.Count; $i++) {
                $cat = [LiveEventPane]::Cats[$i]
                $len = ('' + ($i + 1) + ':' + $cat + ' ').Length
                if ($lx -ge $lx2 -and $lx -lt ($lx2 + $len)) {
                    $this.F[$cat] = -not $this.F[$cat]
                    return $true
                }
                $lx2 += $len
            }
        }
        return $false
    }
}

# --------------------------------------------------------------------------
# LiveMemPane
# --------------------------------------------------------------------------

class LiveMemPane : LivePane {
    LiveMemPane([object]$reader) : base($reader) {
        $this.Title = 'memory'
    }

    [void] Render([MuxScreen]$s, [int]$x, [int]$y, [int]$w, [int]$h, [bool]$focused) {
        $r = [LiveLinkReader]$this.Reader
        $line = 0
        $s.TextMax($x, $y, ('active: ' + [LiveLinkReader]::FmtBytes($r.ActiveBytes) + ' / ' + $r.ActiveAllocs + ' allocs'), [Theme]::TextHead, [Theme]::Panel, $w)
        $line = 2

        if ($r.MemBlocks.Count -eq 0) {
            $s.TextMax($x, $y + $line, '(no live blocks)', [Theme]::TextDim, [Theme]::Panel, $w)
        } else {
            $items = @()
            foreach ($k in $r.MemBlocks.Keys) { $items += @{ K = $k; V = $r.MemBlocks[$k] } }
            $items = $items | Sort-Object { $_.V } -Descending
            $maxV = $items[0].V
            if ($maxV -le 0) { $maxV = 1 }
            $labelW = 18
            $barW = [math]::Max(4, $w - $labelW - 12)
            $shown = 0
            foreach ($it in $items) {
                if ($line -ge $h) { break }
                if ($shown -ge 10) { break }
                $lbl = $it.K
                if ($lbl.Length -gt $labelW) { $lbl = $lbl.Substring(0, $labelW) }
                $s.TextMax($x, $y + $line, $lbl, [Theme]::Text, [Theme]::Panel, $labelW)
                $fill = [int][math]::Round($barW * ($it.V / [double]$maxV))
                if ($fill -lt 1) { $fill = 1 }
                $bx = $x + $labelW + 1
                $s.FillRect($bx, $y + $line, $fill, 1, [char]0x2588, [Theme]::Accent, [Theme]::Panel)
                $s.TextMax($bx + $barW + 1, $y + $line, [LiveLinkReader]::FmtBytes($it.V), [Theme]::TextDim, [Theme]::Panel, 11)
                $line++
                $shown++
            }
        }

        if ($r.Hardened -and $line -lt $h - 1) {
            $line++
            $q = 'quarantine: ' + [LiveLinkReader]::FmtBytes($r.Hardened.QuarantineBytes) + ' / ' + $r.Hardened.QuarantineEntries
            $s.TextMax($x, $y + $line, $q, [Theme]::TextDim, [Theme]::Panel, $w)
        }
    }
}

# --------------------------------------------------------------------------
# VuidDetailPane (modal overlay)
# --------------------------------------------------------------------------
# Detail view for one deduplicated validation entry: live wire data (count,
# severity, function, layer message) followed by the matching entry from the
# offline knowledge base when the VUID suffix resolves against the cache in
# $global:IgnisVuidKb. Content is stored as unwrapped segments and wrapped to
# the overlay width at render time; arrows, paging keys, and the wheel
# scroll; Escape (handled by the engine) closes.

class VuidDetailPane : MuxPane {
    # Detail view for one deduplicated validation entry, shown as a modal
    # overlay. Display lines are fully materialized in the constructor as
    # parallel text and color arrays, wrapped at a fixed 72-column width
    # (the overlay is at least that wide on any reasonable terminal, and
    # TextMax truncates gracefully on narrower ones). Nothing is computed
    # per frame beyond slicing by scroll position.
    #
    # Construction rules followed here, learned the hard way in this
    # project: no local variable shares a property name (class parser
    # trap), no automatic variables ($Matches is avoided in favor of an
    # explicit [regex]::Match call), no helper-method indirection, and the
    # knowledge base is read from $global:IgnisVuidKb because a global is
    # unconditionally visible in class method scope.
    [string[]]$LineText
    [int[]]$LineColor
    [int]$Scroll = 0

    VuidDetailPane([string]$vuid, [hashtable]$entry) {
        $this.Title = 'vuid: ' + $vuid

        $bodyT = [System.Collections.Generic.List[string]]::new()
        $bodyC = [System.Collections.Generic.List[int]]::new()

        $sev = 0
        try { $sev = [int]$entry['Sev'] } catch { }
        $sevName = 'INFO'
        $sevCol = [MuxScreen]::Rgb(120, 200, 220)
        if ($sev -eq 2) { $sevName = 'ERROR'; $sevCol = [MuxScreen]::Rgb(240, 100, 100) }
        elseif ($sev -eq 1) { $sevName = 'WARNING'; $sevCol = [MuxScreen]::Rgb(240, 210, 110) }

        $bodyT.Add('severity : ' + $sevName + '   hits: ' + $entry['Hits']); $bodyC.Add($sevCol)
        $bodyT.Add('function : ' + $entry['Func']); $bodyC.Add([Theme]::Text)
        $bodyT.Add(''); $bodyC.Add([Theme]::TextDim)
        $bodyT.Add('Layer message'); $bodyC.Add([Theme]::TextHead)
        foreach ($ln in [VuidDetailPane]::WrapText([string]$entry['Msg'], 72)) {
            $bodyT.Add($ln); $bodyC.Add([Theme]::Text)
        }
        $bodyT.Add(''); $bodyC.Add([Theme]::TextDim)

        $suffix = ''
        $m = [regex]::Match($vuid, '-([^-]+)$')
        if ($m.Success) { $suffix = $m.Groups[1].Value }
        $kbEntry = $null
        if ($suffix -and $global:IgnisVuidKb -and $global:IgnisVuidKb['Entries']) {
            foreach ($e in $global:IgnisVuidKb['Entries']) {
                if ($e.Suffix -eq $suffix) { $kbEntry = $e; break }
            }
        }
        if ($null -ne $kbEntry) {
            $bodyT.Add('Knowledge base [' + $kbEntry.Category + ']  ' + $kbEntry.SpecSection); $bodyC.Add([Theme]::TextDim)
            $bodyT.Add([string]$kbEntry.Title); $bodyC.Add([Theme]::TextHead)
            $bodyT.Add(''); $bodyC.Add([Theme]::TextDim)
            $bodyT.Add('What happened'); $bodyC.Add([MuxScreen]::Rgb(240, 210, 110))
            foreach ($ln in [VuidDetailPane]::WrapText([string]$kbEntry.WhatHappened, 72)) {
                $bodyT.Add($ln); $bodyC.Add([Theme]::Text)
            }
            $bodyT.Add(''); $bodyC.Add([Theme]::TextDim)
            $bodyT.Add('Why Vulkan rejected it'); $bodyC.Add([MuxScreen]::Rgb(240, 210, 110))
            foreach ($ln in [VuidDetailPane]::WrapText([string]$kbEntry.WhyRejected, 72)) {
                $bodyT.Add($ln); $bodyC.Add([Theme]::Text)
            }
            $bodyT.Add(''); $bodyC.Add([Theme]::TextDim)
            $bodyT.Add('Ignis fix'); $bodyC.Add([Theme]::Accent)
            # Fix bodies contain code-style indentation; never rewrap them.
            foreach ($ln in ([string]$kbEntry.IgnisFix -split "`n")) {
                $bodyT.Add($ln); $bodyC.Add([Theme]::Text)
            }
        } else {
            $bodyT.Add('No local knowledge base entry for this VUID.'); $bodyC.Add([Theme]::TextDim)
        }

        $this.LineText = $bodyT.ToArray()
        $this.LineColor = $bodyC.ToArray()
    }

    # Greedy word wrap. Class methods return values verbatim (no pipeline
    # enumeration), so a plain return of the array is correct here.
    static [string[]] WrapText([string]$text, [int]$w) {
        if ($null -eq $text) { return @('') }
        if ($w -lt 4) { return @($text) }
        $acc = [System.Collections.Generic.List[string]]::new()
        foreach ($para in ($text -split "`n")) {
            $cur = ''
            foreach ($word in ($para -split ' ')) {
                if ($cur.Length -eq 0) { $cur = $word }
                elseif (($cur.Length + 1 + $word.Length) -le $w) { $cur = $cur + ' ' + $word }
                else { $acc.Add($cur); $cur = $word }
            }
            $acc.Add($cur)
        }
        return $acc.ToArray()
    }

    [void] Render([MuxScreen]$s, [int]$x, [int]$y, [int]$w, [int]$h, [bool]$focused) {
        $n = 0
        if ($null -ne $this.LineText) { $n = $this.LineText.Count }
        if ($n -eq 0) {
            # This line should be impossible; if it ever appears on screen it
            # proves the constructor produced no content, which localizes any
            # remaining fault to construction rather than rendering.
            $s.TextMax($x, $y, '(detail body is empty: constructor produced no lines)', [MuxScreen]::Rgb(240, 100, 100), [Theme]::Panel, $w)
            return
        }
        $rows = $h - 1
        if ($rows -lt 1) { $rows = 1 }
        $maxScroll = [math]::Max(0, $n - $rows)
        if ($this.Scroll -gt $maxScroll) { $this.Scroll = $maxScroll }
        if ($this.Scroll -lt 0) { $this.Scroll = 0 }
        for ($i = 0; $i -lt $rows; $i++) {
            $idx = $this.Scroll + $i
            if ($idx -ge $n) { break }
            $s.TextMax($x, $y + $i, $this.LineText[$idx], $this.LineColor[$idx], [Theme]::Panel, $w)
        }
        # The footer always renders, so any screenshot of the overlay carries
        # its internal state (total lines and scroll window).
        $foot = 'line ' + ($this.Scroll + 1) + '-' + [math]::Min($n, $this.Scroll + $rows) + ' of ' + $n + '   arrows/wheel scroll, Esc closes'
        $s.TextMax($x, $y + $h - 1, $foot, [Theme]::TextDim, [Theme]::Panel, $w)
    }

    [bool] OnKey([object]$ev) {
        if (-not $ev.KeyDown) { return $false }
        switch ($ev.VKey) {
            0x26 { $this.Scroll -= 1; return $true }
            0x28 { $this.Scroll += 1; return $true }
            0x21 { $this.Scroll -= 10; return $true }
            0x22 { $this.Scroll += 10; return $true }
            0x24 { $this.Scroll = 0; return $true }
            0x23 { $this.Scroll = 1000000; return $true }
        }
        return $true
    }

    [bool] OnMouse([object]$ev, [int]$lx, [int]$ly) {
        if ($ev.Wheel -gt 0) { $this.Scroll -= 3; return $true }
        if ($ev.Wheel -lt 0) { $this.Scroll += 3; return $true }
        return $true
    }
}

# --------------------------------------------------------------------------
# LiveValidationPane
# --------------------------------------------------------------------------

class LiveValidationPane : LivePane {
    [int]$Sel = 0
    [string[]]$View

    LiveValidationPane([object]$reader) : base($reader) {
        $this.Title = 'validation (enter: detail)'
        $this.View = @()
    }

    [void] Render([MuxScreen]$s, [int]$x, [int]$y, [int]$w, [int]$h, [bool]$focused) {
        $r = [LiveLinkReader]$this.Reader
        if ($r.Vuids.Count -eq 0) {
            $this.View = @()
            $s.TextMax($x, $y, '(no validation messages)', [Theme]::TextDim, [Theme]::Panel, $w)
            return
        }
        $items = @()
        foreach ($k in $r.Vuids.Keys) { $items += @{ Vuid = $k; E = $r.Vuids[$k] } }
        $items = $items | Sort-Object { $_['E']['Hits'] } -Descending

        # The accumulator is named rowKeys rather than view. A local whose
        # name matches a class property case-insensitively (here the View
        # property) makes the class compiler misparse an initializer of the
        # form "$view = [T]::new()" as a property assignment and reject the
        # file. Same trap, same rule as documented on MuxScreen.Flush.
        $rowKeys = [System.Collections.Generic.List[string]]::new()
        $line = 0
        foreach ($it in $items) {
            if ($line -ge $h) { break }
            $entry = $it['E']
            $sev = $entry['Sev']
            $col = if ($sev -eq 2) { [MuxScreen]::Rgb(240, 100, 100) } elseif ($sev -eq 1) { [MuxScreen]::Rgb(240, 210, 110) } else { [MuxScreen]::Rgb(120, 200, 220) }
            $tag = if ($sev -eq 2) { 'E' } elseif ($sev -eq 1) { 'W' } else { 'I' }
            $bg = if ($line -eq $this.Sel) { [Theme]::StatusBg } else { [Theme]::Panel }
            if ($line -eq $this.Sel) {
                $s.FillRect($x, $y + $line, $w, 1, [char]' ', $col, $bg)
            }
            $head = ('x{0} [{1}] {2}' -f $entry['Hits'], $tag, $it['Vuid'])
            $s.TextMax($x, $y + $line, $head, $col, $bg, $w)
            $rowKeys.Add([string]$it['Vuid'])
            $line++
        }
        $this.View = $rowKeys.ToArray()
        if ($this.Sel -ge $this.View.Count) { $this.Sel = [math]::Max(0, $this.View.Count - 1) }
    }

    [void] OpenDetail() {
        if ($this.View.Count -eq 0 -or $null -eq $this.Engine) { return }
        if ($this.Sel -lt 0 -or $this.Sel -ge $this.View.Count) { return }
        $key = $this.View[$this.Sel]
        $r = [LiveLinkReader]$this.Reader
        if (-not $r.Vuids.Contains($key)) { return }
        # Construction failures must be visible, not fatal: a throwing
        # constructor previously propagated through the input path and
        # terminated the workspace. Route it into the session log instead.
        try {
            $detail = [VuidDetailPane]::new($key, $r.Vuids[$key])
            $this.Engine.ShowOverlay($detail)
        } catch {
            $this.Engine.LogMsg('vuid detail failed: ' + $_.Exception.Message)
        }
    }

    [bool] OnKey([object]$ev) {
        if ($ev.Ctrl -or -not $ev.KeyDown) { return $false }
        switch ($ev.VKey) {
            0x26 { if ($this.Sel -gt 0) { $this.Sel-- }; return $true }
            0x28 { if ($this.Sel -lt $this.View.Count - 1) { $this.Sel++ }; return $true }
            0x0D { $this.OpenDetail(); return $true }
        }
        return $false
    }

    [bool] OnMouse([object]$ev, [int]$lx, [int]$ly) {
        if ($ev.Left -and $ly -ge 0 -and $ly -lt $this.View.Count) {
            $this.Sel = $ly
            $this.OpenDetail()
            return $true
        }
        return $false
    }
}

# --------------------------------------------------------------------------
# LiveSyncPane
# --------------------------------------------------------------------------

class LiveSyncPane : LivePane {
    LiveSyncPane([object]$reader) : base($reader) {
        $this.Title = 'sync'
    }

    [void] Render([MuxScreen]$s, [int]$x, [int]$y, [int]$w, [int]$h, [bool]$focused) {
        $r = [LiveLinkReader]$this.Reader
        if ($r.SyncMarks.Count -eq 0) {
            $s.TextMax($x, $y, '(no cross-queue issues)', [Theme]::Accent, [Theme]::Panel, $w)
            return
        }
        $now = [DateTime]::Now
        $line = 0
        foreach ($m in $r.SyncMarks) {
            if ($line -ge $h) { break }
            $sev = $m['Sev']
            $col = if ($sev -ge 2) { [MuxScreen]::Rgb(240, 100, 100) } elseif ($sev -eq 1) { [MuxScreen]::Rgb(240, 210, 110) } else { [MuxScreen]::Rgb(120, 200, 220) }
            $label = if ($sev -ge 2) { 'CYCLE ' } elseif ($sev -eq 1) { 'ORPHAN' } else { 'sync  ' }
            $remain = [math]::Max(0.0, ($m['Expire'] - $now).TotalSeconds)
            $head = ('{0} Q{1}  ({2:N0}s)' -f $label, $m['Key'], $remain)
            $s.TextMax($x, $y + $line, $head, $col, [Theme]::Panel, $w)
            $line++
            if ($line -lt $h) {
                $s.TextMax($x + 2, $y + $line, $m['Desc'], [Theme]::TextDim, [Theme]::Panel, [math]::Max(0, $w - 2))
                $line++
            }
        }
    }
}

# --------------------------------------------------------------------------
# LiveGpuPane
# --------------------------------------------------------------------------
# Most recent duration per GPU scope label as ranked bars, longest first.
# Colors alternate by queue family so a queue is identifiable at a glance.

class LiveGpuPane : LivePane {
    LiveGpuPane([object]$reader) : base($reader) {
        $this.Title = 'gpu scopes'
    }

    [void] Render([MuxScreen]$s, [int]$x, [int]$y, [int]$w, [int]$h, [bool]$focused) {
        $r = [LiveLinkReader]$this.Reader
        if ($r.GpuScopes.Count -eq 0) {
            $s.TextMax($x, $y, '(no gpu timestamps yet)', [Theme]::TextDim, [Theme]::Panel, $w)
            return
        }
        $items = @()
        foreach ($k in $r.GpuScopes.Keys) { $items += @{ L = $k; E = $r.GpuScopes[$k] } }
        $items = $items | Sort-Object { $_['E']['Dur'] } -Descending
        $maxDur = [long]$items[0]['E']['Dur']
        if ($maxDur -le 0) { $maxDur = 1 }
        $labelW = 20
        $barW = [math]::Max(4, $w - $labelW - 18)
        $qCols = @(
            [MuxScreen]::Rgb(110, 170, 240),
            [MuxScreen]::Rgb(180, 140, 220),
            [MuxScreen]::Rgb(220, 150, 90),
            [MuxScreen]::Rgb(120, 200, 140)
        )
        $line = 0
        foreach ($it in $items) {
            if ($line -ge $h) { break }
            $e = $it['E']
            $lbl = [string]$it['L']
            if ($lbl.Length -gt $labelW) { $lbl = $lbl.Substring(0, $labelW) }
            $s.TextMax($x, $y + $line, $lbl, [Theme]::Text, [Theme]::Panel, $labelW)
            $qtag = 'Q' + $e['QF'] + '/' + $e['QI']
            $s.TextMax($x + $labelW + 1, $y + $line, $qtag, [Theme]::TextDim, [Theme]::Panel, 5)
            $dur = [long]$e['Dur']
            $fill = [int][math]::Round($barW * ($dur / [double]$maxDur))
            if ($fill -lt 1) { $fill = 1 }
            $col = $qCols[[int]$e['QF'] % $qCols.Count]
            $bx = $x + $labelW + 7
            $s.FillRect($bx, $y + $line, $fill, 1, [char]0x2588, $col, [Theme]::Panel)
            $ms = '{0:N3}ms' -f ($dur / 1e6)
            $s.TextMax($bx + $barW + 1, $y + $line, $ms, [Theme]::TextDim, [Theme]::Panel, 10)
            $line++
        }
    }
}

# --------------------------------------------------------------------------
# LiveSitesPane
# --------------------------------------------------------------------------

class LiveSitesPane : LivePane {
    LiveSitesPane([object]$reader) : base($reader) {
        $this.Title = 'alloc sites'
    }

    [void] Render([MuxScreen]$s, [int]$x, [int]$y, [int]$w, [int]$h, [bool]$focused) {
        $r = [LiveLinkReader]$this.Reader
        if ($r.Sites.Count -eq 0) {
            $s.TextMax($x, $y, '(no allocation site snapshot yet)', [Theme]::TextDim, [Theme]::Panel, $w)
            return
        }
        $items = @($r.Sites | Sort-Object { $_['Idx'] })
        $maxB = [long]0
        foreach ($it in $items) { if ([long]$it['Bytes'] -gt $maxB) { $maxB = [long]$it['Bytes'] } }
        if ($maxB -le 0) { $maxB = 1 }
        $barW = 14
        $line = 0
        foreach ($it in $items) {
            if ($line -ge $h) { break }
            $bytes = [long]$it['Bytes']
            $fill = [int][math]::Round($barW * ($bytes / [double]$maxB))
            if ($fill -lt 1) { $fill = 1 }
            $s.FillRect($x, $y + $line, $fill, 1, [char]0x2588, [Theme]::Accent, [Theme]::Panel)
            $bstr = [LiveLinkReader]::FmtBytes($bytes)
            $s.TextMax($x + $barW + 1, $y + $line, ('{0,10} ' -f $bstr), [Theme]::TextDim, [Theme]::Panel, 11)
            $rest = $w - $barW - 13
            if ($rest -gt 0) {
                $s.TextMax($x + $barW + 13, $y + $line, [string]$it['Func'], [Theme]::Text, [Theme]::Panel, $rest)
            }
            $line++
        }
    }
}

# --------------------------------------------------------------------------
# LiveHardenedPane
# --------------------------------------------------------------------------

class LiveHardenedPane : LivePane {
    LiveHardenedPane([object]$reader) : base($reader) {
        $this.Title = 'hardened'
    }

    [void] Render([MuxScreen]$s, [int]$x, [int]$y, [int]$w, [int]$h, [bool]$focused) {
        $r = [LiveLinkReader]$this.Reader
        $hd = $r.Hardened
        if ($null -eq $hd) {
            $s.TextMax($x, $y, '(no hardened allocator snapshot yet)', [Theme]::TextDim, [Theme]::Panel, $w)
            return
        }
        $rows = @(
            @{ L = 'total allocs'; V = '' + $hd.TotalAllocs; C = [Theme]::Text },
            @{ L = 'total frees '; V = '' + $hd.TotalFrees; C = [Theme]::Text },
            @{ L = 'active      '; V = ('' + $hd.ActiveAllocs + '  (' + [LiveLinkReader]::FmtBytes($hd.ActiveBytes) + ')'); C = [Theme]::Text },
            @{ L = 'quarantine  '; V = ('' + $hd.QuarantineEntries + '  (' + [LiveLinkReader]::FmtBytes($hd.QuarantineBytes) + ')'); C = [Theme]::TextDim },
            @{ L = 'peak        '; V = ('' + $hd.PeakAllocs + '  (' + [LiveLinkReader]::FmtBytes($hd.PeakBytes) + ')'); C = [Theme]::TextDim },
            @{ L = 'corruptions '; V = '' + $hd.Corruptions;
               C = if ($hd.Corruptions -gt 0) { [MuxScreen]::Rgb(240, 100, 100) } else { [Theme]::Accent } }
        )
        $line = 0
        foreach ($row in $rows) {
            if ($line -ge $h) { break }
            $s.TextMax($x, $y + $line, ($row['L'] + ' : ' + $row['V']), [int]$row['C'], [Theme]::Panel, $w)
            $line++
        }
    }
}

# --------------------------------------------------------------------------
# LivePrintfPane
# --------------------------------------------------------------------------

class LivePrintfPane : LivePane {
    [int]$Scroll = 0

    LivePrintfPane([object]$reader) : base($reader) {
        $this.Title = 'printf'
    }

    [void] Render([MuxScreen]$s, [int]$x, [int]$y, [int]$w, [int]$h, [bool]$focused) {
        $r = [LiveLinkReader]$this.Reader
        $feed = $r.PrintfFeed
        $count = $feed.Count
        if ($count -eq 0) {
            $s.TextMax($x, $y, '(no shader printf yet)', [Theme]::TextDim, [Theme]::Panel, $w)
            return
        }
        $maxScroll = [math]::Max(0, $count - $h)
        if ($this.Scroll -gt $maxScroll) { $this.Scroll = $maxScroll }
        if ($this.Scroll -lt 0) { $this.Scroll = 0 }
        $begin = [math]::Max(0, $count - $h - $this.Scroll)
        for ($i = 0; $i -lt $h; $i++) {
            $idx = $begin + $i
            if ($idx -ge $count) { break }
            $e = $feed[$idx]
            $prefix = $e['T'] + ' '
            $s.Text($x, $y + $i, $prefix, [Theme]::TextDim, [Theme]::Panel)
            $tw = $w - $prefix.Length
            if ($tw -gt 0) {
                $s.TextMax($x + $prefix.Length, $y + $i, $e['X'], [MuxScreen]::Rgb(120, 200, 220), [Theme]::Panel, $tw)
            }
        }
    }

    [bool] OnKey([object]$ev) {
        if ($ev.Ctrl -or -not $ev.KeyDown) { return $false }
        switch ($ev.VKey) {
            0x26 { $this.Scroll += 1; return $true }
            0x28 { $this.Scroll -= 1; if ($this.Scroll -lt 0) { $this.Scroll = 0 }; return $true }
            0x21 { $this.Scroll += 10; return $true }
            0x22 { $this.Scroll -= 10; if ($this.Scroll -lt 0) { $this.Scroll = 0 }; return $true }
            0x23 { $this.Scroll = 0; return $true }
        }
        return $false
    }

    [bool] OnMouse([object]$ev, [int]$lx, [int]$ly) {
        if ($ev.Wheel -gt 0) { $this.Scroll += 3; return $true }
        if ($ev.Wheel -lt 0) { $this.Scroll -= 3; if ($this.Scroll -lt 0) { $this.Scroll = 0 }; return $true }
        return $false
    }
}