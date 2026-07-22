#Requires -Version 7.0
#
# _mux_native.ps1 - Win32 console interop and compiled screen buffer for the
#                   ignis terminal multiplexer.
#
# PURPOSE
#   Two compiled types delivered through a single guarded Add-Type call:
#
#   IgnisConsole - Raw non-blocking keyboard and mouse input via
#     ReadConsoleInputW, console mode save and restore, and live window size.
#     Unchanged in behavior from the original iteration.
#
#   IgnisScreen - The multiplexer's cell buffer and diff-based flusher,
#     implemented in C# because these are the only per-cell hot loops in the
#     system. Profiling showed that a full-screen diff plus per-cell writes in
#     interpreted PowerShell costs seconds per frame at typical window sizes
#     (roughly 9000 cells), which starved input and collapsed the live link
#     poll rate. The same work compiled runs in well under a millisecond.
#     PowerShell retains all layout, focus, and pane logic; only Set, Text,
#     FillRect, Box, HLine, VLine, Clear, Resize, and Flush execute natively.
#
# RENDERING CONTRACT (IgnisScreen)
#   Cells carry a char plus 24-bit packed foreground and background integers
#   (0xRRGGBB); the sentinel -1 denotes the terminal default color. Flush
#   reconciles a front buffer against the back buffer and emits the minimal
#   VT sequence (cursor moves and SGR color changes) directly to the console,
#   then updates the front buffer. Resize reallocates both buffers, poisons
#   the front buffer so the next Flush repaints everything, and clears the
#   physical screen. All drawing methods clip against the buffer bounds.
#
# CONSOLE MODE CONTRACT (IgnisConsole)
#   Setup clears ENABLE_QUICK_EDIT_MODE, ENABLE_LINE_INPUT, ENABLE_ECHO_INPUT
#   and ENABLE_PROCESSED_INPUT on the input handle and enables window and
#   mouse input, so keys arrive immediately and mouse events reach the
#   application instead of starting a selection. The output mode gains
#   ENABLE_VIRTUAL_TERMINAL_PROCESSING. Both original modes are captured on
#   Setup and reinstated on Restore.
#
# SESSION NOTE
#   Add-Type types are immutable within a PowerShell session. After editing
#   this file, close every pwsh window and start fresh; a session holding the
#   previous definition of either type cannot compile the new one.
#
# PLATFORM
#   Windows only. Compilation is deferred until Initialize-MuxNative is
#   called so that dot-sourcing this file during normal shell operation stays
#   cheap. The call is idempotent within a session.

function Initialize-MuxNative {
    <#
    .SYNOPSIS
    Compile and register the [IgnisConsole] and [IgnisScreen] interop types
    if not already present.

    .DESCRIPTION
    Idempotent. Safe to call multiple times per session. Throws only if the
    underlying Add-Type compilation fails, for example on a non-Windows host,
    or when a stale session already holds an older definition of one of the
    types (start a fresh shell in that case).
    #>
    if (([System.Management.Automation.PSTypeName]'IgnisConsole').Type -and
        ([System.Management.Automation.PSTypeName]'IgnisScreen').Type) {
        return
    }

    Add-Type -Language CSharp -TypeDefinition @'
using System;
using System.Runtime.InteropServices;
using System.Collections.Generic;

// Simplified, PowerShell-friendly input event. All fields are public so the
// managed side can read them by name without reflection helpers.
public class IgnisInputEvent {
    public string Kind;       // "key", "mouse", or "resize"

    // Key fields (Kind == "key")
    public bool KeyDown;
    public int  VKey;         // Win32 virtual key code
    public char KeyChar;      // decoded Unicode char, may be '\0'
    public bool Ctrl;
    public bool Alt;
    public bool Shift;

    // Mouse fields (Kind == "mouse")
    public int  X;            // 0-based cell column
    public int  Y;            // 0-based cell row
    public bool Left;
    public bool Right;
    public bool Middle;
    public bool Moved;        // motion event
    public bool DoubleClick;  // double click event
    public int  Wheel;        // -1 down, 0 none, +1 up

    // Resize fields (Kind == "resize")
    public int W;
    public int H;
}

public static class IgnisConsole {
    const int STD_INPUT_HANDLE  = -10;
    const int STD_OUTPUT_HANDLE = -11;

    [StructLayout(LayoutKind.Sequential)]
    struct COORD { public short X; public short Y; }

    [StructLayout(LayoutKind.Sequential)]
    struct SMALL_RECT { public short Left; public short Top; public short Right; public short Bottom; }

    [StructLayout(LayoutKind.Sequential)]
    struct CONSOLE_SCREEN_BUFFER_INFO {
        public COORD dwSize;
        public COORD dwCursorPosition;
        public ushort wAttributes;
        public SMALL_RECT srWindow;
        public COORD dwMaximumWindowSize;
    }

    [StructLayout(LayoutKind.Sequential)]
    struct KEY_EVENT_RECORD {
        public int    bKeyDown;
        public ushort wRepeatCount;
        public ushort wVirtualKeyCode;
        public ushort wVirtualScanCode;
        public ushort UnicodeChar;
        public uint   dwControlKeyState;
    }

    [StructLayout(LayoutKind.Sequential)]
    struct MOUSE_EVENT_RECORD {
        public short MouseX;
        public short MouseY;
        public uint  dwButtonState;
        public uint  dwControlKeyState;
        public uint  dwEventFlags;
    }

    [StructLayout(LayoutKind.Sequential)]
    struct WINDOW_BUFFER_SIZE_RECORD { public short X; public short Y; }

    [StructLayout(LayoutKind.Explicit)]
    struct INPUT_RECORD {
        [FieldOffset(0)] public ushort EventType;
        [FieldOffset(4)] public KEY_EVENT_RECORD KeyEvent;
        [FieldOffset(4)] public MOUSE_EVENT_RECORD MouseEvent;
        [FieldOffset(4)] public WINDOW_BUFFER_SIZE_RECORD WindowBufferSizeEvent;
    }

    [DllImport("kernel32.dll", SetLastError = true)]
    static extern IntPtr GetStdHandle(int nStdHandle);
    [DllImport("kernel32.dll", SetLastError = true)]
    static extern bool GetConsoleMode(IntPtr h, out uint mode);
    [DllImport("kernel32.dll", SetLastError = true)]
    static extern bool SetConsoleMode(IntPtr h, uint mode);
    [DllImport("kernel32.dll", SetLastError = true)]
    static extern bool GetNumberOfConsoleInputEvents(IntPtr h, out uint n);
    [DllImport("kernel32.dll", SetLastError = true)]
    static extern bool ReadConsoleInputW(IntPtr h, [Out] INPUT_RECORD[] buf, uint len, out uint read);
    [DllImport("kernel32.dll", SetLastError = true)]
    static extern bool FlushConsoleInputBuffer(IntPtr h);
    [DllImport("kernel32.dll", SetLastError = true)]
    static extern bool GetConsoleScreenBufferInfo(IntPtr h, out CONSOLE_SCREEN_BUFFER_INFO info);

    static IntPtr hIn  = IntPtr.Zero;
    static IntPtr hOut = IntPtr.Zero;
    static uint savedIn  = 0;
    static uint savedOut = 0;
    static bool active   = false;

    const uint ENABLE_WINDOW_INPUT                = 0x0008;
    const uint ENABLE_MOUSE_INPUT                 = 0x0010;
    const uint ENABLE_EXTENDED_FLAGS              = 0x0080;
    const uint ENABLE_PROCESSED_OUTPUT            = 0x0001;
    const uint ENABLE_VIRTUAL_TERMINAL_PROCESSING = 0x0004;

    public static void Setup() {
        hIn  = GetStdHandle(STD_INPUT_HANDLE);
        hOut = GetStdHandle(STD_OUTPUT_HANDLE);
        GetConsoleMode(hIn,  out savedIn);
        GetConsoleMode(hOut, out savedOut);

        uint inMode = ENABLE_EXTENDED_FLAGS | ENABLE_WINDOW_INPUT | ENABLE_MOUSE_INPUT;
        SetConsoleMode(hIn, inMode);

        uint outMode = savedOut | ENABLE_PROCESSED_OUTPUT | ENABLE_VIRTUAL_TERMINAL_PROCESSING;
        SetConsoleMode(hOut, outMode);

        FlushConsoleInputBuffer(hIn);
        active = true;
    }

    public static void Restore() {
        if (!active) return;
        SetConsoleMode(hIn,  savedIn);
        SetConsoleMode(hOut, savedOut);
        active = false;
    }

    public static int[] GetSize() {
        CONSOLE_SCREEN_BUFFER_INFO info;
        if (!GetConsoleScreenBufferInfo(hOut, out info)) {
            return new int[] { 80, 25 };
        }
        int w = info.srWindow.Right - info.srWindow.Left + 1;
        int h = info.srWindow.Bottom - info.srWindow.Top + 1;
        if (w < 1) w = 1;
        if (h < 1) h = 1;
        return new int[] { w, h };
    }

    public static IgnisInputEvent[] Poll() {
        uint n;
        if (!GetNumberOfConsoleInputEvents(hIn, out n) || n == 0) {
            return new IgnisInputEvent[0];
        }
        INPUT_RECORD[] buf = new INPUT_RECORD[n];
        uint read;
        if (!ReadConsoleInputW(hIn, buf, n, out read) || read == 0) {
            return new IgnisInputEvent[0];
        }

        List<IgnisInputEvent> outl = new List<IgnisInputEvent>();
        for (uint i = 0; i < read; i++) {
            INPUT_RECORD r = buf[i];

            if (r.EventType == 0x0001) {
                KEY_EVENT_RECORD k = r.KeyEvent;
                IgnisInputEvent e = new IgnisInputEvent();
                e.Kind    = "key";
                e.KeyDown = k.bKeyDown != 0;
                e.VKey    = k.wVirtualKeyCode;
                e.KeyChar = (char)k.UnicodeChar;
                uint c = k.dwControlKeyState;
                e.Ctrl  = (c & 0x000C) != 0;
                e.Alt   = (c & 0x0003) != 0;
                e.Shift = (c & 0x0010) != 0;
                outl.Add(e);
            }
            else if (r.EventType == 0x0002) {
                MOUSE_EVENT_RECORD m = r.MouseEvent;
                IgnisInputEvent e = new IgnisInputEvent();
                e.Kind        = "mouse";
                e.X           = m.MouseX;
                e.Y           = m.MouseY;
                e.Left        = (m.dwButtonState & 0x1) != 0;
                e.Right       = (m.dwButtonState & 0x2) != 0;
                e.Middle      = (m.dwButtonState & 0x4) != 0;
                e.Moved       = (m.dwEventFlags  & 0x1) != 0;
                e.DoubleClick = (m.dwEventFlags  & 0x2) != 0;
                if ((m.dwEventFlags & 0x4) != 0) {
                    short delta = (short)(m.dwButtonState >> 16);
                    e.Wheel = delta > 0 ? 1 : -1;
                }
                outl.Add(e);
            }
            else if (r.EventType == 0x0004) {
                IgnisInputEvent e = new IgnisInputEvent();
                e.Kind = "resize";
                e.W = r.WindowBufferSizeEvent.X;
                e.H = r.WindowBufferSizeEvent.Y;
                outl.Add(e);
            }
        }
        return outl.ToArray();
    }
}

// Compiled cell buffer and diff flusher. See the file header for the
// rendering contract and the rationale for implementing this natively.
public class IgnisScreen {
    int W;
    int H;
    char[] ch;
    int[] fg;
    int[] bg;
    char[] fch;
    int[] ffg;
    int[] fbg;
    System.Text.StringBuilder sb = new System.Text.StringBuilder(16384);

    public IgnisScreen(int w, int h) { Resize(w, h); }

    public int Width  { get { return W; } }
    public int Height { get { return H; } }

    // Reallocate buffers, poison the front buffer to force a full repaint on
    // the next Flush, and clear the physical screen so stale content from a
    // larger prior size is removed.
    public void Resize(int w, int h) {
        if (w < 1) w = 1;
        if (h < 1) h = 1;
        W = w; H = h;
        int n = w * h;
        ch = new char[n]; fg = new int[n]; bg = new int[n];
        fch = new char[n]; ffg = new int[n]; fbg = new int[n];
        for (int i = 0; i < n; i++) { ffg[i] = -2; fbg[i] = -2; }
        Console.Out.Write("\x1b[2J");
    }

    // Reset the back buffer to spaces with default foreground and the given
    // background. Called once per frame before drawing panes.
    public void Clear(int background) {
        Array.Fill(ch, ' ');
        Array.Fill(fg, -1);
        Array.Fill(bg, background);
    }

    // Write one cell, clipped to buffer bounds.
    public void Set(int x, int y, char c, int f, int b) {
        if (x < 0 || y < 0 || x >= W || y >= H) return;
        int i = y * W + x;
        ch[i] = c; fg[i] = f; bg[i] = b;
    }

    // Write a string left to right. When max is non-negative and the string
    // exceeds it, the string is truncated and its last visible cell replaced
    // with '~' to indicate elision. Pass max = -1 for no limit.
    public void Text(int x, int y, string s, int f, int b, int max) {
        if (s == null) return;
        string t = s;
        if (max >= 0 && t.Length > max) {
            if (max >= 1) t = t.Substring(0, max - 1) + "~";
            else t = "";
        }
        for (int k = 0; k < t.Length; k++) Set(x + k, y, t[k], f, b);
    }

    // Fill a rectangular region with a single cell value. Row segments use
    // Array.Fill, which is the reason this method is native at all.
    public void FillRect(int x, int y, int w, int h, char c, int f, int b) {
        if (w <= 0 || h <= 0) return;
        int x0 = Math.Max(0, x);
        int y0 = Math.Max(0, y);
        int x1 = Math.Min(W, x + w);
        int y1 = Math.Min(H, y + h);
        if (x1 <= x0 || y1 <= y0) return;
        int len = x1 - x0;
        for (int yy = y0; yy < y1; yy++) {
            int start = yy * W + x0;
            Array.Fill(ch, c, start, len);
            Array.Fill(fg, f, start, len);
            Array.Fill(bg, b, start, len);
        }
    }

    public void HLine(int x, int y, int w, char c, int f, int b) {
        for (int k = 0; k < w; k++) Set(x + k, y, c, f, b);
    }

    public void VLine(int x, int y, int h, char c, int f, int b) {
        for (int j = 0; j < h; j++) Set(x, y + j, c, f, b);
    }

    // One-cell-thick box border. round != 0 selects rounded corners.
    public void Box(int x, int y, int w, int h, int f, int b, int round) {
        if (w < 2 || h < 2) return;
        char tl, tr, bl, br;
        char hz = '\u2500', vt = '\u2502';
        if (round != 0) { tl = '\u256D'; tr = '\u256E'; bl = '\u2570'; br = '\u256F'; }
        else            { tl = '\u250C'; tr = '\u2510'; bl = '\u2514'; br = '\u2518'; }
        Set(x, y, tl, f, b);
        Set(x + w - 1, y, tr, f, b);
        Set(x, y + h - 1, bl, f, b);
        Set(x + w - 1, y + h - 1, br, f, b);
        for (int k = 1; k < w - 1; k++) {
            Set(x + k, y, hz, f, b);
            Set(x + k, y + h - 1, hz, f, b);
        }
        for (int j = 1; j < h - 1; j++) {
            Set(x, y + j, vt, f, b);
            Set(x + w - 1, y + j, vt, f, b);
        }
    }

    // Reconcile the front buffer to the back buffer, emitting the minimal VT
    // sequence directly to the console, and update the front buffer to match.
    // Skips the write entirely when no cell changed.
    public void Flush() {
        sb.Length = 0;
        sb.Append("\x1b[0m");
        int curF = int.MinValue, curB = int.MinValue;
        bool cursorSet = false;
        int nx = -1, ny = -1;
        for (int y = 0; y < H; y++) {
            int rowBase = y * W;
            for (int x = 0; x < W; x++) {
                int i = rowBase + x;
                char c = ch[i];
                int f = fg[i];
                int b = bg[i];
                if (c == fch[i] && f == ffg[i] && b == fbg[i]) continue;
                if (!(cursorSet && ny == y && nx == x)) {
                    sb.Append("\x1b[").Append(y + 1).Append(';').Append(x + 1).Append('H');
                }
                if (f != curF) {
                    if (f == -1) sb.Append("\x1b[39m");
                    else sb.Append("\x1b[38;2;").Append((f >> 16) & 255).Append(';').Append((f >> 8) & 255).Append(';').Append(f & 255).Append('m');
                    curF = f;
                }
                if (b != curB) {
                    if (b == -1) sb.Append("\x1b[49m");
                    else sb.Append("\x1b[48;2;").Append((b >> 16) & 255).Append(';').Append((b >> 8) & 255).Append(';').Append(b & 255).Append('m');
                    curB = b;
                }
                sb.Append(c);
                fch[i] = c; ffg[i] = f; fbg[i] = b;
                nx = x + 1; ny = y;
                cursorSet = true;
            }
        }
        if (sb.Length > 4) {
            Console.Out.Write(sb.ToString());
            Console.Out.Flush();
        }
    }
}
'@
}