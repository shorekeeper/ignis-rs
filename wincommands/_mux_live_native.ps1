#Requires -Version 7.0
#
# _mux_live_native.ps1 - Win32 shared-memory interop for the ignis live link.
#
# PURPOSE
#   Provides read-only access to the file-mapping ring buffer written by the
#   ignis crate's live_link.rs producer, plus failure diagnostics: when the
#   open path fails, the failing API name and its Win32 error code are captured
#   and exposed so the consumer can display an actionable reason instead of a
#   silent retry loop.
#
# WIRE CONTRACT (must match live_link.rs exactly)
#   ShmHeader (64 bytes):
#     offset  0  u64  magic              (0x49474E5356495A30)
#     offset  8  u32  version            (1)
#     offset 12  u32  writer_pid
#     offset 16  u32  capacity           (power of two)
#     offset 20  u32  record_size        (256)
#     offset 24  u64  write_idx          (monotonic, Release-published)
#     offset 32  u64  read_idx           (unused by this consumer)
#     offset 40  u64  last_heartbeat_ns  (UNIX epoch nanoseconds)
#     offset 48  u8[16] reserved
#
#   TraceRecord (256 bytes): timestamp_ns u64, thread_id u64, kind u32,
#   seq u32, payload u8[232]. Record N lives at byte offset
#   HEADER_SIZE + (N & (capacity - 1)) * 256.
#
# DIAGNOSTICS
#   Open() records, on failure, the name of the API that failed (LastStage)
#   and Marshal.GetLastWin32Error() at that point (LastError). On success both
#   are cleared. Common LastError values: 2 = name not found (producer not
#   running, name mismatch, or different logon session), 5 = access denied
#   (elevation mismatch between producer and consumer).
#
# CONCURRENCY MODEL
#   Single producer, single reader. The consumer reads write_idx first, then
#   the records strictly below it. Torn slots (overwritten during a wrap) are
#   detected by the managed layer through the per-record seq field; this layer
#   only copies bytes. Naturally aligned 8-byte reads via Marshal.ReadInt64
#   are atomic on x64, so write_idx and last_heartbeat_ns never tear.
#
# LIFETIME
#   A mapped view keeps the shared pages resident even after the producer
#   exits, so the consumer never faults on producer death; it observes a
#   heartbeat that stops advancing instead.
#
# PLATFORM
#   Windows only. Compilation is deferred to Initialize-MuxLiveNative and the
#   Add-Type call is guarded, so repeated calls in one session are no-ops.
#   Note that Add-Type types are immutable per session: after editing this
#   file, start a fresh shell so the new definition compiles.

function Initialize-MuxLiveNative {
    <#
    .SYNOPSIS
    Compile and register the [IgnisShm] interop type if not already present.

    .DESCRIPTION
    Idempotent. Safe to call multiple times per session. Throws only if the
    underlying Add-Type compilation fails, for example on a non-Windows host
    where the referenced kernel32 entry points cannot be resolved.
    #>
    if (([System.Management.Automation.PSTypeName]'IgnisShm').Type) {
        return
    }

    Add-Type -Language CSharp -TypeDefinition @'
using System;
using System.Runtime.InteropServices;

public static class IgnisShm {
    const uint FILE_MAP_READ = 0x0004;
    const int  HEADER_SIZE   = 64;
    const int  RECORD_SIZE   = 256;

    [DllImport("kernel32.dll", SetLastError = true, CharSet = CharSet.Unicode)]
    static extern IntPtr OpenFileMappingW(uint access, bool inherit, string name);
    [DllImport("kernel32.dll", SetLastError = true)]
    static extern IntPtr MapViewOfFile(IntPtr h, uint access, uint offHigh, uint offLow, UIntPtr bytes);
    [DllImport("kernel32.dll", SetLastError = true)]
    static extern bool UnmapViewOfFile(IntPtr addr);
    [DllImport("kernel32.dll", SetLastError = true)]
    static extern bool CloseHandle(IntPtr h);

    static IntPtr handle    = IntPtr.Zero;
    static IntPtr basePtr   = IntPtr.Zero;
    static int    lastErr   = 0;
    static string lastStage = "";

    // Win32 error code captured at the most recent Open failure. 0 after a
    // successful Open.
    public static int LastError() { return lastErr; }

    // Name of the API that failed during the most recent Open attempt, or an
    // empty string after a successful Open.
    public static string LastStage() { return lastStage; }

    // Open the named mapping ("Local\<name>") for reading and map its full
    // view. Returns true on success. Any prior mapping is released first.
    // On failure the failing stage and Win32 error are captured for
    // LastStage / LastError.
    public static bool Open(string name) {
        Close();
        lastErr = 0;
        lastStage = "";
        string qualified = "Local\\" + name;
        handle = OpenFileMappingW(FILE_MAP_READ, false, qualified);
        if (handle == IntPtr.Zero) {
            lastErr = Marshal.GetLastWin32Error();
            lastStage = "OpenFileMappingW";
            return false;
        }
        basePtr = MapViewOfFile(handle, FILE_MAP_READ, 0, 0, UIntPtr.Zero);
        if (basePtr == IntPtr.Zero) {
            lastErr = Marshal.GetLastWin32Error();
            lastStage = "MapViewOfFile";
            CloseHandle(handle);
            handle = IntPtr.Zero;
            return false;
        }
        return true;
    }

    // Whether a view is currently mapped.
    public static bool IsOpen() {
        return basePtr != IntPtr.Zero;
    }

    // Release the mapped view and mapping handle. Safe to call when not open.
    public static void Close() {
        if (basePtr != IntPtr.Zero) {
            UnmapViewOfFile(basePtr);
            basePtr = IntPtr.Zero;
        }
        if (handle != IntPtr.Zero) {
            CloseHandle(handle);
            handle = IntPtr.Zero;
        }
    }

    // Read a signed 64-bit little-endian value at an absolute byte offset.
    // Returns 0 when no view is mapped.
    public static long ReadI64(int off) {
        if (basePtr == IntPtr.Zero) {
            return 0;
        }
        return Marshal.ReadInt64(basePtr, off);
    }

    // Read a signed 32-bit little-endian value at an absolute byte offset.
    // Returns 0 when no view is mapped.
    public static int ReadI32(int off) {
        if (basePtr == IntPtr.Zero) {
            return 0;
        }
        return Marshal.ReadInt32(basePtr, off);
    }

    // Gather 'count' records beginning at logical index 'startIdx' into a
    // flat byte array of length count * 256, resolving each logical index to
    // its physical slot through the power-of-two ring mask. Returns an empty
    // array when no view is mapped or count is not positive.
    public static byte[] ReadRange(long startIdx, int count, int capacity) {
        if (basePtr == IntPtr.Zero || count <= 0 || capacity <= 0) {
            return new byte[0];
        }
        byte[] outb = new byte[count * RECORD_SIZE];
        long mask = (long)capacity - 1L;
        for (int i = 0; i < count; i++) {
            long slot = (startIdx + (long)i) & mask;
            IntPtr src = IntPtr.Add(basePtr, HEADER_SIZE + (int)slot * RECORD_SIZE);
            Marshal.Copy(src, outb, i * RECORD_SIZE, RECORD_SIZE);
        }
        return outb;
    }
}
'@
}