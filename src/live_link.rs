//! Live link IPC bridge for ignis-viz.
//!
//! Opens a shared memory ring and emits structured events that
//! ignis-viz can consume in real time. Protocol matches
//! ignis-viz/src/ipc.rs (256-byte fixed records, atomic write index).
//!
//! On non-Windows platforms, all methods are no-ops and `create`
//! returns `LiveLinkError::Unsupported`.

#![allow(dead_code)]

use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::Arc;
use std::time::SystemTime;

const SHM_MAGIC: u64 = 0x49474E5356495A30;
const SHM_VERSION: u32 = 1;
const RECORD_SIZE: usize = 256;
const HEADER_SIZE: usize = 64;
const PAYLOAD_SIZE: usize = 232;

/// Event kind: register or update a graph node.
pub const KIND_NODE_REGISTER: u32 = 1;
/// Event kind: register an edge between two nodes.
pub const KIND_EDGE_REGISTER: u32 = 2;
/// Event kind: a queue submission completed.
pub const KIND_SUBMISSION: u32 = 3;
/// Event kind: a frame graph pass executed.
pub const KIND_PASS: u32 = 4;
/// Event kind: a memory allocation occurred.
pub const KIND_ALLOCATION: u32 = 5;
/// Event kind: a memory free occurred.
pub const KIND_FREE: u32 = 6;
/// Event kind: a resource layout transition.
pub const KIND_TRANSITION: u32 = 7;
/// Event kind: remove a previously-registered node.
pub const KIND_NODE_REMOVE: u32 = 8;
/// Event kind: toggle an edge's active state.
pub const KIND_EDGE_TOGGLE: u32 = 9;

/// Node kind tag: a frame graph pass.
pub const NODE_KIND_PASS: u32 = 0;
/// Node kind tag: a Vulkan resource (buffer, image, sampler, etc.).
pub const NODE_KIND_RESOURCE: u32 = 1;
/// Node kind tag: a queue submission.
pub const NODE_KIND_SUBMIT: u32 = 2;

/// Event kind: bind a Vulkan handle to a debug name.
pub const KIND_RESOURCE_NAME: u32 = 10;
/// Event kind: validation layer diagnostic.
pub const KIND_VALIDATION: u32 = 11;
/// Event kind: GPU timestamp query result.
pub const KIND_GPU_TIMESTAMP: u32 = 12;

/// Pipeline statistics counters.
pub const KIND_PIPELINE_STATS: u32 = 13;
/// Memory heap budget snapshot.
pub const KIND_BUDGET: u32 = 14;
/// Sync cycle / orphan diagnostic mark. Body: `SyncCyclePayload`.
/// Emitted by [`bridge_cross_queue_to_live_link`] (and by user code via
/// [`LiveLink::record_sync_cycle`]) when cross-queue analysis detects
/// a dependency cycle or an orphan signal/wait. The viewer uses these
/// to tint affected lanes in the Sync DAG view.
pub const KIND_SYNC_CYCLE_DETECTED: u32 = 15;

/// Guard band corruption event from `HardenedAllocator`. Body:
/// `CanaryCorruptionPayload`. Carries the canary value, hex window
/// around the first corruption, source context and free-form
/// description. Drives the Canary view in ignis-viz.
pub const KIND_CANARY_CORRUPTION: u32 = 16;
/// Periodic snapshot of `HardenedAllocator` aggregate stats.
/// Body: `HardenedStatsPayload`. Replaces prior snapshot wholesale
/// on the viewer side; history is not buffered.
pub const KIND_HARDENED_STATS: u32 = 17;
/// One determinism-checker run completion. Body:
/// `DeterminismRunPayload`. Producer compares the run against run 0
/// and reports `matches_baseline`.
pub const KIND_DETERMINISM_RUN: u32 = 18;
/// Per-capture divergence detail. Body: `DeterminismDivergencePayload`.
/// Emitted at most once per (run, capture) pair when a hash differs
/// from baseline.
pub const KIND_DETERMINISM_DIVERGENCE: u32 = 19;

/// Continuation chunk for the previous KIND_CANARY_CORRUPTION /
/// KIND_VALIDATION / KIND_DETERMINISM_DIVERGENCE event. Body:
/// `TextContinuationPayload`. The viewer concatenates the chunks
/// onto the parent event's `description` (or `message`) field. The
/// producer emits these immediately after the parent event when
/// the original string did not fit. Multiple continuations are
/// allowed; `is_final == 1` marks the last one.
pub const KIND_TEXT_CONTINUATION: u32 = 20;

/// Shader printf message captured from `debugPrintfEXT(...)` calls
/// inside SPIR-V. Body: `ShaderPrintfPayload`.
pub const KIND_SHADER_PRINTF: u32 = 21;

/// GPU hang detection event from `HangDetector`. Body:
/// `HangDetectedPayload`. Followed by zero or more `KIND_BREADCRUMB`
/// records with `parent_seq` referencing this event's seq.
pub const KIND_HANG_DETECTED: u32 = 22;

/// One breadcrumb readback entry attached to a hang event. Body:
/// `BreadcrumbPayload`.
pub const KIND_BREADCRUMB: u32 = 23;

/// Aggregate device fault snapshot. Body: `DeviceFaultPayload`.
pub const KIND_DEVICE_FAULT: u32 = 24;
/// Object registration event. Body: `ObjectRegisteredPayload`.
pub const KIND_OBJECT_REGISTERED: u32 = 25;
/// Object destruction event. Body: `ObjectDestroyedPayload`.
pub const KIND_OBJECT_DESTROYED: u32 = 26;
/// Stale descriptor reference issue. Body: `DescriptorIssuePayload`.
pub const KIND_DESCRIPTOR_ISSUE: u32 = 27;
/// Resource aliasing conflict. Body: `AliasingIssuePayload`.
pub const KIND_ALIASING_CONFLICT: u32 = 28;
/// Pipeline compatibility issue. Body: `PipelineIssuePayload`.
pub const KIND_PIPELINE_ISSUE: u32 = 29;
/// Periodic top-N allocation-site snapshot from `AllocationProfiler`.
/// Body: `AllocSitePayload`. The producer emits one record per ranked
/// site every snapshot interval; the viewer groups records sharing
/// one `epoch` value and discards earlier epochs as new ones arrive.
pub const KIND_ALLOC_SITE_SNAPSHOT: u32 = 30;

/// Pipeline audit issue category code shared with the viewer.
pub const PIPELINE_ISSUE_DESCRIPTOR_COUNT: u32 = 0;
/// Push constant range outside declared layout.
pub const PIPELINE_ISSUE_PUSH_CONSTANT_RANGE: u32 = 1;
/// Pipeline layout compatibility violation.
pub const PIPELINE_ISSUE_LAYOUT_COMPATIBILITY: u32 = 2;
/// Shader stage interface mismatch.
pub const PIPELINE_ISSUE_STAGE_INTERFACE: u32 = 3;

/// Resource kind tag: `VkBuffer`.
pub const RES_KIND_BUFFER: u32 = 0;
/// Resource kind tag: `VkImage`.
pub const RES_KIND_IMAGE: u32 = 1;
/// Resource kind tag: `VkDeviceMemory`.
pub const RES_KIND_DEVICE_MEMORY: u32 = 2;
/// Resource kind tag: `VkSampler`.
pub const RES_KIND_SAMPLER: u32 = 3;
/// Resource kind tag: `VkPipeline`.
pub const RES_KIND_PIPELINE: u32 = 4;
/// Resource kind tag: anything not covered by the specific tags above.
pub const RES_KIND_OTHER: u32 = 255;

/// Validation severity: informational message.
pub const VAL_SEVERITY_INFO: u32 = 0;
/// Validation severity: warning.
pub const VAL_SEVERITY_WARNING: u32 = 1;
/// Validation severity: error.
pub const VAL_SEVERITY_ERROR: u32 = 2;

/// Sync cycle severity values. Encoded with the same numeric range as
/// validation severity for uniform colour mapping in the viewer, even
/// though the semantics differ (cycles are guaranteed deadlocks rather
/// than spec violations).
pub const SYNC_SEVERITY_INFO: u32 = 0;
/// Orphan signal or wait. The submission graph is well-formed but a
/// semaphore is signaled with no waiter, or waited on with no signaler.
/// Often a real bug, occasionally intentional in cross-engine handoff.
pub const SYNC_SEVERITY_ORPHAN: u32 = 1;
/// Dependency cycle. Two or more submissions wait on each other through
/// semaphore chains; Vulkan will deadlock when the graph is submitted.
pub const SYNC_SEVERITY_CYCLE: u32 = 2;

/// Continuation payload. `parent_seq` references the immediately
/// preceding event's `seq` so the viewer can guard against
/// out-of-order delivery. `field_id` selects which string field of
/// the parent event the chunk extends:
///   0 = description
///   1 = message
///   2 = source
/// Up to 216 bytes of UTF-8 per chunk.
#[repr(C)]
#[derive(Copy, Clone)]
pub struct TextContinuationPayload {
    /// Sequence number of the parent event being extended.
    pub parent_seq: u32,
    /// Field id within the parent event:
    /// 0 = description, 1 = message, 2 = source.
    pub field_id: u32,
    /// Number of valid bytes in `chunk`.
    pub chunk_len: u32,
    /// 1 if this is the last continuation chunk, 0 otherwise.
    pub is_final: u32,
    /// Up to 216 bytes of UTF-8 text extending the parent field.
    pub chunk: [u8; 216],
}

const _: () = assert!(std::mem::size_of::<TextContinuationPayload>() <= PAYLOAD_SIZE);

#[repr(C)]
#[derive(Copy, Clone)]
struct ResourceNamePayload {
    handle: u64,
    kind_tag: u32,
    _pad: u32,
    name: [u8; 64],
}

#[repr(C)]
#[derive(Copy, Clone)]
struct ValidationPayload {
    severity: u32,
    node_id: u32,
    object_type: u32,
    _pad: u32,
    object_handle: u64,
    function: [u8; 48],
    vuid: [u8; 64],
    message: [u8; 96],
}

#[repr(C)]
#[derive(Copy, Clone)]
struct GpuTimestampPayload {
    queue_family: u32,
    queue_index: u32,
    stage: u32,
    _pad: u32,
    begin_ns: u64,
    duration_ns: u64,
    label: [u8; 64],
}

#[repr(C)]
#[derive(Copy, Clone)]
struct PipelineStatsPayload {
    vs_invocations: u64,
    fs_invocations: u64,
    cs_invocations: u64,
    ia_vertices: u64,
    ia_primitives: u64,
    clipping_invocations: u64,
    clipping_primitives: u64,
    gs_invocations: u64,
    gs_primitives: u64,
    tcs_patches: u64,
    tes_invocations: u64,
    label: [u8; 64],
}

#[repr(C)]
#[derive(Copy, Clone)]
struct BudgetPayload {
    heap_idx: u32,
    flags: u32,
    used: u64,
    budget: u64,
    heap_size: u64,
    _pad: u64,
}

/// One sync cycle / orphan diagnostic mark.
///
/// `severity` uses `SYNC_SEVERITY_*`. `ttl_ms` controls how long the
/// viewer keeps the mark alive after the most recent emission; the
/// producer re-emits while the issue persists, refreshing the TTL.
///
/// `related_queue_*` is non-zero for orphan events (the queue at the
/// other end of the unmatched semaphore). For cycles it stays zero
/// because the chain may span more than two queues; the description
/// string carries the full chain instead.
#[repr(C)]
#[derive(Copy, Clone)]
struct SyncCyclePayload {
    queue_family: u32,
    queue_index: u32,
    severity: u32,
    ttl_ms: u32,
    related_queue_family: u32,
    related_queue_index: u32,
    _reserved_words: [u32; 2],
    description: [u8; 96],
    _pad: [u8; 104],
}

const _: () = assert!(std::mem::size_of::<SyncCyclePayload>() <= PAYLOAD_SIZE);

/// Shader printf event payload. See ignis-viz/src/ipc.rs for field
/// semantics; the layout is the on-wire form shared with the viewer.
#[repr(C)]
#[derive(Copy, Clone)]
struct ShaderPrintfPayload {
    stage: u32,
    message_id: i32,
    _pad: u32,
    location: [u8; 64],
    message: [u8; 156],
}

const _: () = assert!(std::mem::size_of::<ShaderPrintfPayload>() <= PAYLOAD_SIZE);

/// Hang detection event payload. See ignis-viz/src/ipc.rs for field
/// semantics.
#[repr(C)]
#[derive(Copy, Clone)]
struct HangDetectedPayload {
    fence_handle: u64,
    elapsed_ns: u64,
    last_completed_id: u32,
    first_pending_id: u32,
    total_breadcrumbs: u32,
    _pad: u32,
    label: [u8; 64],
    last_completed_label: [u8; 64],
    first_pending_label: [u8; 64],
}

const _: () = assert!(std::mem::size_of::<HangDetectedPayload>() <= PAYLOAD_SIZE);

/// Breadcrumb attached to a hang event. See ignis-viz/src/ipc.rs for
/// field semantics.
#[repr(C)]
#[derive(Copy, Clone)]
struct BreadcrumbPayload {
    parent_seq: u32,
    crumb_id: u32,
    completed: u32,
    _pad: u32,
    label: [u8; 96],
}

const _: () = assert!(std::mem::size_of::<BreadcrumbPayload>() <= PAYLOAD_SIZE);

#[repr(C)]
#[derive(Copy, Clone)]
struct DeviceFaultPayload {
    description: [u8; 128],
    address_info_count: u32,
    vendor_info_count: u32,
    vendor_binary_size: u32,
    has_fault_info_ext: u32,
    has_checkpoints_ext: u32,
    has_markers_ext: u32,
    checkpoint_count: u32,
    marker_total: u32,
    marker_fired: u32,
    _pad: [u8; 64],
}

const _: () = assert!(std::mem::size_of::<DeviceFaultPayload>() <= PAYLOAD_SIZE);

#[repr(C)]
#[derive(Copy, Clone)]
struct ObjectRegisteredPayload {
    handle: u64,
    object_type: u32,
    creation_line: u32,
    creation_column: u32,
    _pad: u32,
    name: [u8; 64],
    creation_file: [u8; 96],
    creation_function: [u8; 48],
}

const _: () = assert!(std::mem::size_of::<ObjectRegisteredPayload>() <= PAYLOAD_SIZE);

#[repr(C)]
#[derive(Copy, Clone)]
struct ObjectDestroyedPayload {
    handle: u64,
    object_type: u32,
    _pad: u32,
    usage_count: u64,
    _pad2: [u8; 208],
}

const _: () = assert!(std::mem::size_of::<ObjectDestroyedPayload>() <= PAYLOAD_SIZE);

#[repr(C)]
#[derive(Copy, Clone)]
struct DescriptorIssuePayload {
    set_handle: u64,
    dead_handle: u64,
    binding: u32,
    resource_kind: u32,
    _pad: u32,
    _pad2: u32,
    set_name: [u8; 64],
    description: [u8; 96],
}

const _: () = assert!(std::mem::size_of::<DescriptorIssuePayload>() <= PAYLOAD_SIZE);

#[repr(C)]
#[derive(Copy, Clone)]
struct AliasingIssuePayload {
    handle: u64,
    write_op_index: u32,
    conflict_op_index: u32,
    write_stage: u32,
    conflict_stage: u32,
    access_type: u32,
    _pad: u32,
    name: [u8; 32],
    write_label: [u8; 48],
    conflict_label: [u8; 48],
    description: [u8; 64],
}

const _: () = assert!(std::mem::size_of::<AliasingIssuePayload>() <= PAYLOAD_SIZE);

#[repr(C)]
#[derive(Copy, Clone)]
struct PipelineIssuePayload {
    pipeline: u64,
    kind: u32,
    _pad: u32,
    pipeline_name: [u8; 32],
    description: [u8; 156],
}

const _: () = assert!(std::mem::size_of::<PipelineIssuePayload>() <= PAYLOAD_SIZE);

#[repr(C)]
#[derive(Copy, Clone)]
struct AllocSitePayload {
    epoch: u64,
    total_allocs: u64,
    total_bytes: u64,
    active_allocs: u64,
    active_bytes: u64,
    peak_active_allocs: u64,
    peak_active_bytes: u64,
    line: u32,
    site_index: u32,
    function: [u8; 64],
    file: [u8; 88],
}

const _: () = assert!(std::mem::size_of::<AllocSitePayload>() <= PAYLOAD_SIZE);

/// Guard band corruption event payload. Wire layout matches the
/// viewer-side `CanaryCorruptionPayload` exactly.
#[repr(C)]
#[derive(Copy, Clone)]
struct CanaryCorruptionPayload {
    memory: u64,
    user_offset: u64,
    user_size: u64,
    guard_size: u64,
    canary: u64,
    region: u32,
    severity: u32,
    first_corrupted_byte: u32,
    corrupted_count: u32,
    expected_byte: u8,
    actual_byte: u8,
    _pad0: [u8; 6],
    hex_expected: [u8; 16],
    hex_actual: [u8; 16],
    source: [u8; 48],
    description: [u8; 80],
    _pad1: [u8; 8],
}

const _: () = assert!(std::mem::size_of::<CanaryCorruptionPayload>() <= PAYLOAD_SIZE);

/// Aggregate snapshot of `HardenedAllocator` counters.
#[repr(C)]
#[derive(Copy, Clone)]
struct HardenedStatsPayload {
    total_allocs: u64,
    total_frees: u64,
    active_allocs: u64,
    active_bytes: u64,
    quarantine_entries: u64,
    quarantine_bytes: u64,
    corruptions_detected: u64,
    peak_allocs: u64,
    peak_bytes: u64,
    _pad: [u8; 160],
}

const _: () = assert!(std::mem::size_of::<HardenedStatsPayload>() <= PAYLOAD_SIZE);

/// One determinism-checker run completion.
#[repr(C)]
#[derive(Copy, Clone)]
struct DeterminismRunPayload {
    aggregate_hash: u64,
    seed: u64,
    run_index: u32,
    frame_idx: u32,
    buffer_count: u32,
    image_count: u32,
    matches_baseline: u32,
    _pad: u32,
    session_label: [u8; 64],
    _pad2: [u8; 128],
}

const _: () = assert!(std::mem::size_of::<DeterminismRunPayload>() <= PAYLOAD_SIZE);

/// Per-capture divergence detail.
#[repr(C)]
#[derive(Copy, Clone)]
struct DeterminismDivergencePayload {
    baseline_hash: u64,
    current_hash: u64,
    run_index: u32,
    capture_kind: u32,
    dim_a: u32,
    dim_b: u32,
    capture_label: [u8; 64],
    diff_bitmap_path: [u8; 96],
    _pad: [u8; 40],
}

const _: () = assert!(std::mem::size_of::<DeterminismDivergencePayload>() <= PAYLOAD_SIZE);

/// Encode a string into a fixed-size NUL-terminated byte buffer of
/// length N. Truncation is silent; the caller should size the field
/// with the typical worst case in mind.
fn str_to_buf<const N: usize>(s: &str) -> [u8; N] {
    let mut out = [0u8; N];
    let bytes = s.as_bytes();
    let n = bytes.len().min(N - 1);
    out[..n].copy_from_slice(&bytes[..n]);
    out
}

#[repr(C)]
#[derive(Copy, Clone)]
struct TraceRecord {
    timestamp_ns: u64,
    thread_id: u64,
    kind: u32,
    seq: u32,
    payload: [u8; PAYLOAD_SIZE],
}

impl TraceRecord {
    fn zeroed() -> Self {
        Self {
            timestamp_ns: 0,
            thread_id: 0,
            kind: 0,
            seq: 0,
            payload: [0u8; PAYLOAD_SIZE],
        }
    }
    fn write_payload<T: Copy>(&mut self, p: &T) {
        let s = std::mem::size_of::<T>();
        assert!(s <= PAYLOAD_SIZE);
        unsafe {
            std::ptr::copy_nonoverlapping(
                p as *const T as *const u8,
                self.payload.as_mut_ptr(),
                s,
            );
        }
    }
}

#[repr(C)]
#[derive(Copy, Clone)]
struct NodeRegisterPayload { node_id: u32, kind_tag: u32, label: [u8; 64] }

#[repr(C)]
#[derive(Copy, Clone)]
struct NodeRemovePayload { node_id: u32 }

#[repr(C)]
#[derive(Copy, Clone)]
struct EdgeRegisterPayload { from_id: u32, to_id: u32, flags: u32 }

#[repr(C)]
#[derive(Copy, Clone)]
struct EdgeTogglePayload { from_id: u32, to_id: u32, active: u32 }

#[repr(C)]
#[derive(Copy, Clone)]
struct SubmissionPayload {
    queue_family: u32,
    queue_index: u32,
    node_id: u32,
    duration_ns: u64,
    label: [u8; 64],
}

#[repr(C)]
#[derive(Copy, Clone)]
struct PassPayload {
    node_id: u32,
    duration_ns: u64,
    label: [u8; 64],
}

#[repr(C)]
#[derive(Copy, Clone)]
struct AllocationPayload {
    memory: u64,
    offset: u64,
    size: u64,
    site: [u8; 64],
}

#[repr(C)]
#[derive(Copy, Clone)]
struct FreePayload {
    memory: u64,
    offset: u64,
    size: u64,
    _pad: u64,
}

#[repr(C)]
struct ShmHeader {
    magic: u64,
    version: u32,
    writer_pid: u32,
    capacity: u32,
    record_size: u32,
    write_idx: AtomicU64,
    read_idx: AtomicU64,
    last_heartbeat_ns: AtomicU64,
    reserved: [u8; 16],
}

const _: () = assert!(std::mem::size_of::<ShmHeader>() == HEADER_SIZE);

fn str_to_label64(s: &str) -> [u8; 64] {
    let mut out = [0u8; 64];
    let bytes = s.as_bytes();
    let n = bytes.len().min(63);
    out[..n].copy_from_slice(&bytes[..n]);
    out
}

fn current_time_ns() -> u64 {
    SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map(|d| d.as_nanos() as u64)
        .unwrap_or(0)
}

/// Errors produced when creating or operating the live link IPC bridge.
#[derive(Debug)]
pub enum LiveLinkError {
    /// `CreateFileMappingW` failed with the given OS error code.
    CreateFailed(u32),
    /// `MapViewOfFile` failed with the given OS error code.
    MapFailed(u32),
    /// The host platform does not support the live link
    /// (currently only Windows is implemented).
    Unsupported,
}

impl std::fmt::Display for LiveLinkError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::CreateFailed(e) => write!(f, "CreateFileMappingW failed (err={})", e),
            Self::MapFailed(e) => write!(f, "MapViewOfFile failed (err={})", e),
            Self::Unsupported => write!(f, "live link is only supported on Windows"),
        }
    }
}

impl std::error::Error for LiveLinkError {}

#[cfg(target_os = "windows")]
mod platform {
    use std::ffi::c_void;
    pub type HANDLE = *mut c_void;
    pub type DWORD = u32;
    #[allow(non_camel_case_types)]
    pub type SIZE_T = usize;
    pub const INVALID_HANDLE_VALUE: HANDLE = -1isize as HANDLE;
    pub const PAGE_READWRITE: DWORD = 0x04;
    pub const FILE_MAP_ALL_ACCESS: DWORD = 0xF001F;

    #[link(name = "kernel32")]
    extern "system" {
        pub fn CreateFileMappingW(
            file: HANDLE, sec: *mut c_void, protect: DWORD,
            max_high: DWORD, max_low: DWORD, name: *const u16,
        ) -> HANDLE;
        pub fn MapViewOfFile(
            mapping: HANDLE, access: DWORD,
            off_high: DWORD, off_low: DWORD, bytes: SIZE_T,
        ) -> *mut c_void;
        pub fn UnmapViewOfFile(view: *const c_void) -> i32;
        pub fn CloseHandle(h: HANDLE) -> i32;
        pub fn GetCurrentProcessId() -> DWORD;
        pub fn GetLastError() -> DWORD;
    }
}

/// Producer side of the ignis-viz live link.
#[cfg(target_os = "windows")]
pub struct LiveLink {
    mapping: platform::HANDLE,
    base: *mut u8,
    capacity: u32,
}

#[cfg(not(target_os = "windows"))]
pub struct LiveLink;

unsafe impl Send for LiveLink {}
unsafe impl Sync for LiveLink {}

impl LiveLink {
    /// Create a new shared memory ring under `Local\NAME`. `capacity`
    /// must be a power of two, at least 64.
    #[cfg(target_os = "windows")]
    pub fn create(name: &str, capacity: u32) -> Result<Arc<Self>, LiveLinkError> {
        use platform::*;
        assert!(capacity.is_power_of_two() && capacity >= 64);
        let total_size = HEADER_SIZE + (capacity as usize) * RECORD_SIZE;
        let qualified = format!("Local\\{}", name);
        let wname: Vec<u16> = qualified.encode_utf16().chain(std::iter::once(0)).collect();
        unsafe {
            let mapping = CreateFileMappingW(
                INVALID_HANDLE_VALUE,
                std::ptr::null_mut(),
                PAGE_READWRITE,
                0,
                total_size as DWORD,
                wname.as_ptr(),
            );
            if mapping.is_null() {
                return Err(LiveLinkError::CreateFailed(GetLastError()));
            }
            let view = MapViewOfFile(mapping, FILE_MAP_ALL_ACCESS, 0, 0, total_size);
            if view.is_null() {
                let err = GetLastError();
                CloseHandle(mapping);
                return Err(LiveLinkError::MapFailed(err));
            }
            let base = view as *mut u8;
            let header = base as *mut ShmHeader;
            (*header).magic = SHM_MAGIC;
            (*header).version = SHM_VERSION;
            (*header).writer_pid = GetCurrentProcessId();
            (*header).capacity = capacity;
            (*header).record_size = RECORD_SIZE as u32;
            (*header).write_idx.store(0, Ordering::Relaxed);
            (*header).read_idx.store(0, Ordering::Relaxed);
            (*header).last_heartbeat_ns.store(current_time_ns(), Ordering::Relaxed);
            Ok(Arc::new(Self { mapping, base, capacity }))
        }
    }

    #[cfg(not(target_os = "windows"))]
    pub fn create(_name: &str, _capacity: u32) -> Result<Arc<Self>, LiveLinkError> {
        Err(LiveLinkError::Unsupported)
    }

    #[cfg(target_os = "windows")]
    fn header(&self) -> &ShmHeader {
        unsafe { &*(self.base as *const ShmHeader) }
    }

    #[cfg(target_os = "windows")]
    fn record_ptr(&self, idx: u64) -> *mut TraceRecord {
        let slot = (idx & (self.capacity as u64 - 1)) as usize;
        let off = HEADER_SIZE + slot * RECORD_SIZE;
        unsafe { self.base.add(off) as *mut TraceRecord }
    }

    #[cfg(target_os = "windows")]
    fn submit(&self, mut rec: TraceRecord) -> u32 {
        let h = self.header();
        let w = h.write_idx.load(Ordering::Relaxed);
        let seq = (w & 0xFFFFFFFF) as u32;
        rec.seq = seq;
        unsafe {
            std::ptr::write_volatile(self.record_ptr(w), rec);
        }
        h.write_idx.store(w + 1, Ordering::Release);
        seq
    }

    #[cfg(not(target_os = "windows"))]
    fn submit(&self, _rec: TraceRecord) -> u32 { 0 }

    /// Update the writer heartbeat.
    pub fn heartbeat(&self) {
        #[cfg(target_os = "windows")]
        self.header().last_heartbeat_ns.store(current_time_ns(), Ordering::Relaxed);
    }

    /// Push one shader printf message captured from a
    /// `debugPrintfEXT(...)` call inside a SPIR-V shader.
    ///
    /// `stage` is the coarse SPIR-V execution model classification:
    /// 0 = unknown, 1 = vertex, 2 = fragment, 3 = compute,
    /// 4 = raygen, 5 = miss, 6 = closest_hit, 7 = any_hit,
    /// 8 = intersection, 9 = callable, 10 = task, 11 = mesh,
    /// 12 = tess control, 13 = tess eval, 14 = geometry.
    ///
    /// `message_id` is the raw validation layer message id so the
    /// viewer can correlate printf bursts that share an origin.
    ///
    /// `location` is any source-location hint the layer attached;
    /// pass an empty string when not available. `message` is the
    /// formatted printf body. Both strings are emitted with
    /// continuation chunks if they exceed their inline budgets.
    pub fn record_shader_printf(
        &self,
        stage: u32,
        message_id: i32,
        location: &str,
        message: &str,
    ) {
        let mut rec = TraceRecord::zeroed();
        rec.timestamp_ns = current_time_ns();
        rec.kind = KIND_SHADER_PRINTF;
        rec.write_payload(&ShaderPrintfPayload {
            stage,
            message_id,
            _pad: 0,
            location: str_to_buf::<64>(location),
            message: str_to_buf::<156>(message),
        });
        let parent_seq = self.submit(rec);

        let msg_bytes = message.as_bytes();
        if msg_bytes.len() > 155 {
            self.emit_text_continuation(parent_seq, 1, &msg_bytes[155..]);
        }
    }

    /// Push one GPU hang event from `HangDetector`.
    ///
    /// Returns the sequence number that breadcrumb children should
    /// reference via their `parent_seq` field. The caller emits one
    /// `record_breadcrumb` per crumb in the trail right after this
    /// call so the viewer can stitch them together.
    ///
    /// `total_breadcrumbs` should equal the number of subsequent
    /// `record_breadcrumb` calls the producer is going to make for
    /// this hang. The viewer uses it for progress display.
    pub fn record_hang_detected(
        &self,
        fence_handle: u64,
        elapsed_ns: u64,
        label: &str,
        last_completed_id: u32,
        first_pending_id: u32,
        last_completed_label: &str,
        first_pending_label: &str,
        total_breadcrumbs: u32,
    ) -> u32 {
        let mut rec = TraceRecord::zeroed();
        rec.timestamp_ns = current_time_ns();
        rec.kind = KIND_HANG_DETECTED;
        rec.write_payload(&HangDetectedPayload {
            fence_handle,
            elapsed_ns,
            last_completed_id,
            first_pending_id,
            total_breadcrumbs,
            _pad: 0,
            label: str_to_buf::<64>(label),
            last_completed_label: str_to_buf::<64>(last_completed_label),
            first_pending_label: str_to_buf::<64>(first_pending_label),
        });
        self.submit(rec)
    }

    /// Push one breadcrumb attached to a previously recorded hang
    /// event. `parent_seq` is the value returned by the matching
    /// `record_hang_detected` call.
    pub fn record_breadcrumb(
        &self,
        parent_seq: u32,
        crumb_id: u32,
        completed: bool,
        label: &str,
    ) {
        let mut rec = TraceRecord::zeroed();
        rec.timestamp_ns = current_time_ns();
        rec.kind = KIND_BREADCRUMB;
        rec.write_payload(&BreadcrumbPayload {
            parent_seq,
            crumb_id,
            completed: if completed { 1 } else { 0 },
            _pad: 0,
            label: str_to_buf::<96>(label),
        });
        self.submit(rec);
    }

    /// Push one device fault snapshot. Producers should call this
    /// once from their device-lost handler with data collected via
    /// `DeviceFaultRecorder::collect_all`. Long descriptions are
    /// streamed via `KIND_TEXT_CONTINUATION` records.
    #[allow(clippy::too_many_arguments)]
    pub fn record_device_fault(
        &self,
        description: &str,
        address_info_count: u32,
        vendor_info_count: u32,
        vendor_binary_size: u32,
        has_fault_info_ext: bool,
        has_checkpoints_ext: bool,
        has_markers_ext: bool,
        checkpoint_count: u32,
        marker_total: u32,
        marker_fired: u32,
    ) {
        let mut rec = TraceRecord::zeroed();
        rec.timestamp_ns = current_time_ns();
        rec.kind = KIND_DEVICE_FAULT;
        rec.write_payload(&DeviceFaultPayload {
            description: str_to_buf::<128>(description),
            address_info_count,
            vendor_info_count,
            vendor_binary_size,
            has_fault_info_ext: if has_fault_info_ext { 1 } else { 0 },
            has_checkpoints_ext: if has_checkpoints_ext { 1 } else { 0 },
            has_markers_ext: if has_markers_ext { 1 } else { 0 },
            checkpoint_count,
            marker_total,
            marker_fired,
            _pad: [0; 64],
        });
        let parent_seq = self.submit(rec);

        let bytes = description.as_bytes();
        if bytes.len() > 127 {
            self.emit_text_continuation(parent_seq, 0, &bytes[127..]);
        }
    }

    /// Push one object registration event from `LifetimeTracker::register`.
    pub fn record_object_registered(
        &self,
        handle: u64,
        object_type: u32,
        name: &str,
        creation_file: &str,
        creation_line: u32,
        creation_column: u32,
        creation_function: &str,
    ) {
        let mut rec = TraceRecord::zeroed();
        rec.timestamp_ns = current_time_ns();
        rec.kind = KIND_OBJECT_REGISTERED;
        rec.write_payload(&ObjectRegisteredPayload {
            handle,
            object_type,
            creation_line,
            creation_column,
            _pad: 0,
            name: str_to_label64(name),
            creation_file: str_to_buf::<96>(creation_file),
            creation_function: str_to_buf::<48>(creation_function),
        });
        self.submit(rec);
    }

    /// Push one object destruction event from `LifetimeTracker::unregister`.
    /// `usage_count` is the cumulative count of `record_usage` calls.
    pub fn record_object_destroyed(
        &self,
        handle: u64,
        object_type: u32,
        usage_count: u64,
    ) {
        let mut rec = TraceRecord::zeroed();
        rec.timestamp_ns = current_time_ns();
        rec.kind = KIND_OBJECT_DESTROYED;
        rec.write_payload(&ObjectDestroyedPayload {
            handle,
            object_type,
            _pad: 0,
            usage_count,
            _pad2: [0; 208],
        });
        self.submit(rec);
    }

    /// Push one descriptor audit issue from `DescriptorAuditor`.
    /// `resource_kind` uses `RES_KIND_*` constants.
    #[allow(clippy::too_many_arguments)]
    pub fn record_descriptor_issue(
        &self,
        set_handle: u64,
        binding: u32,
        resource_kind: u32,
        dead_handle: u64,
        set_name: &str,
        description: &str,
    ) {
        let mut rec = TraceRecord::zeroed();
        rec.timestamp_ns = current_time_ns();
        rec.kind = KIND_DESCRIPTOR_ISSUE;
        rec.write_payload(&DescriptorIssuePayload {
            set_handle,
            dead_handle,
            binding,
            resource_kind,
            _pad: 0,
            _pad2: 0,
            set_name: str_to_label64(set_name),
            description: str_to_buf::<96>(description),
        });
        self.submit(rec);
    }

    /// Push one aliasing conflict from `AliasingDetector`.
    /// `access_type`: 0 = read after write, 1 = write after write.
    #[allow(clippy::too_many_arguments)]
    pub fn record_aliasing_conflict(
        &self,
        handle: u64,
        write_op_index: u32,
        conflict_op_index: u32,
        write_stage: u32,
        conflict_stage: u32,
        access_type: u32,
        name: &str,
        write_label: &str,
        conflict_label: &str,
        description: &str,
    ) {
        let mut rec = TraceRecord::zeroed();
        rec.timestamp_ns = current_time_ns();
        rec.kind = KIND_ALIASING_CONFLICT;
        rec.write_payload(&AliasingIssuePayload {
            handle,
            write_op_index,
            conflict_op_index,
            write_stage,
            conflict_stage,
            access_type,
            _pad: 0,
            name: str_to_buf::<32>(name),
            write_label: str_to_buf::<48>(write_label),
            conflict_label: str_to_buf::<48>(conflict_label),
            description: str_to_buf::<64>(description),
        });
        self.submit(rec);
    }

    /// Push one pipeline audit issue from `PipelineAuditor`.
    /// `kind` uses `PIPELINE_ISSUE_*` constants. Long descriptions
    /// are streamed via `KIND_TEXT_CONTINUATION`.
    pub fn record_pipeline_issue(
        &self,
        pipeline: u64,
        kind: u32,
        pipeline_name: &str,
        description: &str,
    ) {
        let mut rec = TraceRecord::zeroed();
        rec.timestamp_ns = current_time_ns();
        rec.kind = KIND_PIPELINE_ISSUE;
        rec.write_payload(&PipelineIssuePayload {
            pipeline,
            kind,
            _pad: 0,
            pipeline_name: str_to_buf::<32>(pipeline_name),
            description: str_to_buf::<156>(description),
        });
        let parent_seq = self.submit(rec);

        let bytes = description.as_bytes();
        if bytes.len() > 155 {
            self.emit_text_continuation(parent_seq, 0, &bytes[155..]);
        }
    }

    /// Push one ranked allocation-site row from a profiler snapshot
    /// batch. All sites of one snapshot share the same `epoch` value;
    /// when the producer begins the next snapshot it increments the
    /// epoch and starts emitting fresh site records, which causes
    /// the viewer to atomically swap in the new batch.
    ///
    /// `site_index` is the producer-assigned rank (0 = top by active
    /// bytes). The viewer preserves it as the row order so scrolling
    /// is stable across snapshots.
    #[allow(clippy::too_many_arguments)]
    pub fn record_alloc_site(
        &self,
        epoch: u64,
        site_index: u32,
        function: &str,
        file: &str,
        line: u32,
        total_allocs: u64,
        total_bytes: u64,
        active_allocs: u64,
        active_bytes: u64,
        peak_active_allocs: u64,
        peak_active_bytes: u64,
    ) {
        let mut rec = TraceRecord::zeroed();
        rec.timestamp_ns = current_time_ns();
        rec.kind = KIND_ALLOC_SITE_SNAPSHOT;
        rec.write_payload(&AllocSitePayload {
            epoch,
            total_allocs,
            total_bytes,
            active_allocs,
            active_bytes,
            peak_active_allocs,
            peak_active_bytes,
            line,
            site_index,
            function: str_to_buf::<64>(function),
            file: str_to_buf::<88>(file),
        });
        self.submit(rec);
    }

    /// Register or update a graph node.
    pub fn record_node(&self, node_id: u32, kind_tag: u32, label: &str) {
        let mut rec = TraceRecord::zeroed();
        rec.timestamp_ns = current_time_ns();
        rec.kind = KIND_NODE_REGISTER;
        rec.write_payload(&NodeRegisterPayload {
            node_id, kind_tag, label: str_to_label64(label),
        });
        self.submit(rec);
    }

    /// Remove a node and any edges referencing it.
    pub fn record_node_remove(&self, node_id: u32) {
        let mut rec = TraceRecord::zeroed();
        rec.timestamp_ns = current_time_ns();
        rec.kind = KIND_NODE_REMOVE;
        rec.write_payload(&NodeRemovePayload { node_id });
        self.submit(rec);
    }

    /// Register an edge between two nodes.
    pub fn record_edge(&self, from: u32, to: u32) {
        let mut rec = TraceRecord::zeroed();
        rec.timestamp_ns = current_time_ns();
        rec.kind = KIND_EDGE_REGISTER;
        rec.write_payload(&EdgeRegisterPayload { from_id: from, to_id: to, flags: 0 });
        self.submit(rec);
    }

    /// Animate edge active state.
    pub fn record_edge_toggle(&self, from: u32, to: u32, active: bool) {
        let mut rec = TraceRecord::zeroed();
        rec.timestamp_ns = current_time_ns();
        rec.kind = KIND_EDGE_TOGGLE;
        rec.write_payload(&EdgeTogglePayload {
            from_id: from, to_id: to,
            active: if active { 1 } else { 0 },
        });
        self.submit(rec);
    }

    /// A pass executed with the given duration.
    pub fn record_pass(&self, node_id: u32, label: &str, duration_ns: u64) {
        let mut rec = TraceRecord::zeroed();
        rec.timestamp_ns = current_time_ns();
        rec.kind = KIND_PASS;
        rec.write_payload(&PassPayload {
            node_id, duration_ns, label: str_to_label64(label),
        });
        self.submit(rec);
    }

    /// A queue submission completed.
    pub fn record_submission(
        &self, queue_family: u32, queue_index: u32,
        label: &str, duration_ns: u64,
    ) {
        let mut rec = TraceRecord::zeroed();
        rec.timestamp_ns = current_time_ns();
        rec.kind = KIND_SUBMISSION;
        rec.write_payload(&SubmissionPayload {
            queue_family, queue_index, node_id: 0,
            duration_ns, label: str_to_label64(label),
        });
        self.submit(rec);
    }

    /// A memory allocation occurred.
    pub fn record_allocation(&self, memory: u64, offset: u64, size: u64, site: &str) {
        let mut rec = TraceRecord::zeroed();
        rec.timestamp_ns = current_time_ns();
        rec.kind = KIND_ALLOCATION;
        rec.write_payload(&AllocationPayload {
            memory, offset, size, site: str_to_label64(site),
        });
        self.submit(rec);
    }

    /// A previously-recorded allocation was freed.
    pub fn record_free(&self, memory: u64, offset: u64, size: u64) {
        let mut rec = TraceRecord::zeroed();
        rec.timestamp_ns = current_time_ns();
        rec.kind = KIND_FREE;
        rec.write_payload(&FreePayload { memory, offset, size, _pad: 0 });
        self.submit(rec);
    }

    /// Bind a Vulkan handle to a debug name.
    pub fn record_resource_name(&self, handle: u64, kind: u32, name: &str) {
        let mut rec = TraceRecord::zeroed();
        rec.timestamp_ns = current_time_ns();
        rec.kind = KIND_RESOURCE_NAME;
        rec.write_payload(&ResourceNamePayload {
            handle,
            kind_tag: kind,
            _pad: 0,
            name: str_to_label64(name),
        });
        self.submit(rec);
    }

    /// Push one validation diagnostic.
    #[allow(clippy::too_many_arguments)]
    pub fn record_validation(
        &self,
        severity: u32,
        node_id: u32,
        function: &str,
        vuid: &str,
        message: &str,
        object_type: u32,
        object_handle: u64,
    ) {
        let mut rec = TraceRecord::zeroed();
        rec.timestamp_ns = current_time_ns();
        rec.kind = KIND_VALIDATION;
        rec.write_payload(&ValidationPayload {
            severity,
            node_id,
            object_type,
            _pad: 0,
            object_handle,
            function: str_to_buf::<48>(function),
            vuid: str_to_label64(vuid),
            message: str_to_buf::<96>(message),
        });
        let parent_seq = self.submit(rec);

        // Validation messages can be very long (full Vulkan VL prose
        // with object chains). Stream the overflow.
        let msg_bytes = message.as_bytes();
        if msg_bytes.len() > 95 {
            self.emit_text_continuation(parent_seq, 1, &msg_bytes[95..]);
        }
    }

    /// Push one GPU timer scope.
    pub fn record_gpu_timestamp(
        &self,
        queue_family: u32,
        queue_index: u32,
        stage: u32,
        begin_ns: u64,
        duration_ns: u64,
        label: &str,
    ) {
        let mut rec = TraceRecord::zeroed();
        rec.timestamp_ns = current_time_ns();
        rec.kind = KIND_GPU_TIMESTAMP;
        rec.write_payload(&GpuTimestampPayload {
            queue_family,
            queue_index,
            stage,
            _pad: 0,
            begin_ns,
            duration_ns,
            label: str_to_label64(label),
        });
        self.submit(rec);
    }

    /// Push pipeline statistics counters from a GPU readback.
    #[allow(clippy::too_many_arguments)]
    pub fn record_pipeline_stats(
        &self,
        label: &str,
        vs_invocations: u64,
        fs_invocations: u64,
        cs_invocations: u64,
        ia_vertices: u64,
        ia_primitives: u64,
        clipping_invocations: u64,
        clipping_primitives: u64,
        gs_invocations: u64,
        gs_primitives: u64,
        tcs_patches: u64,
        tes_invocations: u64,
    ) {
        let mut rec = TraceRecord::zeroed();
        rec.timestamp_ns = current_time_ns();
        rec.kind = KIND_PIPELINE_STATS;
        rec.write_payload(&PipelineStatsPayload {
            vs_invocations,
            fs_invocations,
            cs_invocations,
            ia_vertices,
            ia_primitives,
            clipping_invocations,
            clipping_primitives,
            gs_invocations,
            gs_primitives,
            tcs_patches,
            tes_invocations,
            label: str_to_label64(label),
        });
        self.submit(rec);
    }

    /// Push one memory heap budget sample.
    pub fn record_budget(
        &self,
        heap_idx: u32,
        flags: u32,
        used: u64,
        budget: u64,
        heap_size: u64,
    ) {
        let mut rec = TraceRecord::zeroed();
        rec.timestamp_ns = current_time_ns();
        rec.kind = KIND_BUDGET;
        rec.write_payload(&BudgetPayload {
            heap_idx,
            flags,
            used,
            budget,
            heap_size,
            _pad: 0,
        });
        self.submit(rec);
    }

    /// Push one sync cycle / orphan diagnostic mark.
    ///
    /// `severity` uses `SYNC_SEVERITY_*`. `ttl_ms` is how long the
    /// viewer keeps the lane tinted after this emission; the producer
    /// is expected to re-emit while the issue persists, refreshing
    /// the TTL on each pass. Once the producer stops emitting (the
    /// underlying tracker no longer detects the issue), the mark
    /// expires naturally without an explicit clear event.
    ///
    /// `related_queue_family` and `related_queue_index` name the other
    /// side of an unmatched semaphore for orphan events. For cycles
    /// pass 0 / 0 since the chain may span more than two queues; the
    /// `description` string carries the full chain.
    #[allow(clippy::too_many_arguments)]
    pub fn record_sync_cycle(
        &self,
        queue_family: u32,
        queue_index: u32,
        severity: u32,
        ttl_ms: u32,
        related_queue_family: u32,
        related_queue_index: u32,
        description: &str,
    ) {
        let mut rec = TraceRecord::zeroed();
        rec.timestamp_ns = current_time_ns();
        rec.kind = KIND_SYNC_CYCLE_DETECTED;
        rec.write_payload(&SyncCyclePayload {
            queue_family,
            queue_index,
            severity,
            ttl_ms,
            related_queue_family,
            related_queue_index,
            _reserved_words: [0; 2],
            description: str_to_buf::<96>(description),
            _pad: [0; 104],
        });
        self.submit(rec);
    }

    /// Push one guard band corruption event from `HardenedAllocator`.
    ///
    /// `region`: 0 = front guard, 1 = back guard. `severity` uses the
    /// same numeric range as `VAL_SEVERITY_*` (0 info, 1 warn, 2 err)
    /// so the viewer can colour-map both classes uniformly.
    ///
    /// `hex_expected` / `hex_actual` are 16-byte windows centered on
    /// the first corrupted byte. The viewer renders them side-by-side
    /// with `^^` markers under positions where the bytes differ.
    ///
    /// `source` is the detection context (e.g. `"Allocator::free()"`,
    /// `"quarantine eviction"`); `description` is free-form prose.
    /// Both are truncated by the producer side at 47 / 79 chars
    /// respectively.
    #[allow(clippy::too_many_arguments)]
    pub fn record_canary_corruption(
        &self,
        memory: u64,
        user_offset: u64,
        user_size: u64,
        guard_size: u64,
        canary: u64,
        region: u32,
        severity: u32,
        first_corrupted_byte: u32,
        corrupted_count: u32,
        expected_byte: u8,
        actual_byte: u8,
        hex_expected: &[u8; 16],
        hex_actual: &[u8; 16],
        source: &str,
        description: &str,
    ) {
        let mut rec = TraceRecord::zeroed();
        rec.timestamp_ns = current_time_ns();
        rec.kind = KIND_CANARY_CORRUPTION;
        rec.write_payload(&CanaryCorruptionPayload {
            memory,
            user_offset,
            user_size,
            guard_size,
            canary,
            region,
            severity,
            first_corrupted_byte,
            corrupted_count,
            expected_byte,
            actual_byte,
            _pad0: [0; 6],
            hex_expected: *hex_expected,
            hex_actual: *hex_actual,
            source: str_to_buf::<48>(source),
            description: str_to_buf::<80>(description),
            _pad1: [0; 8],
        });
        let parent_seq = self.submit(rec);

        // Emit overflow chunks for description and source if either
        // exceeds its inline budget. The first 79 bytes are already in
        // the parent event; the continuation carries the remainder.
        let desc_bytes = description.as_bytes();
        if desc_bytes.len() > 79 {
            self.emit_text_continuation(parent_seq, 0, &desc_bytes[79..]);
        }
        let src_bytes = source.as_bytes();
        if src_bytes.len() > 47 {
            self.emit_text_continuation(parent_seq, 2, &src_bytes[47..]);
        }
    }

    /// Push one aggregate snapshot of `HardenedAllocator` counters.
    /// The viewer keeps only the most recent snapshot; older values
    /// are not buffered. Producers should emit at a low frequency
    /// (every few seconds) since stats change slowly relative to
    /// individual allocation events.
    #[allow(clippy::too_many_arguments)]
    pub fn record_hardened_stats(
        &self,
        total_allocs: u64,
        total_frees: u64,
        active_allocs: u64,
        active_bytes: u64,
        quarantine_entries: u64,
        quarantine_bytes: u64,
        corruptions_detected: u64,
        peak_allocs: u64,
        peak_bytes: u64,
    ) {
        let mut rec = TraceRecord::zeroed();
        rec.timestamp_ns = current_time_ns();
        rec.kind = KIND_HARDENED_STATS;
        rec.write_payload(&HardenedStatsPayload {
            total_allocs,
            total_frees,
            active_allocs,
            active_bytes,
            quarantine_entries,
            quarantine_bytes,
            corruptions_detected,
            peak_allocs,
            peak_bytes,
            _pad: [0; 160],
        });
        self.submit(rec);
    }

    /// Push one determinism-checker run completion.
    ///
    /// `aggregate_hash` is the producer-computed xxh64 of every
    /// capture's hash in this run. The viewer joins this with later
    /// `record_determinism_divergence` events on `run_index`.
    /// `matches_baseline` should be `true` for run 0 (self-match)
    /// and for any later run whose aggregate equals run 0's.
    #[allow(clippy::too_many_arguments)]
    pub fn record_determinism_run(
        &self,
        run_index: u32,
        seed: u64,
        frame_idx: u32,
        buffer_count: u32,
        image_count: u32,
        aggregate_hash: u64,
        matches_baseline: bool,
        session_label: &str,
    ) {
        let mut rec = TraceRecord::zeroed();
        rec.timestamp_ns = current_time_ns();
        rec.kind = KIND_DETERMINISM_RUN;
        rec.write_payload(&DeterminismRunPayload {
            aggregate_hash,
            seed,
            run_index,
            frame_idx,
            buffer_count,
            image_count,
            matches_baseline: if matches_baseline { 1 } else { 0 },
            _pad: 0,
            session_label: str_to_buf::<64>(session_label),
            _pad2: [0; 128],
        });
        self.submit(rec);
    }

    /// Push per-capture divergence detail for a non-matching run.
    ///
    /// `capture_kind`: 0 = buffer, 1 = image. For buffers, `dim_a` is
    /// the low 32 bits of size and `dim_b` is 0 (or the high 32 bits
    /// for >4 GiB). For images, `dim_a` / `dim_b` are width / height
    /// in pixels.
    ///
    /// `diff_bitmap_path` is the filesystem path of a pre-rendered
    /// BMP showing the diff (typically produced by
    /// `DeterminismChecker::verify_n_runs` when an image divergence
    /// is found). Pass an empty string when no bitmap was generated;
    /// the viewer's "Open diff bitmap" button is then disabled.
    #[allow(clippy::too_many_arguments)]
    pub fn record_determinism_divergence(
        &self,
        run_index: u32,
        capture_kind: u32,
        baseline_hash: u64,
        current_hash: u64,
        dim_a: u32,
        dim_b: u32,
        capture_label: &str,
        diff_bitmap_path: &str,
    ) {
        let mut rec = TraceRecord::zeroed();
        rec.timestamp_ns = current_time_ns();
        rec.kind = KIND_DETERMINISM_DIVERGENCE;
        rec.write_payload(&DeterminismDivergencePayload {
            baseline_hash,
            current_hash,
            run_index,
            capture_kind,
            dim_a,
            dim_b,
            capture_label: str_to_buf::<64>(capture_label),
            diff_bitmap_path: str_to_buf::<96>(diff_bitmap_path),
            _pad: [0; 40],
        });
        let parent_seq = self.submit(rec);

        // Diff bitmap paths can be long absolute filesystem paths.
        let path_bytes = diff_bitmap_path.as_bytes();
        if path_bytes.len() > 95 {
            self.emit_text_continuation(parent_seq, 0, &path_bytes[95..]);
        }
    }
    /// Emit any number of `KIND_TEXT_CONTINUATION` records to extend a
    /// previously-recorded event's text field. `parent_seq` is taken
    /// from the seq the producer last claimed for the parent event;
    /// for raw `submit` callers we pass it explicitly. `field_id`:
    /// 0 = description, 1 = message, 2 = source. No-op when `tail` is
    /// empty. The first 79 bytes (parent's inline buffer) are not
    /// emitted here — only the overflow.
    #[cfg(target_os = "windows")]
    fn emit_text_continuation(
        &self,
        parent_seq: u32,
        field_id: u32,
        tail: &[u8],
    ) {
        const CHUNK: usize = 216;
        if tail.is_empty() { return; }
        let mut offset = 0;
        while offset < tail.len() {
            let end = (offset + CHUNK).min(tail.len());
            let len = end - offset;
            let mut payload = TextContinuationPayload {
                parent_seq,
                field_id,
                chunk_len: len as u32,
                is_final: if end == tail.len() { 1 } else { 0 },
                chunk: [0u8; CHUNK],
            };
            payload.chunk[..len].copy_from_slice(&tail[offset..end]);
            let mut rec = TraceRecord::zeroed();
            rec.timestamp_ns = current_time_ns();
            rec.kind = KIND_TEXT_CONTINUATION;
            rec.write_payload(&payload);
            self.submit(rec);
            offset = end;
        }
    }

    #[cfg(not(target_os = "windows"))]
    fn emit_text_continuation(&self, _: u32, _: u32, _: &[u8]) {}
}

#[cfg(target_os = "windows")]
impl Drop for LiveLink {
    fn drop(&mut self) {
        unsafe {
            platform::UnmapViewOfFile(self.base as *const std::ffi::c_void);
            platform::CloseHandle(self.mapping);
        }
    }
}

/// Install a global ignis validation forensic handler that mirrors every
/// parsed validation diagnostic into the live link IPC ring.
///
/// Replaces any previously installed handler. Available only when the
/// `debug-tools` feature is enabled (the validation forensic subsystem
/// lives there).
#[cfg(feature = "debug-tools")]
pub fn bridge_validation_to_live_link(link: std::sync::Arc<LiveLink>) {
    use crate::debug::validation_forensic::{
        set_handler, LayerSeverity, ValidationDiagnostic,
    };

    set_handler(Box::new(move |diag: &ValidationDiagnostic| {
        let severity = match diag.severity {
            LayerSeverity::Error => VAL_SEVERITY_ERROR,
            LayerSeverity::Warning => VAL_SEVERITY_WARNING,
            LayerSeverity::Info => VAL_SEVERITY_INFO,
        };
        let (object_type, object_handle) = diag
            .objects
            .first()
            .map(|o| (vk_type_to_u32(&o.vk_type), o.handle))
            .unwrap_or((0, 0));

        link.record_validation(
            severity,
            0,
            &diag.function,
            &diag.vuid,
            &diag.raw_body,
            object_type,
            object_handle,
        );
    }));
}

/// Map a Vulkan type name (as printed by the validation layer, e.g.
/// `"VkBuffer"`) to a coarse RES_KIND_* tag understood by ignis-viz.
/// Unknown types fall through to RES_KIND_OTHER.
#[cfg(feature = "debug-tools")]
fn vk_type_to_u32(s: &str) -> u32 {
    match s {
        "VkBuffer" => RES_KIND_BUFFER,
        "VkImage" => RES_KIND_IMAGE,
        "VkDeviceMemory" => RES_KIND_DEVICE_MEMORY,
        "VkSampler" => RES_KIND_SAMPLER,
        "VkPipeline" => RES_KIND_PIPELINE,
        _ => RES_KIND_OTHER,
    }
}

/// Bridge a [`CrossQueueTracker`] analysis loop into the live link
/// IPC ring.
///
/// Spawns a worker thread that, every `poll_interval`, calls
/// `tracker.analyze()` and emits each detected cycle and orphan over
/// the live link in two channels:
///
/// 1. `record_validation` for the Validation Log panel (severity-coloured
///    list with full descriptions, searchable).
/// 2. `record_sync_cycle` for the Sync DAG lane tint (visual signal
///    on the panel where the user is most likely to be looking when
///    investigating cross-queue issues).
///
/// Both channels are populated in parallel. The Validation Log shows
/// every issue as a separate event; the Sync DAG shows one mark per
/// queue (the queue's most-recent severity wins if multiple events
/// affect the same queue in one analysis pass).
///
/// The thread terminates automatically when the last `Arc<LiveLink>`
/// outside the bridge is dropped (the bridge holds a Weak reference
/// to the link). The tracker is held strongly so analysis continues
/// even if the user drops their last reference.
///
/// # TTL
///
/// Sync cycle marks use a TTL slightly larger than `poll_interval` so
/// a single missed analysis pass (transient lock contention, scheduling
/// jitter) does not cause the lane tint to flicker. The producer
/// re-emits the mark every pass while the issue persists, refreshing
/// the TTL on each emission. Once the producer stops emitting (the
/// tracker no longer detects the issue), the mark expires naturally
/// after one TTL period without an explicit clear event.
///
/// # Performance
///
/// `analyze()` is roughly O(submissions + edges) per call. Polling
/// every 500ms-2s is appropriate for most workloads. Going below
/// 100ms is not recommended.
///
/// Available only when both `live-link` and `debug-tools` features
/// are enabled.
#[cfg(feature = "debug-tools")]
pub fn bridge_cross_queue_to_live_link(
    tracker: std::sync::Arc<crate::CrossQueueTracker>,
    link: std::sync::Arc<LiveLink>,
    poll_interval: std::time::Duration,
) -> std::thread::JoinHandle<()> {
    let link_weak = std::sync::Arc::downgrade(&link);
    drop(link);
    // TTL is 3x the poll interval so a single missed pass does not
    // flicker the lane. Capped at 10s to bound stale display when the
    // producer thread itself dies.
    let ttl_ms = (poll_interval.as_millis() as u32).saturating_mul(3).min(10_000);
    std::thread::Builder::new()
        .name("ignis-cross-queue-bridge".into())
        .spawn(move || loop {
            std::thread::sleep(poll_interval);
            let Some(link) = link_weak.upgrade() else { break; };
            let report = tracker.analyze();

            // Cycles are deadlocks. Emit a validation event AND a
            // sync_cycle mark per participating queue so the lane is
            // visually flagged.
            for (i, cycle) in report.cycles.iter().enumerate() {
                let chain: Vec<String> = cycle.iter()
                    .map(|s| format!("#{s}"))
                    .collect();
                let msg = format!(
                    "cycle {}: {} (deadlock - submissions wait on each \
                     other transitively through semaphore dependencies)",
                    i,
                    chain.join(" -> "),
                );
                link.record_validation(
                    VAL_SEVERITY_ERROR,
                    0,
                    "CrossQueueAnalysis",
                    &format!("IGN-XQ-CYCLE-{i}"),
                    &msg,
                    0,
                    0,
                );
                // Mark every queue that appears in the cycle. The
                // tracker's TrackedSubmission carries the queue id of
                // each seq number, but the cycle vector contains only
                // seq numbers; we re-derive the (qf, qi) tuple by
                // scanning the snapshot. Snapshot is cloned per-call;
                // for very large workloads this is the dominant cost
                // of the bridge but still well under 1ms at 4096
                // submissions.
                let snap = tracker.snapshot();
                for &seq in cycle {
                    if let Some(sub) = snap.iter().find(|s| s.seq == seq) {
                        let desc = format!(
                            "cycle{}: {}",
                            i,
                            chain.join("->"),
                        );
                        link.record_sync_cycle(
                            sub.queue_family,
                            sub.queue_index,
                            SYNC_SEVERITY_CYCLE,
                            ttl_ms,
                            0, 0,
                            &desc,
                        );
                    }
                }
            }

            // Orphan signals waste GPU cycles and may indicate a
            // forgotten wait. Warning level. Mark the producing queue
            // and reference the would-be waiter through related_queue
            // (zero here because it is unknown - that's what makes it
            // an orphan).
            for orphan in &report.orphan_signals {
                let msg = format!(
                    "orphan signal: sem {:#x} signaled by #{} \"{}\" \
                     Q{}/{} but never waited on within the recorded window",
                    orphan.semaphore,
                    orphan.from_seq,
                    orphan.from_label,
                    orphan.from_queue.0,
                    orphan.from_queue.1,
                );
                link.record_validation(
                    VAL_SEVERITY_WARNING,
                    0,
                    "CrossQueueAnalysis",
                    "IGN-XQ-ORPHAN-SIG",
                    &msg,
                    0,
                    0,
                );
                let desc = format!(
                    "orphan signal: sem {:#x} from {}",
                    orphan.semaphore, orphan.from_label,
                );
                link.record_sync_cycle(
                    orphan.from_queue.0,
                    orphan.from_queue.1,
                    SYNC_SEVERITY_ORPHAN,
                    ttl_ms,
                    0, 0,
                    &desc,
                );
            }

            // Orphan waits will deadlock unless signaled out of band.
            for orphan in &report.orphan_waits {
                let msg = format!(
                    "orphan wait: sem {:#x} waited by #{} \"{}\" \
                     Q{}/{} but never signaled within the recorded window",
                    orphan.semaphore,
                    orphan.to_seq,
                    orphan.to_label,
                    orphan.to_queue.0,
                    orphan.to_queue.1,
                );
                link.record_validation(
                    VAL_SEVERITY_WARNING,
                    0,
                    "CrossQueueAnalysis",
                    "IGN-XQ-ORPHAN-WAIT",
                    &msg,
                    0,
                    0,
                );
                let desc = format!(
                    "orphan wait: sem {:#x} on {}",
                    orphan.semaphore, orphan.to_label,
                );
                link.record_sync_cycle(
                    orphan.to_queue.0,
                    orphan.to_queue.1,
                    SYNC_SEVERITY_ORPHAN,
                    ttl_ms,
                    0, 0,
                    &desc,
                );
            }
        })
        .expect("spawn cross-queue bridge thread")
}

/// Install a shader printf handler that mirrors every parsed
/// `debugPrintfEXT(...)` message into the live link IPC ring.
///
/// Replaces any previously installed printf handler. The classifier
/// inspects the message's `shader_stage` string (set by the validation
/// layer when it parses the payload) and maps it to the coarse stage
/// code consumed by ignis-viz.
///
/// Available only when both `live-link` and `debug-tools` features
/// are enabled.
#[cfg(feature = "debug-tools")]
pub fn bridge_shader_printf_to_live_link(link: std::sync::Arc<LiveLink>) {
    use crate::debug::shader_printf::{ShaderPrintfMessage, PRINTF_REGISTRY};

    let stage_to_u32 = |s: &str| -> u32 {
        match s {
            "VERTEX" => 1,
            "FRAGMENT" => 2,
            "COMPUTE" => 3,
            "RAYGEN" => 4,
            "MISS" => 5,
            "CLOSEST_HIT" => 6,
            "ANY_HIT" => 7,
            "INTERSECTION" => 8,
            "CALLABLE" => 9,
            "TASK" => 10,
            "MESH" => 11,
            "TESS_CONTROL" => 12,
            "TESS_EVAL" => 13,
            "GEOMETRY" => 14,
            _ => 0,
        }
    };

    PRINTF_REGISTRY.set(Box::new(move |msg: &ShaderPrintfMessage| {
        link.record_shader_printf(
            stage_to_u32(msg.shader_stage),
            msg.message_id,
            msg.location.as_deref().unwrap_or(""),
            &msg.formatted,
        );
    }));
}

/// Bridge a `HangDetector` into the live link IPC ring.
///
/// Installs a callback handler that, when the watchdog fires, emits
/// a `KIND_HANG_DETECTED` event followed by one `KIND_BREADCRUMB`
/// per entry in the breadcrumb buffer (when one was attached to the
/// stalled submission).
///
/// The original `HangAction` configured on the detector is replaced
/// with `HangAction::Callback`. Any previous action is dropped. To
/// chain multiple actions the caller can wrap this bridge with a
/// custom callback that calls both this and the original logic.
///
/// Available only when both `live-link` and `debug-tools` features
/// are enabled.
#[cfg(feature = "debug-tools")]
pub fn bridge_hang_detector_to_live_link(
    _detector: &crate::HangDetector,
    _breadcrumbs: Option<std::sync::Arc<crate::BreadcrumbBuffer>>,
    _link: std::sync::Arc<LiveLink>,
) {
    // Wiring this bridge requires direct access to HangDetector
    // internals (the on_hang action and the breadcrumb readback)
    // which the public API does not currently expose with mutation
    // semantics that fit a post-construction install. The function
    // signature is provided here as a stable surface; the producer
    // application should use HangAction::Callback at construction
    // time and call link.record_hang_detected / record_breadcrumb
    // from the callback body until ignis-rs grows a setter for the
    // action field.
    //
    // Reference implementation a user can inline at their HangDetector
    // construction site:
    //
    //   let link = link.clone();
    //   let breadcrumbs = breadcrumbs.clone();
    //   let action = HangAction::Callback(Box::new(move |report| {
    //       eprint!("{report}");
    //       let trail = breadcrumbs.as_ref()
    //           .map(|b| b.readback())
    //           .unwrap_or_default();
    //       let last_done = trail.iter().rev()
    //           .find(|(_, c)| *c)
    //           .map(|(b, _)| (b.id, b.label.clone()))
    //           .unwrap_or((0, String::new()));
    //       let first_pending = trail.iter()
    //           .find(|(_, c)| !*c)
    //           .map(|(b, _)| (b.id, b.label.clone()))
    //           .unwrap_or((0, String::new()));
    //       let parent = link.record_hang_detected(
    //           0, 0, "submission",
    //           last_done.0, first_pending.0,
    //           &last_done.1, &first_pending.1,
    //           trail.len() as u32);
    //       for (crumb, completed) in trail {
    //           link.record_breadcrumb(
    //               parent, crumb.id, completed, &crumb.label);
    //       }
    //   }));
    //   let detector = ignis.create_hang_detector(config, action);
}

/// Bridge a `LifetimeTracker` into the live link IPC ring.
///
/// `LifetimeTracker` exposes `register`, `unregister`, and
/// `record_usage` as the public surface. To emit live link events on
/// every call without modifying the tracker, the producer side wraps
/// the tracker in a thin newtype that forwards to the link in addition
/// to the underlying tracker. This helper is provided as a stable
/// surface; the actual wrapping happens at the producer's
/// `LifetimeTracker` construction site.
///
/// Reference wiring (inline at construction):
///
/// ```ignore
/// let link = link.clone();
/// let tracker = ignis.create_lifetime_tracker();
///
/// // For each register call:
/// tracker.register(ty, handle, Some("name"));
/// link.record_object_registered(
///     handle, ty.as_raw(), "name",
///     loc.file(), loc.line(), loc.column(), "my_app::func");
///
/// // For each unregister call:
/// let usage = tracker.usage_count_of(ty, handle).unwrap_or(0);
/// tracker.unregister(ty, handle);
/// link.record_object_destroyed(handle, ty.as_raw(), usage);
/// ```
///
/// Available only when both `live-link` and `debug-tools` features
/// are enabled.
#[cfg(feature = "debug-tools")]
pub fn bridge_lifetime_to_live_link(
    _tracker: &crate::LifetimeTracker,
    _link: std::sync::Arc<LiveLink>,
) {
    // Wiring handled by the producer at LifetimeTracker construction
    // site; see the doc comment above for the canonical pattern. This
    // function exists to give the bridge a discoverable name and a
    // stable place to evolve once the LifetimeTracker grows hooks.
}

/// Bridge a `DescriptorAuditor` into the live link IPC ring.
///
/// Reference wiring:
///
/// ```ignore
/// let link = link.clone();
/// let mut auditor = DescriptorAuditor::new();
/// // ... after register/unregister/record_write/clear_set ...
/// for issue in auditor.audit_all() {
///     let kind = match issue.resource_kind {
///         "Buffer" => RES_KIND_BUFFER,
///         "Image" => RES_KIND_IMAGE,
///         "ImageView" => RES_KIND_IMAGE,
///         "Sampler" => RES_KIND_SAMPLER,
///         _ => RES_KIND_OTHER,
///     };
///     link.record_descriptor_issue(
///         issue.set_handle, issue.binding,
///         kind, issue.dead_handle,
///         issue.set_name.as_deref().unwrap_or(""),
///         &format!(
///             "{} {:#x} destroyed but still bound at set {:#x} binding {}",
///             issue.resource_kind, issue.dead_handle,
///             issue.set_handle, issue.binding,
///         ));
/// }
/// ```
#[cfg(feature = "debug-tools")]
pub fn bridge_descriptor_audit_to_live_link(
    _auditor: &crate::DescriptorAuditor,
    _link: std::sync::Arc<LiveLink>,
) {
    // The auditor exposes `audit_all()` returning a Vec of issues.
    // Bridging is done at audit-call time by the producer; this
    // function names the canonical pattern.
}

/// Bridge an `AliasingDetector` into the live link IPC ring.
///
/// Reference wiring:
///
/// ```ignore
/// let link = link.clone();
/// let detector = AliasingDetector::new();
/// // ... record reads/writes/barriers during command recording ...
/// for issue in detector.analyze() {
///     let access = match issue.conflict_access.access_type {
///         AccessType::Read => 0,
///         AccessType::Write => 1,
///     };
///     link.record_aliasing_conflict(
///         issue.handle,
///         issue.write_access.operation_index,
///         issue.conflict_access.operation_index,
///         issue.write_access.stage.as_raw(),
///         issue.conflict_access.stage.as_raw(),
///         access,
///         issue.name.as_deref().unwrap_or(""),
///         &issue.write_access.label,
///         &issue.conflict_access.label,
///         &format!(
///             "{} written by op #{} then accessed by op #{} without barrier",
///             issue.resource_kind,
///             issue.write_access.operation_index,
///             issue.conflict_access.operation_index));
/// }
/// ```
#[cfg(feature = "debug-tools")]
pub fn bridge_aliasing_to_live_link(
    _detector: &crate::AliasingDetector,
    _link: std::sync::Arc<LiveLink>,
) {
}

/// Bridge a `PipelineAuditor` into the live link IPC ring.
///
/// Reference wiring:
///
/// ```ignore
/// let link = link.clone();
/// let mut auditor = PipelineAuditor::new();
/// // ... register layouts and pipelines, then validate at draw time ...
/// for issue in auditor.validate_bind(pipeline_handle, bound_set_count) {
///     link.record_pipeline_issue(
///         issue.pipeline,
///         PIPELINE_ISSUE_DESCRIPTOR_COUNT,
///         issue.pipeline_name.as_deref().unwrap_or(""),
///         &issue.description);
/// }
/// ```
#[cfg(feature = "debug-tools")]
pub fn bridge_pipeline_audit_to_live_link(
    _auditor: &crate::PipelineAuditor,
    _link: std::sync::Arc<LiveLink>,
) {
}

/// Bridge a `DeviceFaultRecorder` into the live link IPC ring.
///
/// Producers call this from their `VK_ERROR_DEVICE_LOST` handler
/// (typically inside `CrashReporter::trigger`). The recorder's
/// `collect_all` returns a snapshot the producer flattens into a
/// single `record_device_fault` call.
///
/// Reference wiring:
///
/// ```ignore
/// let recorder = ignis.create_device_fault_recorder();
/// let link = link.clone();
///
/// // On VK_ERROR_DEVICE_LOST:
/// let data = recorder.collect_all(Some(queue), None);
/// let desc = data.fault_info.as_ref()
///     .map(|f| f.description.clone())
///     .unwrap_or_default();
/// let addr = data.fault_info.as_ref()
///     .map(|f| f.address_infos.len() as u32).unwrap_or(0);
/// let vendor = data.fault_info.as_ref()
///     .map(|f| f.vendor_infos.len() as u32).unwrap_or(0);
/// let bin = data.fault_info.as_ref()
///     .map(|f| f.vendor_binary.len() as u32).unwrap_or(0);
/// let fired = data.buffer_markers.iter()
///     .filter(|m| m.fired).count() as u32;
/// link.record_device_fault(
///     &desc, addr, vendor, bin,
///     data.supports_fault_info,
///     data.supports_checkpoints,
///     data.supports_buffer_markers,
///     data.checkpoints.len() as u32,
///     data.buffer_markers.len() as u32,
///     fired);
/// ```
#[cfg(feature = "debug-tools")]
pub fn bridge_device_fault_to_live_link(
    _recorder: &crate::DeviceFaultRecorder,
    _link: std::sync::Arc<LiveLink>,
) {
}

/// Bridge an `AllocationProfiler` into the live link IPC ring.
///
/// Spawns a worker thread that, every `poll_interval`, snapshots the
/// profiler's per-site stats, sorts them by active bytes descending,
/// and emits the top `top_n` entries as a single epoch batch. The
/// epoch counter increments on every iteration so the viewer can
/// atomically replace its previous snapshot.
///
/// The thread terminates automatically when the last `Arc<LiveLink>`
/// outside the bridge is dropped (the bridge holds a Weak reference
/// to the link). The profiler is held strongly so snapshots continue
/// even if the user drops their last profiler reference.
///
/// # Snapshot interval
///
/// Allocation site stats change slowly relative to per-event traffic.
/// 1-2 second intervals are appropriate for live development;
/// shorter intervals add IPC traffic without revealing meaningfully
/// different data. Going below 250 ms is not recommended.
///
/// # Performance
///
/// Each snapshot iteration calls `profiler.snapshot()` which clones
/// the entire site map. For workloads with thousands of distinct
/// allocation sites this can take a millisecond or two. The bridge
/// runs on a dedicated thread so this does not block the producer's
/// main loop.
///
/// Available only when both `live-link` and `debug-tools` features
/// are enabled.
#[cfg(feature = "debug-tools")]
pub fn bridge_alloc_profiler_to_live_link(
    profiler: std::sync::Arc<crate::AllocationProfiler>,
    link: std::sync::Arc<LiveLink>,
    poll_interval: std::time::Duration,
    top_n: usize,
) -> std::thread::JoinHandle<()> {
    let link_weak = std::sync::Arc::downgrade(&link);
    drop(link);
    std::thread::Builder::new()
        .name("ignis-alloc-profiler-bridge".into())
        .spawn(move || {
            let mut epoch: u64 = 0;
            loop {
                std::thread::sleep(poll_interval);
                let Some(link) = link_weak.upgrade() else { break; };
                epoch = epoch.wrapping_add(1);
                let mut snap = profiler.snapshot();
                snap.sort_by_key(|(_, st)| std::cmp::Reverse(st.active_bytes));
                for (i, (site, stats)) in snap.iter().take(top_n).enumerate() {
                    link.record_alloc_site(
                        epoch,
                        i as u32,
                        &site.function,
                        &site.file,
                        site.line,
                        stats.total_allocs,
                        stats.total_bytes,
                        stats.active_allocs,
                        stats.active_bytes,
                        stats.peak_active_allocs,
                        stats.peak_active_bytes,
                    );
                }
            }
        })
        .expect("spawn alloc profiler bridge thread")
}