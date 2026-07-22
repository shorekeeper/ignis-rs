//! Allocation site profiler (heaptrack for GPU memory).
//!
//! [`AllocationProfiler`] is a decorator allocator that captures the call
//! site of every allocation via runtime backtrace, accumulates per-site
//! statistics (count, bytes, peak), and produces ranked reports of the
//! biggest memory consumers.
//!
//! # Mechanism
//!
//! On each `allocate` call the profiler walks the current backtrace,
//! filters out internal Rust runtime and ignis library frames, and treats
//! the first remaining frame as the user's call site. The site
//! `(function, file, line)` is used as a hash map key for accumulating
//! [`SiteStats`]: total allocations, total bytes, currently active
//! allocations, currently active bytes, peak active count, peak active
//! bytes.
//!
//! Each individual allocation is also recorded with its `VkDeviceMemory`,
//! offset, size, and timestamp, enabling the [`MemoryVisualizer`] to
//! render layout diagrams from the same data.
//!
//! # Performance
//!
//! Backtrace capture costs a few microseconds per allocation. Acceptable
//! for development and testing builds. For production builds running tens
//! of thousands of allocations per frame, either disable backtraces via
//! [`disable_backtraces`](AllocationProfiler::disable_backtraces) (statistics
//! still tracked, all sites collapse to `<unknown>`) or skip wrapping the
//! allocator entirely.
//!
//! # Composability
//!
//! Wraps any [`Allocator`]. Decorators stack:
//! `AllocationProfiler` -> `HardenedAllocator` -> `BlockAllocator`.
//! Each layer sees the same call site through the inherited backtrace
//! frames, so wrapping order does not affect site attribution.
//!
//! [`MemoryVisualizer`]: super::memory_viz::MemoryVisualizer
//!
//! # Example
//!
//! ```rust,no_run
//! # use ignis::*;
//! # use std::sync::Arc;
//! # fn example(ignis: &Ignis) -> Result<()> {
//! let profiler = ignis.create_profiled_block_allocator();
//!
//! // Use as a regular allocator (Arc<AllocationProfiler> coerces to Arc<dyn Allocator>).
//! let allocator: Arc<dyn Allocator> = profiler.clone();
//! let _buf = ignis.create_buffer_with(&allocator, &BufferInfo::staging(1024))?;
//! let _vbo = ignis.create_buffer_with(&allocator, &BufferInfo::vertex(4096, MemoryLocation::GpuOnly))?;
//!
//! // Reports remain accessible through the AllocationProfiler handle.
//! eprintln!("{}", profiler.report_top_sites(10));
//! # Ok(())
//! # }
//! ```

use std::collections::HashMap;
use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant};

use ash::vk;
use ash::vk::Handle;

use crate::error::Result;
use crate::memory::allocator::{Allocation, Allocator};
use crate::memory::resources::MemoryLocation;

/// Source location identifying where an allocation was performed.
///
/// Identity is `(function, file, line)`. Two allocations from the same
/// expression in the same Rust function produce equal `CallSite` values
/// and thus contribute to the same [`SiteStats`].
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct CallSite {
    /// Demangled function name with Rust hash suffix stripped.
    pub function: String,
    /// Source file path as reported by the backtrace machinery.
    pub file: String,
    /// 1-based source line number.
    pub line: u32,
}

impl std::fmt::Display for CallSite {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        if self.line > 0 {
            write!(f, "{}:{} ({})", self.file, self.line, self.function)
        } else {
            write!(f, "<unknown site>")
        }
    }
}

impl CallSite {
    /// Sentinel returned when backtrace parsing fails or capture is
    /// disabled. All allocations with a missing call site collapse to
    /// this single entry, so they remain countable but not attributable.
    pub fn unknown() -> Self {
        Self {
            function: "<unknown>".to_string(),
            file: "<unknown>".to_string(),
            line: 0,
        }
    }
}

/// Aggregate statistics for one call site over the profiler's lifetime.
///
/// All counters are monotonically updated by `allocate`/`free` calls.
/// `peak_*` fields track the maximum value of `active_*` ever observed,
/// which is useful for capacity planning even if the resource has since
/// been freed.
#[derive(Debug, Clone, Default)]
pub struct SiteStats {
    /// Total allocations performed at this site (lifetime cumulative).
    pub total_allocs: u64,
    /// Total bytes allocated at this site (lifetime cumulative).
    pub total_bytes: u64,
    /// Currently live allocations from this site.
    pub active_allocs: u64,
    /// Currently live bytes from this site.
    pub active_bytes: u64,
    /// Peak simultaneous live allocations.
    pub peak_active_allocs: u64,
    /// Peak simultaneous live bytes.
    pub peak_active_bytes: u64,
}

/// Snapshot of a single live allocation.
///
/// Returned by [`AllocationProfiler::live_allocations`] and consumed by
/// the [`MemoryVisualizer`](super::memory_viz::MemoryVisualizer) for
/// rendering memory layout diagrams.
#[derive(Debug, Clone)]
pub struct LiveAllocation {
    /// `VkDeviceMemory` containing this allocation.
    pub memory: vk::DeviceMemory,
    /// Byte offset within `memory`.
    pub offset: vk::DeviceSize,
    /// Allocation size in bytes.
    pub size: vk::DeviceSize,
    /// Memory type index from the device's memory properties.
    pub memory_type_index: u32,
    /// Call site that produced this allocation.
    pub site: CallSite,
    /// Time elapsed since the allocation was made.
    pub age: Duration,
}

#[derive(Debug, Clone)]
struct AllocRecord {
    site: CallSite,
    size: u64,
    memory_type_index: u32,
    memory: vk::DeviceMemory,
    offset: vk::DeviceSize,
    timestamp: Instant,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
struct AllocKey {
    memory_raw: u64,
    offset: u64,
}

struct ProfilerState {
    sites: HashMap<CallSite, SiteStats>,
    live_allocs: HashMap<AllocKey, AllocRecord>,
    total_allocations: u64,
    total_frees: u64,
    capture_backtraces: bool,
    trace: Option<Arc<crate::resource_trace::ResourceTrace>>,
    /// Optional bridge to the cross-process live link channel.
    /// When set, every allocation and free is mirrored into the IPC
    /// ring so an attached ignis-viz instance sees them in real time.
    /// Compiled out unless the `live-link` feature is active to avoid
    /// pulling the live_link module into builds that do not need it.
    #[cfg(feature = "live-link")]
    live_link: Option<Arc<crate::live_link::LiveLink>>
}

/// Decorator allocator that records per-call-site statistics.
///
/// Implements [`Allocator`] so it can be passed wherever any allocator is
/// expected. Returns from `Ignis::create_alloc_profiler` as
/// `Arc<AllocationProfiler>`; the profiler-specific methods (snapshot,
/// reports) remain accessible through that handle while a separate
/// `Arc<dyn Allocator>` clone can be used as the allocator.
pub struct AllocationProfiler {
    inner: Arc<dyn Allocator>,
    state: Mutex<ProfilerState>,
}

impl AllocationProfiler {
    /// Wrap the given allocator with profiling.
    pub fn new(inner: Arc<dyn Allocator>) -> Arc<Self> {
        Arc::new(Self {
            inner,
            state: Mutex::new(ProfilerState {
                sites: HashMap::new(),
                live_allocs: HashMap::new(),
                total_allocations: 0,
                total_frees: 0,
                capture_backtraces: true,
                trace: None,
                #[cfg(feature = "live-link")]
                live_link: None,
            }),
        })
    }

    /// Disable backtrace capture. All subsequent allocations record their
    /// site as [`CallSite::unknown`]. Useful for benchmarking the cost of
    /// the profiler vs the underlying allocator, or for shipping builds
    /// where call site attribution is not needed but counters are.
    pub fn disable_backtraces(&self) {
        self.state.lock().unwrap().capture_backtraces = false;
    }

    /// Re-enable backtrace capture (default state after construction).
    pub fn enable_backtraces(&self) {
        self.state.lock().unwrap().capture_backtraces = true;
    }

    /// Mirror every allocation and free into the given resource trace.
    /// Pass `None` to disable trace mirroring (default state).
    pub fn with_trace(&self, trace: Option<Arc<crate::resource_trace::ResourceTrace>>) {
        self.state.lock().unwrap().trace = trace;
    }

    /// Snapshot all per-site statistics. The returned vector is unordered;
    /// callers that need ranking should sort by the field of interest.
    pub fn snapshot(&self) -> Vec<(CallSite, SiteStats)> {
        let state = self.state.lock().unwrap();
        state
            .sites
            .iter()
            .map(|(s, st)| (s.clone(), st.clone()))
            .collect()
    }

    /// Total allocations recorded since profiler creation.
    pub fn total_allocations(&self) -> u64 {
        self.state.lock().unwrap().total_allocations
    }

    /// Total frees recorded since profiler creation.
    pub fn total_frees(&self) -> u64 {
        self.state.lock().unwrap().total_frees
    }

    /// Currently live allocation count across all sites.
    pub fn active_allocations(&self) -> u64 {
        self.state.lock().unwrap().live_allocs.len() as u64
    }

    /// Currently live bytes across all sites.
    pub fn active_bytes(&self) -> u64 {
        self.state
            .lock()
            .unwrap()
            .live_allocs
            .values()
            .map(|r| r.size)
            .sum()
    }

    /// All currently live allocations.
    ///
    /// Each entry is a clone of the internal record. Suitable for feeding
    /// to [`MemoryVisualizer`](super::memory_viz::MemoryVisualizer) or for
    /// custom analyses (e.g. distribution histograms by size).
    pub fn live_allocations(&self) -> Vec<LiveAllocation> {
        let state = self.state.lock().unwrap();
        state
            .live_allocs
            .values()
            .map(|r| LiveAllocation {
                memory: r.memory,
                offset: r.offset,
                size: r.size,
                memory_type_index: r.memory_type_index,
                site: r.site.clone(),
                age: r.timestamp.elapsed(),
            })
            .collect()
    }

    /// Generate a formatted report of the top N call sites by active bytes.
    ///
    /// Uses the same diagnostic frame as the rest of the crate (severity:
    /// info, code: `IGN-PROF`). Each entry shows active bytes, active
    /// count, peak values, the call site, and lifetime cumulative totals.
    pub fn report_top_sites(&self, n: usize) -> String {
        use crate::diagnostic::{
            format_bytes, write_diagnostic_end, write_header, write_pipe, write_pipe_empty,
            write_pipe_raw, write_section, Severity, Style,
        };

        let s = Style::detect();
        let mut sites = self.snapshot();
        sites.sort_by_key(|(_, st)| std::cmp::Reverse(st.active_bytes));

        let total_active_bytes: u64 = sites.iter().map(|(_, st)| st.active_bytes).sum();
        let total_active_allocs: u64 = sites.iter().map(|(_, st)| st.active_allocs).sum();
        let total_lifetime_allocs: u64 = sites.iter().map(|(_, st)| st.total_allocs).sum();

        let mut o = String::with_capacity(2048);

        let count_to_show = n.min(sites.len());
        write_header(
            &mut o,
            &s,
            &Severity::Info,
            "IGN-PROF",
            &format!(
                "top {} of {} allocation sites by active bytes",
                count_to_show,
                sites.len()
            ),
        );
        write_pipe_empty(&mut o, &s);

        write_pipe(
            &mut o,
            &s,
            &format!(
                "active: {} across {} allocs",
                format_bytes(total_active_bytes),
                total_active_allocs
            ),
        );
        write_pipe(
            &mut o,
            &s,
            &format!("lifetime allocations: {}", total_lifetime_allocs),
        );

        if count_to_show == 0 {
            write_pipe_empty(&mut o, &s);
            write_pipe(&mut o, &s, "no allocations recorded");
            write_diagnostic_end(&mut o, &s, &Severity::Info);
            return o;
        }

        write_section(&mut o, &s, "Top Sites");

        for (i, (site, stats)) in sites.iter().take(count_to_show).enumerate() {
            let pct = if total_active_bytes > 0 {
                (stats.active_bytes as f64 / total_active_bytes as f64) * 100.0
            } else {
                0.0
            };
            write_pipe_raw(
                &mut o,
                &s,
                &format!(
                    "  {} {} {} ({:>5.1}%) | {} active | peak {} / {}",
                    s.bold(&format!("#{:<3}", i + 1)),
                    s.bold_yellow(&format!("{:>10}", format_bytes(stats.active_bytes))),
                    s.dim("active"),
                    pct,
                    stats.active_allocs,
                    format_bytes(stats.peak_active_bytes),
                    stats.peak_active_allocs,
                ),
            );
            write_pipe_raw(
                &mut o,
                &s,
                &format!("       {} {}", s.dim("at"), s.underline(&site.to_string())),
            );
            write_pipe_raw(
                &mut o,
                &s,
                &format!(
                    "       {} cumulative: {} across {} allocs",
                    s.dim("=="),
                    format_bytes(stats.total_bytes),
                    stats.total_allocs,
                ),
            );
        }

        write_diagnostic_end(&mut o, &s, &Severity::Info);
        o
    }

    /// Reset all statistics. Live allocation tracking is preserved (so
    /// frees of pre-reset allocations still update active counts), but
    /// the historical totals and peaks return to zero.
    pub fn reset_stats(&self) {
        let mut state = self.state.lock().unwrap();
        for stats in state.sites.values_mut() {
            stats.total_allocs = stats.active_allocs;
            stats.total_bytes = stats.active_bytes;
            stats.peak_active_allocs = stats.active_allocs;
            stats.peak_active_bytes = stats.active_bytes;
        }
        state.total_allocations = state.live_allocs.len() as u64;
        state.total_frees = 0;
    }
    /// Mirror every allocation and free into the given live link ring.
    /// Pass `None` to detach. Operates orthogonally to `with_trace`:
    /// both can be active simultaneously and will see the same events.
    ///
    /// The live link is held under the same mutex as the rest of the
    /// profiler state. Bridging adds one extra atomic ring write per
    /// allocation, negligible compared to the cost of the call site
    /// backtrace capture itself.
    #[cfg(feature = "live-link")]
    pub fn with_live_link(
        &self,
        link: Option<Arc<crate::live_link::LiveLink>>,
    ) {
        self.state.lock().unwrap().live_link = link;
    }
}

impl Allocator for AllocationProfiler {
    fn allocate(
        &self,
        requirements: &vk::MemoryRequirements,
        location: MemoryLocation,
    ) -> Result<Allocation> {
        // Allocate first so we don't capture a backtrace for failing paths.
        let alloc = self.inner.allocate(requirements, location)?;

        // Decide whether to capture a backtrace under a brief lock.
        let capture = self.state.lock().unwrap().capture_backtraces;
        let site = if capture {
            detect_call_site()
        } else {
            CallSite::unknown()
        };

        let key = AllocKey {
            memory_raw: alloc.memory.as_raw(),
            offset: alloc.offset,
        };
        let record = AllocRecord {
            site: site.clone(),
            size: alloc.size,
            memory_type_index: alloc.memory_type_index,
            memory: alloc.memory,
            offset: alloc.offset,
            timestamp: Instant::now(),
        };

        let mut state = self.state.lock().unwrap();
        state.total_allocations += 1;
        state.live_allocs.insert(key, record.clone());
        let stats = state.sites.entry(site).or_default();
        stats.total_allocs += 1;
        stats.total_bytes += alloc.size;
        stats.active_allocs += 1;
        stats.active_bytes += alloc.size;
        if stats.active_allocs > stats.peak_active_allocs {
            stats.peak_active_allocs = stats.active_allocs;
        }
        if stats.active_bytes > stats.peak_active_bytes {
            stats.peak_active_bytes = stats.active_bytes;
        }

        let trace_clone = state.trace.clone();
        #[cfg(feature = "live-link")]
        let link_clone = state.live_link.clone();
        drop(state);

        if let Some(trace) = trace_clone {
            trace.record_allocation(
                alloc.memory.as_raw(),
                alloc.offset,
                alloc.size,
                &record.site.to_string(),
            );
        }

        #[cfg(feature = "live-link")]
        if let Some(link) = link_clone {
            link.record_allocation(
                alloc.memory.as_raw(),
                alloc.offset,
                alloc.size,
                &record.site.to_string(),
            );
        }

        Ok(alloc)
    }

    fn free(&self, allocation: &Allocation) {
        let key = AllocKey {
            memory_raw: allocation.memory.as_raw(),
            offset: allocation.offset,
        };

        let trace_clone;
        #[cfg(feature = "live-link")]
        let link_clone;
        {
            let mut state = self.state.lock().unwrap();
            state.total_frees += 1;
            if let Some(record) = state.live_allocs.remove(&key) {
                if let Some(stats) = state.sites.get_mut(&record.site) {
                    stats.active_allocs = stats.active_allocs.saturating_sub(1);
                    stats.active_bytes = stats.active_bytes.saturating_sub(record.size);
                }
            }
            trace_clone = state.trace.clone();
            #[cfg(feature = "live-link")]
            {
                link_clone = state.live_link.clone();
            }
        }

        if let Some(trace) = trace_clone {
            trace.record_free(
                allocation.memory.as_raw(),
                allocation.offset,
                allocation.size,
            );
        }

        #[cfg(feature = "live-link")]
        if let Some(link) = link_clone {
            link.record_free(
                allocation.memory.as_raw(),
                allocation.offset,
                allocation.size,
            );
        }

        self.inner.free(allocation);
    }

    fn name(&self) -> &str {
        "AllocationProfiler"
    }
}

/// Capture the current backtrace and find the first non-ignis frame.
///
/// Walks the textual representation of `std::backtrace::Backtrace` looking
/// for a "function name" line followed by an "at file:line:col" location
/// line. Frames whose function names match [`should_skip_frame`] are
/// ignored. The first surviving location becomes the call site.
fn detect_call_site() -> CallSite {
    let bt = std::backtrace::Backtrace::force_capture();
    let text = bt.to_string();

    let mut current_func: Option<String> = None;

    for line in text.lines() {
        let trimmed = line.trim_start();

        if let Some(loc_str) = trimmed.strip_prefix("at ") {
            let func = current_func.clone().unwrap_or_default();
            if should_skip_frame(&func) {
                continue;
            }
            // Parse "path:line:col" or "path:line". Path may contain ':'
            // on Windows ("C:\..."), so we split from the right.
            let parts: Vec<&str> = loc_str.rsplitn(3, ':').collect();
            if parts.len() == 3 {
                if let Ok(line_no) = parts[1].parse::<u32>() {
                    return CallSite {
                        function: clean_function_name(&func),
                        file: parts[2].to_string(),
                        line: line_no,
                    };
                }
            } else if parts.len() == 2 {
                if let Ok(line_no) = parts[0].parse::<u32>() {
                    return CallSite {
                        function: clean_function_name(&func),
                        file: parts[1].to_string(),
                        line: line_no,
                    };
                }
            }
        } else if trimmed.chars().next().is_some_and(|c| c.is_ascii_digit()) {
            // Function-name line. Format: "N: 0xADDR - funcname" or "N: funcname".
            if let Some(idx) = trimmed.find(" - ") {
                current_func = Some(trimmed[idx + 3..].to_string());
            } else if let Some(colon_idx) = trimmed.find(": ") {
                let after = &trimmed[colon_idx + 2..];
                current_func = Some(after.to_string());
            }
        }
    }

    CallSite::unknown()
}

/// Returns true for stack frames that should be skipped during call-site
/// detection: frames inside ignis itself, the standard library, the
/// allocator runtime, and the backtrace capture machinery.
fn should_skip_frame(func: &str) -> bool {
    func.contains("ignis::")
        || func.starts_with("std::")
        || func.starts_with("core::")
        || func.starts_with("alloc::")
        || func.starts_with("__rust_")
        || func.contains("backtrace::")
        || func.contains("Backtrace")
        || func.contains("AllocationProfiler")
        || func.contains("detect_call_site")
        || func.is_empty()
}

/// Strip the Rust hash suffix from a demangled function name:
/// `my_app::foo::h0123456789abcdef` -> `my_app::foo`.
fn clean_function_name(raw: &str) -> String {
    if let Some(idx) = raw.rfind("::h") {
        let suffix = &raw[idx + 3..];
        if !suffix.is_empty() && suffix.chars().all(|c| c.is_ascii_hexdigit()) {
            return raw[..idx].to_string();
        }
    }
    raw.to_string()
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Mock allocator: hands out monotonically increasing offsets within a
    /// single fake memory handle. Sufficient for testing the profiler's
    /// state machine without a real Vulkan device.
    struct MockAllocator {
        next_offset: Mutex<u64>,
        memory_raw: u64,
    }

    impl MockAllocator {
        fn new() -> Self {
            Self {
                next_offset: Mutex::new(0),
                memory_raw: 1,
            }
        }
    }

    impl Allocator for MockAllocator {
        fn allocate(
            &self,
            req: &vk::MemoryRequirements,
            _loc: MemoryLocation,
        ) -> Result<Allocation> {
            let mut off = self.next_offset.lock().unwrap();
            let offset = *off;
            *off += req.size.max(1);
            Ok(Allocation {
                memory: vk::DeviceMemory::from_raw(self.memory_raw),
                offset,
                size: req.size,
                mapped_ptr: None,
                memory_type_index: 0,
            })
        }

        fn free(&self, _allocation: &Allocation) {
            // Mock: no-op.
        }

        fn name(&self) -> &str {
            "MockAllocator"
        }
    }

    fn req(size: u64) -> vk::MemoryRequirements {
        vk::MemoryRequirements {
            size,
            alignment: 1,
            memory_type_bits: 1,
        }
    }

    #[test]
    fn tracks_total_allocations() {
        let inner = Arc::new(MockAllocator::new());
        let profiler = AllocationProfiler::new(inner);
        let _a = profiler.allocate(&req(64), MemoryLocation::GpuOnly).unwrap();
        let _b = profiler.allocate(&req(128), MemoryLocation::GpuOnly).unwrap();
        assert_eq!(profiler.total_allocations(), 2);
        assert_eq!(profiler.active_allocations(), 2);
        assert_eq!(profiler.active_bytes(), 192);
        assert_eq!(profiler.total_frees(), 0);
    }

    #[test]
    fn tracks_frees_and_decrements() {
        let inner = Arc::new(MockAllocator::new());
        let profiler = AllocationProfiler::new(inner);
        let a = profiler.allocate(&req(64), MemoryLocation::GpuOnly).unwrap();
        let b = profiler.allocate(&req(128), MemoryLocation::GpuOnly).unwrap();
        profiler.free(&a);
        assert_eq!(profiler.total_frees(), 1);
        assert_eq!(profiler.active_allocations(), 1);
        assert_eq!(profiler.active_bytes(), 128);
        profiler.free(&b);
        assert_eq!(profiler.total_frees(), 2);
        assert_eq!(profiler.active_allocations(), 0);
        assert_eq!(profiler.active_bytes(), 0);
    }

    #[test]
    fn site_stats_aggregate_correctly() {
        let inner = Arc::new(MockAllocator::new());
        let profiler = AllocationProfiler::new(inner);
        // Disable backtraces so all 5 allocations attribute to one site.
        profiler.disable_backtraces();

        let mut allocs = Vec::new();
        for _ in 0..5 {
            allocs.push(profiler.allocate(&req(100), MemoryLocation::GpuOnly).unwrap());
        }

        let snap = profiler.snapshot();
        assert_eq!(snap.len(), 1, "all allocs collapse to the unknown site");
        let (site, stats) = &snap[0];
        assert_eq!(*site, CallSite::unknown());
        assert_eq!(stats.total_allocs, 5);
        assert_eq!(stats.total_bytes, 500);
        assert_eq!(stats.active_allocs, 5);
        assert_eq!(stats.active_bytes, 500);
        assert_eq!(stats.peak_active_allocs, 5);
        assert_eq!(stats.peak_active_bytes, 500);

        for a in &allocs {
            profiler.free(a);
        }
    }

    #[test]
    fn peak_tracking_retains_max() {
        let inner = Arc::new(MockAllocator::new());
        let profiler = AllocationProfiler::new(inner);
        profiler.disable_backtraces();

        let a = profiler.allocate(&req(100), MemoryLocation::GpuOnly).unwrap();
        let b = profiler.allocate(&req(200), MemoryLocation::GpuOnly).unwrap();
        let c = profiler.allocate(&req(300), MemoryLocation::GpuOnly).unwrap();
        // Peak: 600 bytes / 3 allocs.

        profiler.free(&a);
        profiler.free(&b);
        // Active: 300 bytes / 1 alloc, but peaks must remain at 600/3.

        let snap = profiler.snapshot();
        let (_, stats) = &snap[0];
        assert_eq!(stats.active_bytes, 300);
        assert_eq!(stats.active_allocs, 1);
        assert_eq!(stats.peak_active_bytes, 600);
        assert_eq!(stats.peak_active_allocs, 3);

        profiler.free(&c);
    }

    #[test]
    fn live_allocations_reflect_state() {
        let inner = Arc::new(MockAllocator::new());
        let profiler = AllocationProfiler::new(inner);
        profiler.disable_backtraces();

        let a = profiler.allocate(&req(64), MemoryLocation::GpuOnly).unwrap();
        let b = profiler.allocate(&req(128), MemoryLocation::GpuOnly).unwrap();

        let live = profiler.live_allocations();
        assert_eq!(live.len(), 2);
        let total: u64 = live.iter().map(|l| l.size).sum();
        assert_eq!(total, 192);

        profiler.free(&a);
        let live = profiler.live_allocations();
        assert_eq!(live.len(), 1);
        assert_eq!(live[0].size, 128);

        profiler.free(&b);
    }

    #[test]
    fn report_with_no_allocs_does_not_panic() {
        let inner = Arc::new(MockAllocator::new());
        let profiler = AllocationProfiler::new(inner);
        let report = profiler.report_top_sites(10);
        assert!(report.contains("IGN-PROF"));
    }

    #[test]
    fn report_contains_top_sites() {
        let inner = Arc::new(MockAllocator::new());
        let profiler = AllocationProfiler::new(inner);
        profiler.disable_backtraces();
        let _a = profiler.allocate(&req(2048), MemoryLocation::GpuOnly).unwrap();
        let report = profiler.report_top_sites(5);
        assert!(report.contains("IGN-PROF"));
        assert!(report.contains("2.0 KiB"));
    }

    #[test]
    fn reset_stats_zeros_history_keeps_active() {
        let inner = Arc::new(MockAllocator::new());
        let profiler = AllocationProfiler::new(inner);
        profiler.disable_backtraces();

        let a = profiler.allocate(&req(100), MemoryLocation::GpuOnly).unwrap();
        let b = profiler.allocate(&req(200), MemoryLocation::GpuOnly).unwrap();
        profiler.free(&a);

        // total=2, frees=1, active=200/1
        profiler.reset_stats();

        assert_eq!(profiler.total_frees(), 0);
        // Lifetime allocations should now equal currently live count.
        assert_eq!(profiler.total_allocations(), 1);
        assert_eq!(profiler.active_allocations(), 1);
        assert_eq!(profiler.active_bytes(), 200);

        profiler.free(&b);
    }

    #[test]
    fn clean_function_name_strips_hash_suffix() {
        assert_eq!(
            clean_function_name("my_app::foo::h0123456789abcdef"),
            "my_app::foo"
        );
        assert_eq!(clean_function_name("my_app::foo"), "my_app::foo");
        assert_eq!(clean_function_name("foo::bar::h"), "foo::bar::h");
        assert_eq!(clean_function_name("foo::bar::hxyz"), "foo::bar::hxyz");
    }

    #[test]
    fn skip_frame_filter_classifies_correctly() {
        assert!(should_skip_frame("ignis::Ignis::create_buffer"));
        assert!(should_skip_frame("ignis::memory::allocator::BlockAllocator::allocate"));
        assert!(should_skip_frame("std::sync::Arc::new"));
        assert!(should_skip_frame("core::ops::Drop::drop"));
        assert!(should_skip_frame("alloc::sync::Arc"));
        assert!(should_skip_frame("backtrace::capture::Backtrace::create"));
        assert!(should_skip_frame(""));
        assert!(!should_skip_frame("my_app::renderer::load"));
        assert!(!should_skip_frame("test_executable::main"));
        assert!(!should_skip_frame("game::scene::Scene::update"));
    }

    #[test]
    fn callsite_display_format() {
        let site = CallSite {
            function: "my::func".to_string(),
            file: "src/foo.rs".to_string(),
            line: 42,
        };
        assert_eq!(format!("{}", site), "src/foo.rs:42 (my::func)");

        let unknown = CallSite::unknown();
        assert_eq!(format!("{}", unknown), "<unknown site>");
    }

    #[test]
    fn concurrent_allocations_are_safe() {
        // Ensure the Mutex-based state machine handles parallel access.
        let inner = Arc::new(MockAllocator::new());
        let profiler = AllocationProfiler::new(inner);
        profiler.disable_backtraces();

        let prof_clone = Arc::clone(&profiler);
        let handles: Vec<_> = (0..4)
            .map(|_| {
                let p = Arc::clone(&prof_clone);
                std::thread::spawn(move || {
                    let mut allocs = Vec::new();
                    for _ in 0..50 {
                        allocs.push(p.allocate(&req(32), MemoryLocation::GpuOnly).unwrap());
                    }
                    for a in &allocs {
                        p.free(a);
                    }
                })
            })
            .collect();

        for h in handles {
            h.join().unwrap();
        }

        assert_eq!(profiler.total_allocations(), 200);
        assert_eq!(profiler.total_frees(), 200);
        assert_eq!(profiler.active_allocations(), 0);
    }
}