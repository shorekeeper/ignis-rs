//! Resource access timeline (ring buffer of GPU operations).
//!
//! [`ResourceTrace`] is a thread-safe, lock-protected ring buffer of
//! [`TraceEvent`]s. Each event represents one observable GPU-related
//! operation: a queue submission, a memory allocation, a free, an image
//! or buffer layout transition, a frame graph pass, or a user-defined
//! custom event. Events are timestamped with nanosecond resolution
//! relative to the trace creation time, tagged with the producer thread,
//! and held in FIFO order. When the buffer is full, the oldest event is
//! evicted.
//!
//! Two consumers:
//!
//! 1. **Chrome Tracing JSON exporter** ([`export_chrome_string`],
//!    [`export_chrome_json`]). Output can be opened in `chrome://tracing`,
//!    Perfetto UI, or any other Chrome Trace Format viewer. Submissions
//!    and passes appear as duration events ("phase X"); allocations,
//!    frees, and transitions appear as instant events ("phase i").
//!
//! 2. **Real-time debug window** ([`debug_window`](crate::debug_window),
//!    feature `debug-window`), which renders a live timeline panel from
//!    the same data via `snapshot()`.
//!
//! The trace is opt-in. No subsystem produces events unless it has been
//! given an `Arc<ResourceTrace>` explicitly. [`AllocationProfiler`] is
//! the main producer right now via
//! [`AllocationProfiler::with_trace`](crate::AllocationProfiler::with_trace).
//! Other subsystems (queue, frame graph, command recorder) can record
//! manually via the `record_*` methods.
//!
//! [`AllocationProfiler`]: crate::AllocationProfiler
//! [`export_chrome_string`]: ResourceTrace::export_chrome_string
//! [`export_chrome_json`]: ResourceTrace::export_chrome_json

use std::collections::VecDeque;
use std::path::Path;
use std::sync::{Arc, Mutex};
use std::time::Instant;

/// Different kinds of events captured by the trace.
#[derive(Debug, Clone)]
pub enum TraceEventKind {
    /// A queue submission completed (or was recorded). `duration_ns` may
    /// be 0 if the GPU duration is unknown at capture time.
    Submission {
        /// Queue family index.
        queue_family: u32,
        /// Queue index within the family.
        queue_index: u32,
        /// User-supplied label for the submission.
        label: String,
        /// Approximate GPU-side duration if measured, else 0.
        duration_ns: u64,
    },
    /// A memory allocation occurred.
    Allocation {
        /// Raw `VkDeviceMemory` handle.
        memory: u64,
        /// Byte offset within the memory object.
        offset: u64,
        /// Allocation size in bytes.
        size: u64,
        /// Source location (file:line:col or function name) of the
        /// allocation, if known.
        site: String,
    },
    /// A memory free occurred.
    Free {
        /// Raw `VkDeviceMemory` handle.
        memory: u64,
        /// Byte offset within the memory object.
        offset: u64,
        /// Allocation size in bytes that was freed.
        size: u64,
    },
    /// A resource layout / access transition was emitted.
    Transition {
        /// `"Image"` or `"Buffer"`.
        resource_kind: &'static str,
        /// Raw resource handle.
        handle: u64,
        /// Human-readable transition description (e.g. `"UNDEFINED -> COLOR_ATTACHMENT_OPTIMAL"`).
        description: String,
    },
    /// A frame graph pass executed.
    Pass {
        /// Pass name as registered in the frame graph.
        name: String,
        /// CPU recording duration in nanoseconds.
        duration_ns: u64,
    },
    /// User-defined custom event.
    Custom {
        /// Free-form category string for grouping.
        category: String,
        /// Event name.
        name: String,
        /// Optional argument string (free-form, will be embedded in JSON).
        args: String,
    },
}

/// One trace event with its time, thread, and kind-specific data.
#[derive(Debug, Clone)]
pub struct TraceEvent {
    /// Nanoseconds since trace creation.
    pub timestamp_ns: u64,
    /// Hash of the producing thread's id (for visual lane separation).
    pub thread_id: u64,
    /// Event payload.
    pub kind: TraceEventKind,
}

impl TraceEvent {
    /// Short human-readable category name useful for filtering and
    /// color-mapping. Mirrors Chrome Trace Format `cat` semantics.
    pub fn category(&self) -> &'static str {
        match self.kind {
            TraceEventKind::Submission { .. } => "submit",
            TraceEventKind::Allocation { .. } => "alloc",
            TraceEventKind::Free { .. } => "free",
            TraceEventKind::Transition { .. } => "transition",
            TraceEventKind::Pass { .. } => "pass",
            TraceEventKind::Custom { .. } => "custom",
        }
    }
}

/// Aggregate stats over the current trace contents.
#[derive(Debug, Clone, Default)]
pub struct TraceStats {
    /// Number of submission events.
    pub submissions: u64,
    /// Number of allocation events.
    pub allocations: u64,
    /// Number of free events.
    pub frees: u64,
    /// Number of transition events.
    pub transitions: u64,
    /// Number of pass events.
    pub passes: u64,
    /// Number of custom events.
    pub custom: u64,
    /// Total events currently in the ring.
    pub total: u64,
    /// Lifetime cumulative count, including evicted events.
    pub lifetime_total: u64,
}

/// Thread-safe ring buffer of GPU trace events.
pub struct ResourceTrace {
    start: Instant,
    events: Mutex<VecDeque<TraceEvent>>,
    capacity: usize,
    lifetime_total: Mutex<u64>,
}

impl ResourceTrace {
    /// Create a new trace with the given ring capacity.
    ///
    /// Once `capacity` events accumulate, every new event evicts the
    /// oldest one. A typical capacity for a debug window session is
    /// 4000 to 16000 events.
    pub fn new(capacity: usize) -> Arc<Self> {
        Arc::new(Self {
            start: Instant::now(),
            events: Mutex::new(VecDeque::with_capacity(capacity.min(1024))),
            capacity: capacity.max(1),
            lifetime_total: Mutex::new(0),
        })
    }

    /// Wall time since trace creation, in nanoseconds.
    pub fn now_ns(&self) -> u64 {
        self.start.elapsed().as_nanos() as u64
    }

    fn push(&self, kind: TraceEventKind) {
        let evt = TraceEvent {
            timestamp_ns: self.now_ns(),
            thread_id: thread_id_hash(),
            kind,
        };
        let mut q = self.events.lock().unwrap();
        if q.len() >= self.capacity {
            q.pop_front();
        }
        q.push_back(evt);
        let mut tot = self.lifetime_total.lock().unwrap();
        *tot = tot.saturating_add(1);
    }

    /// Record a queue submission event.
    pub fn record_submission(
        &self,
        queue_family: u32,
        queue_index: u32,
        label: &str,
        duration_ns: u64,
    ) {
        self.push(TraceEventKind::Submission {
            queue_family,
            queue_index,
            label: label.to_string(),
            duration_ns,
        });
    }

    /// Record an allocation event.
    pub fn record_allocation(&self, memory: u64, offset: u64, size: u64, site: &str) {
        self.push(TraceEventKind::Allocation {
            memory,
            offset,
            size,
            site: site.to_string(),
        });
    }

    /// Record a free event.
    pub fn record_free(&self, memory: u64, offset: u64, size: u64) {
        self.push(TraceEventKind::Free {
            memory,
            offset,
            size,
        });
    }

    /// Record a resource transition.
    pub fn record_transition(
        &self,
        resource_kind: &'static str,
        handle: u64,
        description: &str,
    ) {
        self.push(TraceEventKind::Transition {
            resource_kind,
            handle,
            description: description.to_string(),
        });
    }

    /// Record a frame graph pass execution.
    pub fn record_pass(&self, name: &str, duration_ns: u64) {
        self.push(TraceEventKind::Pass {
            name: name.to_string(),
            duration_ns,
        });
    }

    /// Record a user-defined custom event.
    pub fn record_custom(&self, category: &str, name: &str, args: &str) {
        self.push(TraceEventKind::Custom {
            category: category.to_string(),
            name: name.to_string(),
            args: args.to_string(),
        });
    }

    /// Snapshot all events currently in the ring.
    pub fn snapshot(&self) -> Vec<TraceEvent> {
        self.events.lock().unwrap().iter().cloned().collect()
    }

    /// Number of events currently in the ring.
    pub fn len(&self) -> usize {
        self.events.lock().unwrap().len()
    }

    /// Whether the ring is empty.
    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    /// Iterate over every event in the ring under the internal lock,
    /// without cloning. Use this in hot paths (e.g. per-frame timeline
    /// rendering) to avoid the heap traffic that `snapshot()` causes
    /// when the ring contains thousands of events with `String` payloads.
    ///
    /// The closure runs while the lock is held; do not call back into
    /// `record_*` from within it (would deadlock).
    pub fn for_each<F: FnMut(&TraceEvent)>(&self, mut f: F) {
        let q = self.events.lock().unwrap();
        for e in q.iter() {
            f(e);
        }
    }
    
    /// Aggregate stats over the current trace contents.
    pub fn stats(&self) -> TraceStats {
        let q = self.events.lock().unwrap();
        let mut s = TraceStats {
            total: q.len() as u64,
            lifetime_total: *self.lifetime_total.lock().unwrap(),
            ..Default::default()
        };
        for e in q.iter() {
            match e.kind {
                TraceEventKind::Submission { .. } => s.submissions += 1,
                TraceEventKind::Allocation { .. } => s.allocations += 1,
                TraceEventKind::Free { .. } => s.frees += 1,
                TraceEventKind::Transition { .. } => s.transitions += 1,
                TraceEventKind::Pass { .. } => s.passes += 1,
                TraceEventKind::Custom { .. } => s.custom += 1,
            }
        }
        s
    }

    /// Clear all events. Lifetime counter is preserved.
    pub fn clear(&self) {
        self.events.lock().unwrap().clear();
    }

    /// Render the trace as a Chrome Trace Format JSON string.
    ///
    /// The output is a top-level JSON array (sometimes called
    /// "JSON Array Format" in Chrome's documentation). It loads in
    /// `chrome://tracing`, Perfetto UI, and the SpeedScope viewer.
    pub fn export_chrome_string(&self) -> String {
        let q = self.events.lock().unwrap();
        let mut out = String::with_capacity(q.len() * 128 + 32);
        out.push('[');
        let mut first = true;
        for e in q.iter() {
            if !first {
                out.push(',');
            }
            first = false;
            write_chrome_event(&mut out, e);
        }
        out.push(']');
        out
    }

    /// Save the Chrome trace to a file. Equivalent to
    /// `std::fs::write(path, self.export_chrome_string())`.
    pub fn export_chrome_json(&self, path: impl AsRef<Path>) -> std::io::Result<()> {
        let s = self.export_chrome_string();
        std::fs::write(path, s)
    }
}

fn thread_id_hash() -> u64 {
    // ThreadId is opaque; format-debug it and FNV-1a hash the result.
    let id = std::thread::current().id();
    let s = format!("{:?}", id);
    let mut h: u64 = 0xCBF2_9CE4_8422_2325;
    for b in s.bytes() {
        h ^= b as u64;
        h = h.wrapping_mul(0x100_0000_01B3);
    }
    h
}

/// Write one event as a Chrome trace object into `out`.
fn write_chrome_event(out: &mut String, e: &TraceEvent) {
    let ts_us = (e.timestamp_ns as f64) / 1_000.0;
    match &e.kind {
        TraceEventKind::Submission {
            queue_family,
            queue_index,
            label,
            duration_ns,
        } => {
            let dur_us = (*duration_ns as f64) / 1_000.0;
            out.push_str(&format!(
                r#"{{"name":"{}","cat":"submit","ph":"X","ts":{:.3},"dur":{:.3},"pid":1,"tid":{},"args":{{"queue_family":{},"queue_index":{}}}}}"#,
                json_escape(label),
                ts_us,
                dur_us.max(0.001),
                e.thread_id,
                queue_family,
                queue_index,
            ));
        }
        TraceEventKind::Allocation {
            memory,
            offset,
            size,
            site,
        } => {
            out.push_str(&format!(
                r#"{{"name":"alloc","cat":"alloc","ph":"i","ts":{:.3},"pid":1,"tid":{},"args":{{"memory":"{:#x}","offset":{},"size":{},"site":"{}"}}}}"#,
                ts_us,
                e.thread_id,
                memory,
                offset,
                size,
                json_escape(site),
            ));
        }
        TraceEventKind::Free {
            memory,
            offset,
            size,
        } => {
            out.push_str(&format!(
                r#"{{"name":"free","cat":"free","ph":"i","ts":{:.3},"pid":1,"tid":{},"args":{{"memory":"{:#x}","offset":{},"size":{}}}}}"#,
                ts_us, e.thread_id, memory, offset, size,
            ));
        }
        TraceEventKind::Transition {
            resource_kind,
            handle,
            description,
        } => {
            out.push_str(&format!(
                r#"{{"name":"{}","cat":"transition","ph":"i","ts":{:.3},"pid":1,"tid":{},"args":{{"kind":"{}","handle":"{:#x}"}}}}"#,
                json_escape(description),
                ts_us,
                e.thread_id,
                resource_kind,
                handle,
            ));
        }
        TraceEventKind::Pass { name, duration_ns } => {
            let dur_us = (*duration_ns as f64) / 1_000.0;
            out.push_str(&format!(
                r#"{{"name":"{}","cat":"pass","ph":"X","ts":{:.3},"dur":{:.3},"pid":1,"tid":{}}}"#,
                json_escape(name),
                ts_us,
                dur_us.max(0.001),
                e.thread_id,
            ));
        }
        TraceEventKind::Custom {
            category,
            name,
            args,
        } => {
            out.push_str(&format!(
                r#"{{"name":"{}","cat":"{}","ph":"i","ts":{:.3},"pid":1,"tid":{},"args":{{"data":"{}"}}}}"#,
                json_escape(name),
                json_escape(category),
                ts_us,
                e.thread_id,
                json_escape(args),
            ));
        }
    }
}

fn json_escape(s: &str) -> String {
    let mut out = String::with_capacity(s.len() + 2);
    for c in s.chars() {
        match c {
            '"' => out.push_str("\\\""),
            '\\' => out.push_str("\\\\"),
            '\n' => out.push_str("\\n"),
            '\r' => out.push_str("\\r"),
            '\t' => out.push_str("\\t"),
            c if (c as u32) < 0x20 => {
                out.push_str(&format!("\\u{:04x}", c as u32));
            }
            c => out.push(c),
        }
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn ring_evicts_oldest() {
        let trace = ResourceTrace::new(3);
        trace.record_custom("c", "a", "");
        trace.record_custom("c", "b", "");
        trace.record_custom("c", "c", "");
        trace.record_custom("c", "d", "");
        let snap = trace.snapshot();
        assert_eq!(snap.len(), 3);
        assert!(matches!(&snap[0].kind, TraceEventKind::Custom { name, .. } if name == "b"));
        assert!(matches!(&snap[2].kind, TraceEventKind::Custom { name, .. } if name == "d"));
        assert_eq!(trace.stats().lifetime_total, 4);
    }

    #[test]
    fn stats_count_per_kind() {
        let trace = ResourceTrace::new(100);
        trace.record_allocation(0x1, 0, 100, "src/foo.rs:1");
        trace.record_allocation(0x1, 100, 200, "src/foo.rs:1");
        trace.record_free(0x1, 0, 100);
        trace.record_submission(0, 0, "main_submit", 12_000);
        trace.record_pass("geometry", 5_000);
        trace.record_transition("Image", 0xABCD, "UNDEFINED -> COLOR_ATTACHMENT_OPTIMAL");
        trace.record_custom("user", "frame_start", "");
        let s = trace.stats();
        assert_eq!(s.allocations, 2);
        assert_eq!(s.frees, 1);
        assert_eq!(s.submissions, 1);
        assert_eq!(s.passes, 1);
        assert_eq!(s.transitions, 1);
        assert_eq!(s.custom, 1);
        assert_eq!(s.total, 7);
    }

    #[test]
    fn timestamps_monotonic() {
        let trace = ResourceTrace::new(100);
        for i in 0..16 {
            trace.record_custom("t", &format!("e{i}"), "");
        }
        let snap = trace.snapshot();
        for w in snap.windows(2) {
            assert!(w[0].timestamp_ns <= w[1].timestamp_ns);
        }
    }

    #[test]
    fn export_chrome_string_is_valid_json_array() {
        let trace = ResourceTrace::new(100);
        trace.record_allocation(0x1234, 0, 100, "test::site");
        trace.record_submission(0, 0, "submit_a", 1500);
        trace.record_pass("shadow_pass", 800);
        let json = trace.export_chrome_string();
        assert!(json.starts_with('['));
        assert!(json.ends_with(']'));
        assert!(json.contains("\"cat\":\"alloc\""));
        assert!(json.contains("\"cat\":\"submit\""));
        assert!(json.contains("\"cat\":\"pass\""));
        assert!(json.contains("\"ph\":\"X\""));
        assert!(json.contains("\"ph\":\"i\""));
        // Number of opening braces matches number of events plus one for
        // each "args" object.
        let event_count = json.matches("\"ph\":").count();
        assert_eq!(event_count, 3);
    }

    #[test]
    fn export_chrome_string_handles_empty() {
        let trace = ResourceTrace::new(10);
        let json = trace.export_chrome_string();
        assert_eq!(json, "[]");
    }

    #[test]
    fn json_escape_special_chars() {
        assert_eq!(json_escape("a\"b"), "a\\\"b");
        assert_eq!(json_escape("a\\b"), "a\\\\b");
        assert_eq!(json_escape("a\nb"), "a\\nb");
        assert_eq!(json_escape("plain"), "plain");
    }

    #[test]
    fn category_helper_matches_kind() {
        let trace = ResourceTrace::new(10);
        trace.record_allocation(0, 0, 0, "");
        trace.record_submission(0, 0, "", 0);
        let snap = trace.snapshot();
        assert_eq!(snap[0].category(), "alloc");
        assert_eq!(snap[1].category(), "submit");
    }

    #[test]
    fn clear_resets_buffer_keeps_lifetime() {
        let trace = ResourceTrace::new(10);
        for _ in 0..5 {
            trace.record_custom("c", "x", "");
        }
        trace.clear();
        assert_eq!(trace.len(), 0);
        assert_eq!(trace.stats().lifetime_total, 5);
    }

    #[test]
    fn export_chrome_json_writes_file() {
        let trace = ResourceTrace::new(10);
        trace.record_allocation(0xDEAD, 0, 64, "test_site");
        let mut path = std::env::temp_dir();
        path.push(format!("ignis_trace_test_{}.json", std::process::id()));
        trace.export_chrome_json(&path).unwrap();
        let contents = std::fs::read_to_string(&path).unwrap();
        assert!(contents.contains("test_site"));
        let _ = std::fs::remove_file(&path);
    }
}