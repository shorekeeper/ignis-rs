//! Cross-queue submission tracker.
//!
//! Records every queue submission with its semaphore wait/signal lists
//! and fence handle, then analyzes the recorded graph to detect:
//!
//! - **Orphan signals**: a semaphore signaled by some submission but
//!   never waited on. Wastes GPU cycles and may indicate a forgotten
//!   wait that will manifest as visual corruption later.
//! - **Orphan waits**: a semaphore waited on but never signaled in the
//!   recorded data window. Will deadlock unless signaled out of band.
//! - **Cross-queue edges**: explicit dependencies where one queue's
//!   output feeds another queue's input via semaphore. Useful for
//!   understanding pipeline parallelism.
//! - **Cycles**: two or more submissions whose dependencies form a
//!   directed cycle in the wait/signal graph. Guaranteed deadlock.
//! - **Longest chain**: maximum depth of serialized dependencies. High
//!   values indicate poor parallelism between queues.
//!
//! # Data sources
//!
//! The tracker is a passive recorder. Users either:
//!
//! 1. Call [`record`](CrossQueueTracker::record) directly around their
//!    `vkQueueSubmit` calls, or
//! 2. Attach a [`SubmissionJournal`] and call
//!    [`import_from_journal`](CrossQueueTracker::import_from_journal)
//!    to bulk-load existing data.
//!
//! Both methods can be combined; the tracker assigns a fresh sequence
//! number to every record regardless of source.
//!
//! # Example
//!
//! ```rust,no_run
//! # use ignis::*;
//! # use ash::vk;
//! # fn example(ctx: &Ignis) -> Result<()> {
//! let tracker = ctx.create_cross_queue_tracker();
//!
//! // ... user records submissions ...
//!
//! let report = tracker.analyze();
//! if report.has_cycles() {
//!     panic!("deadlock detected:\n{report}");
//! }
//! if report.has_orphans() {
//!     eprintln!("warning: {report}");
//! }
//! # Ok(())
//! # }
//! ```
//!
//! [`SubmissionJournal`]: super::journal::SubmissionJournal

use std::collections::{HashMap, HashSet};
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::Mutex;
use std::time::Instant;

use ash::vk;
use ash::vk::Handle;

use crate::diagnostic::{
    self, write_diagnostic_end, write_header, write_kv, write_pipe, write_pipe_empty,
    write_pipe_raw, write_section, Severity, Style,
};

#[cfg(feature = "debug-tools")]
use super::journal::SubmissionJournal;

/// Default ring buffer capacity for tracked submissions.
const DEFAULT_CAPACITY: usize = 4096;

/// One recorded submission with its synchronization data.
#[derive(Debug, Clone)]
pub struct TrackedSubmission {
    /// Monotonic sequence number assigned by the tracker.
    pub seq: u64,
    /// Wall-clock timestamp of the record call.
    pub timestamp: Instant,
    /// Queue family index.
    pub queue_family: u32,
    /// Queue index within family.
    pub queue_index: u32,
    /// User-supplied label.
    pub label: String,
    /// Command buffer handles (raw u64 to avoid `vk::CommandBuffer` borrow lifetimes).
    pub command_buffers: Vec<u64>,
    /// Wait semaphore handles.
    pub wait_semaphores: Vec<u64>,
    /// Signal semaphore handles.
    pub signal_semaphores: Vec<u64>,
    /// Fence handle, or 0 if none.
    pub fence: u64,
}

impl TrackedSubmission {
    /// Composite queue identifier `(family, index)`.
    pub fn queue_id(&self) -> (u32, u32) {
        (self.queue_family, self.queue_index)
    }
}

/// One cross-queue dependency edge: signal -> wait via shared semaphore.
#[derive(Debug, Clone)]
pub struct CrossQueueEdge {
    /// Sequence number of the signaling submission.
    pub from_seq: u64,
    /// Queue identifier of the signaling submission.
    pub from_queue: (u32, u32),
    /// Label of the signaling submission.
    pub from_label: String,
    /// Sequence number of the waiting submission.
    pub to_seq: u64,
    /// Queue identifier of the waiting submission.
    pub to_queue: (u32, u32),
    /// Label of the waiting submission.
    pub to_label: String,
    /// Semaphore handle that connects them.
    pub via_semaphore: u64,
}

/// A semaphore signaled but never waited on.
#[derive(Debug, Clone)]
pub struct OrphanSignal {
    /// Semaphore handle.
    pub semaphore: u64,
    /// Sequence number of the signaler.
    pub from_seq: u64,
    /// Label of the signaler.
    pub from_label: String,
    /// Queue identifier of the signaler.
    pub from_queue: (u32, u32),
}

/// A semaphore waited on but never signaled in the recorded data.
#[derive(Debug, Clone)]
pub struct OrphanWait {
    /// Semaphore handle.
    pub semaphore: u64,
    /// Sequence number of the waiter.
    pub to_seq: u64,
    /// Label of the waiter.
    pub to_label: String,
    /// Queue identifier of the waiter.
    pub to_queue: (u32, u32),
}

/// Result of analyzing the tracker contents.
#[derive(Debug, Clone)]
pub struct CrossQueueReport {
    /// Number of submissions analyzed.
    pub submission_count: usize,
    /// Number of distinct queues observed.
    pub queue_count: usize,
    /// All cross-queue dependency edges (signaler and waiter on different queues).
    pub cross_queue_edges: Vec<CrossQueueEdge>,
    /// Same-queue dependency edges (informational).
    pub same_queue_edges: Vec<CrossQueueEdge>,
    /// Orphan signals.
    pub orphan_signals: Vec<OrphanSignal>,
    /// Orphan waits.
    pub orphan_waits: Vec<OrphanWait>,
    /// Detected cycles. Each entry is an ordered list of submission seq
    /// numbers. Empty if the graph is acyclic.
    pub cycles: Vec<Vec<u64>>,
    /// Longest dependency chain (sequence of seq numbers). Empty if
    /// the graph contains cycles.
    pub longest_chain: Vec<u64>,
}

impl CrossQueueReport {
    /// Whether any structural issue was detected (cycles or orphans).
    pub fn has_issues(&self) -> bool {
        self.has_cycles() || self.has_orphans()
    }

    /// Whether the dependency graph has cycles.
    pub fn has_cycles(&self) -> bool {
        !self.cycles.is_empty()
    }

    /// Whether any orphan signals or orphan waits exist.
    pub fn has_orphans(&self) -> bool {
        !self.orphan_signals.is_empty() || !self.orphan_waits.is_empty()
    }
}

impl std::fmt::Display for CrossQueueReport {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let s = Style::detect();
        let mut o = String::with_capacity(2048);

        let sev = if self.has_cycles() {
            Severity::Error
        } else if self.has_orphans() {
            Severity::Warning
        } else {
            Severity::Info
        };

        let title = if self.has_cycles() {
            format!(
                "cross-queue analysis: {} cycle(s) detected (DEADLOCK)",
                self.cycles.len()
            )
        } else if self.has_orphans() {
            format!(
                "cross-queue analysis: {} orphan signal(s), {} orphan wait(s)",
                self.orphan_signals.len(),
                self.orphan_waits.len()
            )
        } else {
            "cross-queue analysis: no issues detected".to_string()
        };

        write_header(&mut o, &s, &sev, "IGN-XQ001", &title);
        write_pipe_empty(&mut o, &s);

        write_kv(&mut o, &s, "Submissions", &self.submission_count.to_string());
        write_kv(&mut o, &s, "Distinct queues", &self.queue_count.to_string());
        write_kv(
            &mut o,
            &s,
            "Cross-queue edges",
            &self.cross_queue_edges.len().to_string(),
        );
        write_kv(
            &mut o,
            &s,
            "Same-queue edges",
            &self.same_queue_edges.len().to_string(),
        );
        let chain_hops = self.longest_chain.len().saturating_sub(1);
        write_kv(
            &mut o,
            &s,
            "Longest chain",
            &format!(
                "{} hop(s){}",
                chain_hops,
                if self.has_cycles() {
                    " (cycles present, chain analysis skipped)"
                } else {
                    ""
                }
            ),
        );

        if !self.cross_queue_edges.is_empty() {
            write_section(&mut o, &s, "Cross-Queue Edges");
            for e in self.cross_queue_edges.iter().take(12) {
                write_pipe_raw(
                    &mut o,
                    &s,
                    &format!(
                        "  #{} {:<24} Q{}/{}  -> #{} {:<24} Q{}/{}  via sem {:#x}",
                        e.from_seq,
                        truncate(&e.from_label, 24),
                        e.from_queue.0,
                        e.from_queue.1,
                        e.to_seq,
                        truncate(&e.to_label, 24),
                        e.to_queue.0,
                        e.to_queue.1,
                        e.via_semaphore,
                    ),
                );
            }
            if self.cross_queue_edges.len() > 12 {
                write_pipe_raw(
                    &mut o,
                    &s,
                    &s.dim(&format!(
                        "  ... {} more",
                        self.cross_queue_edges.len() - 12
                    )),
                );
            }
        }

        if !self.cycles.is_empty() {
            write_section(&mut o, &s, "Cycles (Guaranteed Deadlock)");
            for (i, cycle) in self.cycles.iter().enumerate() {
                let chain: Vec<String> = cycle.iter().map(|n| format!("#{}", n)).collect();
                write_pipe_raw(
                    &mut o,
                    &s,
                    &format!("  cycle {}: {}", i, chain.join(s.bold_red(" -> ").as_str())),
                );
            }
            write_pipe_empty(&mut o, &s);
            write_pipe(
                &mut o,
                &s,
                "two or more submissions wait on each other directly or",
            );
            write_pipe(
                &mut o,
                &s,
                "transitively through a chain of semaphore dependencies.",
            );
            write_pipe(
                &mut o,
                &s,
                "Vulkan will deadlock when this graph is submitted.",
            );
        }

        if !self.orphan_signals.is_empty() {
            write_section(
                &mut o,
                &s,
                &format!(
                    "Orphan Signals ({} total)",
                    self.orphan_signals.len()
                ),
            );
            for o_sig in self.orphan_signals.iter().take(8) {
                write_pipe_raw(
                    &mut o,
                    &s,
                    &format!(
                        "  sem {:#x} signaled by #{} \"{}\" Q{}/{}",
                        o_sig.semaphore,
                        o_sig.from_seq,
                        o_sig.from_label,
                        o_sig.from_queue.0,
                        o_sig.from_queue.1
                    ),
                );
            }
            if self.orphan_signals.len() > 8 {
                write_pipe_raw(
                    &mut o,
                    &s,
                    &s.dim(&format!(
                        "  ... {} more",
                        self.orphan_signals.len() - 8
                    )),
                );
            }
        }

        if !self.orphan_waits.is_empty() {
            write_section(
                &mut o,
                &s,
                &format!("Orphan Waits ({} total)", self.orphan_waits.len()),
            );
            for o_wait in self.orphan_waits.iter().take(8) {
                write_pipe_raw(
                    &mut o,
                    &s,
                    &format!(
                        "  sem {:#x} waited by #{} \"{}\" Q{}/{}",
                        o_wait.semaphore,
                        o_wait.to_seq,
                        o_wait.to_label,
                        o_wait.to_queue.0,
                        o_wait.to_queue.1
                    ),
                );
            }
            if self.orphan_waits.len() > 8 {
                write_pipe_raw(
                    &mut o,
                    &s,
                    &s.dim(&format!(
                        "  ... {} more",
                        self.orphan_waits.len() - 8
                    )),
                );
            }
        }

        if !self.has_issues() && self.submission_count > 0 {
            write_pipe_empty(&mut o, &s);
            write_pipe_raw(
                &mut o,
                &s,
                &s.bold_green("  ✓ no synchronization issues detected"),
            );
        }

        write_diagnostic_end(&mut o, &s, &sev);
        f.write_str(&o)
    }
}

/// Cross-queue submission tracker.
///
/// Constructed via [`Ignis::create_cross_queue_tracker`] or directly via
/// [`CrossQueueTracker::new`]. Thread-safe; record calls are protected
/// by an internal mutex.
///
/// [`Ignis::create_cross_queue_tracker`]: crate::Ignis::create_cross_queue_tracker
pub struct CrossQueueTracker {
    submissions: Mutex<Vec<TrackedSubmission>>,
    capacity: usize,
    next_seq: AtomicU64,
}

impl CrossQueueTracker {
    /// Construct with default capacity (4096 submissions).
    pub fn new() -> Self {
        Self::with_capacity(DEFAULT_CAPACITY)
    }

    /// Construct with a specific ring buffer capacity. When full, the
    /// oldest entry is evicted on each new record.
    pub fn with_capacity(capacity: usize) -> Self {
        Self {
            submissions: Mutex::new(Vec::with_capacity(capacity.min(1024))),
            capacity: capacity.max(1),
            next_seq: AtomicU64::new(1),
        }
    }

    /// Record a submission using Vulkan handles. Convenience for users
    /// who already have `vk::CommandBuffer` / `vk::Semaphore` handles.
    pub fn record(
        &self,
        queue_family: u32,
        queue_index: u32,
        label: &str,
        command_buffers: &[vk::CommandBuffer],
        wait_semaphores: &[vk::Semaphore],
        signal_semaphores: &[vk::Semaphore],
        fence: vk::Fence,
    ) -> u64 {
        self.record_raw(
            queue_family,
            queue_index,
            label,
            &command_buffers.iter().map(|h| h.as_raw()).collect::<Vec<_>>(),
            &wait_semaphores.iter().map(|h| h.as_raw()).collect::<Vec<_>>(),
            &signal_semaphores.iter().map(|h| h.as_raw()).collect::<Vec<_>>(),
            fence.as_raw(),
        )
    }

    /// Record a submission using raw u64 handles.
    pub fn record_raw(
        &self,
        queue_family: u32,
        queue_index: u32,
        label: &str,
        command_buffers: &[u64],
        wait_semaphores: &[u64],
        signal_semaphores: &[u64],
        fence: u64,
    ) -> u64 {
        let seq = self.next_seq.fetch_add(1, Ordering::Relaxed);
        let entry = TrackedSubmission {
            seq,
            timestamp: Instant::now(),
            queue_family,
            queue_index,
            label: label.to_string(),
            command_buffers: command_buffers.to_vec(),
            wait_semaphores: wait_semaphores.to_vec(),
            signal_semaphores: signal_semaphores.to_vec(),
            fence,
        };
        let mut subs = self.submissions.lock().unwrap();
        if subs.len() >= self.capacity {
            subs.remove(0);
        }
        subs.push(entry);
        seq
    }

    /// Bulk-import every entry from a [`SubmissionJournal`] into this
    /// tracker. Each journal entry produces one tracker record with a
    /// fresh sequence number (the original journal sequence is not
    /// preserved).
    #[cfg(feature = "debug-tools")]
    pub fn import_from_journal(&self, journal: &SubmissionJournal) {
        journal.for_each_entry(|e| {
            self.record_raw(
                e.queue_family,
                e.queue_index,
                &e.label,
                &e.command_buffers,
                &e.wait_semaphores,
                &e.signal_semaphores,
                e.fence,
            );
        });
    }

    /// Snapshot all currently tracked submissions in record order.
    pub fn snapshot(&self) -> Vec<TrackedSubmission> {
        self.submissions
            .lock()
            .unwrap_or_else(|e| e.into_inner())
            .clone()
    }

    /// Discard all recorded submissions and reset the sequence counter.
    pub fn clear(&self) {
        self.submissions.lock().unwrap().clear();
        self.next_seq.store(1, Ordering::Relaxed);
    }

    /// Number of currently tracked submissions.
    pub fn len(&self) -> usize {
        self.submissions
            .lock()
            .unwrap_or_else(|e| e.into_inner())
            .len()
    }

    /// Whether the tracker has no submissions.
    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    /// Capacity of the ring buffer.
    pub fn capacity(&self) -> usize {
        self.capacity
    }

    /// Run analysis over the current submissions.
    pub fn analyze(&self) -> CrossQueueReport {
        let subs = self.submissions.lock().unwrap_or_else(|e| e.into_inner());
        analyze_inner(&subs)
    }
}

impl Default for CrossQueueTracker {
    fn default() -> Self {
        Self::new()
    }
}

// ---- Analysis ----------------------------------------------------------

fn analyze_inner(submissions: &[TrackedSubmission]) -> CrossQueueReport {
    // Map semaphore -> ordered list of submission indices that signal/wait on it.
    let mut signalers: HashMap<u64, Vec<usize>> = HashMap::new();
    let mut waiters: HashMap<u64, Vec<usize>> = HashMap::new();
    for (i, s) in submissions.iter().enumerate() {
        for sem in &s.signal_semaphores {
            signalers.entry(*sem).or_default().push(i);
        }
        for sem in &s.wait_semaphores {
            waiters.entry(*sem).or_default().push(i);
        }
    }

    // For each semaphore, pair signalers with waiters in arrival order.
    // Edge from kth signaler to kth waiter. Excess on either side -> orphan.
    let mut cross_queue_edges = Vec::new();
    let mut same_queue_edges = Vec::new();
    let mut adjacency: HashMap<u64, Vec<u64>> = HashMap::new();

    for (sem, sig_indices) in &signalers {
        let empty: Vec<usize> = Vec::new();
        let wait_indices = waiters.get(sem).unwrap_or(&empty);
        let pair_count = sig_indices.len().min(wait_indices.len());
        for k in 0..pair_count {
            let from = &submissions[sig_indices[k]];
            let to = &submissions[wait_indices[k]];
            let edge = CrossQueueEdge {
                from_seq: from.seq,
                from_queue: from.queue_id(),
                from_label: from.label.clone(),
                to_seq: to.seq,
                to_queue: to.queue_id(),
                to_label: to.label.clone(),
                via_semaphore: *sem,
            };
            if from.queue_id() != to.queue_id() {
                cross_queue_edges.push(edge);
            } else {
                same_queue_edges.push(edge);
            }
            adjacency.entry(from.seq).or_default().push(to.seq);
        }
    }

    // Orphans: excess signalers (no matching waiter) or excess waiters.
    let mut orphan_signals = Vec::new();
    let mut orphan_waits = Vec::new();
    for (sem, sig_indices) in &signalers {
        let wait_count = waiters.get(sem).map(|v| v.len()).unwrap_or(0);
        if sig_indices.len() > wait_count {
            for &i in &sig_indices[wait_count..] {
                let s = &submissions[i];
                orphan_signals.push(OrphanSignal {
                    semaphore: *sem,
                    from_seq: s.seq,
                    from_label: s.label.clone(),
                    from_queue: s.queue_id(),
                });
            }
        }
    }
    for (sem, wait_indices) in &waiters {
        let sig_count = signalers.get(sem).map(|v| v.len()).unwrap_or(0);
        if wait_indices.len() > sig_count {
            for &i in &wait_indices[sig_count..] {
                let s = &submissions[i];
                orphan_waits.push(OrphanWait {
                    semaphore: *sem,
                    to_seq: s.seq,
                    to_label: s.label.clone(),
                    to_queue: s.queue_id(),
                });
            }
        }
    }
    orphan_signals.sort_by_key(|o| o.from_seq);
    orphan_waits.sort_by_key(|o| o.to_seq);

    let cycles = detect_cycles(&adjacency);
    let longest_chain = if cycles.is_empty() {
        find_longest_chain(&adjacency, submissions)
    } else {
        Vec::new()
    };

    let mut queue_set: HashSet<(u32, u32)> = HashSet::new();
    for s in submissions {
        queue_set.insert(s.queue_id());
    }

    CrossQueueReport {
        submission_count: submissions.len(),
        queue_count: queue_set.len(),
        cross_queue_edges,
        same_queue_edges,
        orphan_signals,
        orphan_waits,
        cycles,
        longest_chain,
    }
}

/// Detect all cycles via DFS with on-stack tracking. Returns each cycle
/// as an ordered list of submission seq numbers starting from the entry
/// point of the cycle.
fn detect_cycles(adj: &HashMap<u64, Vec<u64>>) -> Vec<Vec<u64>> {
    let mut cycles: Vec<Vec<u64>> = Vec::new();
    let mut visited: HashSet<u64> = HashSet::new();
    let mut on_stack: HashSet<u64> = HashSet::new();
    let mut path: Vec<u64> = Vec::new();

    let mut nodes: Vec<u64> = adj.keys().copied().collect();
    nodes.sort();
    for &start in &nodes {
        if !visited.contains(&start) {
            dfs_find_cycles(start, adj, &mut visited, &mut on_stack, &mut path, &mut cycles);
        }
    }
    // Deduplicate cycles that may have been found from different entry points.
    cycles.sort();
    cycles.dedup();
    cycles
}

fn dfs_find_cycles(
    node: u64,
    adj: &HashMap<u64, Vec<u64>>,
    visited: &mut HashSet<u64>,
    on_stack: &mut HashSet<u64>,
    path: &mut Vec<u64>,
    cycles: &mut Vec<Vec<u64>>,
) {
    visited.insert(node);
    on_stack.insert(node);
    path.push(node);

    if let Some(neighbors) = adj.get(&node) {
        for &next in neighbors {
            if !visited.contains(&next) {
                dfs_find_cycles(next, adj, visited, on_stack, path, cycles);
            } else if on_stack.contains(&next) {
                if let Some(start_idx) = path.iter().position(|&x| x == next) {
                    let mut cycle: Vec<u64> = path[start_idx..].to_vec();
                    cycle.push(next);
                    cycles.push(cycle);
                }
            }
        }
    }

    on_stack.remove(&node);
    path.pop();
}

/// Find the longest path in the DAG. Memoized DFS — caller guarantees
/// no cycles before invoking.
fn find_longest_chain(
    adj: &HashMap<u64, Vec<u64>>,
    submissions: &[TrackedSubmission],
) -> Vec<u64> {
    let mut all_nodes: HashSet<u64> = HashSet::new();
    for s in submissions {
        all_nodes.insert(s.seq);
    }
    for (&from, tos) in adj {
        all_nodes.insert(from);
        for &to in tos {
            all_nodes.insert(to);
        }
    }

    let mut best: Vec<u64> = Vec::new();
    let mut memo: HashMap<u64, Vec<u64>> = HashMap::new();
    for &node in &all_nodes {
        let chain = longest_from(node, adj, &mut memo);
        if chain.len() > best.len() {
            best = chain;
        }
    }
    best
}

fn longest_from(
    node: u64,
    adj: &HashMap<u64, Vec<u64>>,
    memo: &mut HashMap<u64, Vec<u64>>,
) -> Vec<u64> {
    if let Some(cached) = memo.get(&node) {
        return cached.clone();
    }
    let mut best: Vec<u64> = Vec::new();
    if let Some(neighbors) = adj.get(&node) {
        for &next in neighbors {
            let sub = longest_from(next, adj, memo);
            if sub.len() > best.len() {
                best = sub;
            }
        }
    }
    let mut result = vec![node];
    result.extend(best);
    memo.insert(node, result.clone());
    result
}

fn truncate(s: &str, max: usize) -> &str {
    if s.len() <= max {
        s
    } else {
        match s.char_indices().nth(max) {
            Some((idx, _)) => &s[..idx],
            None => s,
        }
    }
}

// Suppress unused-import warning when the diagnostic module is not used
// by code paths the compiler can see (some configurations).
#[allow(dead_code)]
fn _force_diagnostic_use(_: &diagnostic::Style) {}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn empty_tracker_has_no_issues() {
        let t = CrossQueueTracker::new();
        let r = t.analyze();
        assert!(!r.has_issues());
        assert_eq!(r.submission_count, 0);
        assert_eq!(r.queue_count, 0);
    }

    #[test]
    fn linear_chain_no_cycles() {
        let t = CrossQueueTracker::new();
        // A signals 1; B waits 1, signals 2; C waits 2.
        t.record_raw(0, 0, "A", &[], &[], &[1], 0);
        t.record_raw(0, 0, "B", &[], &[1], &[2], 0);
        t.record_raw(0, 0, "C", &[], &[2], &[], 0);
        let r = t.analyze();
        assert!(r.cycles.is_empty());
        assert_eq!(r.longest_chain.len(), 3);
        assert_eq!(r.same_queue_edges.len(), 2);
        assert_eq!(r.cross_queue_edges.len(), 0);
    }

    #[test]
    fn detects_simple_cycle() {
        let t = CrossQueueTracker::new();
        // A waits sem_y, signals sem_x. B waits sem_x, signals sem_y.
        t.record_raw(0, 0, "A", &[], &[2], &[1], 0);
        t.record_raw(1, 0, "B", &[], &[1], &[2], 0);
        let r = t.analyze();
        assert!(r.has_cycles(), "expected cycle, got: {r:?}");
        assert!(r.longest_chain.is_empty());
    }

    #[test]
    fn detects_orphan_signal() {
        let t = CrossQueueTracker::new();
        t.record_raw(0, 0, "A", &[], &[], &[42], 0);
        let r = t.analyze();
        assert_eq!(r.orphan_signals.len(), 1);
        assert_eq!(r.orphan_signals[0].semaphore, 42);
        assert_eq!(r.orphan_waits.len(), 0);
    }

    #[test]
    fn detects_orphan_wait() {
        let t = CrossQueueTracker::new();
        t.record_raw(0, 0, "A", &[], &[42], &[], 0);
        let r = t.analyze();
        assert_eq!(r.orphan_waits.len(), 1);
        assert_eq!(r.orphan_signals.len(), 0);
    }

    #[test]
    fn cross_queue_edge_classified_correctly() {
        let t = CrossQueueTracker::new();
        t.record_raw(0, 0, "graphics", &[], &[], &[7], 0);
        t.record_raw(1, 0, "compute", &[], &[7], &[], 0);
        let r = t.analyze();
        assert_eq!(r.cross_queue_edges.len(), 1);
        assert_eq!(r.same_queue_edges.len(), 0);
        assert_eq!(r.queue_count, 2);
    }

    #[test]
    fn ring_buffer_evicts_oldest() {
        let t = CrossQueueTracker::with_capacity(3);
        t.record_raw(0, 0, "A", &[], &[], &[], 0);
        t.record_raw(0, 0, "B", &[], &[], &[], 0);
        t.record_raw(0, 0, "C", &[], &[], &[], 0);
        t.record_raw(0, 0, "D", &[], &[], &[], 0);
        let snap = t.snapshot();
        assert_eq!(snap.len(), 3);
        assert_eq!(snap[0].label, "B");
        assert_eq!(snap[2].label, "D");
    }

    #[test]
    fn longest_chain_three_hops() {
        let t = CrossQueueTracker::new();
        t.record_raw(0, 0, "A", &[], &[], &[1], 0);
        t.record_raw(0, 0, "B", &[], &[1], &[2], 0);
        t.record_raw(0, 0, "C", &[], &[2], &[3], 0);
        t.record_raw(0, 0, "D", &[], &[3], &[], 0);
        let r = t.analyze();
        assert_eq!(r.longest_chain.len(), 4);
    }

    #[test]
    fn diamond_pattern_no_cycle() {
        let t = CrossQueueTracker::new();
        // A signals 1+2; B waits 1, signals 3; C waits 2, signals 4; D waits 3+4.
        t.record_raw(0, 0, "A", &[], &[], &[1, 2], 0);
        t.record_raw(1, 0, "B", &[], &[1], &[3], 0);
        t.record_raw(1, 0, "C", &[], &[2], &[4], 0);
        t.record_raw(0, 0, "D", &[], &[3, 4], &[], 0);
        let r = t.analyze();
        assert!(r.cycles.is_empty());
        assert!(r.cross_queue_edges.len() >= 2);
        assert!(r.longest_chain.len() >= 3);
    }
}