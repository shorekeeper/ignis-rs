//! Submission flight recorder (black box).
//!
//! [`SubmissionJournal`] maintains a ring buffer of queue submissions,
//! capturing timestamps, queue identity, command buffers, semaphores,
//! and fences. On `VK_ERROR_DEVICE_LOST` or any other failure, the
//! journal provides a chronological record of what was in flight.

use std::collections::VecDeque;
use std::sync::atomic::{AtomicU64, AtomicU8, Ordering};
use std::sync::Mutex;
use std::time::Instant;

use ash::vk;
use ash::vk::Handle;

use crate::diagnostic::{self, Severity, Style};

/// Completion status of a journal entry.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[repr(u8)]
pub enum EntryStatus {
    /// Submitted to the GPU, not yet signaled.
    Pending = 0,
    /// Fence signaled, work completed successfully.
    Completed = 1,
    /// Vulkan error occurred (device lost, etc.).
    Error = 2,
}

/// A single submission record.
pub struct JournalEntry {
    /// Monotonically increasing sequence number.
    pub sequence: u64,
    /// Wall-clock time of submission.
    pub timestamp: Instant,
    /// Queue family index.
    pub queue_family: u32,
    /// Queue index within family.
    pub queue_index: u32,
    /// Human-readable label for this submission.
    pub label: String,
    /// Command buffer handles (raw u64 for storage efficiency).
    pub command_buffers: Vec<u64>,
    /// Wait semaphore handles.
    pub wait_semaphores: Vec<u64>,
    /// Signal semaphore handles.
    pub signal_semaphores: Vec<u64>,
    /// Fence handle.
    pub fence: u64,
    /// Completion status.
    status: AtomicU8,
    /// Error code if status is Error.
    pub error_code: Mutex<Option<vk::Result>>,
}

impl JournalEntry {
    /// Get the current status.
    pub fn status(&self) -> EntryStatus {
        match self.status.load(Ordering::Relaxed) {
            1 => EntryStatus::Completed,
            2 => EntryStatus::Error,
            _ => EntryStatus::Pending,
        }
    }
}

/// Flight recorder for GPU submissions.
///
/// Thread-safe ring buffer with configurable capacity. Old entries
/// are evicted when the journal is full.
pub struct SubmissionJournal {
    entries: Mutex<VecDeque<JournalEntry>>,
    capacity: usize,
    next_sequence: AtomicU64,
    creation_time: Instant,
}

impl SubmissionJournal {
    /// Create a journal with the given capacity (number of entries).
    pub fn new(capacity: usize) -> Self {
        Self {
            entries: Mutex::new(VecDeque::with_capacity(capacity)),
            capacity,
            next_sequence: AtomicU64::new(1),
            creation_time: Instant::now(),
        }
    }

    /// Record a submission.
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
        let seq = self.next_sequence.fetch_add(1, Ordering::Relaxed);

        let entry = JournalEntry {
            sequence: seq,
            timestamp: Instant::now(),
            queue_family,
            queue_index,
            label: label.to_string(),
            command_buffers: command_buffers.iter().map(|h| h.as_raw()).collect(),
            wait_semaphores: wait_semaphores.iter().map(|h| h.as_raw()).collect(),
            signal_semaphores: signal_semaphores.iter().map(|h| h.as_raw()).collect(),
            fence: fence.as_raw(),
            status: AtomicU8::new(EntryStatus::Pending as u8),
            error_code: Mutex::new(None),
        };

        let mut entries = self.entries.lock().unwrap();
        if entries.len() >= self.capacity {
            entries.pop_front();
        }
        entries.push_back(entry);

        seq
    }

    /// Mark a submission as completed.
    pub fn mark_completed(&self, fence: vk::Fence) {
        let raw = fence.as_raw();
        let entries = self.entries.lock().unwrap();
        for entry in entries.iter() {
            if entry.fence == raw {
                entry
                    .status
                    .store(EntryStatus::Completed as u8, Ordering::Relaxed);
                return;
            }
        }
    }

    /// Mark a submission as failed with an error code.
    pub fn mark_error(&self, fence: vk::Fence, error: vk::Result) {
        let raw = fence.as_raw();
        let entries = self.entries.lock().unwrap();
        for entry in entries.iter() {
            if entry.fence == raw {
                entry
                    .status
                    .store(EntryStatus::Error as u8, Ordering::Relaxed);
                *entry.error_code.lock().unwrap() = Some(error);
                return;
            }
        }
    }

    /// Number of entries currently in the journal.
    pub fn len(&self) -> usize {
        self.entries.lock().unwrap().len()
    }

    /// Whether the journal is empty.
    pub fn is_empty(&self) -> bool {
        self.entries.lock().unwrap().is_empty()
    }

    /// Dump the last `n` entries as a formatted diagnostic.
    pub fn dump_last(&self, n: usize) -> String {
        let entries = self.entries.lock().unwrap();
        let start = entries.len().saturating_sub(n);
        let slice: Vec<&JournalEntry> = entries.iter().skip(start).collect();
        format_journal_dump(&slice, &self.creation_time, None)
    }

    /// Dump all entries.
    pub fn dump_all(&self) -> String {
        let entries = self.entries.lock().unwrap();
        let all: Vec<&JournalEntry> = entries.iter().collect();
        format_journal_dump(&all, &self.creation_time, None)
    }

    /// Dump with an error context (e.g., `VK_ERROR_DEVICE_LOST`).
    pub fn dump_with_error(&self, error: vk::Result) -> String {
        let entries = self.entries.lock().unwrap();
        let all: Vec<&JournalEntry> = entries.iter().collect();
        format_journal_dump(&all, &self.creation_time, Some(error))
    }
}

fn format_journal_dump(
    entries: &[&JournalEntry],
    base_time: &Instant,
    error: Option<vk::Result>,
) -> String {
    let s = Style::detect();
    let mut o = String::with_capacity(512 + entries.len() * 256);

    let (code, msg) = if let Some(e) = error {
        (
            "IGN-J001",
            format!(
                "{} — submission journal dump ({} entries)",
                diagnostic::vk_result_name(e),
                entries.len()
            ),
        )
    } else {
        (
            "IGN-J002",
            format!("submission journal dump ({} entries)", entries.len()),
        )
    };

    let sev = if error.is_some() {
        Severity::Error
    } else {
        Severity::Info
    };

    if error.is_some() {
        diagnostic::write_full_diagnostic(&mut o, &s, &sev, code, &msg, true, false);
    } else {
        diagnostic::write_header(&mut o, &s, &sev, code, &msg);
    }
    diagnostic::write_pipe_empty(&mut o, &s);

    // Stats summary
    let pending = entries.iter().filter(|e| e.status() == EntryStatus::Pending).count();
    let completed = entries.iter().filter(|e| e.status() == EntryStatus::Completed).count();
    let errored = entries.iter().filter(|e| e.status() == EntryStatus::Error).count();

    diagnostic::write_pipe(
        &mut o,
        &s,
        &format!(
            "status: {} completed, {} pending, {} error",
            s.green(&completed.to_string()),
            s.bold_yellow(&pending.to_string()),
            if errored > 0 { s.bold_red(&errored.to_string()) } else { "0".to_string() },
        ),
    );
    diagnostic::write_pipe_empty(&mut o, &s);

    // Entry listing
    for entry in entries {
        let offset = entry.timestamp.duration_since(*base_time);
        let t = diagnostic::format_duration(offset);

        let status_str = match entry.status() {
            EntryStatus::Pending => s.bold_yellow("⏳ PENDING"),
            EntryStatus::Completed => s.green("✓ OK"),
            EntryStatus::Error => {
                let code = entry
                    .error_code
                    .lock()
                    .unwrap()
                    .map(|e| diagnostic::vk_result_name(e).to_string())
                    .unwrap_or_else(|| "unknown".to_string());
                s.bold_red(&format!("✗ ERROR({code})"))
            }
        };

        diagnostic::write_pipe(
            &mut o,
            &s,
            &format!(
                "T+{:<12} #{:<4} Queue[{},{}]  \"{}\"  {}",
                t,
                entry.sequence,
                entry.queue_family,
                entry.queue_index,
                entry.label,
                status_str,
            ),
        );

        if !entry.command_buffers.is_empty()
            || !entry.wait_semaphores.is_empty()
        {
            let cmds = entry.command_buffers.len();
            let waits = entry.wait_semaphores.len();
            let sigs = entry.signal_semaphores.len();
            diagnostic::write_pipe(
                &mut o,
                &s,
                &format!(
                    "             cmd={cmds} wait_sem={waits} sig_sem={sigs} fence={:#x}",
                    entry.fence,
                ),
            );
        }
    }

    diagnostic::write_pipe_empty(&mut o, &s);

    if error.is_some() {
        diagnostic::write_note(
            &mut o,
            &s,
            "PENDING entries at time of error were in-flight on the GPU\n\
             these submissions may have caused or been affected by the error",
        );
        diagnostic::write_help(
            &mut o,
            &s,
            "device lost often indicates GPU memory corruption,\n\
             shader infinite loop, or driver bug\n\
             check the last PENDING submission for problematic shaders or resources",
        );
    }

    diagnostic::write_diagnostic_end(&mut o, &s, &sev);

    o
}