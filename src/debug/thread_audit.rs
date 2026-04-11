//! Thread safety auditor for command pools.
//!
//! [`AuditedPool`] wraps a [`CommandPool`] and detects concurrent access
//! from multiple threads, which violates the Vulkan specification.

use std::sync::Mutex;
use std::thread::ThreadId;

use ash::vk;

use crate::command::{CommandPool, CommandRecorder};
use crate::diagnostic::{self, Severity, Style};
use crate::error::Result;

/// Thread identity record.
struct ThreadRecord {
    thread_id: ThreadId,
    thread_name: String,
    last_operation: String,
}

/// Action on thread violation.
#[derive(Default)]
pub enum ThreadViolationAction {
    /// Log to stderr.
    Log,
    /// Panic.
    #[default]
    Panic,
    /// Custom callback.
    Callback(Box<dyn Fn(&str) + Send + Sync>),
}

impl std::fmt::Debug for ThreadViolationAction {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Log => write!(f, "Log"),
            Self::Panic => write!(f, "Panic"),
            Self::Callback(_) => write!(f, "Callback(...)"),
        }
    }
}

/// A command pool wrapper that audits thread safety.
///
/// Tracks which thread last used the pool and reports violations
/// when a different thread accesses it concurrently.
pub struct AuditedPool {
    inner: CommandPool,
    owner: Mutex<Option<ThreadRecord>>,
    on_violation: ThreadViolationAction,
}

impl AuditedPool {
    /// Wrap a command pool with thread auditing.
    pub fn new(pool: CommandPool) -> Self {
        Self {
            inner: pool,
            owner: Mutex::new(None),
            on_violation: ThreadViolationAction::default(),
        }
    }

    /// Set the action on thread violation.
    pub fn on_violation(mut self, action: ThreadViolationAction) -> Self {
        self.on_violation = action;
        self
    }

    /// Access the inner command pool (bypasses auditing).
    pub fn inner(&self) -> &CommandPool {
        &self.inner
    }

    /// Get the pool handle.
    pub fn handle(&self) -> vk::CommandPool {
        self.inner.handle()
    }

    /// Queue family index.
    pub fn family_index(&self) -> u32 {
        self.inner.family_index()
    }

    /// Release thread ownership (e.g., after a frame boundary when you
    /// know the pool is idle and intend to use it from another thread).
    pub fn release_ownership(&self) {
        *self.owner.lock().unwrap() = None;
    }

    fn check_thread(&self, operation: &str) {
        let current = std::thread::current();
        let current_id = current.id();
        let current_name = current.name().unwrap_or("<unnamed>").to_string();

        let mut owner = self.owner.lock().unwrap();

        match &*owner {
            None => {
                *owner = Some(ThreadRecord {
                    thread_id: current_id,
                    thread_name: current_name,
                    last_operation: operation.to_string(),
                });
            }
            Some(rec) if rec.thread_id == current_id => {
                // Same thread, update operation.
                let rec_mut = owner.as_mut().unwrap();
                rec_mut.last_operation = operation.to_string();
            }
            Some(rec) => {
                // Different thread! Violation.
                let report = format_violation(
                    self.inner.handle(),
                    self.inner.family_index(),
                    &rec.thread_name,
                    rec.thread_id,
                    &rec.last_operation,
                    &current_name,
                    current_id,
                    operation,
                );

                match &self.on_violation {
                    ThreadViolationAction::Log => eprint!("{report}"),
                    ThreadViolationAction::Panic => panic!("{report}"),
                    ThreadViolationAction::Callback(f) => f(&report),
                }
            }
        }
    }

    /// Allocate a primary command buffer.
    pub fn allocate_primary(&self) -> Result<vk::CommandBuffer> {
        self.check_thread("allocate_primary()");
        self.inner.allocate_primary()
    }

    /// Allocate a secondary command buffer.
    pub fn allocate_secondary(&self) -> Result<vk::CommandBuffer> {
        self.check_thread("allocate_secondary()");
        self.inner.allocate_secondary()
    }

    /// Allocate multiple command buffers.
    pub fn allocate(
        &self,
        level: vk::CommandBufferLevel,
        count: u32,
    ) -> Result<Vec<vk::CommandBuffer>> {
        self.check_thread(&format!("allocate({count})"));
        self.inner.allocate(level, count)
    }

    /// Reset the pool.
    pub fn reset(&self) -> Result<()> {
        self.check_thread("reset()");
        self.inner.reset()
    }

    /// Begin recording a primary command buffer.
    pub fn begin_primary(&self, buffer: vk::CommandBuffer) -> Result<CommandRecorder<'_>> {
        self.check_thread("begin_primary()");
        self.inner.begin_primary(buffer)
    }
}

fn format_violation(
    pool: vk::CommandPool,
    family: u32,
    owner_name: &str,
    owner_id: ThreadId,
    owner_op: &str,
    accessor_name: &str,
    accessor_id: ThreadId,
    accessor_op: &str,
) -> String {
    use ash::vk::Handle;
    let s = Style::detect();
    let mut o = String::with_capacity(2048);

    diagnostic::write_full_diagnostic(
        &mut o,
        &s,
        &Severity::Error,
        "IGN-T001",
        "command pool accessed from wrong thread",
        true,
        true,
    );
    diagnostic::write_location(
        &mut o,
        &s,
        &format!("CommandPool({:#x}) family={family}", pool.as_raw()),
    );
    diagnostic::write_pipe_empty(&mut o, &s);

    // ── Thread comparison ──
    diagnostic::write_section(&mut o, &s, "Thread Conflict");
    diagnostic::write_pipe(
        &mut o,
        &s,
        &format!(
            "pool owner:    thread {} ({:?})",
            s.bold_green(&format!("\"{owner_name}\"")),
            owner_id,
        ),
    );
    diagnostic::write_pipe(&mut o, &s, &format!("  last op:     {owner_op}"));
    diagnostic::write_pipe_empty(&mut o, &s);
    diagnostic::write_pipe(
        &mut o,
        &s,
        &format!(
            "accessed from: thread {} ({:?})",
            s.bold_red(&format!("\"{accessor_name}\"")),
            accessor_id,
        ),
    );
    diagnostic::write_pipe(
        &mut o,
        &s,
        &format!("  operation:   {}", s.bold_red(accessor_op)),
    );

    // ── Vulkan spec quote ──
    diagnostic::write_separator(&mut o, &s);
    diagnostic::write_section(&mut o, &s, "Vulkan Specification");
    diagnostic::write_pipe_raw(&mut o, &s, &s.dim("§3.3.1 External Synchronization:"));
    diagnostic::write_pipe_raw(
        &mut o,
        &s,
        &s.dim("\"The following Vulkan objects must not be accessed"),
    );
    diagnostic::write_pipe_raw(
        &mut o,
        &s,
        &s.dim(" concurrently from multiple host threads:"),
    );
    diagnostic::write_pipe_raw(
        &mut o,
        &s,
        &s.dim(" VkCommandPool, VkDescriptorPool, VkQueue\""),
    );

    // ── Remediation options ──
    diagnostic::write_separator(&mut o, &s);
    diagnostic::write_section(&mut o, &s, "Remediation Options (best to worst)");
    diagnostic::write_numbered(
        &mut o,
        &s,
        1,
        "Use ParallelRecorder — one pool per thread, zero contention (recommended)",
    );
    diagnostic::write_numbered(
        &mut o,
        &s,
        2,
        "Create separate CommandPool per thread manually",
    );
    diagnostic::write_numbered(
        &mut o,
        &s,
        3,
        "Wrap pool access in std::sync::Mutex (correct but poor throughput)",
    );
    diagnostic::write_numbered(
        &mut o,
        &s,
        4,
        "Call release_ownership() at frame boundaries for intentional thread transfer",
    );

    diagnostic::write_pipe_empty(&mut o, &s);
    diagnostic::write_note(
        &mut o,
        &s,
        "this violation may cause data races, command buffer corruption,\n\
         or undefined behavior depending on GPU driver implementation",
    );

    diagnostic::write_diagnostic_end(&mut o, &s, &Severity::Error);

    o
}
