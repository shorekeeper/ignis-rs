//! Resource aliasing detector.
//!
//! Tracks read/write accesses to resources within a recording session
//! and detects conflicts: resources read without a barrier after being
//! written, or written concurrently without synchronization.

use std::collections::HashMap;
use std::fmt::Write;

use ash::vk;

use crate::diagnostic::{self, Severity, Style};

/// How a resource is being accessed.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum AccessType {
    /// Read access (e.g., sampling a texture, reading a buffer).
    Read,
    /// Write access (e.g., rendering to a color attachment).
    Write,
}

/// A recorded resource access.
#[derive(Debug, Clone)]
struct ResourceAccess {
    access_type: AccessType,
    stage: vk::PipelineStageFlags,
    operation_index: u32,
    label: String,
}

/// A detected aliasing issue.
#[derive(Debug, Clone)]
pub struct AliasingIssue {
    /// Raw handle of the resource.
    pub handle: u64,
    /// Whether this is a buffer or image.
    pub resource_kind: &'static str,
    /// Optional debug name.
    pub name: Option<String>,
    /// The write access that produced new data.
    pub write_access: AliasingAccess,
    /// The conflicting access (read or write without barrier).
    pub conflict_access: AliasingAccess,
}

/// One side of an aliasing conflict.
#[derive(Debug, Clone)]
pub struct AliasingAccess {
    /// Read or Write.
    pub access_type: AccessType,
    /// Pipeline stage.
    pub stage: vk::PipelineStageFlags,
    /// Operation index (draw/dispatch sequential number).
    pub operation_index: u32,
    /// Label of the operation.
    pub label: String,
}

/// Per-resource tracking state.
struct ResourceState {
    name: Option<String>,
    kind: &'static str,
    accesses: Vec<ResourceAccess>,
    /// Reset when a barrier covers this resource.
    last_barrier_index: Option<u32>,
}

/// Tracks resource accesses and detects conflicts.
///
/// Use within a single command buffer recording session:
///
/// ```rust,no_run
/// # use ignis::aliasing::*; use ash::vk;
/// let mut det = AliasingDetector::new();
///
/// det.note_write(0x42, "image", Some("color"), vk::PipelineStageFlags::COLOR_ATTACHMENT_OUTPUT, 0, "geometry_pass");
/// // Missing barrier here!
/// det.note_read(0x42, "image", Some("color"), vk::PipelineStageFlags::FRAGMENT_SHADER, 1, "lighting");
///
/// let issues = det.analyze();
/// if !issues.is_empty() {
///     eprintln!("{}", det.report());
/// }
/// ```
pub struct AliasingDetector {
    resources: HashMap<u64, ResourceState>,
}

impl AliasingDetector {
    /// Create a new detector.
    pub fn new() -> Self {
        Self {
            resources: HashMap::new(),
        }
    }

    fn get_or_create(
        &mut self,
        handle: u64,
        kind: &'static str,
        name: Option<&str>,
    ) -> &mut ResourceState {
        self.resources
            .entry(handle)
            .or_insert_with(|| ResourceState {
                name: name.map(String::from),
                kind,
                accesses: Vec::new(),
                last_barrier_index: None,
            })
    }

    /// Record a read access.
    pub fn note_read(
        &mut self,
        handle: u64,
        kind: &'static str,
        name: Option<&str>,
        stage: vk::PipelineStageFlags,
        operation_index: u32,
        label: &str,
    ) {
        let state = self.get_or_create(handle, kind, name);
        state.accesses.push(ResourceAccess {
            access_type: AccessType::Read,
            stage,
            operation_index,
            label: label.to_string(),
        });
    }

    /// Record a write access.
    pub fn note_write(
        &mut self,
        handle: u64,
        kind: &'static str,
        name: Option<&str>,
        stage: vk::PipelineStageFlags,
        operation_index: u32,
        label: &str,
    ) {
        let state = self.get_or_create(handle, kind, name);
        state.accesses.push(ResourceAccess {
            access_type: AccessType::Write,
            stage,
            operation_index,
            label: label.to_string(),
        });
    }

    /// Record a barrier that synchronizes a resource.
    pub fn note_barrier(&mut self, handle: u64, at_operation: u32) {
        if let Some(state) = self.resources.get_mut(&handle) {
            state.last_barrier_index = Some(at_operation);
        }
    }

    /// Analyze all recorded accesses for conflicts.
    pub fn analyze(&self) -> Vec<AliasingIssue> {
        let mut issues = Vec::new();

        for (&handle, state) in &self.resources {
            let accesses = &state.accesses;

            for i in 0..accesses.len() {
                for j in (i + 1)..accesses.len() {
                    let a = &accesses[i];
                    let b = &accesses[j];

                    // Conflict: at least one is a write, no barrier between them.
                    if a.access_type == AccessType::Read && b.access_type == AccessType::Read {
                        continue;
                    }

                    // Check if a barrier covers the gap.
                    let barrier_between = state
                        .last_barrier_index
                        .is_some_and(|bi| bi > a.operation_index && bi <= b.operation_index);

                    if !barrier_between {
                        let (write, conflict) = if a.access_type == AccessType::Write {
                            (a, b)
                        } else {
                            (b, a)
                        };

                        issues.push(AliasingIssue {
                            handle,
                            resource_kind: state.kind,
                            name: state.name.clone(),
                            write_access: AliasingAccess {
                                access_type: write.access_type,
                                stage: write.stage,
                                operation_index: write.operation_index,
                                label: write.label.clone(),
                            },
                            conflict_access: AliasingAccess {
                                access_type: conflict.access_type,
                                stage: conflict.stage,
                                operation_index: conflict.operation_index,
                                label: conflict.label.clone(),
                            },
                        });
                    }
                }
            }
        }

        issues
    }

    /// Generate a formatted report of all detected issues.
    pub fn report(&self) -> String {
        let issues = self.analyze();
        if issues.is_empty() {
            return String::new();
        }
        format_aliasing_report(&issues)
    }

    /// Reset all tracking state.
    pub fn clear(&mut self) {
        self.resources.clear();
    }
}

impl Default for AliasingDetector {
    fn default() -> Self {
        Self::new()
    }
}

fn format_aliasing_report(issues: &[AliasingIssue]) -> String {
    let s = Style::detect();
    let mut o = String::with_capacity(issues.len() * 1024);

    for (i, issue) in issues.iter().enumerate() {
        let name_str = issue
            .name
            .as_deref()
            .map(|n| format!(" \"{}\"", s.bold_cyan(n)))
            .unwrap_or_default();

        diagnostic::write_full_diagnostic(
            &mut o,
            &s,
            &Severity::Error,
            "IGN-A001",
            &format!("resource aliasing without synchronization (#{})", i + 1),
            i == 0, // env block only on first issue
            i == 0, // backtrace only on first issue
        );
        diagnostic::write_location(
            &mut o,
            &s,
            &format!("{}({:#x}){name_str}", issue.resource_kind, issue.handle),
        );
        diagnostic::write_pipe_empty(&mut o, &s);

        // ── Execution timeline visualization ──
        diagnostic::write_section(&mut o, &s, "Execution Timeline");

        let w_idx = issue.write_access.operation_index;
        let c_idx = issue.conflict_access.operation_index;
        let min_idx = w_idx.min(c_idx);
        let max_idx = w_idx.max(c_idx);

        for idx in min_idx..=max_idx {
            if idx == w_idx {
                diagnostic::write_pipe_raw(
                    &mut o,
                    &s,
                    &format!(
                        "  op #{:<4} {} at {}  \"{}\"",
                        idx,
                        s.bold_red("██ WRITE"),
                        s.dim(&diagnostic::stage_flags_short(issue.write_access.stage)),
                        issue.write_access.label,
                    ),
                );
            } else if idx == c_idx {
                let kind = match issue.conflict_access.access_type {
                    AccessType::Read => s.bold_yellow("██ READ "),
                    AccessType::Write => s.bold_red("██ WRITE"),
                };
                diagnostic::write_pipe_raw(
                    &mut o,
                    &s,
                    &format!(
                        "  op #{:<4} {} at {}  \"{}\"",
                        idx,
                        kind,
                        s.dim(&diagnostic::stage_flags_short(issue.conflict_access.stage)),
                        issue.conflict_access.label,
                    ),
                );
            } else {
                diagnostic::write_pipe_raw(
                    &mut o,
                    &s,
                    &format!("  op #{:<4} {}", idx, s.dim("·· (other operations)")),
                );
            }

            // Show missing barrier between write and conflict
            if idx == w_idx.min(c_idx) && idx < w_idx.max(c_idx) {
                diagnostic::write_pipe_raw(
                    &mut o,
                    &s,
                    &format!(
                        "          {} {}",
                        s.bold_red("╳╳╳╳╳╳╳╳"),
                        s.bold_red("NO BARRIER — undefined behavior"),
                    ),
                );
            }
        }

        diagnostic::write_pipe_empty(&mut o, &s);
        diagnostic::write_note(
            &mut o,
            &s,
            "accessing a resource written without a synchronization barrier\n\
             causes undefined results: visual corruption, stale data,\n\
             or device lost on some drivers",
        );
        diagnostic::write_help(
            &mut o,
            &s,
            "insert a pipeline_barrier() between the conflicting operations\n\
             or use ResourceTracker::transition_image() / transition_buffer()\n\
             for automatic minimal barrier computation",
        );

        diagnostic::write_diagnostic_end(&mut o, &s, &Severity::Error);

        if i < issues.len() - 1 {
            let _ = writeln!(o);
        }
    }

    o
}
