//! Pipeline barrier analyzer and optimizer.
//!
//! Records barriers as they are issued and analyzes them for:
//!
//! - **Overly broad stages/access**: `ALL_COMMANDS`/`MEMORY_READ|WRITE`
//!   serializes the entire GPU pipeline
//! - **Redundant barriers**: back-to-back barriers on the same resource
//! - **Suggested minimization**: computes tighter stage/access masks
//!   based on tracked resource usage

use ash::vk;

use crate::diagnostic::{self, Severity, Style};

/// A recorded pipeline barrier.
#[derive(Debug, Clone)]
pub struct BarrierRecord {
    /// Sequential barrier index within the recording session.
    pub index: u32,
    /// Source pipeline stage mask.
    pub src_stage: vk::PipelineStageFlags,
    /// Destination pipeline stage mask.
    pub dst_stage: vk::PipelineStageFlags,
    /// Source access mask (from memory barriers).
    pub src_access: vk::AccessFlags,
    /// Destination access mask.
    pub dst_access: vk::AccessFlags,
    /// Optional label for context.
    pub label: String,
}

/// An optimization suggestion.
#[derive(Debug, Clone)]
pub struct BarrierSuggestion {
    /// Index of the barrier being analyzed.
    pub barrier_index: u32,
    /// The label of the original barrier.
    pub label: String,
    /// Category of the suggestion.
    pub kind: SuggestionKind,
    /// Human-readable description.
    pub description: String,
    /// Suggested replacement for source stage mask.
    pub suggested_src_stage: Option<vk::PipelineStageFlags>,
    /// Suggested replacement for destination stage mask.
    pub suggested_dst_stage: Option<vk::PipelineStageFlags>,
    /// Suggested replacement for source access mask.
    pub suggested_src_access: Option<vk::AccessFlags>,
    /// Suggested replacement for destination access mask.
    pub suggested_dst_access: Option<vk::AccessFlags>,
}

/// Category of optimization.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SuggestionKind {
    /// Stage mask is broader than necessary.
    BroadStage,
    /// Access mask is broader than necessary.
    BroadAccess,
    /// Barrier appears redundant.
    Redundant,
}

impl std::fmt::Display for SuggestionKind {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::BroadStage => write!(f, "overly broad stage mask"),
            Self::BroadAccess => write!(f, "overly broad access mask"),
            Self::Redundant => write!(f, "potentially redundant barrier"),
        }
    }
}

/// Collects and analyzes pipeline barriers.
pub struct BarrierAnalyzer {
    barriers: Vec<BarrierRecord>,
    next_index: u32,
}

impl BarrierAnalyzer {
    /// Create a new analyzer.
    pub fn new() -> Self {
        Self {
            barriers: Vec::new(),
            next_index: 0,
        }
    }

    /// Record a barrier for analysis.
    pub fn record(
        &mut self,
        src_stage: vk::PipelineStageFlags,
        dst_stage: vk::PipelineStageFlags,
        src_access: vk::AccessFlags,
        dst_access: vk::AccessFlags,
        label: &str,
    ) {
        let index = self.next_index;
        self.next_index += 1;
        self.barriers.push(BarrierRecord {
            index,
            src_stage,
            dst_stage,
            src_access,
            dst_access,
            label: label.to_string(),
        });
    }

    /// Analyze all recorded barriers and produce suggestions.
    pub fn analyze(&self) -> Vec<BarrierSuggestion> {
        let mut suggestions = Vec::new();

        for barrier in &self.barriers {
            // Check for ALL_COMMANDS stages.
            if barrier.src_stage == vk::PipelineStageFlags::ALL_COMMANDS
                || barrier.dst_stage == vk::PipelineStageFlags::ALL_COMMANDS
            {
                let (suggested_src, suggested_dst) =
                    suggest_stages(barrier.src_access, barrier.dst_access);

                suggestions.push(BarrierSuggestion {
                    barrier_index: barrier.index,
                    label: barrier.label.clone(),
                    kind: SuggestionKind::BroadStage,
                    description: format!(
                        "ALL_COMMANDS stage serializes the GPU pipeline\n\
                         src={:?} dst={:?}",
                        barrier.src_stage, barrier.dst_stage,
                    ),
                    suggested_src_stage: Some(suggested_src),
                    suggested_dst_stage: Some(suggested_dst),
                    suggested_src_access: None,
                    suggested_dst_access: None,
                });
            }

            // Check for broad access masks.
            let both_rw = vk::AccessFlags::MEMORY_READ | vk::AccessFlags::MEMORY_WRITE;
            if barrier.src_access.contains(both_rw) || barrier.dst_access.contains(both_rw) {
                suggestions.push(BarrierSuggestion {
                    barrier_index: barrier.index,
                    label: barrier.label.clone(),
                    kind: SuggestionKind::BroadAccess,
                    description: format!(
                        "MEMORY_READ|MEMORY_WRITE is too broad\n\
                         src_access={:?} dst_access={:?}",
                        barrier.src_access, barrier.dst_access,
                    ),
                    suggested_src_stage: None,
                    suggested_dst_stage: None,
                    suggested_src_access: Some(narrow_access(
                        barrier.src_access,
                        barrier.src_stage,
                    )),
                    suggested_dst_access: Some(narrow_access(
                        barrier.dst_access,
                        barrier.dst_stage,
                    )),
                });
            }
        }

        // Check for redundant consecutive barriers.
        for w in self.barriers.windows(2) {
            let a = &w[0];
            let b = &w[1];

            if a.src_stage == b.src_stage
                && a.dst_stage == b.dst_stage
                && a.src_access == b.src_access
                && a.dst_access == b.dst_access
            {
                suggestions.push(BarrierSuggestion {
                    barrier_index: b.index,
                    label: b.label.clone(),
                    kind: SuggestionKind::Redundant,
                    description: format!(
                        "barrier #{} is identical to preceding barrier #{}",
                        b.index, a.index,
                    ),
                    suggested_src_stage: None,
                    suggested_dst_stage: None,
                    suggested_src_access: None,
                    suggested_dst_access: None,
                });
            }
        }

        suggestions
    }

    /// Generate a formatted report.
    pub fn report(&self) -> String {
        let suggestions = self.analyze();
        if suggestions.is_empty() {
            return String::new();
        }
        format_barrier_report(&suggestions)
    }

    /// Reset all recorded barriers.
    pub fn clear(&mut self) {
        self.barriers.clear();
        self.next_index = 0;
    }
}

impl Default for BarrierAnalyzer {
    fn default() -> Self {
        Self::new()
    }
}

/// Suggest tighter stage masks from access flags.
fn suggest_stages(
    src_access: vk::AccessFlags,
    dst_access: vk::AccessFlags,
) -> (vk::PipelineStageFlags, vk::PipelineStageFlags) {
    (access_to_stage(src_access), access_to_stage(dst_access))
}

fn access_to_stage(access: vk::AccessFlags) -> vk::PipelineStageFlags {
    let mut stage = vk::PipelineStageFlags::empty();

    if access.contains(vk::AccessFlags::COLOR_ATTACHMENT_READ)
        || access.contains(vk::AccessFlags::COLOR_ATTACHMENT_WRITE)
    {
        stage |= vk::PipelineStageFlags::COLOR_ATTACHMENT_OUTPUT;
    }
    if access.contains(vk::AccessFlags::DEPTH_STENCIL_ATTACHMENT_READ)
        || access.contains(vk::AccessFlags::DEPTH_STENCIL_ATTACHMENT_WRITE)
    {
        stage |= vk::PipelineStageFlags::EARLY_FRAGMENT_TESTS
            | vk::PipelineStageFlags::LATE_FRAGMENT_TESTS;
    }
    if access.contains(vk::AccessFlags::SHADER_READ)
        || access.contains(vk::AccessFlags::SHADER_WRITE)
    {
        stage |= vk::PipelineStageFlags::FRAGMENT_SHADER
            | vk::PipelineStageFlags::VERTEX_SHADER
            | vk::PipelineStageFlags::COMPUTE_SHADER;
    }
    if access.contains(vk::AccessFlags::TRANSFER_READ)
        || access.contains(vk::AccessFlags::TRANSFER_WRITE)
    {
        stage |= vk::PipelineStageFlags::TRANSFER;
    }
    if access.contains(vk::AccessFlags::INDEX_READ)
        || access.contains(vk::AccessFlags::VERTEX_ATTRIBUTE_READ)
    {
        stage |= vk::PipelineStageFlags::VERTEX_INPUT;
    }
    if access.contains(vk::AccessFlags::INDIRECT_COMMAND_READ) {
        stage |= vk::PipelineStageFlags::DRAW_INDIRECT;
    }
    if access.contains(vk::AccessFlags::HOST_READ) || access.contains(vk::AccessFlags::HOST_WRITE) {
        stage |= vk::PipelineStageFlags::HOST;
    }

    if stage.is_empty() {
        // Fallback: if we can't determine, use ALL_COMMANDS.
        vk::PipelineStageFlags::ALL_COMMANDS
    } else {
        stage
    }
}

fn narrow_access(access: vk::AccessFlags, stage: vk::PipelineStageFlags) -> vk::AccessFlags {
    let both_rw = vk::AccessFlags::MEMORY_READ | vk::AccessFlags::MEMORY_WRITE;
    if !access.contains(both_rw) {
        return access;
    }

    // Try to narrow MEMORY_READ|WRITE to specific access bits based on stage.
    let mut narrow = vk::AccessFlags::empty();

    if stage.contains(vk::PipelineStageFlags::COLOR_ATTACHMENT_OUTPUT) {
        narrow |= vk::AccessFlags::COLOR_ATTACHMENT_READ | vk::AccessFlags::COLOR_ATTACHMENT_WRITE;
    }
    if stage.contains(vk::PipelineStageFlags::EARLY_FRAGMENT_TESTS)
        || stage.contains(vk::PipelineStageFlags::LATE_FRAGMENT_TESTS)
    {
        narrow |= vk::AccessFlags::DEPTH_STENCIL_ATTACHMENT_READ
            | vk::AccessFlags::DEPTH_STENCIL_ATTACHMENT_WRITE;
    }
    if stage.contains(vk::PipelineStageFlags::FRAGMENT_SHADER)
        || stage.contains(vk::PipelineStageFlags::VERTEX_SHADER)
        || stage.contains(vk::PipelineStageFlags::COMPUTE_SHADER)
    {
        narrow |= vk::AccessFlags::SHADER_READ | vk::AccessFlags::SHADER_WRITE;
    }
    if stage.contains(vk::PipelineStageFlags::TRANSFER) {
        narrow |= vk::AccessFlags::TRANSFER_READ | vk::AccessFlags::TRANSFER_WRITE;
    }

    if narrow.is_empty() {
        access
    } else {
        narrow
    }
}

fn format_barrier_report(suggestions: &[BarrierSuggestion]) -> String {
    let s = Style::detect();
    let mut o = String::with_capacity(suggestions.len() * 512);

    diagnostic::write_header(
        &mut o,
        &s,
        &Severity::Warning,
        "IGN-O001",
        &format!("{} barrier optimization(s) suggested", suggestions.len()),
    );
    diagnostic::write_pipe_empty(&mut o, &s);

    for sug in suggestions {
        diagnostic::write_pipe(
            &mut o,
            &s,
            &format!(
                "barrier #{} \"{}\": {}",
                sug.barrier_index,
                s.bold_cyan(&sug.label),
                s.bold_yellow(&sug.kind.to_string()),
            ),
        );

        for line in sug.description.lines() {
            diagnostic::write_pipe(&mut o, &s, &format!("  {line}"));
        }

        let has_suggestion = sug.suggested_src_stage.is_some()
            || sug.suggested_dst_stage.is_some()
            || sug.suggested_src_access.is_some()
            || sug.suggested_dst_access.is_some();

        if has_suggestion {
            diagnostic::write_pipe_empty(&mut o, &s);
            diagnostic::write_pipe(&mut o, &s, &format!("{}", s.green("suggested:")));

            if let Some(src) = sug.suggested_src_stage {
                diagnostic::write_pipe(&mut o, &s, &format!("  src_stage = {:?}", src));
            }
            if let Some(dst) = sug.suggested_dst_stage {
                diagnostic::write_pipe(&mut o, &s, &format!("  dst_stage = {:?}", dst));
            }
            if let Some(src) = sug.suggested_src_access {
                diagnostic::write_pipe(&mut o, &s, &format!("  src_access = {:?}", src));
            }
            if let Some(dst) = sug.suggested_dst_access {
                diagnostic::write_pipe(&mut o, &s, &format!("  dst_access = {:?}", dst));
            }
        }

        diagnostic::write_pipe_empty(&mut o, &s);
    }

    diagnostic::write_help(
        &mut o,
        &s,
        "overly broad barriers serialize GPU work and can\nreduce performance by 10-40%\nuse ResourceTracker::transition() for automatic minimal barriers",
    );

    o
}
