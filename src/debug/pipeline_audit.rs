//! Pipeline compatibility checker.
//!
//! Tracks pipeline layouts and verifies that descriptor sets bound
//! at draw/dispatch time are compatible with the active pipeline's
//! layout.

use std::collections::HashMap;
use std::fmt::Write;

use ash::vk;
use ash::vk::Handle;

use crate::diagnostic::{self, Severity, Style};

/// Metadata about a registered pipeline layout.
#[derive(Debug, Clone)]
pub struct LayoutRecord {
    /// Raw handle.
    pub handle: u64,
    /// Number of descriptor set layouts.
    pub set_count: u32,
    /// Hashes of each set layout for compatibility comparison.
    pub set_layout_hashes: Vec<u64>,
    /// Push constant ranges.
    pub push_constant_ranges: Vec<vk::PushConstantRange>,
}

/// Metadata about a registered pipeline.
#[derive(Debug, Clone)]
pub struct PipelineRecord {
    /// Raw handle.
    pub handle: u64,
    /// Optional debug name.
    pub name: Option<String>,
    /// The pipeline layout used.
    pub layout: u64,
    /// SPIR-V hash per stage (for change detection).
    pub shader_hashes: Vec<u64>,
}

/// A detected compatibility issue.
#[derive(Debug, Clone)]
pub struct PipelineIssue {
    /// The pipeline handle.
    pub pipeline: u64,
    /// Pipeline name.
    pub pipeline_name: Option<String>,
    /// Description of the issue.
    pub description: String,
}

/// Tracks pipelines and their layouts for compatibility validation.
pub struct PipelineAuditor {
    layouts: HashMap<u64, LayoutRecord>,
    pipelines: HashMap<u64, PipelineRecord>,
}

impl PipelineAuditor {
    /// Create a new auditor.
    pub fn new() -> Self {
        Self {
            layouts: HashMap::new(),
            pipelines: HashMap::new(),
        }
    }

    /// Register a pipeline layout.
    pub fn register_layout(
        &mut self,
        layout: vk::PipelineLayout,
        set_layout_hashes: &[u64],
        push_ranges: &[vk::PushConstantRange],
    ) {
        self.layouts.insert(
            layout.as_raw(),
            LayoutRecord {
                handle: layout.as_raw(),
                set_count: set_layout_hashes.len() as u32,
                set_layout_hashes: set_layout_hashes.to_vec(),
                push_constant_ranges: push_ranges.to_vec(),
            },
        );
    }

    /// Register a pipeline.
    pub fn register_pipeline(
        &mut self,
        pipeline: vk::Pipeline,
        name: Option<&str>,
        layout: vk::PipelineLayout,
        shader_hashes: &[u64],
    ) {
        self.pipelines.insert(
            pipeline.as_raw(),
            PipelineRecord {
                handle: pipeline.as_raw(),
                name: name.map(String::from),
                layout: layout.as_raw(),
                shader_hashes: shader_hashes.to_vec(),
            },
        );
    }

    /// Validate that the bound descriptor set count matches the pipeline layout.
    pub fn validate_bind(
        &self,
        pipeline: vk::Pipeline,
        bound_set_count: u32,
    ) -> Vec<PipelineIssue> {
        let mut issues = Vec::new();

        let Some(pipe_rec) = self.pipelines.get(&pipeline.as_raw()) else {
            return issues;
        };

        let Some(layout_rec) = self.layouts.get(&pipe_rec.layout) else {
            return issues;
        };

        if bound_set_count < layout_rec.set_count {
            issues.push(PipelineIssue {
                pipeline: pipe_rec.handle,
                pipeline_name: pipe_rec.name.clone(),
                description: format!(
                    "pipeline expects {} descriptor set(s) but only {} bound",
                    layout_rec.set_count, bound_set_count,
                ),
            });
        }

        issues
    }

    /// Validate push constant size against the pipeline layout.
    pub fn validate_push_constants(
        &self,
        pipeline: vk::Pipeline,
        stage_flags: vk::ShaderStageFlags,
        offset: u32,
        size: u32,
    ) -> Vec<PipelineIssue> {
        let mut issues = Vec::new();

        let Some(pipe_rec) = self.pipelines.get(&pipeline.as_raw()) else {
            return issues;
        };

        let Some(layout_rec) = self.layouts.get(&pipe_rec.layout) else {
            return issues;
        };

        let end = offset + size;
        let covered = layout_rec.push_constant_ranges.iter().any(|r| {
            r.stage_flags.contains(stage_flags) && r.offset <= offset && (r.offset + r.size) >= end
        });

        if !covered {
            issues.push(PipelineIssue {
                pipeline: pipe_rec.handle,
                pipeline_name: pipe_rec.name.clone(),
                description: format!(
                    "push constants at offset={offset} size={size} stage={stage_flags:?}\n\
                     not covered by any push constant range in the pipeline layout",
                ),
            });
        }

        issues
    }

    /// Format a report for detected issues.
    pub fn report(&self, issues: &[PipelineIssue]) -> String {
        if issues.is_empty() {
            return String::new();
        }
        format_pipeline_report(issues)
    }
}

impl Default for PipelineAuditor {
    fn default() -> Self {
        Self::new()
    }
}

fn format_pipeline_report(issues: &[PipelineIssue]) -> String {
    let s = Style::detect();
    let mut o = String::with_capacity(issues.len() * 512);

    for (i, issue) in issues.iter().enumerate() {
        let name_str = issue
            .pipeline_name
            .as_deref()
            .map(|n| format!(" \"{}\"", s.bold_cyan(n)))
            .unwrap_or_default();

        diagnostic::write_full_diagnostic(
            &mut o,
            &s,
            &Severity::Error,
            "IGN-P001",
            &format!("pipeline compatibility issue (#{i})"),
            i == 0,
            i == 0,
        );
        diagnostic::write_location(
            &mut o,
            &s,
            &format!("VkPipeline({:#x}){name_str}", issue.pipeline),
        );
        diagnostic::write_pipe_empty(&mut o, &s);

        for line in issue.description.lines() {
            diagnostic::write_pipe(&mut o, &s, line);
        }

        diagnostic::write_pipe_empty(&mut o, &s);
        diagnostic::write_help(
            &mut o,
            &s,
            "verify pipeline layout matches the descriptor set layouts\n\
             and push constant ranges used at bind/push time\n\
             use PipelineAuditor::register_layout() to track layouts",
        );
        diagnostic::write_diagnostic_end(&mut o, &s, &Severity::Error);

        if i < issues.len() - 1 {
            let _ = writeln!(o);
        }
    }

    o
}