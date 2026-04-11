//! Descriptor set validator.
//!
//! Tracks which resources are alive and what descriptors reference.
//! Detects use-after-free on descriptors: a buffer or image was destroyed
//! but a descriptor set still references it.

use std::collections::{HashMap, HashSet};

use ash::vk;
use ash::vk::Handle;

use crate::diagnostic::{self, Severity, Style};

/// Reference to a resource bound in a descriptor.
#[derive(Debug, Clone)]
pub enum BoundResource {
    /// A buffer binding.
    Buffer {
        /// Raw buffer handle.
        handle: u64,
        /// Byte offset within the buffer.
        offset: vk::DeviceSize,
        /// Byte range from offset (or `VK_WHOLE_SIZE`).
        range: vk::DeviceSize,
    },
    /// An image binding.
    Image {
        /// Raw image view handle.
        view_handle: u64,
        /// Raw image handle backing the view.
        image_handle: u64,
        /// Expected image layout during shader access.
        layout: vk::ImageLayout,
    },
    /// A sampler binding.
    Sampler {
        /// Raw sampler handle.
        handle: u64,
    },
    /// Combined image sampler.
    CombinedImageSampler {
        /// Raw image view handle.
        view_handle: u64,
        /// Raw image handle backing the view.
        image_handle: u64,
        /// Raw sampler handle.
        sampler_handle: u64,
        /// Expected image layout during shader access.
        layout: vk::ImageLayout,
    },
}

/// A single binding within a descriptor set.
#[derive(Debug, Clone)]
struct DescriptorBinding {
    binding: u32,
    resource: BoundResource,
}

/// A detected descriptor issue.
#[derive(Debug, Clone)]
pub struct DescriptorIssue {
    /// The descriptor set containing the stale reference.
    pub set_handle: u64,
    /// Binding index within the set.
    pub binding: u32,
    /// Type of the destroyed resource.
    pub resource_kind: &'static str,
    /// Raw handle of the destroyed resource.
    pub dead_handle: u64,
    /// Optional name of the descriptor set.
    pub set_name: Option<String>,
}

/// Tracks descriptor writes and resource liveness.
pub struct DescriptorAuditor {
    live_resources: HashSet<u64>,
    sets: HashMap<u64, Vec<DescriptorBinding>>,
    set_names: HashMap<u64, String>,
}

impl DescriptorAuditor {
    /// Create a new auditor.
    pub fn new() -> Self {
        Self {
            live_resources: HashSet::new(),
            sets: HashMap::new(),
            set_names: HashMap::new(),
        }
    }

    /// Register a resource as alive.
    pub fn register_resource(&mut self, handle: u64) {
        self.live_resources.insert(handle);
    }

    /// Unregister a resource (on destroy).
    pub fn unregister_resource(&mut self, handle: u64) {
        self.live_resources.remove(&handle);
    }

    /// Name a descriptor set for better diagnostics.
    pub fn name_set(&mut self, set: vk::DescriptorSet, name: &str) {
        self.set_names.insert(set.as_raw(), name.to_string());
    }

    /// Record a descriptor write.
    pub fn record_write(&mut self, set: vk::DescriptorSet, binding: u32, resource: BoundResource) {
        let entries = self.sets.entry(set.as_raw()).or_default();
        // Remove old binding at this index and replace.
        entries.retain(|b| b.binding != binding);
        entries.push(DescriptorBinding { binding, resource });
    }

    /// Remove all bindings for a descriptor set (e.g., on pool reset).
    pub fn clear_set(&mut self, set: vk::DescriptorSet) {
        self.sets.remove(&set.as_raw());
    }

    /// Validate that all resources referenced by a set are still alive.
    pub fn validate_set(&self, set: vk::DescriptorSet) -> Vec<DescriptorIssue> {
        let raw = set.as_raw();
        let name = self.set_names.get(&raw).cloned();

        let Some(bindings) = self.sets.get(&raw) else {
            return Vec::new();
        };

        let mut issues = Vec::new();

        for binding in bindings {
            let handles_to_check = match &binding.resource {
                BoundResource::Buffer { handle, .. } => vec![(*handle, "Buffer")],
                BoundResource::Image {
                    view_handle,
                    image_handle,
                    ..
                } => {
                    vec![(*view_handle, "ImageView"), (*image_handle, "Image")]
                }
                BoundResource::Sampler { handle } => vec![(*handle, "Sampler")],
                BoundResource::CombinedImageSampler {
                    view_handle,
                    image_handle,
                    sampler_handle,
                    ..
                } => vec![
                    (*view_handle, "ImageView"),
                    (*image_handle, "Image"),
                    (*sampler_handle, "Sampler"),
                ],
            };

            for (handle, kind) in handles_to_check {
                if handle != 0 && !self.live_resources.contains(&handle) {
                    issues.push(DescriptorIssue {
                        set_handle: raw,
                        binding: binding.binding,
                        resource_kind: kind,
                        dead_handle: handle,
                        set_name: name.clone(),
                    });
                }
            }
        }

        issues
    }

    /// Generate a formatted report for detected issues.
    pub fn report(&self, issues: &[DescriptorIssue]) -> String {
        if issues.is_empty() {
            return String::new();
        }
        format_descriptor_report(issues)
    }
}

impl Default for DescriptorAuditor {
    fn default() -> Self {
        Self::new()
    }
}

fn format_descriptor_report(issues: &[DescriptorIssue]) -> String {
    let s = Style::detect();
    let mut o = String::with_capacity(issues.len() * 512);

    diagnostic::write_full_diagnostic(
        &mut o,
        &s,
        &Severity::Error,
        "IGN-D001",
        &format!("{} stale descriptor reference(s) detected", issues.len()),
        true,
        true,
    );
    diagnostic::write_pipe_empty(&mut o, &s);

    for (i, issue) in issues.iter().enumerate() {
        let name_str = issue
            .set_name
            .as_deref()
            .map(|n| format!(" \"{}\"", s.bold_cyan(n)))
            .unwrap_or_default();

        diagnostic::write_pipe(
            &mut o,
            &s,
            &format!(
                "{} DescriptorSet({:#x}){name_str} binding={}",
                s.dim(&format!("[{i}]")),
                issue.set_handle,
                issue.binding,
            ),
        );
        diagnostic::write_pipe(
            &mut o,
            &s,
            &format!(
                "    references {} {:#x} {}",
                issue.resource_kind,
                issue.dead_handle,
                s.bold_red("← DESTROYED"),
            ),
        );
        diagnostic::write_pipe_empty(&mut o, &s);
    }

    diagnostic::write_note(
        &mut o,
        &s,
        "using a descriptor referencing a destroyed resource causes\n\
         undefined behavior: GPU crash, visual corruption, or device lost\n\
         the descriptor set is still bound but its backing resource is gone",
    );
    diagnostic::write_help(
        &mut o,
        &s,
        "update or invalidate descriptor sets before destroying resources\n\
         or delay resource destruction via DeletionQueue until the\n\
         descriptor set is no longer bound to any in-flight command buffer",
    );

    diagnostic::write_diagnostic_end(&mut o, &s, &Severity::Error);

    o
}
