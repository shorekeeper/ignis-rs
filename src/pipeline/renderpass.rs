//! Render pass and subpass builder.
//!
//! Provides [`RenderPassBuilder`] for constructing `VkRenderPass` objects
//! with attachments, subpasses, and subpass dependencies.
//!
//! For Vulkan 1.3+ dynamic rendering, you may skip render passes entirely
//! and use `VK_KHR_dynamic_rendering` / `vkCmdBeginRendering` directly
//! through the raw device handle.

use std::sync::Arc;

use ash::vk;

use crate::device::SharedState;
use crate::error::{Error, Result};

/// Configuration for a render pass attachment.
#[derive(Debug, Clone, Copy)]
pub struct AttachmentConfig {
    /// Image format.
    pub format: vk::Format,
    /// Multisample count.
    pub samples: vk::SampleCountFlags,
    /// How to handle the attachment at the beginning of the render pass.
    pub load_op: vk::AttachmentLoadOp,
    /// How to handle the attachment at the end of the render pass.
    pub store_op: vk::AttachmentStoreOp,
    /// Stencil load operation.
    pub stencil_load_op: vk::AttachmentLoadOp,
    /// Stencil store operation.
    pub stencil_store_op: vk::AttachmentStoreOp,
    /// Image layout at the start of the render pass.
    pub initial_layout: vk::ImageLayout,
    /// Image layout to transition to after the render pass.
    pub final_layout: vk::ImageLayout,
}

impl Default for AttachmentConfig {
    fn default() -> Self {
        Self {
            format: vk::Format::UNDEFINED,
            samples: vk::SampleCountFlags::TYPE_1,
            load_op: vk::AttachmentLoadOp::CLEAR,
            store_op: vk::AttachmentStoreOp::STORE,
            stencil_load_op: vk::AttachmentLoadOp::DONT_CARE,
            stencil_store_op: vk::AttachmentStoreOp::DONT_CARE,
            initial_layout: vk::ImageLayout::UNDEFINED,
            final_layout: vk::ImageLayout::COLOR_ATTACHMENT_OPTIMAL,
        }
    }
}

/// An attachment reference within a subpass.
#[derive(Debug, Clone, Copy)]
pub struct AttachmentRef {
    /// Index into the render pass attachment array.
    pub attachment: u32,
    /// Image layout for this attachment during the subpass.
    pub layout: vk::ImageLayout,
}

/// Configuration for a single subpass.
#[derive(Debug, Clone, Default)]
pub struct SubpassConfig {
    /// Color attachment references.
    pub color_attachments: Vec<AttachmentRef>,
    /// Depth/stencil attachment reference (if any).
    pub depth_stencil_attachment: Option<AttachmentRef>,
    /// Input attachment references (for reading from previous subpasses).
    pub input_attachments: Vec<AttachmentRef>,
    /// Indices of attachments to preserve (not used by this subpass but needed later).
    pub preserve_attachments: Vec<u32>,
}

/// A subpass dependency specification.
#[derive(Debug, Clone, Copy)]
pub struct SubpassDependency {
    /// Source subpass index, or `vk::SUBPASS_EXTERNAL`.
    pub src_subpass: u32,
    /// Destination subpass index, or `vk::SUBPASS_EXTERNAL`.
    pub dst_subpass: u32,
    /// Source pipeline stages.
    pub src_stage_mask: vk::PipelineStageFlags,
    /// Destination pipeline stages.
    pub dst_stage_mask: vk::PipelineStageFlags,
    /// Source access flags.
    pub src_access_mask: vk::AccessFlags,
    /// Destination access flags.
    pub dst_access_mask: vk::AccessFlags,
    /// Dependency flags.
    pub dependency_flags: vk::DependencyFlags,
}

/// Builder for constructing a `VkRenderPass`.
///
/// # Example
///
/// ```rust,no_run
/// # use ignis::*; use ash::vk;
/// # fn example(ignis: &Ignis) -> Result<()> {
/// let pass = ignis.render_pass_builder()
///     .attachment(AttachmentConfig {
///         format: vk::Format::B8G8R8A8_SRGB,
///         load_op: vk::AttachmentLoadOp::CLEAR,
///         store_op: vk::AttachmentStoreOp::STORE,
///         final_layout: vk::ImageLayout::PRESENT_SRC_KHR,
///         ..Default::default()
///     })
///     .attachment(AttachmentConfig {
///         format: vk::Format::D32_SFLOAT,
///         load_op: vk::AttachmentLoadOp::CLEAR,
///         store_op: vk::AttachmentStoreOp::DONT_CARE,
///         final_layout: vk::ImageLayout::DEPTH_STENCIL_ATTACHMENT_OPTIMAL,
///         ..Default::default()
///     })
///     .subpass(SubpassConfig {
///         color_attachments: vec![AttachmentRef {
///             attachment: 0,
///             layout: vk::ImageLayout::COLOR_ATTACHMENT_OPTIMAL,
///         }],
///         depth_stencil_attachment: Some(AttachmentRef {
///             attachment: 1,
///             layout: vk::ImageLayout::DEPTH_STENCIL_ATTACHMENT_OPTIMAL,
///         }),
///         ..Default::default()
///     })
///     .dependency(SubpassDependency {
///         src_subpass: vk::SUBPASS_EXTERNAL,
///         dst_subpass: 0,
///         src_stage_mask: vk::PipelineStageFlags::COLOR_ATTACHMENT_OUTPUT
///             | vk::PipelineStageFlags::EARLY_FRAGMENT_TESTS,
///         dst_stage_mask: vk::PipelineStageFlags::COLOR_ATTACHMENT_OUTPUT
///             | vk::PipelineStageFlags::EARLY_FRAGMENT_TESTS,
///         src_access_mask: vk::AccessFlags::empty(),
///         dst_access_mask: vk::AccessFlags::COLOR_ATTACHMENT_WRITE
///             | vk::AccessFlags::DEPTH_STENCIL_ATTACHMENT_WRITE,
///         dependency_flags: vk::DependencyFlags::empty(),
///     })
///     .build()?;
/// # Ok(())
/// # }
/// ```
pub struct RenderPassBuilder {
    shared: Arc<SharedState>,
    attachments: Vec<AttachmentConfig>,
    subpasses: Vec<SubpassConfig>,
    dependencies: Vec<SubpassDependency>,
}

impl RenderPassBuilder {
    pub(crate) fn new(shared: Arc<SharedState>) -> Self {
        Self {
            shared,
            attachments: Vec::new(),
            subpasses: Vec::new(),
            dependencies: Vec::new(),
        }
    }

    /// Add an attachment to the render pass.
    pub fn attachment(mut self, config: AttachmentConfig) -> Self {
        self.attachments.push(config);
        self
    }

    /// Add a subpass to the render pass.
    pub fn subpass(mut self, config: SubpassConfig) -> Self {
        self.subpasses.push(config);
        self
    }

    /// Add a subpass dependency.
    pub fn dependency(mut self, dep: SubpassDependency) -> Self {
        self.dependencies.push(dep);
        self
    }

    /// Build the render pass.
    ///
    /// # Errors
    ///
    /// Returns [`Error::InvalidConfig`] if no subpasses are configured,
    /// or a Vulkan error if render pass creation fails.
    pub fn build(self) -> Result<RenderPassHandle> {
        if self.subpasses.is_empty() {
            return Err(Error::InvalidConfig("at least one subpass is required"));
        }

        // Convert attachments.
        let attachment_descs: Vec<vk::AttachmentDescription> = self
            .attachments
            .iter()
            .map(|a| {
                vk::AttachmentDescription::default()
                    .format(a.format)
                    .samples(a.samples)
                    .load_op(a.load_op)
                    .store_op(a.store_op)
                    .stencil_load_op(a.stencil_load_op)
                    .stencil_store_op(a.stencil_store_op)
                    .initial_layout(a.initial_layout)
                    .final_layout(a.final_layout)
            })
            .collect();

        // Convert subpass attachment references.
        // We need to keep these alive during render pass creation.
        let mut color_refs: Vec<Vec<vk::AttachmentReference>> = Vec::new();
        let mut depth_refs: Vec<Option<vk::AttachmentReference>> = Vec::new();
        let mut input_refs: Vec<Vec<vk::AttachmentReference>> = Vec::new();

        for sp in &self.subpasses {
            color_refs.push(
                sp.color_attachments
                    .iter()
                    .map(|r| vk::AttachmentReference {
                        attachment: r.attachment,
                        layout: r.layout,
                    })
                    .collect(),
            );
            depth_refs.push(
                sp.depth_stencil_attachment
                    .map(|r| vk::AttachmentReference {
                        attachment: r.attachment,
                        layout: r.layout,
                    }),
            );
            input_refs.push(
                sp.input_attachments
                    .iter()
                    .map(|r| vk::AttachmentReference {
                        attachment: r.attachment,
                        layout: r.layout,
                    })
                    .collect(),
            );
        }

        let subpass_descs: Vec<vk::SubpassDescription> = self
            .subpasses
            .iter()
            .enumerate()
            .map(|(i, sp)| {
                let mut desc = vk::SubpassDescription::default()
                    .pipeline_bind_point(vk::PipelineBindPoint::GRAPHICS)
                    .color_attachments(&color_refs[i])
                    .input_attachments(&input_refs[i])
                    .preserve_attachments(&sp.preserve_attachments);

                if let Some(ref depth_ref) = depth_refs[i] {
                    desc = desc.depth_stencil_attachment(depth_ref);
                }

                desc
            })
            .collect();

        let dependency_descs: Vec<vk::SubpassDependency> = self
            .dependencies
            .iter()
            .map(|d| {
                vk::SubpassDependency::default()
                    .src_subpass(d.src_subpass)
                    .dst_subpass(d.dst_subpass)
                    .src_stage_mask(d.src_stage_mask)
                    .dst_stage_mask(d.dst_stage_mask)
                    .src_access_mask(d.src_access_mask)
                    .dst_access_mask(d.dst_access_mask)
                    .dependency_flags(d.dependency_flags)
            })
            .collect();

        let create_info = vk::RenderPassCreateInfo::default()
            .attachments(&attachment_descs)
            .subpasses(&subpass_descs)
            .dependencies(&dependency_descs);

        // SAFETY: all referenced data lives on the stack and is valid.
        let handle = unsafe { self.shared.device.create_render_pass(&create_info, None)? };

        Ok(RenderPassHandle {
            shared: self.shared,
            handle,
        })
    }
}

/// An owned render pass handle with automatic cleanup.
pub struct RenderPassHandle {
    shared: Arc<SharedState>,
    handle: vk::RenderPass,
}

impl RenderPassHandle {
    /// Get the raw render pass handle.
    #[inline]
    pub fn handle(&self) -> vk::RenderPass {
        self.handle
    }
}

impl Drop for RenderPassHandle {
    fn drop(&mut self) {
        unsafe {
            self.shared.device.destroy_render_pass(self.handle, None);
        }
    }
}
