//! GPU resource state tracking with per-subresource image tracking
//! and buffer barrier support.
//!
//! Key improvements over the previous version:
//!
//! - **Buffer tracking**: barriers for compute read-after-write, etc.
//! - **Per-subresource image tracking**: individual mip levels and layers
//!   can be in different layouts (needed for mipmap generation).
//! - **Explicit pipeline stages**: no automatic guessing. The caller
//!   specifies the intended usage, and the tracker computes the barrier.
//!
//! # Stage Inference
//!
//! The old `infer_access_and_stage` assumed SHADER_READ_ONLY_OPTIMAL
//! always meant FRAGMENT_SHADER. This is wrong for compute reads.
//! The new API requires the caller to specify the target usage via
//! [`ImageUsageContext`] or [`BufferUsageContext`], which unambiguously
//! determines the pipeline stage.

use std::collections::HashMap;

use ash::vk;

use crate::command::CommandRecorder;

/// How a pipeline stage and access mask are determined for an image.
///
/// Instead of guessing from layout alone, the caller specifies intent.
#[derive(Debug, Clone, Copy)]
pub enum ImageUsageContext {
    /// Color attachment (read + write).
    ColorAttachment,
    /// Depth/stencil attachment (read + write).
    DepthStencilAttachment,
    /// Depth/stencil read-only (e.g., shadow map sampling).
    DepthStencilReadOnly,
    /// Sampled in a fragment shader.
    FragmentShaderRead,
    /// Sampled in a vertex shader.
    VertexShaderRead,
    /// Sampled in a compute shader.
    ComputeShaderRead,
    /// Storage image written by a compute shader.
    ComputeShaderWrite,
    /// Storage image read + written by a compute shader.
    ComputeShaderReadWrite,
    /// Transfer source.
    TransferSrc,
    /// Transfer destination.
    TransferDst,
    /// Presentation.
    PresentSrc,
    /// General (read + write from any stage, broad barrier).
    General,
    /// Custom specification.
    Custom {
        /// Image layout.
        layout: vk::ImageLayout,
        /// Access flags.
        access: vk::AccessFlags,
        /// Pipeline stage flags.
        stage: vk::PipelineStageFlags,
    },
}

impl ImageUsageContext {
    /// Compute the layout, access mask, and pipeline stage.
    pub fn resolve(self) -> (vk::ImageLayout, vk::AccessFlags, vk::PipelineStageFlags) {
        match self {
            Self::ColorAttachment => (
                vk::ImageLayout::COLOR_ATTACHMENT_OPTIMAL,
                vk::AccessFlags::COLOR_ATTACHMENT_READ | vk::AccessFlags::COLOR_ATTACHMENT_WRITE,
                vk::PipelineStageFlags::COLOR_ATTACHMENT_OUTPUT,
            ),
            Self::DepthStencilAttachment => (
                vk::ImageLayout::DEPTH_STENCIL_ATTACHMENT_OPTIMAL,
                vk::AccessFlags::DEPTH_STENCIL_ATTACHMENT_READ
                    | vk::AccessFlags::DEPTH_STENCIL_ATTACHMENT_WRITE,
                vk::PipelineStageFlags::EARLY_FRAGMENT_TESTS
                    | vk::PipelineStageFlags::LATE_FRAGMENT_TESTS,
            ),
            Self::DepthStencilReadOnly => (
                vk::ImageLayout::DEPTH_STENCIL_READ_ONLY_OPTIMAL,
                vk::AccessFlags::DEPTH_STENCIL_ATTACHMENT_READ,
                vk::PipelineStageFlags::EARLY_FRAGMENT_TESTS
                    | vk::PipelineStageFlags::LATE_FRAGMENT_TESTS,
            ),
            Self::FragmentShaderRead => (
                vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL,
                vk::AccessFlags::SHADER_READ,
                vk::PipelineStageFlags::FRAGMENT_SHADER,
            ),
            Self::VertexShaderRead => (
                vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL,
                vk::AccessFlags::SHADER_READ,
                vk::PipelineStageFlags::VERTEX_SHADER,
            ),
            Self::ComputeShaderRead => (
                vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL,
                vk::AccessFlags::SHADER_READ,
                vk::PipelineStageFlags::COMPUTE_SHADER,
            ),
            Self::ComputeShaderWrite => (
                vk::ImageLayout::GENERAL,
                vk::AccessFlags::SHADER_WRITE,
                vk::PipelineStageFlags::COMPUTE_SHADER,
            ),
            Self::ComputeShaderReadWrite => (
                vk::ImageLayout::GENERAL,
                vk::AccessFlags::SHADER_READ | vk::AccessFlags::SHADER_WRITE,
                vk::PipelineStageFlags::COMPUTE_SHADER,
            ),
            Self::TransferSrc => (
                vk::ImageLayout::TRANSFER_SRC_OPTIMAL,
                vk::AccessFlags::TRANSFER_READ,
                vk::PipelineStageFlags::TRANSFER,
            ),
            Self::TransferDst => (
                vk::ImageLayout::TRANSFER_DST_OPTIMAL,
                vk::AccessFlags::TRANSFER_WRITE,
                vk::PipelineStageFlags::TRANSFER,
            ),
            Self::PresentSrc => (
                vk::ImageLayout::PRESENT_SRC_KHR,
                vk::AccessFlags::empty(),
                vk::PipelineStageFlags::BOTTOM_OF_PIPE,
            ),
            Self::General => (
                vk::ImageLayout::GENERAL,
                vk::AccessFlags::MEMORY_READ | vk::AccessFlags::MEMORY_WRITE,
                vk::PipelineStageFlags::ALL_COMMANDS,
            ),
            Self::Custom { layout, access, stage } => (layout, access, stage),
        }
    }
}

/// How a buffer is being used (determines access mask and stage).
#[derive(Debug, Clone, Copy)]
pub enum BufferUsageContext {
    /// Vertex buffer input.
    VertexInput,
    /// Index buffer input.
    IndexInput,
    /// Uniform buffer read in vertex shader.
    UniformVertex,
    /// Uniform buffer read in fragment shader.
    UniformFragment,
    /// Uniform buffer read in compute shader.
    UniformCompute,
    /// Storage buffer read by compute shader.
    StorageComputeRead,
    /// Storage buffer write by compute shader.
    StorageComputeWrite,
    /// Storage buffer read + write by compute shader.
    StorageComputeReadWrite,
    /// Transfer source.
    TransferSrc,
    /// Transfer destination.
    TransferDst,
    /// Indirect draw arguments.
    IndirectDraw,
    /// Custom specification.
    Custom {
        /// Access flags.
        access: vk::AccessFlags,
        /// Pipeline stage flags.
        stage: vk::PipelineStageFlags,
    },
}

impl BufferUsageContext {
    /// Compute access mask and pipeline stage.
    pub fn resolve(self) -> (vk::AccessFlags, vk::PipelineStageFlags) {
        match self {
            Self::VertexInput => (
                vk::AccessFlags::VERTEX_ATTRIBUTE_READ,
                vk::PipelineStageFlags::VERTEX_INPUT,
            ),
            Self::IndexInput => (
                vk::AccessFlags::INDEX_READ,
                vk::PipelineStageFlags::VERTEX_INPUT,
            ),
            Self::UniformVertex => (
                vk::AccessFlags::UNIFORM_READ,
                vk::PipelineStageFlags::VERTEX_SHADER,
            ),
            Self::UniformFragment => (
                vk::AccessFlags::UNIFORM_READ,
                vk::PipelineStageFlags::FRAGMENT_SHADER,
            ),
            Self::UniformCompute => (
                vk::AccessFlags::UNIFORM_READ,
                vk::PipelineStageFlags::COMPUTE_SHADER,
            ),
            Self::StorageComputeRead => (
                vk::AccessFlags::SHADER_READ,
                vk::PipelineStageFlags::COMPUTE_SHADER,
            ),
            Self::StorageComputeWrite => (
                vk::AccessFlags::SHADER_WRITE,
                vk::PipelineStageFlags::COMPUTE_SHADER,
            ),
            Self::StorageComputeReadWrite => (
                vk::AccessFlags::SHADER_READ | vk::AccessFlags::SHADER_WRITE,
                vk::PipelineStageFlags::COMPUTE_SHADER,
            ),
            Self::TransferSrc => (
                vk::AccessFlags::TRANSFER_READ,
                vk::PipelineStageFlags::TRANSFER,
            ),
            Self::TransferDst => (
                vk::AccessFlags::TRANSFER_WRITE,
                vk::PipelineStageFlags::TRANSFER,
            ),
            Self::IndirectDraw => (
                vk::AccessFlags::INDIRECT_COMMAND_READ,
                vk::PipelineStageFlags::DRAW_INDIRECT,
            ),
            Self::Custom { access, stage } => (access, stage),
        }
    }
}

/// Key for per-subresource tracking.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
struct SubresourceKey {
    image: vk::Image,
    mip_level: u32,
    array_layer: u32,
}

/// State of a single image subresource.
#[derive(Debug, Clone, Copy)]
pub struct SubresourceState {
    /// Current image layout.
    pub layout: vk::ImageLayout,
    /// Current access mask.
    pub access: vk::AccessFlags,
    /// Current pipeline stage.
    pub stage: vk::PipelineStageFlags,
    /// Queue family that owns this subresource.
    pub queue_family: u32,
}

/// State of a tracked buffer.
#[derive(Debug, Clone, Copy)]
pub struct BufferState {
    /// Current access mask.
    pub access: vk::AccessFlags,
    /// Current pipeline stage.
    pub stage: vk::PipelineStageFlags,
    /// Queue family that owns this buffer.
    pub queue_family: u32,
}

/// Image layout transition barrier.
#[derive(Debug, Clone)]
pub struct ImageTransition {
    /// The image being transitioned.
    pub image: vk::Image,
    /// Layout before the transition.
    pub old_layout: vk::ImageLayout,
    /// Layout after the transition.
    pub new_layout: vk::ImageLayout,
    /// Source access mask (what the image was doing before).
    pub src_access: vk::AccessFlags,
    /// Destination access mask (what the image will do after).
    pub dst_access: vk::AccessFlags,
    /// Source pipeline stage.
    pub src_stage: vk::PipelineStageFlags,
    /// Destination pipeline stage.
    pub dst_stage: vk::PipelineStageFlags,
    /// Subresource range affected by the transition.
    pub subresource_range: vk::ImageSubresourceRange,
    /// Source queue family index for ownership transfer.
    pub src_queue_family: u32,
    /// Destination queue family index for ownership transfer.
    pub dst_queue_family: u32,
}

impl ImageTransition {
    /// Convert to a `VkImageMemoryBarrier`.
    pub fn to_barrier(&self) -> vk::ImageMemoryBarrier<'_> {
        vk::ImageMemoryBarrier::default()
            .src_access_mask(self.src_access)
            .dst_access_mask(self.dst_access)
            .old_layout(self.old_layout)
            .new_layout(self.new_layout)
            .src_queue_family_index(self.src_queue_family)
            .dst_queue_family_index(self.dst_queue_family)
            .image(self.image)
            .subresource_range(self.subresource_range)
    }
}

/// Buffer memory barrier.
#[derive(Debug, Clone)]
pub struct BufferTransition {
    /// The buffer being transitioned.
    pub buffer: vk::Buffer,
    /// Source access mask.
    pub src_access: vk::AccessFlags,
    /// Destination access mask.
    pub dst_access: vk::AccessFlags,
    /// Source pipeline stage.
    pub src_stage: vk::PipelineStageFlags,
    /// Destination pipeline stage.
    pub dst_stage: vk::PipelineStageFlags,
    /// Byte offset of the affected region.
    pub offset: vk::DeviceSize,
    /// Size of the affected region in bytes.
    pub size: vk::DeviceSize,
    /// Source queue family index.
    pub src_queue_family: u32,
    /// Destination queue family index.
    pub dst_queue_family: u32,
}

impl BufferTransition {
    /// Convert to a `VkBufferMemoryBarrier`.
    pub fn to_barrier(&self) -> vk::BufferMemoryBarrier<'_> {
        vk::BufferMemoryBarrier::default()
            .src_access_mask(self.src_access)
            .dst_access_mask(self.dst_access)
            .src_queue_family_index(self.src_queue_family)
            .dst_queue_family_index(self.dst_queue_family)
            .buffer(self.buffer)
            .offset(self.offset)
            .size(self.size)
    }
}

/// Image tracking metadata.
struct TrackedImage {
    mip_levels: u32,
    array_layers: u32,
    aspect: vk::ImageAspectFlags,
}

/// Resource tracker with per-subresource image tracking and buffer support.
pub struct ResourceTracker {
    subresources: HashMap<SubresourceKey, SubresourceState>,
    images: HashMap<vk::Image, TrackedImage>,
    buffers: HashMap<vk::Buffer, BufferState>,
}

impl ResourceTracker {
    /// Create an empty tracker.
    pub fn new() -> Self {
        Self {
            subresources: HashMap::new(),
            images: HashMap::new(),
            buffers: HashMap::new(),
        }
    }

    /// Start tracking an image. Initializes all subresources to the given layout.
    pub fn track_image(
        &mut self,
        image: vk::Image,
        initial_layout: vk::ImageLayout,
        mip_levels: u32,
        array_layers: u32,
        aspect: vk::ImageAspectFlags,
    ) {
        self.images.insert(image, TrackedImage {
            mip_levels,
            array_layers,
            aspect,
        });
        for mip in 0..mip_levels {
            for layer in 0..array_layers {
                self.subresources.insert(
                    SubresourceKey { image, mip_level: mip, array_layer: layer },
                    SubresourceState {
                        layout: initial_layout,
                        access: vk::AccessFlags::empty(),
                        stage: vk::PipelineStageFlags::TOP_OF_PIPE,
                        queue_family: vk::QUEUE_FAMILY_IGNORED,
                    },
                );
            }
        }
    }

    /// Stop tracking an image.
    pub fn untrack_image(&mut self, image: vk::Image) {
        if let Some(meta) = self.images.remove(&image) {
            for mip in 0..meta.mip_levels {
                for layer in 0..meta.array_layers {
                    self.subresources.remove(&SubresourceKey {
                        image,
                        mip_level: mip,
                        array_layer: layer,
                    });
                }
            }
        }
    }

    /// Transition an entire image to a new usage.
    ///
    /// Returns `None` if the image is not tracked.
    pub fn transition_image(
        &mut self,
        image: vk::Image,
        usage: ImageUsageContext,
    ) -> Option<ImageTransition> {
        let meta = self.images.get(&image)?;
        let aspect = meta.aspect;
        let mip_levels = meta.mip_levels;
        let array_layers = meta.array_layers;

        let (new_layout, dst_access, dst_stage) = usage.resolve();

        // Find the "worst case" source state across all subresources.
        let first_key = SubresourceKey {
            image,
            mip_level: 0,
            array_layer: 0,
        };
        let first = self.subresources.get(&first_key)?;

        if first.layout == new_layout
            && first.access == dst_access
            && first.stage == dst_stage
        {
            // Check if all subresources match (potential no-op).
            let all_same = (0..mip_levels).all(|m| {
                (0..array_layers).all(|l| {
                    let k = SubresourceKey {
                        image,
                        mip_level: m,
                        array_layer: l,
                    };
                    self.subresources.get(&k).map_or(false, |s| {
                        s.layout == new_layout
                    })
                })
            });
            if all_same {
                return None;
            }
        }

        let old_layout = first.layout;
        let src_access = first.access;
        let src_stage = first.stage;
        let src_qf = first.queue_family;

        // Update all subresources.
        for mip in 0..mip_levels {
            for layer in 0..array_layers {
                let k = SubresourceKey {
                    image,
                    mip_level: mip,
                    array_layer: layer,
                };
                if let Some(s) = self.subresources.get_mut(&k) {
                    s.layout = new_layout;
                    s.access = dst_access;
                    s.stage = dst_stage;
                }
            }
        }

        Some(ImageTransition {
            image,
            old_layout,
            new_layout,
            src_access,
            dst_access,
            src_stage,
            dst_stage,
            subresource_range: vk::ImageSubresourceRange {
                aspect_mask: aspect,
                base_mip_level: 0,
                level_count: mip_levels,
                base_array_layer: 0,
                layer_count: array_layers,
            },
            src_queue_family: src_qf,
            dst_queue_family: src_qf,
        })
    }

    /// Transition a specific mip level of an image.
    ///
    /// Essential for mipmap generation where each level has a different layout.
    pub fn transition_mip(
        &mut self,
        image: vk::Image,
        mip_level: u32,
        usage: ImageUsageContext,
    ) -> Option<ImageTransition> {
        let meta = self.images.get(&image)?;
        let aspect = meta.aspect;
        let array_layers = meta.array_layers;

        let (new_layout, dst_access, dst_stage) = usage.resolve();

        let first_key = SubresourceKey {
            image,
            mip_level,
            array_layer: 0,
        };
        let first = self.subresources.get(&first_key)?;

        if first.layout == new_layout {
            return None;
        }

        let old_layout = first.layout;
        let src_access = first.access;
        let src_stage = first.stage;
        let src_qf = first.queue_family;

        for layer in 0..array_layers {
            let k = SubresourceKey {
                image,
                mip_level,
                array_layer: layer,
            };
            if let Some(s) = self.subresources.get_mut(&k) {
                s.layout = new_layout;
                s.access = dst_access;
                s.stage = dst_stage;
            }
        }

        Some(ImageTransition {
            image,
            old_layout,
            new_layout,
            src_access,
            dst_access,
            src_stage,
            dst_stage,
            subresource_range: vk::ImageSubresourceRange {
                aspect_mask: aspect,
                base_mip_level: mip_level,
                level_count: 1,
                base_array_layer: 0,
                layer_count: array_layers,
            },
            src_queue_family: src_qf,
            dst_queue_family: src_qf,
        })
    }

    /// Get the current state of a subresource.
    pub fn subresource_state(
        &self,
        image: vk::Image,
        mip_level: u32,
        array_layer: u32,
    ) -> Option<&SubresourceState> {
        self.subresources.get(&SubresourceKey {
            image,
            mip_level,
            array_layer,
        })
    }

    // Buffer tracking.

    /// Start tracking a buffer.
    pub fn track_buffer(&mut self, buffer: vk::Buffer) {
        self.buffers.insert(
            buffer,
            BufferState {
                access: vk::AccessFlags::empty(),
                stage: vk::PipelineStageFlags::TOP_OF_PIPE,
                queue_family: vk::QUEUE_FAMILY_IGNORED,
            },
        );
    }

    /// Stop tracking a buffer.
    pub fn untrack_buffer(&mut self, buffer: vk::Buffer) {
        self.buffers.remove(&buffer);
    }

    /// Transition a buffer to a new usage.
    ///
    /// Returns `None` if the buffer is not tracked or already in the
    /// requested state.
    pub fn transition_buffer(
        &mut self,
        buffer: vk::Buffer,
        usage: BufferUsageContext,
    ) -> Option<BufferTransition> {
        let state = self.buffers.get_mut(&buffer)?;
        let (dst_access, dst_stage) = usage.resolve();

        if state.access == dst_access && state.stage == dst_stage {
            return None;
        }

        let src_access = state.access;
        let src_stage = state.stage;
        let src_qf = state.queue_family;

        state.access = dst_access;
        state.stage = dst_stage;

        Some(BufferTransition {
            buffer,
            src_access,
            dst_access,
            src_stage,
            dst_stage,
            offset: 0,
            size: vk::WHOLE_SIZE,
            src_queue_family: src_qf,
            dst_queue_family: src_qf,
        })
    }

    /// Number of tracked images.
    pub fn image_count(&self) -> usize {
        self.images.len()
    }

    /// Number of tracked buffers.
    pub fn buffer_count(&self) -> usize {
        self.buffers.len()
    }

    /// Clear all tracking state.
    pub fn clear(&mut self) {
        self.subresources.clear();
        self.images.clear();
        self.buffers.clear();
    }
}

impl Default for ResourceTracker {
    fn default() -> Self {
        Self::new()
    }
}

// CommandRecorder integration.

impl CommandRecorder<'_> {
    /// Apply image and buffer transitions as a single pipeline barrier.
    pub fn apply_image_transitions(&self, transitions: &[ImageTransition]) {
        if transitions.is_empty() {
            return;
        }
        let barriers: Vec<vk::ImageMemoryBarrier<'_>> =
            transitions.iter().map(|t| t.to_barrier()).collect();
        let src = transitions
            .iter()
            .fold(vk::PipelineStageFlags::empty(), |a, t| a | t.src_stage);
        let dst = transitions
            .iter()
            .fold(vk::PipelineStageFlags::empty(), |a, t| a | t.dst_stage);
        unsafe {
            self.device.cmd_pipeline_barrier(
                self.buffer,
                src,
                dst,
                vk::DependencyFlags::empty(),
                &[],
                &[],
                &barriers,
            );
        }
    }

    /// Apply buffer transitions.
    pub fn apply_buffer_transitions(&self, transitions: &[BufferTransition]) {
        if transitions.is_empty() {
            return;
        }
        let barriers: Vec<vk::BufferMemoryBarrier<'_>> =
            transitions.iter().map(|t| t.to_barrier()).collect();
        let src = transitions
            .iter()
            .fold(vk::PipelineStageFlags::empty(), |a, t| a | t.src_stage);
        let dst = transitions
            .iter()
            .fold(vk::PipelineStageFlags::empty(), |a, t| a | t.dst_stage);
        unsafe {
            self.device.cmd_pipeline_barrier(
                self.buffer,
                src,
                dst,
                vk::DependencyFlags::empty(),
                &[],
                &barriers,
                &[],
            );
        }
    }
}