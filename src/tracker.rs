//! GPU resource state tracking and barrier computation.
//!
//! [`ResourceTracker`] maintains per-image layout state and computes
//! the appropriate memory barriers when transitioning between layouts.
//! It automatically infers source/destination access masks and pipeline
//! stages from the layouts, eliminating the most common source of
//! Vulkan barrier bugs.
//!
//! # Design Philosophy
//!
//! The tracker is **advisory**, not automatic:
//!
//! - It does NOT inject barriers into command buffers behind your back.
//! - It computes [`ImageTransition`] structs that describe what barrier
//!   is needed.
//! - You apply the barriers yourself via
//!   [`CommandRecorder::apply_transitions`](crate::CommandRecorder).
//!
//! This means `wgpu` / `vulkano` users can safely ignore this module.
//! It cannot conflict with their internal barrier management.
//!
//! # Limitations
//!
//! - Tracks layout per-image, not per-subresource-range. If you need to
//!   track individual mip levels or array layers in different layouts,
//!   use the raw barrier API.
//! - Single-threaded. If recording command buffers on multiple threads,
//!   wrap in `Mutex` or use one tracker per thread with manual merging.
//!
//! # Example
//!
//! ```rust,no_run
//! # use ignis::*; use ignis::tracker::*; use ash::vk;
//! # fn example(recorder: &CommandRecorder, image: vk::Image) {
//! let mut tracker = ResourceTracker::new();
//! tracker.track_image(image, vk::ImageLayout::UNDEFINED);
//!
//! // Transition to transfer destination for an upload.
//! let t1 = tracker.transition(image, vk::ImageLayout::TRANSFER_DST_OPTIMAL)
//!     .expect("image is tracked");
//! recorder.apply_transitions(&[t1]);
//!
//! // ... copy data into image ...
//!
//! // Transition to shader-readable for sampling.
//! let t2 = tracker.transition(image, vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL)
//!     .expect("image is tracked");
//! recorder.apply_transitions(&[t2]);
//! # }
//! ```

use std::collections::HashMap;

use ash::vk;

use crate::command::CommandRecorder;

/// Tracked state of a single image.
#[derive(Debug, Clone, Copy)]
pub struct ImageState {
    /// Current layout.
    pub layout: vk::ImageLayout,
    /// Access mask describing how the image was last accessed.
    pub access: vk::AccessFlags,
    /// Pipeline stage of the last access.
    pub stage: vk::PipelineStageFlags,
    /// Queue family that currently owns the image.
    /// `vk::QUEUE_FAMILY_IGNORED` if ownership tracking is not used.
    pub queue_family: u32,
}

impl ImageState {
    /// Create a new image state with common defaults.
    pub fn new(layout: vk::ImageLayout) -> Self {
        let (access, stage) = infer_access_and_stage(layout);
        Self {
            layout,
            access,
            stage,
            queue_family: vk::QUEUE_FAMILY_IGNORED,
        }
    }
}

/// Describes a computed image layout transition.
///
/// Contains all the information needed to emit a `VkImageMemoryBarrier`.
/// Apply via [`CommandRecorder::apply_transitions`].
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
    /// Convert to a `VkImageMemoryBarrier` suitable for
    /// `vkCmdPipelineBarrier`.
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

/// Tracks the current state of GPU images and computes layout transitions.
///
/// See [module documentation](self) for design rationale and examples.
pub struct ResourceTracker {
    images: HashMap<vk::Image, ImageState>,
    default_subresource_range: vk::ImageSubresourceRange,
}

impl ResourceTracker {
    /// Create an empty tracker.
    pub fn new() -> Self {
        Self {
            images: HashMap::new(),
            default_subresource_range: vk::ImageSubresourceRange {
                aspect_mask: vk::ImageAspectFlags::COLOR,
                base_mip_level: 0,
                level_count: vk::REMAINING_MIP_LEVELS,
                base_array_layer: 0,
                layer_count: vk::REMAINING_ARRAY_LAYERS,
            },
        }
    }

    /// Begin tracking an image with the given initial layout.
    ///
    /// If the image is already tracked, its state is overwritten.
    pub fn track_image(&mut self, image: vk::Image, initial_layout: vk::ImageLayout) {
        self.images.insert(image, ImageState::new(initial_layout));
    }

    /// Begin tracking an image with a fully specified initial state.
    pub fn track_image_with_state(&mut self, image: vk::Image, state: ImageState) {
        self.images.insert(image, state);
    }

    /// Stop tracking an image.
    pub fn untrack_image(&mut self, image: vk::Image) {
        self.images.remove(&image);
    }

    /// Query the current tracked state of an image.
    pub fn image_state(&self, image: vk::Image) -> Option<&ImageState> {
        self.images.get(&image)
    }

    /// Compute a transition to a new layout.
    ///
    /// Returns `None` if:
    /// - The image is not tracked
    /// - The image is already in `new_layout`
    ///
    /// On success, updates the tracked state to reflect `new_layout`.
    pub fn transition(
        &mut self,
        image: vk::Image,
        new_layout: vk::ImageLayout,
    ) -> Option<ImageTransition> {
        let state = self.images.get_mut(&image)?;

        if state.layout == new_layout {
            return None;
        }

        let old_layout = state.layout;
        let src_access = state.access;
        let src_stage = state.stage;
        let src_qf = state.queue_family;

        let (dst_access, dst_stage) = infer_access_and_stage(new_layout);

        // Update tracked state.
        state.layout = new_layout;
        state.access = dst_access;
        state.stage = dst_stage;

        Some(ImageTransition {
            image,
            old_layout,
            new_layout,
            src_access,
            dst_access,
            src_stage,
            dst_stage,
            subresource_range: self.default_subresource_range,
            src_queue_family: src_qf,
            dst_queue_family: src_qf,
        })
    }

    /// Compute a transition with an explicit subresource range.
    ///
    /// Useful for depth images, cube maps, or individual mip levels.
    /// Note: the tracker still stores only one state per image, so
    /// this does NOT enable true per-subresource tracking.
    pub fn transition_subresource(
        &mut self,
        image: vk::Image,
        new_layout: vk::ImageLayout,
        range: vk::ImageSubresourceRange,
    ) -> Option<ImageTransition> {
        let mut t = self.transition(image, new_layout)?;
        t.subresource_range = range;
        Some(t)
    }

    /// Compute a transition with custom destination access and stage.
    ///
    /// Overrides the automatic inference. Useful when the default
    /// heuristic does not match your use case (e.g. compute shader
    /// reads from `SHADER_READ_ONLY_OPTIMAL` should use
    /// `COMPUTE_SHADER` stage, not `FRAGMENT_SHADER`).
    pub fn transition_custom(
        &mut self,
        image: vk::Image,
        new_layout: vk::ImageLayout,
        dst_access: vk::AccessFlags,
        dst_stage: vk::PipelineStageFlags,
    ) -> Option<ImageTransition> {
        let state = self.images.get_mut(&image)?;

        if state.layout == new_layout {
            return None;
        }

        let old_layout = state.layout;
        let src_access = state.access;
        let src_stage = state.stage;
        let src_qf = state.queue_family;

        state.layout = new_layout;
        state.access = dst_access;
        state.stage = dst_stage;

        Some(ImageTransition {
            image,
            old_layout,
            new_layout,
            src_access,
            dst_access,
            src_stage,
            dst_stage,
            subresource_range: self.default_subresource_range,
            src_queue_family: src_qf,
            dst_queue_family: src_qf,
        })
    }

    /// Compute a queue family ownership transfer.
    ///
    /// This generates a release barrier (for the source queue) or an
    /// acquire barrier (for the destination queue). You must record
    /// BOTH: the release on the source queue and the acquire on the
    /// destination queue.
    ///
    /// Updates the tracked queue family.
    pub fn transfer_ownership(
        &mut self,
        image: vk::Image,
        new_layout: vk::ImageLayout,
        dst_queue_family: u32,
    ) -> Option<ImageTransition> {
        let state = self.images.get_mut(&image)?;

        let old_layout = state.layout;
        let src_access = state.access;
        let src_stage = state.stage;
        let src_qf = state.queue_family;

        let (dst_access, dst_stage) = infer_access_and_stage(new_layout);

        state.layout = new_layout;
        state.access = dst_access;
        state.stage = dst_stage;
        state.queue_family = dst_queue_family;

        Some(ImageTransition {
            image,
            old_layout,
            new_layout,
            src_access,
            dst_access,
            src_stage,
            dst_stage,
            subresource_range: self.default_subresource_range,
            src_queue_family: src_qf,
            dst_queue_family,
        })
    }

    /// Set the default subresource range used by [`transition`](Self::transition).
    ///
    /// Defaults to all color mips and layers. Change this if you
    /// primarily work with depth images.
    pub fn set_default_subresource_range(&mut self, range: vk::ImageSubresourceRange) {
        self.default_subresource_range = range;
    }

    /// Number of tracked images.
    pub fn tracked_count(&self) -> usize {
        self.images.len()
    }

    /// Remove all tracked images.
    pub fn clear(&mut self) {
        self.images.clear();
    }
}

impl Default for ResourceTracker {
    fn default() -> Self {
        Self::new()
    }
}

/// Infer typical access flags and pipeline stage from an image layout.
///
/// These are reasonable defaults for the most common use cases. When the
/// heuristic does not match your situation, use
/// [`ResourceTracker::transition_custom`].
///
/// | Layout | Access | Stage |
/// |---|---|---|
/// | `UNDEFINED` | - | `TOP_OF_PIPE` |
/// | `GENERAL` | `MEMORY_READ \| MEMORY_WRITE` | `ALL_COMMANDS` |
/// | `COLOR_ATTACHMENT_OPTIMAL` | `COLOR_ATTACHMENT_WRITE` | `COLOR_ATTACHMENT_OUTPUT` |
/// | `DEPTH_STENCIL_ATTACHMENT_OPTIMAL` | `DEPTH_STENCIL_ATTACHMENT_WRITE` | `EARLY_FRAGMENT_TESTS` |
/// | `DEPTH_STENCIL_READ_ONLY_OPTIMAL` | `DEPTH_STENCIL_ATTACHMENT_READ` | `EARLY_FRAGMENT_TESTS` |
/// | `SHADER_READ_ONLY_OPTIMAL` | `SHADER_READ` | `FRAGMENT_SHADER` |
/// | `TRANSFER_SRC_OPTIMAL` | `TRANSFER_READ` | `TRANSFER` |
/// | `TRANSFER_DST_OPTIMAL` | `TRANSFER_WRITE` | `TRANSFER` |
/// | `PRESENT_SRC_KHR` | - | `BOTTOM_OF_PIPE` |
pub fn infer_access_and_stage(
    layout: vk::ImageLayout,
) -> (vk::AccessFlags, vk::PipelineStageFlags) {
    match layout {
        vk::ImageLayout::UNDEFINED | vk::ImageLayout::PREINITIALIZED => (
            vk::AccessFlags::empty(),
            vk::PipelineStageFlags::TOP_OF_PIPE,
        ),
        vk::ImageLayout::GENERAL => (
            vk::AccessFlags::MEMORY_READ | vk::AccessFlags::MEMORY_WRITE,
            vk::PipelineStageFlags::ALL_COMMANDS,
        ),
        vk::ImageLayout::COLOR_ATTACHMENT_OPTIMAL => (
            vk::AccessFlags::COLOR_ATTACHMENT_READ | vk::AccessFlags::COLOR_ATTACHMENT_WRITE,
            vk::PipelineStageFlags::COLOR_ATTACHMENT_OUTPUT,
        ),
        vk::ImageLayout::DEPTH_STENCIL_ATTACHMENT_OPTIMAL => (
            vk::AccessFlags::DEPTH_STENCIL_ATTACHMENT_READ
                | vk::AccessFlags::DEPTH_STENCIL_ATTACHMENT_WRITE,
            vk::PipelineStageFlags::EARLY_FRAGMENT_TESTS
                | vk::PipelineStageFlags::LATE_FRAGMENT_TESTS,
        ),
        vk::ImageLayout::DEPTH_STENCIL_READ_ONLY_OPTIMAL => (
            vk::AccessFlags::DEPTH_STENCIL_ATTACHMENT_READ,
            vk::PipelineStageFlags::EARLY_FRAGMENT_TESTS
                | vk::PipelineStageFlags::LATE_FRAGMENT_TESTS,
        ),
        vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL => (
            vk::AccessFlags::SHADER_READ,
            vk::PipelineStageFlags::FRAGMENT_SHADER,
        ),
        vk::ImageLayout::TRANSFER_SRC_OPTIMAL => (
            vk::AccessFlags::TRANSFER_READ,
            vk::PipelineStageFlags::TRANSFER,
        ),
        vk::ImageLayout::TRANSFER_DST_OPTIMAL => (
            vk::AccessFlags::TRANSFER_WRITE,
            vk::PipelineStageFlags::TRANSFER,
        ),
        vk::ImageLayout::PRESENT_SRC_KHR => (
            vk::AccessFlags::empty(),
            vk::PipelineStageFlags::BOTTOM_OF_PIPE,
        ),
        _ => (
            vk::AccessFlags::MEMORY_READ | vk::AccessFlags::MEMORY_WRITE,
            vk::PipelineStageFlags::ALL_COMMANDS,
        ),
    }
}

/// Extension methods on [`CommandRecorder`] for applying tracked transitions.
impl CommandRecorder<'_> {
    /// Apply a batch of image transitions as a single pipeline barrier.
    ///
    /// Computes the union of all source and destination stages, then
    /// issues one `vkCmdPipelineBarrier` call with all barriers.
    ///
    /// Does nothing if `transitions` is empty.
    ///
    /// # Example
    ///
    /// ```rust,no_run
    /// # use ignis::*; use ignis::tracker::*; use ash::vk;
    /// # fn example(recorder: &CommandRecorder, img_a: vk::Image, img_b: vk::Image) {
    /// let mut tracker = ResourceTracker::new();
    /// tracker.track_image(img_a, vk::ImageLayout::UNDEFINED);
    /// tracker.track_image(img_b, vk::ImageLayout::UNDEFINED);
    ///
    /// let transitions: Vec<_> = [
    ///     tracker.transition(img_a, vk::ImageLayout::COLOR_ATTACHMENT_OPTIMAL),
    ///     tracker.transition(img_b, vk::ImageLayout::DEPTH_STENCIL_ATTACHMENT_OPTIMAL),
    /// ].into_iter().flatten().collect();
    ///
    /// recorder.apply_transitions(&transitions);
    /// # }
    /// ```
    pub fn apply_transitions(&self, transitions: &[ImageTransition]) {
        if transitions.is_empty() {
            return;
        }

        let barriers: Vec<vk::ImageMemoryBarrier<'_>> =
            transitions.iter().map(|t| t.to_barrier()).collect();

        let src_stage = transitions
            .iter()
            .fold(vk::PipelineStageFlags::empty(), |acc, t| acc | t.src_stage);
        let dst_stage = transitions
            .iter()
            .fold(vk::PipelineStageFlags::empty(), |acc, t| acc | t.dst_stage);

        unsafe {
            self.device.cmd_pipeline_barrier(
                self.buffer,
                src_stage,
                dst_stage,
                vk::DependencyFlags::empty(),
                &[],
                &[],
                &barriers,
            );
        }
    }

    /// Transition a single image layout without a tracker.
    ///
    /// Stateless convenience method that infers access masks and stages
    /// from the old and new layouts. For tracked state management, prefer
    /// [`ResourceTracker`].
    pub fn transition_image_layout(
        &self,
        image: vk::Image,
        old_layout: vk::ImageLayout,
        new_layout: vk::ImageLayout,
        subresource_range: vk::ImageSubresourceRange,
    ) {
        let (src_access, src_stage) = infer_access_and_stage(old_layout);
        let (dst_access, dst_stage) = infer_access_and_stage(new_layout);

        let barrier = vk::ImageMemoryBarrier::default()
            .src_access_mask(src_access)
            .dst_access_mask(dst_access)
            .old_layout(old_layout)
            .new_layout(new_layout)
            .src_queue_family_index(vk::QUEUE_FAMILY_IGNORED)
            .dst_queue_family_index(vk::QUEUE_FAMILY_IGNORED)
            .image(image)
            .subresource_range(subresource_range);

        unsafe {
            self.device.cmd_pipeline_barrier(
                self.buffer,
                src_stage,
                dst_stage,
                vk::DependencyFlags::empty(),
                &[],
                &[],
                std::slice::from_ref(&barrier),
            );
        }
    }
}
