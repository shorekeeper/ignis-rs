//! Command pool and buffer management with parallel recording support.
//!
//! This module provides:
//!
//! - [`CommandPool`]: a wrapper around `VkCommandPool` with buffer allocation
//! - [`CommandRecorder`]: a thin safe wrapper for recording Vulkan commands
//! - [`ParallelRecorder`]: multi-threaded secondary command buffer recording
//!   using `std::thread::scope`
//!
//! # Thread Safety
//!
//! Each `CommandPool` is tied to a single thread (per Vulkan spec). For
//! multi-threaded recording, use [`ParallelRecorder`] which manages one
//! pool per worker thread.

use std::sync::Arc;

use ash::vk;

use crate::device::SharedState;
use crate::error::{Error, Result};

/// Wraps a `VkCommandPool` with buffer allocation and lifecycle management.
///
/// Created via [`Ignis::create_command_pool`](crate::Ignis::create_command_pool).
///
/// # Thread Safety
///
/// A `CommandPool` must only be used from a single thread at a time.
/// It is `Send` (can be moved between threads) but not `Sync`.
pub struct CommandPool {
    pub(crate) shared: Arc<SharedState>,
    handle: vk::CommandPool,
    family_index: u32,
}

impl CommandPool {
    /// Create a new command pool for the given queue family.
    pub(crate) fn new(shared: Arc<SharedState>, family_index: u32) -> Result<Self> {
        let info = vk::CommandPoolCreateInfo::default()
            .queue_family_index(family_index)
            .flags(vk::CommandPoolCreateFlags::RESET_COMMAND_BUFFER);

        // SAFETY: device and family index are valid.
        let handle = unsafe { shared.device.create_command_pool(&info, None)? };

        Ok(Self {
            shared,
            handle,
            family_index,
        })
    }

    /// Allocate a single primary command buffer.
    pub fn allocate_primary(&self) -> Result<vk::CommandBuffer> {
        self.allocate(vk::CommandBufferLevel::PRIMARY, 1)
            .map(|v| v[0])
    }

    /// Allocate a single secondary command buffer.
    pub fn allocate_secondary(&self) -> Result<vk::CommandBuffer> {
        self.allocate(vk::CommandBufferLevel::SECONDARY, 1)
            .map(|v| v[0])
    }

    /// Allocate `count` command buffers at the specified level.
    pub fn allocate(
        &self,
        level: vk::CommandBufferLevel,
        count: u32,
    ) -> Result<Vec<vk::CommandBuffer>> {
        let alloc_info = vk::CommandBufferAllocateInfo::default()
            .command_pool(self.handle)
            .level(level)
            .command_buffer_count(count);

        // SAFETY: pool is valid, device is valid.
        let buffers = unsafe { self.shared.device.allocate_command_buffers(&alloc_info)? };
        Ok(buffers)
    }

    /// Reset the entire pool, recycling all allocated command buffers.
    pub fn reset(&self) -> Result<()> {
        // SAFETY: pool is valid.
        unsafe {
            self.shared
                .device
                .reset_command_pool(self.handle, vk::CommandPoolResetFlags::empty())?;
        }
        Ok(())
    }

    /// Begin recording a primary command buffer for one-time submission.
    ///
    /// Convenience method that calls `vkBeginCommandBuffer` with
    /// `ONE_TIME_SUBMIT` flag.
    pub fn begin_primary(&self, buffer: vk::CommandBuffer) -> Result<CommandRecorder<'_>> {
        let begin_info = vk::CommandBufferBeginInfo::default()
            .flags(vk::CommandBufferUsageFlags::ONE_TIME_SUBMIT);

        // SAFETY: buffer was allocated from this pool, pool is valid.
        unsafe {
            self.shared
                .device
                .begin_command_buffer(buffer, &begin_info)?;
        }

        Ok(CommandRecorder {
            device: &self.shared.device,
            buffer,
        })
    }

    /// Get the raw pool handle.
    #[inline]
    pub fn handle(&self) -> vk::CommandPool {
        self.handle
    }

    /// Queue family this pool was created for.
    #[inline]
    pub fn family_index(&self) -> u32 {
        self.family_index
    }
}

impl Drop for CommandPool {
    fn drop(&mut self) {
        unsafe {
            self.shared.device.destroy_command_pool(self.handle, None);
        }
    }
}

/// A thin safe wrapper around a command buffer in recording state.
///
/// Provides safe methods for common Vulkan commands and raw access for
/// anything not covered. The underlying command buffer is expected to have
/// been begun via [`CommandPool::begin_primary`] or the parallel recorder.
///
/// # Ending Recording
///
/// Call [`end`](CommandRecorder::end) when finished, or let the recorder
/// go out of scope (which does NOT end recording - you must call `end`).
pub struct CommandRecorder<'a> {
    pub(crate) device: &'a ash::Device,
    pub(crate) buffer: vk::CommandBuffer,
}

impl CommandRecorder<'_> {
    /// End command buffer recording.
    pub fn end(self) -> Result<vk::CommandBuffer> {
        // SAFETY: buffer is in recording state.
        unsafe { self.device.end_command_buffer(self.buffer)? };
        Ok(self.buffer)
    }

    /// Bind a pipeline.
    pub fn bind_pipeline(&self, bind_point: vk::PipelineBindPoint, pipeline: vk::Pipeline) {
        unsafe {
            self.device
                .cmd_bind_pipeline(self.buffer, bind_point, pipeline);
        }
    }

    /// Bind descriptor sets.
    pub fn bind_descriptor_sets(
        &self,
        bind_point: vk::PipelineBindPoint,
        layout: vk::PipelineLayout,
        first_set: u32,
        sets: &[vk::DescriptorSet],
        dynamic_offsets: &[u32],
    ) {
        unsafe {
            self.device.cmd_bind_descriptor_sets(
                self.buffer,
                bind_point,
                layout,
                first_set,
                sets,
                dynamic_offsets,
            );
        }
    }

    /// Bind a vertex buffer.
    pub fn bind_vertex_buffers(
        &self,
        first_binding: u32,
        buffers: &[vk::Buffer],
        offsets: &[vk::DeviceSize],
    ) {
        unsafe {
            self.device
                .cmd_bind_vertex_buffers(self.buffer, first_binding, buffers, offsets);
        }
    }

    /// Bind an index buffer.
    pub fn bind_index_buffer(
        &self,
        buffer: vk::Buffer,
        offset: vk::DeviceSize,
        index_type: vk::IndexType,
    ) {
        unsafe {
            self.device
                .cmd_bind_index_buffer(self.buffer, buffer, offset, index_type);
        }
    }

    /// Draw primitives.
    pub fn draw(
        &self,
        vertex_count: u32,
        instance_count: u32,
        first_vertex: u32,
        first_instance: u32,
    ) {
        unsafe {
            self.device.cmd_draw(
                self.buffer,
                vertex_count,
                instance_count,
                first_vertex,
                first_instance,
            );
        }
    }

    /// Draw indexed primitives.
    pub fn draw_indexed(
        &self,
        index_count: u32,
        instance_count: u32,
        first_index: u32,
        vertex_offset: i32,
        first_instance: u32,
    ) {
        unsafe {
            self.device.cmd_draw_indexed(
                self.buffer,
                index_count,
                instance_count,
                first_index,
                vertex_offset,
                first_instance,
            );
        }
    }

    /// Dispatch compute work groups.
    pub fn dispatch(&self, group_count_x: u32, group_count_y: u32, group_count_z: u32) {
        unsafe {
            self.device
                .cmd_dispatch(self.buffer, group_count_x, group_count_y, group_count_z);
        }
    }

    /// Insert a pipeline barrier.
    pub fn pipeline_barrier(
        &self,
        src_stage: vk::PipelineStageFlags,
        dst_stage: vk::PipelineStageFlags,
        dependency_flags: vk::DependencyFlags,
        memory_barriers: &[vk::MemoryBarrier<'_>],
        buffer_barriers: &[vk::BufferMemoryBarrier<'_>],
        image_barriers: &[vk::ImageMemoryBarrier<'_>],
    ) {
        unsafe {
            self.device.cmd_pipeline_barrier(
                self.buffer,
                src_stage,
                dst_stage,
                dependency_flags,
                memory_barriers,
                buffer_barriers,
                image_barriers,
            );
        }
    }

    /// Execute secondary command buffers from a primary buffer.
    pub fn execute_commands(&self, secondaries: &[vk::CommandBuffer]) {
        unsafe {
            self.device.cmd_execute_commands(self.buffer, secondaries);
        }
    }

    /// Push constants.
    pub fn push_constants(
        &self,
        layout: vk::PipelineLayout,
        stage_flags: vk::ShaderStageFlags,
        offset: u32,
        data: &[u8],
    ) {
        unsafe {
            self.device
                .cmd_push_constants(self.buffer, layout, stage_flags, offset, data);
        }
    }

    /// Get the raw command buffer handle for direct Vulkan calls.
    #[inline]
    pub fn raw_buffer(&self) -> vk::CommandBuffer {
        self.buffer
    }

    /// Get the raw device handle for direct Vulkan calls.
    #[inline]
    pub fn raw_device(&self) -> &ash::Device {
        self.device
    }
}

/// Inheritance information for secondary command buffers recorded inside
/// a render pass.
#[derive(Debug, Clone, Copy, Default)]
pub struct CommandBufferInheritance {
    /// The render pass the secondary buffers will be used within.
    pub render_pass: vk::RenderPass,
    /// Index of the subpass the secondary buffers will execute in.
    pub subpass: u32,
    /// Optional framebuffer hint for driver optimization.
    pub framebuffer: vk::Framebuffer,
}

/// Multi-threaded command recorder using one command pool per worker thread.
///
/// Created via
/// [`Ignis::create_parallel_recorder`](crate::Ignis::create_parallel_recorder).
///
/// Uses `std::thread::scope` (stable since Rust 1.63) for scoped parallel
/// recording. Each worker thread gets its own command pool and records
/// into a secondary command buffer.
///
/// # Example
///
/// ```rust,no_run
/// # use ignis::*; use ash::vk;
/// # fn example(ignis: &Ignis, inheritance: CommandBufferInheritance) -> Result<()> {
/// let recorder = ignis.create_parallel_recorder(QueueType::Graphics, 4)?;
///
/// let tasks: Vec<Box<dyn Fn(&CommandRecorder) + Send + Sync>> = vec![
///     Box::new(|rec| { rec.draw(100, 1, 0, 0); }),
///     Box::new(|rec| { rec.draw(200, 1, 100, 0); }),
///     Box::new(|rec| { rec.draw(300, 1, 300, 0); }),
///     Box::new(|rec| { rec.draw(400, 1, 600, 0); }),
/// ];
///
/// let secondaries = recorder.record(&inheritance, &tasks)?;
///
/// // Execute all secondary buffers from a primary command buffer:
/// // recorder.execute_commands(&secondaries);
/// # Ok(())
/// # }
/// ```
pub struct ParallelRecorder {
    shared: Arc<SharedState>,
    pools: Vec<vk::CommandPool>,
    family_index: u32,
}

impl ParallelRecorder {
    /// Create a parallel recorder with `thread_count` command pools.
    pub(crate) fn new(
        shared: Arc<SharedState>,
        family_index: u32,
        thread_count: u32,
    ) -> Result<Self> {
        let mut pools = Vec::with_capacity(thread_count as usize);
        for _ in 0..thread_count {
            let info = vk::CommandPoolCreateInfo::default()
                .queue_family_index(family_index)
                .flags(
                    vk::CommandPoolCreateFlags::RESET_COMMAND_BUFFER
                        | vk::CommandPoolCreateFlags::TRANSIENT,
                );
            // SAFETY: device and family index are valid.
            let pool = unsafe { shared.device.create_command_pool(&info, None)? };
            pools.push(pool);
        }
        Ok(Self {
            shared,
            pools,
            family_index,
        })
    }

    /// The number of worker threads (pools) available.
    #[inline]
    pub fn thread_count(&self) -> u32 {
        self.pools.len() as u32
    }

    /// Queue family index.
    #[inline]
    pub fn family_index(&self) -> u32 {
        self.family_index
    }

    /// Reset all command pools, recycling previously allocated buffers.
    pub fn reset_all(&self) -> Result<()> {
        for &pool in &self.pools {
            unsafe {
                self.shared
                    .device
                    .reset_command_pool(pool, vk::CommandPoolResetFlags::empty())?;
            }
        }
        Ok(())
    }

    /// Record secondary command buffers in parallel.
    ///
    /// Each task closure receives a [`CommandRecorder`] and should record
    /// rendering commands into it. The number of tasks must not exceed
    /// [`thread_count`](ParallelRecorder::thread_count); excess tasks are
    /// silently ignored.
    ///
    /// All pools are reset before recording. The returned command buffers
    /// are in the executable state and can be passed to
    /// [`CommandRecorder::execute_commands`].
    ///
    /// # Errors
    ///
    /// Returns [`Error::ThreadPanic`] if any worker thread panics,
    /// or a Vulkan error for command buffer allocation/recording failures.
    pub fn record<F>(
        &self,
        inheritance: &CommandBufferInheritance,
        tasks: &[F],
    ) -> Result<Vec<vk::CommandBuffer>>
    where
        F: Fn(&CommandRecorder<'_>) + Send + Sync,
    {
        self.reset_all()?;

        let task_count = tasks.len().min(self.pools.len());
        let device = &self.shared.device;

        // Capture inheritance by value for use across threads.
        let rp = inheritance.render_pass;
        let subpass = inheritance.subpass;
        let fb = inheritance.framebuffer;

        std::thread::scope(|scope| {
            let handles: Vec<_> = (0..task_count)
                .map(|i| {
                    let pool = self.pools[i];
                    let task = &tasks[i];

                    scope.spawn(move || -> Result<vk::CommandBuffer> {
                        // Allocate a secondary command buffer from this thread's pool.
                        let alloc_info = vk::CommandBufferAllocateInfo::default()
                            .command_pool(pool)
                            .level(vk::CommandBufferLevel::SECONDARY)
                            .command_buffer_count(1);

                        // SAFETY: pool belongs to this thread exclusively.
                        let cmd = unsafe { device.allocate_command_buffers(&alloc_info)? }[0];

                        // Begin recording with render pass inheritance.
                        let inherit_info = vk::CommandBufferInheritanceInfo::default()
                            .render_pass(rp)
                            .subpass(subpass)
                            .framebuffer(fb);

                        let begin_info = vk::CommandBufferBeginInfo::default()
                            .flags(
                                vk::CommandBufferUsageFlags::ONE_TIME_SUBMIT
                                    | vk::CommandBufferUsageFlags::RENDER_PASS_CONTINUE,
                            )
                            .inheritance_info(&inherit_info);

                        // SAFETY: cmd is freshly allocated and not yet recording.
                        unsafe { device.begin_command_buffer(cmd, &begin_info)? };

                        let recorder = CommandRecorder {
                            device,
                            buffer: cmd,
                        };

                        task(&recorder);

                        // SAFETY: cmd is in recording state.
                        unsafe { device.end_command_buffer(cmd)? };

                        Ok(cmd)
                    })
                })
                .collect();

            // Collect results, propagating the first error.
            let mut buffers = Vec::with_capacity(task_count);
            for handle in handles {
                match handle.join() {
                    Ok(result) => buffers.push(result?),
                    Err(_) => return Err(Error::ThreadPanic),
                }
            }
            Ok(buffers)
        })
    }
}

impl Drop for ParallelRecorder {
    fn drop(&mut self) {
        for &pool in &self.pools {
            unsafe {
                self.shared.device.destroy_command_pool(pool, None);
            }
        }
    }
}

/// Configuration for a single color attachment in dynamic rendering.
///
/// Used by [`DynamicRenderPassBuilder`].
#[derive(Clone, Copy)]
pub struct ColorAttachmentInfo {
    /// Image view to render into.
    pub image_view: vk::ImageView,
    /// Layout of the image during rendering.
    pub image_layout: vk::ImageLayout,
    /// How to initialize the attachment at the start of rendering.
    pub load_op: vk::AttachmentLoadOp,
    /// How to handle the attachment at the end of rendering.
    pub store_op: vk::AttachmentStoreOp,
    /// Clear value (used if `load_op` is `CLEAR`).
    pub clear_value: vk::ClearValue,
    /// Image view of the resolve target, or `null` if no MSAA resolve.
    pub resolve_image_view: vk::ImageView,
    /// Layout of the resolve target image.
    pub resolve_image_layout: vk::ImageLayout,
    /// Resolve mode for MSAA.
    pub resolve_mode: vk::ResolveModeFlags,
}

impl std::fmt::Debug for ColorAttachmentInfo {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("ColorAttachmentInfo")
            .field("image_view", &self.image_view)
            .field("image_layout", &self.image_layout)
            .field("load_op", &self.load_op)
            .field("store_op", &self.store_op)
            .field("clear_value", &"ClearValue { .. }")
            .field("resolve_image_view", &self.resolve_image_view)
            .field("resolve_image_layout", &self.resolve_image_layout)
            .field("resolve_mode", &self.resolve_mode)
            .finish()
    }
}

impl Default for ColorAttachmentInfo {
    fn default() -> Self {
        Self {
            image_view: vk::ImageView::null(),
            image_layout: vk::ImageLayout::COLOR_ATTACHMENT_OPTIMAL,
            load_op: vk::AttachmentLoadOp::CLEAR,
            store_op: vk::AttachmentStoreOp::STORE,
            clear_value: vk::ClearValue {
                color: vk::ClearColorValue {
                    float32: [0.0, 0.0, 0.0, 1.0],
                },
            },
            resolve_image_view: vk::ImageView::null(),
            resolve_image_layout: vk::ImageLayout::UNDEFINED,
            resolve_mode: vk::ResolveModeFlags::NONE,
        }
    }
}

/// Configuration for a depth or stencil attachment in dynamic rendering.
#[derive(Clone, Copy)]
pub struct DepthStencilAttachmentInfo {
    /// Image view.
    pub image_view: vk::ImageView,
    /// Layout during rendering.
    pub image_layout: vk::ImageLayout,
    /// Load operation.
    pub load_op: vk::AttachmentLoadOp,
    /// Store operation.
    pub store_op: vk::AttachmentStoreOp,
    /// Clear value.
    pub clear_value: vk::ClearValue,
}

impl std::fmt::Debug for DepthStencilAttachmentInfo {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("DepthStencilAttachmentInfo")
            .field("image_view", &self.image_view)
            .field("image_layout", &self.image_layout)
            .field("load_op", &self.load_op)
            .field("store_op", &self.store_op)
            .field("clear_value", &"ClearValue { .. }")
            .finish()
    }
}

impl Default for DepthStencilAttachmentInfo {
    fn default() -> Self {
        Self {
            image_view: vk::ImageView::null(),
            image_layout: vk::ImageLayout::DEPTH_STENCIL_ATTACHMENT_OPTIMAL,
            load_op: vk::AttachmentLoadOp::CLEAR,
            store_op: vk::AttachmentStoreOp::STORE,
            clear_value: vk::ClearValue {
                depth_stencil: vk::ClearDepthStencilValue {
                    depth: 1.0,
                    stencil: 0,
                },
            },
        }
    }
}

/// Builder for Vulkan 1.3 dynamic rendering (no `VkRenderPass` needed).
///
/// Constructs and issues a `vkCmdBeginRendering` call on a command buffer.
/// Call [`begin`](DynamicRenderPassBuilder::begin) to start rendering, then
/// record draw commands, then call
/// [`CommandRecorder::end_rendering`](CommandRecorder::end_rendering).
///
/// # Example
///
/// ```rust,no_run
/// # use ignis::command::*; use ash::vk;
/// # fn example(recorder: &CommandRecorder, color_view: vk::ImageView,
/// #            depth_view: vk::ImageView) {
/// DynamicRenderPassBuilder::new()
///     .render_area(vk::Rect2D {
///         offset: vk::Offset2D { x: 0, y: 0 },
///         extent: vk::Extent2D { width: 1920, height: 1080 },
///     })
///     .color_attachment(ColorAttachmentInfo {
///         image_view: color_view,
///         ..Default::default()
///     })
///     .depth_attachment(DepthStencilAttachmentInfo {
///         image_view: depth_view,
///         ..Default::default()
///     })
///     .begin(recorder);
///
/// // ... draw commands ...
///
/// recorder.end_rendering();
/// # }
/// ```
///
/// # Requirements
///
/// Requires Vulkan 1.3 or `VK_KHR_dynamic_rendering`.
pub struct DynamicRenderPassBuilder {
    render_area: vk::Rect2D,
    layer_count: u32,
    view_mask: u32,
    color_attachments: Vec<ColorAttachmentInfo>,
    depth_attachment: Option<DepthStencilAttachmentInfo>,
    stencil_attachment: Option<DepthStencilAttachmentInfo>,
}

impl DynamicRenderPassBuilder {
    /// Create an empty builder.
    pub fn new() -> Self {
        Self {
            render_area: vk::Rect2D::default(),
            layer_count: 1,
            view_mask: 0,
            color_attachments: Vec::new(),
            depth_attachment: None,
            stencil_attachment: None,
        }
    }

    /// Set the render area.
    pub fn render_area(mut self, area: vk::Rect2D) -> Self {
        self.render_area = area;
        self
    }

    /// Set the number of layers for multiview rendering.
    pub fn layer_count(mut self, count: u32) -> Self {
        self.layer_count = count;
        self
    }

    /// Set the view mask for multiview rendering.
    pub fn view_mask(mut self, mask: u32) -> Self {
        self.view_mask = mask;
        self
    }

    /// Add a color attachment.
    pub fn color_attachment(mut self, info: ColorAttachmentInfo) -> Self {
        self.color_attachments.push(info);
        self
    }

    /// Set the depth attachment.
    pub fn depth_attachment(mut self, info: DepthStencilAttachmentInfo) -> Self {
        self.depth_attachment = Some(info);
        self
    }

    /// Set the stencil attachment.
    pub fn stencil_attachment(mut self, info: DepthStencilAttachmentInfo) -> Self {
        self.stencil_attachment = Some(info);
        self
    }

    /// Issue `vkCmdBeginRendering` on the given command recorder.
    ///
    /// After this call, record draw commands, then call
    /// `recorder.end_rendering()`.
    pub fn begin(self, recorder: &CommandRecorder<'_>) {
        let color_infos: Vec<vk::RenderingAttachmentInfo<'_>> = self
            .color_attachments
            .iter()
            .map(|a| {
                let mut info = vk::RenderingAttachmentInfo::default()
                    .image_view(a.image_view)
                    .image_layout(a.image_layout)
                    .load_op(a.load_op)
                    .store_op(a.store_op)
                    .clear_value(a.clear_value);

                if a.resolve_image_view != vk::ImageView::null() {
                    info = info
                        .resolve_image_view(a.resolve_image_view)
                        .resolve_image_layout(a.resolve_image_layout)
                        .resolve_mode(a.resolve_mode);
                }

                info
            })
            .collect();

        let depth_info = self.depth_attachment.map(|a| {
            vk::RenderingAttachmentInfo::default()
                .image_view(a.image_view)
                .image_layout(a.image_layout)
                .load_op(a.load_op)
                .store_op(a.store_op)
                .clear_value(a.clear_value)
        });

        let stencil_info = self.stencil_attachment.map(|a| {
            vk::RenderingAttachmentInfo::default()
                .image_view(a.image_view)
                .image_layout(a.image_layout)
                .load_op(a.load_op)
                .store_op(a.store_op)
                .clear_value(a.clear_value)
        });

        let mut rendering_info = vk::RenderingInfo::default()
            .render_area(self.render_area)
            .layer_count(self.layer_count)
            .view_mask(self.view_mask)
            .color_attachments(&color_infos);

        if let Some(ref depth) = depth_info {
            rendering_info = rendering_info.depth_attachment(depth);
        }
        if let Some(ref stencil) = stencil_info {
            rendering_info = rendering_info.stencil_attachment(stencil);
        }

        unsafe {
            recorder
                .device
                .cmd_begin_rendering(recorder.buffer, &rendering_info);
        }
    }
}

impl Default for DynamicRenderPassBuilder {
    fn default() -> Self {
        Self::new()
    }
}

// Additional methods on CommandRecorder for dynamic rendering.
impl CommandRecorder<'_> {
    /// End dynamic rendering started by [`DynamicRenderPassBuilder::begin`].
    ///
    /// Requires Vulkan 1.3 or `VK_KHR_dynamic_rendering`.
    pub fn end_rendering(&self) {
        unsafe {
            self.device.cmd_end_rendering(self.buffer);
        }
    }

    /// Begin a traditional render pass.
    ///
    /// For Vulkan 1.2 and earlier, or when explicit render passes are
    /// needed (e.g. for subpass dependencies, input attachments).
    pub fn begin_render_pass(
        &self,
        render_pass: vk::RenderPass,
        framebuffer: vk::Framebuffer,
        render_area: vk::Rect2D,
        clear_values: &[vk::ClearValue],
        contents: vk::SubpassContents,
    ) {
        let info = vk::RenderPassBeginInfo::default()
            .render_pass(render_pass)
            .framebuffer(framebuffer)
            .render_area(render_area)
            .clear_values(clear_values);

        unsafe {
            self.device
                .cmd_begin_render_pass(self.buffer, &info, contents);
        }
    }

    /// End the current render pass.
    pub fn end_render_pass(&self) {
        unsafe {
            self.device.cmd_end_render_pass(self.buffer);
        }
    }

    /// Advance to the next subpass.
    pub fn next_subpass(&self, contents: vk::SubpassContents) {
        unsafe {
            self.device.cmd_next_subpass(self.buffer, contents);
        }
    }

    /// Set the viewport dynamically.
    pub fn set_viewport(&self, first: u32, viewports: &[vk::Viewport]) {
        unsafe {
            self.device.cmd_set_viewport(self.buffer, first, viewports);
        }
    }

    /// Set the scissor rectangles dynamically.
    pub fn set_scissor(&self, first: u32, scissors: &[vk::Rect2D]) {
        unsafe {
            self.device.cmd_set_scissor(self.buffer, first, scissors);
        }
    }

    /// Copy data between buffers.
    pub fn copy_buffer(&self, src: vk::Buffer, dst: vk::Buffer, regions: &[vk::BufferCopy]) {
        unsafe {
            self.device.cmd_copy_buffer(self.buffer, src, dst, regions);
        }
    }

    /// Copy data from a buffer to an image.
    pub fn copy_buffer_to_image(
        &self,
        src_buffer: vk::Buffer,
        dst_image: vk::Image,
        dst_layout: vk::ImageLayout,
        regions: &[vk::BufferImageCopy],
    ) {
        unsafe {
            self.device.cmd_copy_buffer_to_image(
                self.buffer,
                src_buffer,
                dst_image,
                dst_layout,
                regions,
            );
        }
    }

    /// Copy regions between images.
    pub fn copy_image(
        &self,
        src: vk::Image,
        src_layout: vk::ImageLayout,
        dst: vk::Image,
        dst_layout: vk::ImageLayout,
        regions: &[vk::ImageCopy],
    ) {
        unsafe {
            self.device
                .cmd_copy_image(self.buffer, src, src_layout, dst, dst_layout, regions);
        }
    }

    /// Blit (scaled copy) regions between images.
    pub fn blit_image(
        &self,
        src: vk::Image,
        src_layout: vk::ImageLayout,
        dst: vk::Image,
        dst_layout: vk::ImageLayout,
        regions: &[vk::ImageBlit],
        filter: vk::Filter,
    ) {
        unsafe {
            self.device.cmd_blit_image(
                self.buffer,
                src,
                src_layout,
                dst,
                dst_layout,
                regions,
                filter,
            );
        }
    }

    /// Clear a color image to a uniform value.
    pub fn clear_color_image(
        &self,
        image: vk::Image,
        layout: vk::ImageLayout,
        color: &vk::ClearColorValue,
        ranges: &[vk::ImageSubresourceRange],
    ) {
        unsafe {
            self.device
                .cmd_clear_color_image(self.buffer, image, layout, color, ranges);
        }
    }

    /// Clear a depth/stencil image.
    pub fn clear_depth_stencil_image(
        &self,
        image: vk::Image,
        layout: vk::ImageLayout,
        value: &vk::ClearDepthStencilValue,
        ranges: &[vk::ImageSubresourceRange],
    ) {
        unsafe {
            self.device
                .cmd_clear_depth_stencil_image(self.buffer, image, layout, value, ranges);
        }
    }

    /// Fill a buffer region with a 32-bit value.
    pub fn fill_buffer(
        &self,
        buffer: vk::Buffer,
        offset: vk::DeviceSize,
        size: vk::DeviceSize,
        data: u32,
    ) {
        unsafe {
            self.device
                .cmd_fill_buffer(self.buffer, buffer, offset, size, data);
        }
    }

    /// Update a buffer region with inline data (max 65536 bytes).
    pub fn update_buffer(&self, buffer: vk::Buffer, offset: vk::DeviceSize, data: &[u8]) {
        unsafe {
            self.device
                .cmd_update_buffer(self.buffer, buffer, offset, data);
        }
    }

    /// Draw primitives with indirect parameters from a buffer.
    pub fn draw_indirect(
        &self,
        buffer: vk::Buffer,
        offset: vk::DeviceSize,
        draw_count: u32,
        stride: u32,
    ) {
        unsafe {
            self.device
                .cmd_draw_indirect(self.buffer, buffer, offset, draw_count, stride);
        }
    }

    /// Draw indexed primitives with indirect parameters from a buffer.
    pub fn draw_indexed_indirect(
        &self,
        buffer: vk::Buffer,
        offset: vk::DeviceSize,
        draw_count: u32,
        stride: u32,
    ) {
        unsafe {
            self.device
                .cmd_draw_indexed_indirect(self.buffer, buffer, offset, draw_count, stride);
        }
    }

    /// Dispatch compute with indirect parameters from a buffer.
    pub fn dispatch_indirect(&self, buffer: vk::Buffer, offset: vk::DeviceSize) {
        unsafe {
            self.device
                .cmd_dispatch_indirect(self.buffer, buffer, offset);
        }
    }

    /// Set line width dynamically.
    pub fn set_line_width(&self, width: f32) {
        unsafe {
            self.device.cmd_set_line_width(self.buffer, width);
        }
    }

    /// Set depth bias dynamically.
    pub fn set_depth_bias(&self, constant_factor: f32, clamp: f32, slope_factor: f32) {
        unsafe {
            self.device
                .cmd_set_depth_bias(self.buffer, constant_factor, clamp, slope_factor);
        }
    }

    /// Set blend constants dynamically.
    pub fn set_blend_constants(&self, constants: &[f32; 4]) {
        unsafe {
            self.device.cmd_set_blend_constants(self.buffer, constants);
        }
    }

    /// Set depth bounds dynamically.
    pub fn set_depth_bounds(&self, min: f32, max: f32) {
        unsafe {
            self.device.cmd_set_depth_bounds(self.buffer, min, max);
        }
    }

    /// Copy data from an image to a buffer.
    pub fn copy_image_to_buffer(
        &self,
        src_image: vk::Image,
        src_layout: vk::ImageLayout,
        dst_buffer: vk::Buffer,
        regions: &[vk::BufferImageCopy],
    ) {
        unsafe {
            self.device.cmd_copy_image_to_buffer(
                self.buffer,
                src_image,
                src_layout,
                dst_buffer,
                regions,
            );
        }
    }

    /// Trace rays using a ray tracing pipeline.
    ///
    /// Requires a ray tracing pipeline to be bound and the
    /// `VK_KHR_ray_tracing_pipeline` extension.
    ///
    /// # Safety
    ///
    /// The SBT regions must reference valid buffer addresses with
    /// correctly aligned shader group handles. The ray tracing pipeline
    /// function loader must be available in the shared state.
    pub unsafe fn trace_rays(
        &self,
        shared: &crate::device::SharedState,
        raygen: &vk::StridedDeviceAddressRegionKHR,
        miss: &vk::StridedDeviceAddressRegionKHR,
        hit: &vk::StridedDeviceAddressRegionKHR,
        callable: &vk::StridedDeviceAddressRegionKHR,
        width: u32,
        height: u32,
        depth: u32,
    ) {
        if let Some(rt_fn) = &shared.rt_pipeline_fn {
            rt_fn.cmd_trace_rays(
                self.buffer,
                raygen,
                miss,
                hit,
                callable,
                width,
                height,
                depth,
            );
        }
    }
}
