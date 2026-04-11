//! Command buffer state machine validator.
//!
//! [`ValidatedRecorder`] wraps a [`CommandRecorder`] and validates that
//! commands are issued in the correct recording state. Detects:
//!
//! - Draw calls outside a render pass
//! - Dispatch calls inside a render pass
//! - Nested render pass begins
//! - Ending a render pass that was never started
//! - Transfer commands inside a render pass
//! - Recording after end
//!
//! All validation is CPU-side only and has zero GPU overhead.

use ash::vk;

use crate::command::CommandRecorder;
use crate::diagnostic::{self, Severity, Style};

/// The current recording state of a command buffer.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum RecordingState {
    /// Recording has begun, no render pass active.
    Recording,
    /// Inside a traditional render pass at the given subpass index.
    InRenderPass {
        /// Current subpass index (0-based).
        subpass: u32,
    },
    /// Inside a dynamic rendering session (Vulkan 1.3).
    InDynamicRendering,
    /// Recording has ended.
    Ended,
}

impl std::fmt::Display for RecordingState {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Recording => write!(f, "Recording"),
            Self::InRenderPass { subpass } => write!(f, "InRenderPass(subpass={subpass})"),
            Self::InDynamicRendering => write!(f, "InDynamicRendering"),
            Self::Ended => write!(f, "Ended"),
        }
    }
}

/// Category of a command for error messages.
#[derive(Debug, Clone, Copy)]
pub enum CommandCategory {
    /// Draw commands (draw, `draw_indexed`).
    Draw,
    /// Compute dispatch.
    Dispatch,
    /// Transfer commands (`copy_buffer`, `copy_buffer_to_image`).
    Transfer,
    /// Begin a render pass.
    BeginRenderPass,
    /// End a render pass.
    EndRenderPass,
    /// Advance to the next subpass.
    NextSubpass,
    /// Begin dynamic rendering.
    BeginRendering,
    /// End dynamic rendering.
    EndRendering,
    /// End recording.
    End,
    /// State-agnostic commands (bind, viewport, scissor, barrier).
    Any,
}

impl std::fmt::Display for CommandCategory {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Draw => write!(f, "draw"),
            Self::Dispatch => write!(f, "dispatch"),
            Self::Transfer => write!(f, "transfer"),
            Self::BeginRenderPass => write!(f, "begin_render_pass"),
            Self::EndRenderPass => write!(f, "end_render_pass"),
            Self::NextSubpass => write!(f, "next_subpass"),
            Self::BeginRendering => write!(f, "begin_rendering"),
            Self::EndRendering => write!(f, "end_rendering"),
            Self::End => write!(f, "end"),
            Self::Any => write!(f, "any"),
        }
    }
}

/// An entry in the command history for diagnostic context.
#[derive(Debug, Clone)]
struct HistoryEntry {
    command: String,
    state_after: RecordingState,
}

/// Action on validation error.
#[derive(Default)]
pub enum StateErrorAction {
    /// Log to stderr and continue.
    Log,
    /// Panic with full diagnostic.
    #[default]
    Panic,
    /// Custom callback receiving the formatted report.
    Callback(Box<dyn Fn(&str) + Send + Sync>),
}


impl std::fmt::Debug for StateErrorAction {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Log => write!(f, "Log"),
            Self::Panic => write!(f, "Panic"),
            Self::Callback(_) => write!(f, "Callback(...)"),
        }
    }
}

/// A command recorder wrapper that validates state transitions.
///
/// Delegates all calls to the inner [`CommandRecorder`] after verifying
/// the current recording state is valid for the command.
///
/// Created via [`ValidatedRecorder::wrap`].
pub struct ValidatedRecorder<'a> {
    inner: CommandRecorder<'a>,
    state: RecordingState,
    history: Vec<HistoryEntry>,
    max_history: usize,
    on_error: StateErrorAction,
    bound_graphics_pipeline: bool,
    bound_compute_pipeline: bool,
}

impl<'a> ValidatedRecorder<'a> {
    /// Wrap a command recorder with state validation.
    pub fn wrap(recorder: CommandRecorder<'a>) -> Self {
        Self {
            inner: recorder,
            state: RecordingState::Recording,
            history: Vec::new(),
            max_history: 32,
            on_error: StateErrorAction::default(),
            bound_graphics_pipeline: false,
            bound_compute_pipeline: false,
        }
    }

    /// Set the maximum history entries kept for diagnostics.
    pub fn max_history(mut self, n: usize) -> Self {
        self.max_history = n;
        self
    }

    /// Set the action on validation error.
    pub fn on_error(mut self, action: StateErrorAction) -> Self {
        self.on_error = action;
        self
    }

    /// Get the current recording state.
    pub fn state(&self) -> &RecordingState {
        &self.state
    }

    /// Access the inner recorder for operations not covered by this wrapper.
    pub fn inner(&self) -> &CommandRecorder<'a> {
        &self.inner
    }

    /// Get the raw command buffer handle.
    pub fn raw_buffer(&self) -> vk::CommandBuffer {
        self.inner.raw_buffer()
    }

    fn record_history(&mut self, command: &str) {
        if self.history.len() >= self.max_history {
            self.history.remove(0);
        }
        self.history.push(HistoryEntry {
            command: command.to_string(),
            state_after: self.state.clone(),
        });
    }

    fn check(&self, command: &str, cat: CommandCategory) -> bool {
        let valid = match cat {
            CommandCategory::Draw => matches!(
                self.state,
                RecordingState::InRenderPass { .. } | RecordingState::InDynamicRendering
            ),
            CommandCategory::Dispatch => matches!(self.state, RecordingState::Recording),
            CommandCategory::Transfer => matches!(self.state, RecordingState::Recording),
            CommandCategory::BeginRenderPass => matches!(self.state, RecordingState::Recording),
            CommandCategory::EndRenderPass => {
                matches!(self.state, RecordingState::InRenderPass { .. })
            }
            CommandCategory::NextSubpass => {
                matches!(self.state, RecordingState::InRenderPass { .. })
            }
            CommandCategory::BeginRendering => matches!(self.state, RecordingState::Recording),
            CommandCategory::EndRendering => {
                matches!(self.state, RecordingState::InDynamicRendering)
            }
            CommandCategory::End => matches!(self.state, RecordingState::Recording),
            CommandCategory::Any => !matches!(self.state, RecordingState::Ended),
        };

        if !valid {
            let expected = match cat {
                CommandCategory::Draw => "InRenderPass or InDynamicRendering",
                CommandCategory::Dispatch => "Recording (outside render pass)",
                CommandCategory::Transfer => "Recording (outside render pass)",
                CommandCategory::BeginRenderPass => "Recording (outside render pass)",
                CommandCategory::EndRenderPass => "InRenderPass",
                CommandCategory::NextSubpass => "InRenderPass",
                CommandCategory::BeginRendering => "Recording (outside render pass)",
                CommandCategory::EndRendering => "InDynamicRendering",
                CommandCategory::End => "Recording (outside render pass)",
                CommandCategory::Any => "any state except Ended",
            };
            let report = format_state_error(command, &self.state, expected, &self.history);
            self.dispatch_error(&report);
        }

        valid
    }

    fn dispatch_error(&self, report: &str) {
        match &self.on_error {
            StateErrorAction::Log => eprint!("{report}"),
            StateErrorAction::Panic => panic!("{report}"),
            StateErrorAction::Callback(f) => f(report),
        }
    }

    // Delegated commands with validation.

    // Delegated commands with validation.
    // On validation failure the Vulkan call is SKIPPED to prevent
    // driver crashes. The diagnostic is still emitted.

    /// End recording.
    pub fn end(mut self) -> crate::Result<vk::CommandBuffer> {
        // Always attempt end even if state is wrong - not ending a
        // command buffer is worse (resource leak). But still report.
        let _ = self.check("end()", CommandCategory::End);
        self.state = RecordingState::Ended;
        self.record_history("end()");
        self.inner.end()
    }

    /// Bind a pipeline.
    pub fn bind_pipeline(&mut self, bind_point: vk::PipelineBindPoint, pipeline: vk::Pipeline) {
        if !self.check("bind_pipeline()", CommandCategory::Any) {
            return;
        }
        match bind_point {
            vk::PipelineBindPoint::GRAPHICS => self.bound_graphics_pipeline = true,
            vk::PipelineBindPoint::COMPUTE => self.bound_compute_pipeline = true,
            _ => {}
        }
        self.record_history("bind_pipeline()");
        self.inner.bind_pipeline(bind_point, pipeline);
    }

    /// Bind descriptor sets.
    pub fn bind_descriptor_sets(
        &mut self,
        bind_point: vk::PipelineBindPoint,
        layout: vk::PipelineLayout,
        first_set: u32,
        sets: &[vk::DescriptorSet],
        dynamic_offsets: &[u32],
    ) {
        if !self.check("bind_descriptor_sets()", CommandCategory::Any) {
            return;
        }
        self.record_history("bind_descriptor_sets()");
        self.inner
            .bind_descriptor_sets(bind_point, layout, first_set, sets, dynamic_offsets);
    }

    /// Bind vertex buffers.
    pub fn bind_vertex_buffers(
        &mut self,
        first_binding: u32,
        buffers: &[vk::Buffer],
        offsets: &[vk::DeviceSize],
    ) {
        if !self.check("bind_vertex_buffers()", CommandCategory::Any) {
            return;
        }
        self.record_history("bind_vertex_buffers()");
        self.inner
            .bind_vertex_buffers(first_binding, buffers, offsets);
    }

    /// Bind index buffer.
    pub fn bind_index_buffer(
        &mut self,
        buffer: vk::Buffer,
        offset: vk::DeviceSize,
        index_type: vk::IndexType,
    ) {
        if !self.check("bind_index_buffer()", CommandCategory::Any) {
            return;
        }
        self.record_history("bind_index_buffer()");
        self.inner.bind_index_buffer(buffer, offset, index_type);
    }

    /// Draw primitives.
    pub fn draw(
        &mut self,
        vertex_count: u32,
        instance_count: u32,
        first_vertex: u32,
        first_instance: u32,
    ) {
        if !self.check(
            &format!("draw({vertex_count}, {instance_count}, {first_vertex}, {first_instance})"),
            CommandCategory::Draw,
        ) {
            return;
        }
        if !self.bound_graphics_pipeline {
            let report = format_binding_error("draw", "graphics pipeline");
            self.dispatch_error(&report);
            return;
        }
        self.record_history(&format!("draw({vertex_count}, ...)"));
        self.inner
            .draw(vertex_count, instance_count, first_vertex, first_instance);
    }

    /// Draw indexed primitives.
    pub fn draw_indexed(
        &mut self,
        index_count: u32,
        instance_count: u32,
        first_index: u32,
        vertex_offset: i32,
        first_instance: u32,
    ) {
        if !self.check(
            &format!("draw_indexed({index_count}, ...)"),
            CommandCategory::Draw,
        ) {
            return;
        }
        if !self.bound_graphics_pipeline {
            let report = format_binding_error("draw_indexed", "graphics pipeline");
            self.dispatch_error(&report);
            return;
        }
        self.record_history(&format!("draw_indexed({index_count}, ...)"));
        self.inner.draw_indexed(
            index_count,
            instance_count,
            first_index,
            vertex_offset,
            first_instance,
        );
    }

    /// Dispatch compute.
    pub fn dispatch(&mut self, gx: u32, gy: u32, gz: u32) {
        if !self.check(&format!("dispatch({gx}, {gy}, {gz})"), CommandCategory::Dispatch) {
            return;
        }
        if !self.bound_compute_pipeline {
            let report = format_binding_error("dispatch", "compute pipeline");
            self.dispatch_error(&report);
            return;
        }
        self.record_history(&format!("dispatch({gx}, {gy}, {gz})"));
        self.inner.dispatch(gx, gy, gz);
    }

    /// Pipeline barrier.
    pub fn pipeline_barrier(
        &mut self,
        src: vk::PipelineStageFlags,
        dst: vk::PipelineStageFlags,
        dep: vk::DependencyFlags,
        mem: &[vk::MemoryBarrier<'_>],
        buf: &[vk::BufferMemoryBarrier<'_>],
        img: &[vk::ImageMemoryBarrier<'_>],
    ) {
        if !self.check("pipeline_barrier()", CommandCategory::Any) {
            return;
        }
        self.record_history("pipeline_barrier()");
        self.inner.pipeline_barrier(src, dst, dep, mem, buf, img);
    }

    /// Execute secondary command buffers.
    pub fn execute_commands(&mut self, secondaries: &[vk::CommandBuffer]) {
        if !self.check("execute_commands()", CommandCategory::Any) {
            return;
        }
        self.record_history(&format!("execute_commands({})", secondaries.len()));
        self.inner.execute_commands(secondaries);
    }

    /// Push constants.
    pub fn push_constants(
        &mut self,
        layout: vk::PipelineLayout,
        stages: vk::ShaderStageFlags,
        offset: u32,
        data: &[u8],
    ) {
        if !self.check("push_constants()", CommandCategory::Any) {
            return;
        }
        self.record_history("push_constants()");
        self.inner.push_constants(layout, stages, offset, data);
    }

    /// Set viewport.
    pub fn set_viewport(&mut self, first: u32, viewports: &[vk::Viewport]) {
        if !self.check("set_viewport()", CommandCategory::Any) {
            return;
        }
        self.record_history("set_viewport()");
        self.inner.set_viewport(first, viewports);
    }

    /// Set scissor.
    pub fn set_scissor(&mut self, first: u32, scissors: &[vk::Rect2D]) {
        if !self.check("set_scissor()", CommandCategory::Any) {
            return;
        }
        self.record_history("set_scissor()");
        self.inner.set_scissor(first, scissors);
    }

    /// Copy buffer to buffer.
    pub fn copy_buffer(&mut self, src: vk::Buffer, dst: vk::Buffer, regions: &[vk::BufferCopy]) {
        if !self.check("copy_buffer()", CommandCategory::Transfer) {
            return;
        }
        self.record_history("copy_buffer()");
        self.inner.copy_buffer(src, dst, regions);
    }

    /// Copy buffer to image.
    pub fn copy_buffer_to_image(
        &mut self,
        src: vk::Buffer,
        dst: vk::Image,
        layout: vk::ImageLayout,
        regions: &[vk::BufferImageCopy],
    ) {
        if !self.check("copy_buffer_to_image()", CommandCategory::Transfer) {
            return;
        }
        self.record_history("copy_buffer_to_image()");
        self.inner.copy_buffer_to_image(src, dst, layout, regions);
    }

    /// Begin a traditional render pass.
    pub fn begin_render_pass(
        &mut self,
        render_pass: vk::RenderPass,
        framebuffer: vk::Framebuffer,
        render_area: vk::Rect2D,
        clear_values: &[vk::ClearValue],
        contents: vk::SubpassContents,
    ) {
        if !self.check("begin_render_pass()", CommandCategory::BeginRenderPass) {
            return;
        }
        self.state = RecordingState::InRenderPass { subpass: 0 };
        self.record_history("begin_render_pass()");
        self.inner.begin_render_pass(
            render_pass,
            framebuffer,
            render_area,
            clear_values,
            contents,
        );
    }

    /// End the current render pass.
    pub fn end_render_pass(&mut self) {
        if !self.check("end_render_pass()", CommandCategory::EndRenderPass) {
            return;
        }
        self.state = RecordingState::Recording;
        self.record_history("end_render_pass()");
        self.inner.end_render_pass();
    }

    /// Advance to the next subpass.
    pub fn next_subpass(&mut self, contents: vk::SubpassContents) {
        if !self.check("next_subpass()", CommandCategory::NextSubpass) {
            return;
        }
        if let RecordingState::InRenderPass { ref mut subpass } = self.state {
            *subpass += 1;
        }
        self.record_history("next_subpass()");
        self.inner.next_subpass(contents);
    }

    /// End dynamic rendering.
    pub fn end_rendering(&mut self) {
        if !self.check("end_rendering()", CommandCategory::EndRendering) {
            return;
        }
        self.state = RecordingState::Recording;
        self.record_history("end_rendering()");
        self.inner.end_rendering();
    }

    /// Notify the validator that dynamic rendering has been started
    /// externally (via `DynamicRenderPassBuilder::begin`).
    pub fn notify_begin_rendering(&mut self) {
        if !self.check("begin_rendering()", CommandCategory::BeginRendering) {
            return;
        }
        self.state = RecordingState::InDynamicRendering;
        self.record_history("begin_rendering()");
    }
}

fn format_state_error(
    command: &str,
    current_state: &RecordingState,
    expected: &str,
    history: &[HistoryEntry],
) -> String {
    let s = Style::detect();
    let mut o = String::with_capacity(1024);

    diagnostic::write_header(
        &mut o,
        &s,
        &Severity::Error,
        "IGN-S001",
        "invalid command in current recording state",
    );
    diagnostic::write_location(&mut o, &s, &format!("command: {}", s.bold(command)));
    diagnostic::write_pipe_empty(&mut o, &s);

    diagnostic::write_pipe(
        &mut o,
        &s,
        &format!("current state: {}", s.bold_red(&current_state.to_string())),
    );
    diagnostic::write_pipe(
        &mut o,
        &s,
        &format!("expected:      {}", s.bold_green(expected)),
    );
    diagnostic::write_pipe_empty(&mut o, &s);

    if !history.is_empty() {
        diagnostic::write_pipe(&mut o, &s, "state trace (recent):");
        let start = history.len().saturating_sub(8);
        for (i, entry) in history[start..].iter().enumerate() {
            let marker = if i == history[start..].len() - 1 {
                s.bold_red("->")
            } else {
                s.dim("  ")
            };
            diagnostic::write_pipe(
                &mut o,
                &s,
                &format!("  {marker} {:>30} -> {}", entry.command, entry.state_after,),
            );
        }
        diagnostic::write_pipe(
            &mut o,
            &s,
            &format!(
                "  {} {:>30} -> {}",
                s.bold_red("->"),
                s.bold_red(command),
                s.bold_red("ERROR"),
            ),
        );
        diagnostic::write_pipe_empty(&mut o, &s);
    }

    let help = match command {
        c if c.starts_with("draw") => {
            "draw commands require an active render pass or dynamic rendering\n\
             call begin_render_pass() or DynamicRenderPassBuilder::begin() first"
        }
        c if c.starts_with("dispatch") => {
            "dispatch must be called outside a render pass\n\
             call end_render_pass() or end_rendering() before dispatching compute"
        }
        c if c.starts_with("copy") => {
            "transfer commands must be outside a render pass\n\
             call end_render_pass() or end_rendering() before transfers"
        }
        _ => "check the Vulkan specification for valid command sequences",
    };
    diagnostic::write_help(&mut o, &s, help);

    o
}

fn format_binding_error(command: &str, missing: &str) -> String {
    let s = Style::detect();
    let mut o = String::with_capacity(512);

    diagnostic::write_header(
        &mut o,
        &s,
        &Severity::Error,
        "IGN-S002",
        &format!("{command} called without bound {missing}"),
    );
    diagnostic::write_pipe_empty(&mut o, &s);
    diagnostic::write_pipe(
        &mut o,
        &s,
        &format!(
            "command {} requires a {} to be bound",
            s.bold_red(command),
            s.bold(missing),
        ),
    );
    diagnostic::write_pipe_empty(&mut o, &s);
    diagnostic::write_help(
        &mut o,
        &s,
        &format!(
            "call bind_pipeline({}) before {command}",
            if missing.contains("graphics") {
                "GRAPHICS, pipeline"
            } else {
                "COMPUTE, pipeline"
            }
        ),
    );

    o
}