//! Pipeline builders for graphics, compute, and ray tracing pipelines.
//!
//! Each builder collects configuration incrementally and constructs the
//! Vulkan pipeline via a final `build()` call. Shader module references
//! are stored as raw `VkShaderModule` handles; the caller is responsible
//! for keeping the modules alive until the pipeline is built.
//!
//! # Specialization Constants
//!
//! All pipeline builders support specialization constants via the
//! `.specialization()` method. This sets per-stage constant data that
//! is baked into the pipeline at creation time, enabling shader
//! permutations without recompilation.
//!
//! # Pipeline Layout
//!
//! [`PipelineLayoutBuilder`] provides an ergonomic way to construct
//! `VkPipelineLayout` objects with RAII cleanup, matching the builder
//! pattern used by render passes and pipelines.
//!
//! # Ray Tracing
//!
//! The [`RayTracingPipelineBuilder`] and [`RayTracingPipeline`] provide
//! first-class support for `VK_KHR_ray_tracing_pipeline`, including
//! shader group configuration and SBT layout computation.

use std::ffi::CString;
use std::sync::Arc;

use ash::vk;

use crate::device::SharedState;
use crate::error::{Error, Result};

/// Internal helper to align a value up to the given alignment.
#[inline]
fn align_up(value: u64, alignment: u64) -> u64 {
    (value + alignment - 1) & !(alignment - 1)
}

/// A shader stage configuration for pipeline builders.
///
/// Owns the entry point name and optional specialization data so they
/// can outlive the builder method call.
#[derive(Clone)]
pub(crate) struct ShaderStageConfig {
    pub stage: vk::ShaderStageFlags,
    pub module: vk::ShaderModule,
    pub entry_point: CString,
    pub specialization_data: Option<Vec<u8>>,
    pub specialization_map: Vec<vk::SpecializationMapEntry>,
}

/// Builder for a graphics pipeline.
///
/// Provides a subset of the most common configuration options. For full
/// control, retrieve the raw device via [`Ignis::device`](crate::Ignis::device)
/// and construct the pipeline directly.
///
/// # Required State
///
/// At minimum, you must provide:
/// - At least one shader stage (vertex + fragment typically)
/// - A pipeline layout
/// - A render pass and subpass index, OR enable dynamic rendering
pub struct GraphicsPipelineBuilder {
    shared: Arc<SharedState>,
    stages: Vec<ShaderStageConfig>,
    vertex_bindings: Vec<vk::VertexInputBindingDescription>,
    vertex_attributes: Vec<vk::VertexInputAttributeDescription>,
    topology: vk::PrimitiveTopology,
    viewport_count: u32,
    scissor_count: u32,
    polygon_mode: vk::PolygonMode,
    cull_mode: vk::CullModeFlags,
    front_face: vk::FrontFace,
    depth_test: bool,
    depth_write: bool,
    depth_compare_op: vk::CompareOp,
    color_blend_attachments: Vec<vk::PipelineColorBlendAttachmentState>,
    dynamic_states: Vec<vk::DynamicState>,
    layout: vk::PipelineLayout,
    render_pass: vk::RenderPass,
    subpass: u32,
    cache: vk::PipelineCache,
}
impl GraphicsPipelineBuilder {
    /// Create a new builder with sensible defaults.
    pub(crate) fn new(shared: Arc<SharedState>) -> Self {
        Self {
            shared,
            stages: Vec::new(),
            vertex_bindings: Vec::new(),
            vertex_attributes: Vec::new(),
            topology: vk::PrimitiveTopology::TRIANGLE_LIST,
            viewport_count: 1,
            scissor_count: 1,
            polygon_mode: vk::PolygonMode::FILL,
            cull_mode: vk::CullModeFlags::BACK,
            front_face: vk::FrontFace::COUNTER_CLOCKWISE,
            depth_test: true,
            depth_write: true,
            depth_compare_op: vk::CompareOp::LESS,
            color_blend_attachments: vec![vk::PipelineColorBlendAttachmentState::default()
                .blend_enable(false)
                .color_write_mask(vk::ColorComponentFlags::RGBA)],
            dynamic_states: vec![vk::DynamicState::VIEWPORT, vk::DynamicState::SCISSOR],
            layout: vk::PipelineLayout::null(),
            render_pass: vk::RenderPass::null(),
            subpass: 0,
            cache: vk::PipelineCache::null(),
        }
    }

    /// Add a shader stage.
    ///
    /// `entry_point` is the name of the entry function (usually "main").
    pub fn shader_stage(
        mut self,
        stage: vk::ShaderStageFlags,
        module: vk::ShaderModule,
        entry_point: &str,
    ) -> Self {
        self.stages.push(ShaderStageConfig {
            stage,
            module,
            entry_point: CString::new(entry_point).unwrap(),
            specialization_data: None,
            specialization_map: Vec::new(),
        });
        self
    }

    /// Set specialization constant data for the last added shader stage.
    ///
    /// `map_entries` describes the layout of `data`. Each entry maps
    /// a constant ID to an offset and size within `data`.
    ///
    /// # Example
    ///
    /// ```rust,no_run
    /// # use ignis::*; use ash::vk;
    /// # fn example(builder: GraphicsPipelineBuilder,
    /// #            vs: vk::ShaderModule) -> GraphicsPipelineBuilder {
    /// builder
    ///     .shader_stage(vk::ShaderStageFlags::VERTEX, vs, "main")
    ///     .specialization(
    ///         &[vk::SpecializationMapEntry {
    ///             constant_id: 0,
    ///             offset: 0,
    ///             size: 4,
    ///         }],
    ///         &42u32.to_ne_bytes(),
    ///     )
    /// # }
    /// ```
    pub fn specialization(
        mut self,
        map_entries: &[vk::SpecializationMapEntry],
        data: &[u8],
    ) -> Self {
        if let Some(last) = self.stages.last_mut() {
            last.specialization_map = map_entries.to_vec();
            last.specialization_data = Some(data.to_vec());
        }
        self
    }

    /// Add a vertex input binding.
    pub fn vertex_binding(
        mut self,
        binding: u32,
        stride: u32,
        input_rate: vk::VertexInputRate,
    ) -> Self {
        self.vertex_bindings
            .push(vk::VertexInputBindingDescription {
                binding,
                stride,
                input_rate,
            });
        self
    }

    /// Add a vertex input attribute.
    pub fn vertex_attribute(
        mut self,
        location: u32,
        binding: u32,
        format: vk::Format,
        offset: u32,
    ) -> Self {
        self.vertex_attributes
            .push(vk::VertexInputAttributeDescription {
                location,
                binding,
                format,
                offset,
            });
        self
    }

    /// Set the primitive topology.
    pub fn topology(mut self, topology: vk::PrimitiveTopology) -> Self {
        self.topology = topology;
        self
    }

    /// Set the polygon mode (fill, line, point).
    pub fn polygon_mode(mut self, mode: vk::PolygonMode) -> Self {
        self.polygon_mode = mode;
        self
    }

    /// Set the cull mode.
    pub fn cull_mode(mut self, mode: vk::CullModeFlags) -> Self {
        self.cull_mode = mode;
        self
    }

    /// Set the front face winding order.
    pub fn front_face(mut self, face: vk::FrontFace) -> Self {
        self.front_face = face;
        self
    }

    /// Enable or disable depth testing.
    pub fn depth_test(mut self, enable: bool) -> Self {
        self.depth_test = enable;
        self
    }

    /// Enable or disable depth writing.
    pub fn depth_write(mut self, enable: bool) -> Self {
        self.depth_write = enable;
        self
    }

    /// Set the depth comparison operator.
    pub fn depth_compare_op(mut self, op: vk::CompareOp) -> Self {
        self.depth_compare_op = op;
        self
    }

    /// Set additional dynamic states.
    pub fn dynamic_states(mut self, states: &[vk::DynamicState]) -> Self {
        self.dynamic_states = states.to_vec();
        self
    }

    /// Set the pipeline layout (required).
    pub fn layout(mut self, layout: vk::PipelineLayout) -> Self {
        self.layout = layout;
        self
    }

    /// Set the render pass and subpass index (required unless using dynamic rendering).
    pub fn render_pass(mut self, render_pass: vk::RenderPass, subpass: u32) -> Self {
        self.render_pass = render_pass;
        self.subpass = subpass;
        self
    }

    /// Set the pipeline cache for faster creation on subsequent runs.
    pub fn cache(mut self, cache: vk::PipelineCache) -> Self {
        self.cache = cache;
        self
    }

    /// Build the graphics pipeline.
    ///
    /// # Errors
    ///
    /// Returns [`Error::InvalidConfig`] if the layout is null or no shader
    /// stages are present. Returns a Vulkan error if pipeline creation fails.
    pub fn build(self) -> Result<vk::Pipeline> {
        if self.layout == vk::PipelineLayout::null() {
            return Err(Error::InvalidConfig("pipeline layout is required"));
        }
        if self.stages.is_empty() {
            return Err(Error::InvalidConfig(
                "at least one shader stage is required",
            ));
        }

        // Build specialization infos. These must live until create_graphics_pipelines returns.
        let spec_infos: Vec<Option<vk::SpecializationInfo<'_>>> = self
            .stages
            .iter()
            .map(|s| {
                s.specialization_data.as_ref().map(|data| {
                    vk::SpecializationInfo::default()
                        .map_entries(&s.specialization_map)
                        .data(data)
                })
            })
            .collect();

        // Build shader stage create infos (referencing owned CStrings and spec infos).
        let stage_infos: Vec<vk::PipelineShaderStageCreateInfo<'_>> = self
            .stages
            .iter()
            .enumerate()
            .map(|(i, s)| {
                let mut info = vk::PipelineShaderStageCreateInfo::default()
                    .stage(s.stage)
                    .module(s.module)
                    .name(&s.entry_point);
                if let Some(ref spec) = spec_infos[i] {
                    info = info.specialization_info(spec);
                }
                info
            })
            .collect();

        let vertex_input = vk::PipelineVertexInputStateCreateInfo::default()
            .vertex_binding_descriptions(&self.vertex_bindings)
            .vertex_attribute_descriptions(&self.vertex_attributes);

        let input_assembly = vk::PipelineInputAssemblyStateCreateInfo::default()
            .topology(self.topology)
            .primitive_restart_enable(false);

        let viewport_state = vk::PipelineViewportStateCreateInfo::default()
            .viewport_count(self.viewport_count)
            .scissor_count(self.scissor_count);

        let rasterization = vk::PipelineRasterizationStateCreateInfo::default()
            .depth_clamp_enable(false)
            .rasterizer_discard_enable(false)
            .polygon_mode(self.polygon_mode)
            .cull_mode(self.cull_mode)
            .front_face(self.front_face)
            .depth_bias_enable(false)
            .line_width(1.0);

        let multisampling = vk::PipelineMultisampleStateCreateInfo::default()
            .rasterization_samples(vk::SampleCountFlags::TYPE_1)
            .sample_shading_enable(false);

        let depth_stencil = vk::PipelineDepthStencilStateCreateInfo::default()
            .depth_test_enable(self.depth_test)
            .depth_write_enable(self.depth_write)
            .depth_compare_op(self.depth_compare_op)
            .depth_bounds_test_enable(false)
            .stencil_test_enable(false);

        let color_blend = vk::PipelineColorBlendStateCreateInfo::default()
            .logic_op_enable(false)
            .attachments(&self.color_blend_attachments);

        let dynamic_state =
            vk::PipelineDynamicStateCreateInfo::default().dynamic_states(&self.dynamic_states);

        let create_info = vk::GraphicsPipelineCreateInfo::default()
            .stages(&stage_infos)
            .vertex_input_state(&vertex_input)
            .input_assembly_state(&input_assembly)
            .viewport_state(&viewport_state)
            .rasterization_state(&rasterization)
            .multisample_state(&multisampling)
            .depth_stencil_state(&depth_stencil)
            .color_blend_state(&color_blend)
            .dynamic_state(&dynamic_state)
            .layout(self.layout)
            .render_pass(self.render_pass)
            .subpass(self.subpass);

        // SAFETY: all referenced data is valid and lives on the stack.
        let pipelines = unsafe {
            self.shared.device.create_graphics_pipelines(
                self.cache,
                std::slice::from_ref(&create_info),
                None,
            )
        }
        .map_err(|(_, e)| Error::Vulkan(e))?;

        Ok(pipelines[0])
    }
}

/// Builder for a compute pipeline.
pub struct ComputePipelineBuilder {
    shared: Arc<SharedState>,
    stage: Option<ShaderStageConfig>,
    layout: vk::PipelineLayout,
    cache: vk::PipelineCache,
}

impl ComputePipelineBuilder {
    pub(crate) fn new(shared: Arc<SharedState>) -> Self {
        Self {
            shared,
            stage: None,
            layout: vk::PipelineLayout::null(),
            cache: vk::PipelineCache::null(),
        }
    }

    /// Set the compute shader.
    pub fn shader(mut self, module: vk::ShaderModule, entry_point: &str) -> Self {
        self.stage = Some(ShaderStageConfig {
            stage: vk::ShaderStageFlags::COMPUTE,
            module,
            entry_point: CString::new(entry_point).unwrap(),
            specialization_data: None,
            specialization_map: Vec::new(),
        });
        self
    }

    /// Set specialization constants for the compute shader.
    ///
    /// `map_entries` describes the layout of `data`. Each entry maps
    /// a constant ID to an offset and size within `data`.
    ///
    /// Must be called after [`shader`](Self::shader).
    ///
    /// # Example
    ///
    /// ```rust,no_run
    /// # use ignis::*; use ash::vk;
    /// # fn example(builder: ComputePipelineBuilder,
    /// #            cs: vk::ShaderModule) -> ComputePipelineBuilder {
    /// builder
    ///     .shader(cs, "main")
    ///     .specialization(
    ///         &[
    ///             vk::SpecializationMapEntry { constant_id: 0, offset: 0, size: 4 },
    ///             vk::SpecializationMapEntry { constant_id: 1, offset: 4, size: 4 },
    ///         ],
    ///         &{
    ///             let mut buf = [0u8; 8];
    ///             buf[0..4].copy_from_slice(&64u32.to_ne_bytes()); // local_size_x
    ///             buf[4..8].copy_from_slice(&1u32.to_ne_bytes());  // enable_feature
    ///             buf
    ///         },
    ///     )
    /// # }
    /// ```
    pub fn specialization(
        mut self,
        map_entries: &[vk::SpecializationMapEntry],
        data: &[u8],
    ) -> Self {
        if let Some(ref mut stage) = self.stage {
            stage.specialization_map = map_entries.to_vec();
            stage.specialization_data = Some(data.to_vec());
        }
        self
    }

    /// Set the pipeline layout.
    pub fn layout(mut self, layout: vk::PipelineLayout) -> Self {
        self.layout = layout;
        self
    }

    /// Set the pipeline cache.
    pub fn cache(mut self, cache: vk::PipelineCache) -> Self {
        self.cache = cache;
        self
    }

    /// Build the compute pipeline.
    ///
    /// # Errors
    ///
    /// Returns [`Error::InvalidConfig`] if the shader or layout is missing.
    pub fn build(self) -> Result<vk::Pipeline> {
        let stage_cfg = self
            .stage
            .as_ref()
            .ok_or(Error::InvalidConfig("compute shader is required"))?;

        if self.layout == vk::PipelineLayout::null() {
            return Err(Error::InvalidConfig("pipeline layout is required"));
        }

        // Build specialization info (must live until create_compute_pipelines returns).
        let spec_info = stage_cfg.specialization_data.as_ref().map(|data| {
            vk::SpecializationInfo::default()
                .map_entries(&stage_cfg.specialization_map)
                .data(data)
        });

        let mut stage_info = vk::PipelineShaderStageCreateInfo::default()
            .stage(stage_cfg.stage)
            .module(stage_cfg.module)
            .name(&stage_cfg.entry_point);
        if let Some(ref spec) = spec_info {
            stage_info = stage_info.specialization_info(spec);
        }

        let create_info = vk::ComputePipelineCreateInfo::default()
            .stage(stage_info)
            .layout(self.layout);

        let pipelines = unsafe {
            self.shared.device.create_compute_pipelines(
                self.cache,
                std::slice::from_ref(&create_info),
                None,
            )
        }
        .map_err(|(_, e)| Error::Vulkan(e))?;

        Ok(pipelines[0])
    }
}

/// Shader group configuration for a ray tracing pipeline.
#[derive(Debug, Clone)]
pub enum ShaderGroup {
    /// A general shader group (raygen, miss, or callable).
    /// The `shader_index` references a stage in the stages array.
    General {
        /// Index of the shader stage.
        shader_index: u32,
    },
    /// A triangle hit group with optional any-hit shader.
    TrianglesHit {
        /// Index of the closest-hit shader stage.
        closest_hit: u32,
        /// Optional index of the any-hit shader stage.
        any_hit: Option<u32>,
    },
    /// A procedural hit group with an intersection shader.
    ProceduralHit {
        /// Index of the closest-hit shader stage.
        closest_hit: u32,
        /// Optional index of the any-hit shader stage.
        any_hit: Option<u32>,
        /// Index of the intersection shader stage.
        intersection: u32,
    },
}

/// Builder for a ray tracing pipeline.
///
/// Requires the `VK_KHR_ray_tracing_pipeline` extension to be enabled.
///
/// # Example
///
/// ```rust,no_run
/// # use ignis::*; use ash::vk;
/// # fn example(ignis: &Ignis, raygen: vk::ShaderModule,
/// #            miss: vk::ShaderModule, hit: vk::ShaderModule,
/// #            layout: vk::PipelineLayout) -> Result<()> {
/// let pipeline = ignis.raytracing_pipeline_builder()?
///     .stage(vk::ShaderStageFlags::RAYGEN_KHR, raygen, "main")
///     .stage(vk::ShaderStageFlags::MISS_KHR, miss, "main")
///     .stage(vk::ShaderStageFlags::CLOSEST_HIT_KHR, hit, "main")
///     .group(ShaderGroup::General { shader_index: 0 })    // raygen
///     .group(ShaderGroup::General { shader_index: 1 })    // miss
///     .group(ShaderGroup::TrianglesHit {                   // hit
///         closest_hit: 2,
///         any_hit: None,
///     })
///     .max_recursion_depth(2)
///     .layout(layout)
///     .build()?;
/// # Ok(())
/// # }
/// ```
pub struct RayTracingPipelineBuilder {
    shared: Arc<SharedState>,
    stages: Vec<ShaderStageConfig>,
    groups: Vec<ShaderGroup>,
    max_recursion_depth: u32,
    layout: vk::PipelineLayout,
    cache: vk::PipelineCache,
}

impl RayTracingPipelineBuilder {
    pub(crate) fn new(shared: Arc<SharedState>) -> Self {
        Self {
            shared,
            stages: Vec::new(),
            groups: Vec::new(),
            max_recursion_depth: 1,
            layout: vk::PipelineLayout::null(),
            cache: vk::PipelineCache::null(),
        }
    }

    /// Add a shader stage.
    pub fn stage(
        mut self,
        stage_flags: vk::ShaderStageFlags,
        module: vk::ShaderModule,
        entry_point: &str,
    ) -> Self {
        self.stages.push(ShaderStageConfig {
            stage: stage_flags,
            module,
            entry_point: CString::new(entry_point).unwrap(),
            specialization_data: None,
            specialization_map: Vec::new(),
        });
        self
    }

    /// Set specialization constants for the last added shader stage.
    ///
    /// Must be called immediately after [`stage`](Self::stage).
    pub fn specialization(
        mut self,
        map_entries: &[vk::SpecializationMapEntry],
        data: &[u8],
    ) -> Self {
        if let Some(last) = self.stages.last_mut() {
            last.specialization_map = map_entries.to_vec();
            last.specialization_data = Some(data.to_vec());
        }
        self
    }

    /// Add a shader group.
    pub fn group(mut self, group: ShaderGroup) -> Self {
        self.groups.push(group);
        self
    }

    /// Set the maximum ray recursion depth.
    pub fn max_recursion_depth(mut self, depth: u32) -> Self {
        self.max_recursion_depth = depth;
        self
    }

    /// Set the pipeline layout.
    pub fn layout(mut self, layout: vk::PipelineLayout) -> Self {
        self.layout = layout;
        self
    }

    /// Set the pipeline cache.
    pub fn cache(mut self, cache: vk::PipelineCache) -> Self {
        self.cache = cache;
        self
    }

    /// Build the ray tracing pipeline.
    ///
    /// # Errors
    ///
    /// Returns [`Error::FeatureNotEnabled`] if the RT extension is not loaded,
    /// [`Error::InvalidConfig`] if required fields are missing, or a Vulkan
    /// error if pipeline creation fails.
    pub fn build(self) -> Result<RayTracingPipeline> {
        let rt_fn = self
            .shared
            .rt_pipeline_fn
            .as_ref()
            .ok_or(Error::FeatureNotEnabled("VK_KHR_ray_tracing_pipeline"))?;

        if self.layout == vk::PipelineLayout::null() {
            return Err(Error::InvalidConfig("pipeline layout is required"));
        }
        if self.stages.is_empty() {
            return Err(Error::InvalidConfig(
                "at least one shader stage is required",
            ));
        }
        if self.groups.is_empty() {
            return Err(Error::InvalidConfig(
                "at least one shader group is required",
            ));
        }

        // Build specialization infos.
        let spec_infos: Vec<Option<vk::SpecializationInfo<'_>>> = self
            .stages
            .iter()
            .map(|s| {
                s.specialization_data.as_ref().map(|data| {
                    vk::SpecializationInfo::default()
                        .map_entries(&s.specialization_map)
                        .data(data)
                })
            })
            .collect();

        // Build stage infos.
        let stage_infos: Vec<vk::PipelineShaderStageCreateInfo<'_>> = self
            .stages
            .iter()
            .enumerate()
            .map(|(i, s)| {
                let mut info = vk::PipelineShaderStageCreateInfo::default()
                    .stage(s.stage)
                    .module(s.module)
                    .name(&s.entry_point);
                if let Some(ref spec) = spec_infos[i] {
                    info = info.specialization_info(spec);
                }
                info
            })
            .collect();

        // Build group infos.
        let group_infos: Vec<vk::RayTracingShaderGroupCreateInfoKHR<'_>> = self
            .groups
            .iter()
            .map(|g| match g {
                ShaderGroup::General { shader_index } => {
                    vk::RayTracingShaderGroupCreateInfoKHR::default()
                        .ty(vk::RayTracingShaderGroupTypeKHR::GENERAL)
                        .general_shader(*shader_index)
                        .closest_hit_shader(vk::SHADER_UNUSED_KHR)
                        .any_hit_shader(vk::SHADER_UNUSED_KHR)
                        .intersection_shader(vk::SHADER_UNUSED_KHR)
                }
                ShaderGroup::TrianglesHit {
                    closest_hit,
                    any_hit,
                } => vk::RayTracingShaderGroupCreateInfoKHR::default()
                    .ty(vk::RayTracingShaderGroupTypeKHR::TRIANGLES_HIT_GROUP)
                    .general_shader(vk::SHADER_UNUSED_KHR)
                    .closest_hit_shader(*closest_hit)
                    .any_hit_shader(any_hit.unwrap_or(vk::SHADER_UNUSED_KHR))
                    .intersection_shader(vk::SHADER_UNUSED_KHR),
                ShaderGroup::ProceduralHit {
                    closest_hit,
                    any_hit,
                    intersection,
                } => vk::RayTracingShaderGroupCreateInfoKHR::default()
                    .ty(vk::RayTracingShaderGroupTypeKHR::PROCEDURAL_HIT_GROUP)
                    .general_shader(vk::SHADER_UNUSED_KHR)
                    .closest_hit_shader(*closest_hit)
                    .any_hit_shader(any_hit.unwrap_or(vk::SHADER_UNUSED_KHR))
                    .intersection_shader(*intersection),
            })
            .collect();

        let create_info = vk::RayTracingPipelineCreateInfoKHR::default()
            .stages(&stage_infos)
            .groups(&group_infos)
            .max_pipeline_ray_recursion_depth(self.max_recursion_depth)
            .layout(self.layout);

        // SAFETY: all referenced data is valid and on the stack.
        let pipelines = unsafe {
            rt_fn.create_ray_tracing_pipelines(
                vk::DeferredOperationKHR::null(),
                self.cache,
                std::slice::from_ref(&create_info),
                None,
            )
        }
        .map_err(|(_, e)| Error::Vulkan(e))?;

        Ok(RayTracingPipeline {
            shared: Arc::clone(&self.shared),
            handle: pipelines[0],
            group_count: self.groups.len() as u32,
        })
    }
}

/// A compiled ray tracing pipeline with SBT layout computation.
pub struct RayTracingPipeline {
    shared: Arc<SharedState>,
    handle: vk::Pipeline,
    group_count: u32,
}

impl RayTracingPipeline {
    /// Get the raw pipeline handle.
    #[inline]
    pub fn handle(&self) -> vk::Pipeline {
        self.handle
    }

    /// Number of shader groups in this pipeline.
    #[inline]
    pub fn group_count(&self) -> u32 {
        self.group_count
    }

    /// Compute the shader binding table layout.
    ///
    /// Returns the raw shader group handle data and alignment information
    /// needed to create SBT buffers. The caller is responsible for
    /// allocating GPU buffers and copying the handle data with proper
    /// alignment.
    ///
    /// # Arguments
    ///
    /// * `raygen_count` - Number of raygen groups (typically 1)
    /// * `miss_count` - Number of miss groups
    /// * `hit_count` - Number of hit groups
    /// * `callable_count` - Number of callable groups
    ///
    /// The sum of all counts must equal [`group_count`](Self::group_count).
    ///
    /// # Errors
    ///
    /// Returns [`Error::FeatureNotEnabled`] if RT properties are unavailable,
    /// or a Vulkan error if handle retrieval fails.
    pub fn sbt_layout(
        &self,
        raygen_count: u32,
        miss_count: u32,
        hit_count: u32,
        callable_count: u32,
    ) -> Result<ShaderBindingTableLayout> {
        let rt_fn = self
            .shared
            .rt_pipeline_fn
            .as_ref()
            .ok_or(Error::FeatureNotEnabled("VK_KHR_ray_tracing_pipeline"))?;
        let rt_props = self
            .shared
            .rt_properties
            .as_ref()
            .ok_or(Error::FeatureNotEnabled("ray tracing properties"))?;

        let handle_size = u64::from(rt_props.shader_group_handle_size);
        let handle_alignment = u64::from(rt_props.shader_group_handle_alignment);
        let base_alignment = u64::from(rt_props.shader_group_base_alignment);

        let handle_size_aligned = align_up(handle_size, handle_alignment);

        // Retrieve all shader group handles.
        let data_size = (self.group_count as usize) * (handle_size as usize);
        // SAFETY: pipeline is valid, group_count matches.
        let handle_data = unsafe {
            rt_fn.get_ray_tracing_shader_group_handles(
                self.handle,
                0,
                self.group_count,
                data_size,
            )?
        };

        // Compute region sizes (aligned to base_alignment).
        let raygen_size = align_up(
            u64::from(raygen_count) * handle_size_aligned,
            base_alignment,
        );
        let miss_size = align_up(u64::from(miss_count) * handle_size_aligned, base_alignment);
        let hit_size = align_up(u64::from(hit_count) * handle_size_aligned, base_alignment);
        let callable_size = align_up(
            u64::from(callable_count) * handle_size_aligned,
            base_alignment,
        );

        let total_size = raygen_size + miss_size + hit_size + callable_size;

        Ok(ShaderBindingTableLayout {
            handle_data,
            handle_size: handle_size as u32,
            handle_size_aligned: handle_size_aligned as u32,
            base_alignment: base_alignment as u32,
            raygen_offset: 0,
            raygen_size,
            raygen_stride: handle_size_aligned,
            miss_offset: raygen_size,
            miss_size,
            miss_stride: handle_size_aligned,
            hit_offset: raygen_size + miss_size,
            hit_size,
            hit_stride: handle_size_aligned,
            callable_offset: raygen_size + miss_size + hit_size,
            callable_size,
            callable_stride: handle_size_aligned,
            total_size,
        })
    }
}

impl Drop for RayTracingPipeline {
    fn drop(&mut self) {
        unsafe {
            self.shared.device.destroy_pipeline(self.handle, None);
        }
    }
}

/// Computed shader binding table layout and raw handle data.
///
/// Use this information to allocate a GPU buffer of [`total_size`] bytes,
/// copy handle data at the specified offsets with proper alignment, and
/// construct `VkStridedDeviceAddressRegionKHR` for `vkCmdTraceRaysKHR`.
///
/// # SBT Buffer Layout
///
/// ```text
/// | raygen region | miss region | hit region | callable region |
/// |<- base align ->|<- base a. ->|<- base a.->|<-  base a.   ->|
/// ```
///
/// Within each region, handles are stored at `handle_size_aligned` stride.
pub struct ShaderBindingTableLayout {
    /// Raw shader group handle data from the driver.
    pub handle_data: Vec<u8>,
    /// Size of a single handle (unaligned).
    pub handle_size: u32,
    /// Size of a single handle aligned to `shader_group_handle_alignment`.
    pub handle_size_aligned: u32,
    /// Base alignment for each SBT region.
    pub base_alignment: u32,
    /// Byte offset of the raygen region within the SBT buffer.
    pub raygen_offset: u64,
    /// Size of the raygen region in bytes.
    pub raygen_size: u64,
    /// Stride between raygen entries.
    pub raygen_stride: u64,
    /// Byte offset of the miss region.
    pub miss_offset: u64,
    /// Size of the miss region in bytes.
    pub miss_size: u64,
    /// Stride between miss entries.
    pub miss_stride: u64,
    /// Byte offset of the hit region.
    pub hit_offset: u64,
    /// Size of the hit region in bytes.
    pub hit_size: u64,
    /// Stride between hit entries.
    pub hit_stride: u64,
    /// Byte offset of the callable region.
    pub callable_offset: u64,
    /// Size of the callable region in bytes.
    pub callable_size: u64,
    /// Stride between callable entries.
    pub callable_stride: u64,
    /// Total size of the SBT buffer in bytes.
    pub total_size: u64,
}

impl ShaderBindingTableLayout {
    /// Build a `VkStridedDeviceAddressRegionKHR` for the raygen region.
    ///
    /// `base_address` is the device address of the SBT buffer.
    pub fn raygen_region(
        &self,
        base_address: vk::DeviceAddress,
    ) -> vk::StridedDeviceAddressRegionKHR {
        vk::StridedDeviceAddressRegionKHR {
            device_address: base_address + self.raygen_offset,
            stride: self.raygen_stride,
            size: self.raygen_size,
        }
    }

    /// Build a `VkStridedDeviceAddressRegionKHR` for the miss region.
    pub fn miss_region(
        &self,
        base_address: vk::DeviceAddress,
    ) -> vk::StridedDeviceAddressRegionKHR {
        vk::StridedDeviceAddressRegionKHR {
            device_address: base_address + self.miss_offset,
            stride: self.miss_stride,
            size: self.miss_size,
        }
    }

    /// Build a `VkStridedDeviceAddressRegionKHR` for the hit region.
    pub fn hit_region(&self, base_address: vk::DeviceAddress) -> vk::StridedDeviceAddressRegionKHR {
        vk::StridedDeviceAddressRegionKHR {
            device_address: base_address + self.hit_offset,
            stride: self.hit_stride,
            size: self.hit_size,
        }
    }

    /// Build a `VkStridedDeviceAddressRegionKHR` for the callable region.
    pub fn callable_region(
        &self,
        base_address: vk::DeviceAddress,
    ) -> vk::StridedDeviceAddressRegionKHR {
        vk::StridedDeviceAddressRegionKHR {
            device_address: base_address + self.callable_offset,
            stride: self.callable_stride,
            size: self.callable_size,
        }
    }
}

/// Builder for a `VkPipelineLayout`.
///
/// RAII wrapper that creates and owns the layout. Eliminates raw
/// `vkCreatePipelineLayout` calls.
///
/// # Example
///
/// ```rust,no_run
/// # use ignis::*; use ash::vk;
/// # fn example(ignis: &Ignis, set_layout: vk::DescriptorSetLayout) -> Result<()> {
/// let layout = ignis.pipeline_layout_builder()
///     .descriptor_set_layout(set_layout)
///     .push_constant_range(vk::ShaderStageFlags::VERTEX, 0, 64)
///     .build()?;
/// // Use layout.handle() in pipeline builders.
/// # Ok(())
/// # }
/// ```
pub struct PipelineLayoutBuilder {
    shared: Arc<SharedState>,
    set_layouts: Vec<vk::DescriptorSetLayout>,
    push_constant_ranges: Vec<vk::PushConstantRange>,
}

impl PipelineLayoutBuilder {
    pub(crate) fn new(shared: Arc<SharedState>) -> Self {
        Self {
            shared,
            set_layouts: Vec::new(),
            push_constant_ranges: Vec::new(),
        }
    }

    /// Add a descriptor set layout.
    pub fn descriptor_set_layout(mut self, layout: vk::DescriptorSetLayout) -> Self {
        self.set_layouts.push(layout);
        self
    }

    /// Add multiple descriptor set layouts at once.
    pub fn descriptor_set_layouts(mut self, layouts: &[vk::DescriptorSetLayout]) -> Self {
        self.set_layouts.extend_from_slice(layouts);
        self
    }

    /// Add a push constant range.
    pub fn push_constant_range(
        mut self,
        stage_flags: vk::ShaderStageFlags,
        offset: u32,
        size: u32,
    ) -> Self {
        self.push_constant_ranges.push(vk::PushConstantRange {
            stage_flags,
            offset,
            size,
        });
        self
    }

    /// Build the pipeline layout.
    ///
    /// # Errors
    ///
    /// Returns a Vulkan error if layout creation fails.
    pub fn build(self) -> Result<PipelineLayoutHandle> {
        let ci = vk::PipelineLayoutCreateInfo::default()
            .set_layouts(&self.set_layouts)
            .push_constant_ranges(&self.push_constant_ranges);
        let handle = unsafe { self.shared.device.create_pipeline_layout(&ci, None)? };
        Ok(PipelineLayoutHandle {
            shared: self.shared,
            handle,
        })
    }
}

/// An owned `VkPipelineLayout` with automatic cleanup on drop.
///
/// Created via [`PipelineLayoutBuilder::build`].
pub struct PipelineLayoutHandle {
    shared: Arc<SharedState>,
    handle: vk::PipelineLayout,
}

impl PipelineLayoutHandle {
    /// Get the raw pipeline layout handle.
    #[inline]
    pub fn handle(&self) -> vk::PipelineLayout {
        self.handle
    }
}

impl Drop for PipelineLayoutHandle {
    fn drop(&mut self) {
        unsafe {
            self.shared
                .device
                .destroy_pipeline_layout(self.handle, None);
        }
    }
}
