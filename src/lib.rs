#![warn(missing_docs)]

//! # Ignis - Asynchronous Vulkan Queue Orchestration
//!
//! A lightweight orchestration layer on top of [`ash`] providing:
//!
//! - **Async GPU futures**: submit work, poll or `.await` completion
//! - **Per-frame synchronization**: managed fences and semaphores for N frames in flight
//! - **Multi-threaded command recording**: parallel secondary buffer recording via
//!   `std::thread::scope`
//! - **Ray tracing pipeline**: first-class `VK_KHR_ray_tracing_pipeline` support
//!   with SBT layout computation
//! - **Interoperability**: works alongside `wgpu`, `vulkano`, `egui`, or bare `ash`
//!   through the [`DeviceHandle`] trait and external device mode
//!
//! # Device Modes
//!
//! Ignis supports two modes of device ownership:
//!
//! - **Managed** ([`Ignis::managed`]): ignis creates and owns the Vulkan instance,
//!   device, and queues. Suitable for standalone applications.
//! - **External** ([`Ignis::external`]): ignis wraps handles provided by the caller.
//!   The caller retains ownership and must ensure the handles outlive ignis.
//!   Suitable for integration with existing engines.
//!
//! # Features
//!
//! | Feature | What it adds |
//! |---|---|
//! | `tracking` | `ResourceTracker` (per-subresource barriers), `DeletionQueue` |
//! | `descriptors` | Descriptor set/pool builders, `DescriptorArena`, `DescriptorRing` |
//! | `slab-allocator` | Production hardened `SlabAllocator` |
//! | `swapchain` | Swapchain and surface management |
//! | `interop` | `QueueBroker`, `InteropSync` for cross-engine sharing |
//! | `debug-tools` | 12 validation/diagnostic modules |
//! | `full` | All of the above |
//!
//! # Example
//!
//! ```rust,no_run
//! use ignis::{Ignis, ManagedConfig, QueueType};
//!
//! let config = ManagedConfig::new("MyApp", ash::vk::API_VERSION_1_3);
//! let ignis = Ignis::managed(config).expect("failed to initialize ignis");
//!
//! let gfx = ignis.queue(QueueType::Graphics).unwrap();
//! // ... record commands, submit via gfx.submit(), await GpuFuture ...
//! ```

// Std imports.
use std::sync::{Arc, OnceLock};
use std::time::Duration;

// External crate imports.
use ash::vk;

// Core modules (always compiled).
pub mod command;
pub mod device;
pub mod diagnostic;
pub mod error;
pub mod queue;
pub mod shader;
pub mod sync;
pub mod format;

// Grouped modules (always compiled, internally feature-gated).
pub mod memory;
pub mod pipeline;
pub mod tracking;

// Feature-gated top-level modules.
#[cfg(feature = "swapchain")]
pub mod surface;

#[cfg(feature = "interop")]
pub mod interop;

#[cfg(feature = "debug-tools")]
pub mod debug;

// Internal shared state (not re-exported).
use device::SharedState;

// Core re-exports: error handling.
pub use error::{Error, Result};

// Core re-exports: device and configuration.
pub use device::{
    ExternalDeviceInfo, ManagedConfig, PhysicalDeviceInfo, QueueAllocation, RayTracingProperties,
};

// Core re-exports: queues and async submission.
pub use queue::{AsyncQueue, GpuFuture, SubmitBuilder};
pub use sync::{FrameContext, FrameSync};
pub use tracking::watcher::FenceWatcher;
pub use tracking::timeline::{QueueTimeline, TimelineWatcher};

// Core re-exports: command recording.
pub use command::{
    ColorAttachmentInfo, CommandBufferInheritance, CommandPool, CommandRecorder,
    DepthStencilAttachmentInfo, DynamicRenderPassBuilder, ParallelRecorder,
};

// Core re-exports: memory and allocation.
pub use memory::allocator::{Allocation, Allocator, BlockAllocator, DedicatedAllocator};
pub use memory::resources::{Buffer, BufferInfo, Image, ImageInfo, MemoryLocation};

// Core re-exports: format utilities.
pub use format::{
    dispatch_size, dispatch_size_3d, format_aspect_mask, format_byte_size,
    format_block_extent, is_compressed_format, is_depth_format, is_stencil_format,
    mip_levels_for_size,
};

// Core re-exports: staging, frame alloc, typed buffer, readback.
pub use memory::staging::{StagingRegion, StagingRing};
pub use memory::frame_alloc::FrameAllocator;
pub use memory::typed::TypedBuffer;
pub use memory::readback::ReadbackRequest;

// Core re-exports: pipeline cache and layout.
pub use pipeline::cache::PipelineCache;
pub use pipeline::builders::{PipelineLayoutBuilder, PipelineLayoutHandle};

// Core re-exports: fence pool.
pub use sync::FencePool;

// Core re-exports: error context.
pub use error::WithContext;

// Core re-exports: shaders, pipelines, render passes.
pub use pipeline::builders::{
    ComputePipelineBuilder, GraphicsPipelineBuilder, RayTracingPipeline, RayTracingPipelineBuilder,
    ShaderBindingTableLayout, ShaderGroup,
};
pub use pipeline::renderpass::{
    AttachmentConfig, AttachmentRef, RenderPassBuilder, RenderPassHandle, SubpassConfig,
    SubpassDependency,
};
pub use shader::ShaderModule;

// Feature: tracking.
#[cfg(feature = "tracking")]
pub use tracking::tracker::{
    BufferState, BufferTransition, BufferUsageContext, ImageTransition,
    ImageUsageContext, ResourceTracker, SubresourceState,
};
#[cfg(feature = "tracking")]
pub use tracking::deletion::{DeletionGuard, DeletionQueue};

// Feature: slab-allocator.
#[cfg(feature = "slab-allocator")]
pub use memory::slab::{SlabAllocator, SlabConfig, SlabErrorAction, SlabStats, SizeClassStats};

// Feature: descriptors.
#[cfg(feature = "descriptors")]
pub use pipeline::descriptor::{
    DescriptorArena, DescriptorPoolBuilder, DescriptorRing,
    DescriptorSetLayoutBuilder, DescriptorWriter,
};

// Feature: swapchain.
#[cfg(feature = "swapchain")]
pub use surface::swapchain::{Swapchain, SwapchainConfig, SwapchainSupport};

// Feature: interop.
#[cfg(feature = "interop")]
pub use interop::{InteropSync, QueueBroker, QueueGuard};

// Feature: debug-tools - hardened allocator.
#[cfg(feature = "debug-tools")]
pub use debug::hardened::{
    CorruptionAction, CorruptionEvent, FreePattern, GuardRegion, HardenedAllocator, HardenedConfig,
    HardenedStats,
};

// Feature: debug-tools - validation tools.
#[cfg(feature = "debug-tools")]
pub use debug::aliasing::{AliasingDetector, AliasingIssue};
#[cfg(feature = "debug-tools")]
pub use debug::barrier_opt::{BarrierAnalyzer, BarrierSuggestion, SuggestionKind};
#[cfg(feature = "debug-tools")]
pub use debug::budget::{BudgetMonitor, BudgetSnapshot, BudgetThresholds, HeapStatus};
#[cfg(feature = "debug-tools")]
pub use debug::cmd_state::{RecordingState, StateErrorAction, ValidatedRecorder};
#[cfg(feature = "debug-tools")]
pub use debug::descriptor_audit::{BoundResource, DescriptorAuditor, DescriptorIssue};
#[cfg(feature = "debug-tools")]
pub use debug::hang_detector::{BreadcrumbBuffer, HangAction, HangConfig, HangDetector};
#[cfg(feature = "debug-tools")]
pub use debug::journal::{EntryStatus, SubmissionJournal};
#[cfg(feature = "debug-tools")]
pub use debug::lifetime::{LeakAction, LifetimeTracker};
#[cfg(feature = "debug-tools")]
pub use debug::pipeline_audit::{PipelineAuditor, PipelineIssue};
#[cfg(feature = "debug-tools")]
pub use debug::thread_audit::{AuditedPool, ThreadViolationAction};
#[cfg(feature = "debug-tools")]
pub use debug::debug_utils::DebugUtils;
#[cfg(feature = "debug-tools")]
pub use debug::profiler::{GpuProfiler, ScopeHandle, ScopeResult};

/// The type of GPU queue being requested.
///
/// Maps to Vulkan queue capability flags. When requesting a queue from
/// [`Ignis::queue`], ignis finds the best matching queue family.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum QueueType {
    /// Queue supporting graphics operations (`VK_QUEUE_GRAPHICS_BIT`).
    /// Graphics queues also implicitly support transfer and compute.
    Graphics,
    /// Queue supporting compute operations (`VK_QUEUE_COMPUTE_BIT`).
    /// Prefers a dedicated compute queue family when available.
    Compute,
    /// Queue supporting transfer operations (`VK_QUEUE_TRANSFER_BIT`).
    /// Prefers a dedicated transfer queue family when available.
    Transfer,
}

impl QueueType {
    /// Returns the Vulkan queue flags required for this queue type.
    #[inline]
    pub fn required_flags(self) -> vk::QueueFlags {
        match self {
            Self::Graphics => vk::QueueFlags::GRAPHICS,
            Self::Compute => vk::QueueFlags::COMPUTE,
            Self::Transfer => vk::QueueFlags::TRANSFER,
        }
    }
}

/// Trait for types that provide access to Vulkan device handles.
///
/// Implement this trait for your engine's device abstraction to enable
/// interoperability with ignis utilities without constructing a full
/// [`Ignis`] context.
///
/// # Example
///
/// ```rust,no_run
/// use ignis::DeviceHandle;
///
/// struct MyEngine {
///     instance: ash::Instance,
///     device: ash::Device,
///     physical: ash::vk::PhysicalDevice,
///     families: Vec<ash::vk::QueueFamilyProperties>,
/// }
///
/// impl DeviceHandle for MyEngine {
///     fn ash_instance(&self) -> &ash::Instance { &self.instance }
///     fn ash_device(&self) -> &ash::Device { &self.device }
///     fn physical_device(&self) -> ash::vk::PhysicalDevice { self.physical }
///     fn queue_family_properties(&self) -> &[ash::vk::QueueFamilyProperties] {
///         &self.families
///     }
///  }
/// ```
pub trait DeviceHandle: Send + Sync {
    /// Returns a reference to the ash instance.
    fn ash_instance(&self) -> &ash::Instance;
    /// Returns a reference to the ash logical device.
    fn ash_device(&self) -> &ash::Device;
    /// Returns the physical device handle.
    fn physical_device(&self) -> vk::PhysicalDevice;
    /// Returns queue family properties for the physical device.
    fn queue_family_properties(&self) -> &[vk::QueueFamilyProperties];
}

/// The main ignis orchestration context.
///
/// Holds shared Vulkan state and provides factory methods for queues,
/// synchronization primitives, command pools, pipeline builders,
/// allocators, debug tooling, and more.
///
/// Created via [`Ignis::managed`] (ignis owns the device) or
/// [`Ignis::external`] (wraps an existing device).
///
/// All child objects returned by factory methods hold an `Arc` reference
/// to the shared state, keeping the device alive as long as any child exists.
/// In external mode, the caller must ensure the original device outlives
/// all ignis objects.
pub struct Ignis {
    shared: Arc<SharedState>,
    queues: Vec<Arc<AsyncQueue>>,
    default_allocator: OnceLock<Arc<dyn Allocator>>,
}

// SAFETY: Ignis contains only Arc-wrapped state, Vec of Arc, and OnceLock.
// SharedState is Send + Sync (verified by static assert in device.rs).
// AsyncQueue wraps vk::Queue in a Mutex, making it Send + Sync.
unsafe impl Send for Ignis {}
unsafe impl Sync for Ignis {}

impl Ignis {
    /// Create an ignis context that manages its own Vulkan instance and device.
    ///
    /// This is the standalone mode. Ignis will:
    /// 1. Load the Vulkan library
    /// 2. Create a `VkInstance` with requested extensions
    /// 3. Select a physical device (via the configured selector or default heuristic)
    /// 4. Create a logical device with appropriate queue families
    /// 5. Optionally load ray tracing extension functions
    /// 6. Create per-queue timeline semaphores (Vulkan 1.2+)
    ///
    /// The instance and device are destroyed when the last reference to the
    /// shared state is dropped.
    ///
    /// # Errors
    ///
    /// Returns [`Error::LoadFailed`] if the Vulkan library cannot be loaded,
    /// [`Error::NoSuitableDevice`] if no physical device matches requirements,
    /// or a [`Error::Vulkan`] variant for Vulkan API errors.
    pub fn managed(config: ManagedConfig) -> Result<Self> {
        let (shared, allocations) = device::create_managed_device(config)?;
        let shared = Arc::new(shared);
        diagnostic::init_diagnostic_context(&shared);
        let queues = Self::build_queues(&shared, allocations);
        Ok(Self {
            shared,
            queues,
            default_allocator: OnceLock::new(),
        })
    }

    /// Create an ignis context wrapping externally-owned Vulkan handles.
    ///
    /// Use this when integrating with `wgpu`, `vulkano`, or another engine
    /// that already owns the Vulkan device. The caller provides raw handles
    /// and queue allocations.
    ///
    /// # Safety Contract
    ///
    /// The caller must ensure:
    /// - All provided handles are valid
    /// - The `ash::Instance` and `ash::Device` remain alive for the lifetime
    ///   of this `Ignis` and all objects created from it
    /// - Queue handles correspond to actual device queues
    /// - If ray tracing is used, the device was created with the required
    ///   extensions and features enabled
    ///
    /// # Errors
    ///
    /// Returns [`Error::InvalidConfig`] if no queue allocations are provided.
    pub fn external(info: ExternalDeviceInfo) -> Result<Self> {
        let (shared, allocations) = device::create_external_device(info)?;
        let shared = Arc::new(shared);
        diagnostic::init_diagnostic_context(&shared);
        let queues = Self::build_queues(&shared, allocations);
        Ok(Self {
            shared,
            queues,
            default_allocator: OnceLock::new(),
        })
    }

    /// Build queue wrappers with optional timeline semaphores.
    fn build_queues(
        shared: &Arc<SharedState>,
        allocations: Vec<device::QueueAllocation>,
    ) -> Vec<Arc<AsyncQueue>> {
        allocations
            .into_iter()
            .map(|alloc| {
                let timeline = if shared.supports_timelines {
                    QueueTimeline::new(Arc::clone(shared)).ok().map(Arc::new)
                } else {
                    None
                };
                Arc::new(AsyncQueue::new(
                    Arc::clone(shared),
                    alloc.handle,
                    alloc.family_index,
                    alloc.queue_index,
                    alloc.capabilities,
                    timeline,
                ))
            })
            .collect()
    }

    /// Find a queue matching the requested type.
    ///
    /// For [`QueueType::Compute`] and [`QueueType::Transfer`], prefers
    /// dedicated queue families (without graphics capability) when available,
    /// falling back to shared families.
    ///
    /// # Errors
    ///
    /// Returns [`Error::NoSuitableQueueFamily`] if no queue with the required
    /// capability exists.
    pub fn queue(&self, queue_type: QueueType) -> Result<&Arc<AsyncQueue>> {
        let required = queue_type.required_flags();

        // Prefer a dedicated queue (one that has the required flag but NOT graphics,
        // unless graphics is what we want).
        let dedicated = self.queues.iter().find(|q| {
            let caps = q.capabilities();
            caps.contains(required)
                && (queue_type == QueueType::Graphics
                    || !caps.contains(vk::QueueFlags::GRAPHICS))
        });

        dedicated
            .or_else(|| {
                self.queues
                    .iter()
                    .find(|q| q.capabilities().contains(required))
            })
            .ok_or(Error::NoSuitableQueueFamily(queue_type))
    }

    /// Returns all available queues.
    pub fn all_queues(&self) -> &[Arc<AsyncQueue>] {
        &self.queues
    }

    /// Create a per-frame synchronization manager.
    ///
    /// Allocates `frames_in_flight` fences (initially signaled) and pairs
    /// of semaphores for image acquisition and render completion.
    ///
    /// # Errors
    ///
    /// Returns a Vulkan error if fence or semaphore creation fails.
    pub fn create_frame_sync(&self, frames_in_flight: u32) -> Result<FrameSync> {
        FrameSync::new(Arc::clone(&self.shared), frames_in_flight)
    }

    /// Create a command pool for the given queue type.
    ///
    /// The pool is created with `RESET_COMMAND_BUFFER` flag, allowing
    /// individual command buffer resets.
    ///
    /// # Errors
    ///
    /// Returns [`Error::NoSuitableQueueFamily`] if no matching family exists,
    /// or a Vulkan error if pool creation fails.
    pub fn create_command_pool(&self, queue_type: QueueType) -> Result<CommandPool> {
        let queue = self.queue(queue_type)?;
        CommandPool::new(Arc::clone(&self.shared), queue.family_index())
    }

    /// Create a parallel command recorder with one command pool per thread.
    ///
    /// Each of the `thread_count` pools belongs to the same queue family.
    /// Use [`ParallelRecorder::record`] to record secondary command buffers
    /// in parallel via `std::thread::scope`.
    ///
    /// # Errors
    ///
    /// Returns [`Error::NoSuitableQueueFamily`] if no matching family exists,
    /// or a Vulkan error if pool creation fails.
    pub fn create_parallel_recorder(
        &self,
        queue_type: QueueType,
        thread_count: u32,
    ) -> Result<ParallelRecorder> {
        let queue = self.queue(queue_type)?;
        ParallelRecorder::new(Arc::clone(&self.shared), queue.family_index(), thread_count)
    }

    /// Begin building a graphics pipeline.
    pub fn graphics_pipeline_builder(&self) -> GraphicsPipelineBuilder {
        GraphicsPipelineBuilder::new(Arc::clone(&self.shared))
    }

    /// Begin building a compute pipeline.
    pub fn compute_pipeline_builder(&self) -> ComputePipelineBuilder {
        ComputePipelineBuilder::new(Arc::clone(&self.shared))
    }

    /// Begin building a ray tracing pipeline.
    ///
    /// # Errors
    ///
    /// Returns [`Error::FeatureNotEnabled`] if the device was not created
    /// with ray tracing extensions.
    pub fn raytracing_pipeline_builder(&self) -> Result<RayTracingPipelineBuilder> {
        if self.shared.rt_pipeline_fn.is_none() {
            return Err(Error::FeatureNotEnabled("VK_KHR_ray_tracing_pipeline"));
        }
        Ok(RayTracingPipelineBuilder::new(Arc::clone(&self.shared)))
    }

    /// Begin building a render pass.
    pub fn render_pass_builder(&self) -> RenderPassBuilder {
        RenderPassBuilder::new(Arc::clone(&self.shared))
    }

    /// Create a shader module from SPIR-V bytecode.
    ///
    /// # Arguments
    ///
    /// * `spirv` - SPIR-V data as a slice of `u32` words. Must begin with the
    ///   SPIR-V magic number (`0x07230203`).
    ///
    /// # Errors
    ///
    /// Returns [`Error::InvalidSpirv`] if the data is empty or has wrong magic,
    /// or a Vulkan error if module creation fails.
    pub fn create_shader_module(&self, spirv: &[u32]) -> Result<ShaderModule> {
        ShaderModule::new(Arc::clone(&self.shared), spirv)
    }

    /// Access the underlying ash logical device.
    #[inline]
    pub fn device(&self) -> &ash::Device {
        &self.shared.device
    }

    /// Access the underlying ash instance.
    #[inline]
    pub fn instance(&self) -> &ash::Instance {
        &self.shared.instance
    }

    /// Get the physical device handle.
    #[inline]
    pub fn physical_device(&self) -> vk::PhysicalDevice {
        self.shared.physical_device
    }

    /// Get device properties.
    #[inline]
    pub fn device_properties(&self) -> &vk::PhysicalDeviceProperties {
        &self.shared.device_properties
    }

    /// Get device memory properties.
    #[inline]
    pub fn memory_properties(&self) -> &vk::PhysicalDeviceMemoryProperties {
        &self.shared.memory_properties
    }

    /// Get a clone of the internal shared state for advanced interop.
    ///
    /// This allows creating ignis objects (command pools, pipelines, etc.)
    /// outside of the [`Ignis`] struct's factory methods.
    #[inline]
    pub fn shared_state(&self) -> &Arc<SharedState> {
        &self.shared
    }

    /// Whether ray tracing extensions were loaded.
    #[inline]
    pub fn supports_ray_tracing(&self) -> bool {
        self.shared.rt_pipeline_fn.is_some()
    }

    /// Whether timeline semaphores are available (Vulkan 1.2+).
    #[inline]
    pub fn supports_timelines(&self) -> bool {
        self.shared.supports_timelines
    }

    /// Ray tracing pipeline properties, if available.
    #[inline]
    pub fn ray_tracing_properties(&self) -> Option<&RayTracingProperties> {
        self.shared.rt_properties.as_ref()
    }

    /// Ray tracing pipeline extension function loader, if available.
    #[inline]
    pub fn ray_tracing_pipeline_fn(
        &self,
    ) -> Option<&ash::khr::ray_tracing_pipeline::Device> {
        self.shared.rt_pipeline_fn.as_ref()
    }

    /// Acceleration structure extension function loader, if available.
    #[inline]
    pub fn acceleration_structure_fn(
        &self,
    ) -> Option<&ash::khr::acceleration_structure::Device> {
        self.shared.accel_struct_fn.as_ref()
    }

    // Allocator factory methods.

    /// Returns (or lazily creates) the default block allocator.
    ///
    /// Shared across all convenience `create_buffer` / `create_image`
    /// calls. The allocator is created on first use and reused afterwards,
    /// ensuring all convenience-created resources share memory blocks.
    fn default_allocator(&self) -> &Arc<dyn Allocator> {
        self.default_allocator.get_or_init(|| {
            Arc::new(BlockAllocator::new(Arc::clone(&self.shared)))
        })
    }

    /// Create the default block allocator (256 MiB blocks).
    ///
    /// This is the recommended allocator for most workloads. Each Vulkan
    /// memory type gets its own lock, so threads working with different
    /// memory types never contend.
    ///
    /// Returns an `Arc` suitable for passing to
    /// [`create_buffer_with`](Self::create_buffer_with) and
    /// [`create_image_with`](Self::create_image_with).
    pub fn create_block_allocator(&self) -> Arc<dyn Allocator> {
        Arc::new(BlockAllocator::new(Arc::clone(&self.shared)))
    }

    /// Create a block allocator with a custom block size.
    ///
    /// Larger blocks reduce the number of `vkAllocateMemory` calls but
    /// increase peak memory usage. 64-256 MiB is typical.
    pub fn create_block_allocator_with_size(
        &self,
        block_size: vk::DeviceSize,
    ) -> Arc<dyn Allocator> {
        Arc::new(BlockAllocator::with_block_size(
            Arc::clone(&self.shared),
            block_size,
        ))
    }

    /// Create a dedicated allocator (one `VkDeviceMemory` per resource).
    ///
    /// Only suitable for a small number of large resources. Most drivers
    /// limit total allocations to ~4096.
    pub fn create_dedicated_allocator(&self) -> Arc<dyn Allocator> {
        Arc::new(DedicatedAllocator::new(Arc::clone(&self.shared)))
    }

    /// Create a hardened allocator wrapping the default block allocator.
    ///
    /// For development/testing builds. Adds guard bands, canary
    /// verification, quarantine, and optional junk/zero fills.
    ///
    /// # Example
    ///
    /// ```rust,no_run
    /// # use ignis::*; use ash::vk;
    /// # fn example(ignis: &Ignis) -> Result<()> {
    /// let alloc = ignis.create_hardened_allocator(HardenedConfig::default());
    /// let buf = ignis.create_buffer_with(&alloc, &BufferInfo::staging(1024))?;
    /// # Ok(())
    /// # }
    /// ```
    #[cfg(feature = "debug-tools")]
    pub fn create_hardened_allocator(
        &self,
        config: HardenedConfig,
    ) -> Arc<dyn Allocator> {
        let inner = self.create_block_allocator();
        Arc::new(HardenedAllocator::new(
            Arc::clone(&self.shared),
            inner,
            config,
        ))
    }

    /// Create a hardened allocator wrapping a specific inner allocator.
    ///
    /// Composable: wrap any allocator (`BlockAllocator`, `DedicatedAllocator`,
    /// or a foreign gpu-allocator/vk-mem bridge).
    #[cfg(feature = "debug-tools")]
    pub fn create_hardened_allocator_with(
        &self,
        inner: Arc<dyn Allocator>,
        config: HardenedConfig,
    ) -> Arc<dyn Allocator> {
        Arc::new(HardenedAllocator::new(
            Arc::clone(&self.shared),
            inner,
            config,
        ))
    }

    // Buffer and Image creation.

    /// Create a buffer with the default shared block allocator.
    ///
    /// Convenience method that uses a lazily-created shared allocator.
    /// All buffers created through this method share memory blocks,
    /// minimizing `VkDeviceMemory` count.
    ///
    /// For bulk creation with explicit allocator control, prefer
    /// [`create_buffer_with`](Self::create_buffer_with).
    ///
    /// # Example
    ///
    /// ```rust,no_run
    /// # use ignis::*; use ash::vk;
    /// # fn example(ignis: &Ignis) -> Result<()> {
    /// let staging = ignis.create_buffer(&BufferInfo::staging(4096))?;
    /// staging.write(0, &data_bytes);
    /// # Ok(())
    /// # }
    /// ```
    pub fn create_buffer(&self, info: &BufferInfo) -> Result<Buffer> {
        let allocator = self.default_allocator();
        Buffer::new(Arc::clone(&self.shared), Arc::clone(allocator), info)
    }

    /// Create a buffer using a specific allocator.
    ///
    /// Preferred when creating many resources. Share a single allocator
    /// across all calls to minimize `VkDeviceMemory` allocations.
    ///
    /// # Example
    ///
    /// ```rust,no_run
    /// # use ignis::*; use ash::vk;
    /// # fn example(ignis: &Ignis) -> Result<()> {
    /// let alloc = ignis.create_block_allocator();
    /// let vbo = ignis.create_buffer_with(&alloc, &BufferInfo::vertex(1024, MemoryLocation::GpuOnly))?;
    /// let ibo = ignis.create_buffer_with(&alloc, &BufferInfo::index(512, MemoryLocation::GpuOnly))?;
    /// # Ok(())
    /// # }
    /// ```
    pub fn create_buffer_with(
        &self,
        allocator: &Arc<dyn Allocator>,
        info: &BufferInfo,
    ) -> Result<Buffer> {
        Buffer::new(
            Arc::clone(&self.shared),
            Arc::clone(allocator),
            info,
        )
    }

    /// Create an image with the default shared block allocator.
    ///
    /// Convenience method. For bulk creation prefer
    /// [`create_image_with`](Self::create_image_with).
    pub fn create_image(&self, info: &ImageInfo) -> Result<Image> {
        let allocator = self.default_allocator();
        Image::new(Arc::clone(&self.shared), Arc::clone(allocator), info)
    }

    /// Create an image using a specific allocator.
    pub fn create_image_with(
        &self,
        allocator: &Arc<dyn Allocator>,
        info: &ImageInfo,
    ) -> Result<Image> {
        Image::new(
            Arc::clone(&self.shared),
            Arc::clone(allocator),
            info,
        )
    }

    // Swapchain.

    /// Create a swapchain for the given surface.
    ///
    /// The surface must have been created externally (via winit, SDL, etc.)
    /// for the same Vulkan instance. Required extensions:
    /// `VK_KHR_surface` (instance) and `VK_KHR_swapchain` (device).
    ///
    /// # Surface Ownership
    ///
    /// The caller retains ownership of the surface and must destroy it
    /// after the swapchain is dropped.
    #[cfg(feature = "swapchain")]
    pub fn create_swapchain(
        &self,
        surface: vk::SurfaceKHR,
        config: &SwapchainConfig,
        width: u32,
        height: u32,
    ) -> Result<Swapchain> {
        Swapchain::new(Arc::clone(&self.shared), surface, config, width, height)
    }

    /// Query swapchain support for a surface.
    #[cfg(feature = "swapchain")]
    pub fn query_swapchain_support(
        &self,
        surface: vk::SurfaceKHR,
    ) -> Result<SwapchainSupport> {
        Swapchain::query_support(&self.shared, surface)
    }

    // Synchronization and async primitives.

    /// Create a background fence watcher for async polling (Vulkan 1.1 fallback).
    ///
    /// Spawns a dedicated thread that periodically checks fence status and
    /// wakes async tasks. The `poll_interval` controls check frequency.
    ///
    /// For Vulkan 1.2+ devices, prefer [`create_timeline_watcher`](Self::create_timeline_watcher)
    /// which uses `vkWaitSemaphores` for O(1) kernel-side blocking instead
    /// of O(N) fence polling.
    ///
    /// A reasonable default is `Duration::from_micros(200)`.
    pub fn create_fence_watcher(&self, poll_interval: Duration) -> Arc<FenceWatcher> {
        Arc::new(FenceWatcher::new(Arc::clone(&self.shared), poll_interval))
    }

    /// Create a timeline watcher for efficient async completion (Vulkan 1.2+).
    ///
    /// Uses `vkWaitSemaphores(ANY)` which blocks in the kernel at O(1) cost
    /// regardless of pending future count. One watcher services all queues.
    ///
    /// Wake processing is O(queues + `completed_futures`) per wake-up, compared
    /// to `O(total_pending)` for the fence watcher.
    ///
    /// Falls back gracefully if timelines are not available (check
    /// [`supports_timelines`](Self::supports_timelines) first).
    pub fn create_timeline_watcher(&self) -> Arc<TimelineWatcher> {
        Arc::new(TimelineWatcher::new(Arc::clone(&self.shared)))
    }

    // Resource tracker.

    /// Create a new empty resource tracker.
    ///
    /// The tracker supports per-subresource image layout tracking (individual
    /// mip levels and array layers) and buffer barrier tracking. It uses
    /// explicit [`ImageUsageContext`] / [`BufferUsageContext`] enums instead
    /// of guessing pipeline stages from layouts, eliminating the
    /// `SHADER_READ_ONLY` -> `FRAGMENT_SHADER` misattribution bug.
    #[cfg(feature = "tracking")]
    pub fn create_resource_tracker(&self) -> ResourceTracker {
        ResourceTracker::new()
    }

    // Deletion queue.

    /// Create a timeline-based deletion queue.
    ///
    /// Resources are tagged with a (`timeline_semaphore`, value) guard
    /// when retired. They are destroyed only after `poll()` confirms the
    /// GPU has moved past that point. No concept of "frame" - works
    /// correctly with multiple windows, async compute, and independent
    /// transfer queues.
    ///
    /// Call [`DeletionQueue::poll`] periodically (e.g., once per frame)
    /// to process completed entries.
    ///
    /// # Example
    ///
    /// ```rust,no_run
    /// # use ignis::*; use ash::vk;
    /// # fn example(ignis: &Ignis, queue: &AsyncQueue) -> Result<()> {
    /// let dq = ignis.create_deletion_queue();
    /// let timeline = queue.timeline().unwrap();
    ///
    /// let buf = ignis.create_buffer(&BufferInfo::staging(1024))?;
    /// let submit_value = timeline.claim_next_value();
    /// // ... submit work using buf, signaling submit_value ...
    ///
    /// let (handle, alloc, allocation) = buf.into_raw();
    /// dq.retire_buffer_after(
    ///     handle,
    ///     Some((alloc, allocation)),
    ///     DeletionGuard::Timeline {
    ///         timeline: std::sync::Arc::clone(timeline),
    ///         value: submit_value,
    ///     },
    /// );
    ///
    /// // Later, once per frame:
    /// dq.poll();
    /// # Ok(())
    /// # }
    /// ```
    #[cfg(feature = "tracking")]
    pub fn create_deletion_queue(&self) -> Arc<DeletionQueue> {
        Arc::new(DeletionQueue::new(Arc::clone(&self.shared)))
    }

    // Descriptor management.

    /// Begin building a descriptor set layout.
    ///
    /// # Example
    ///
    /// ```rust,no_run
    /// # use ignis::*; use ash::vk;
    /// # fn example(ignis: &Ignis) -> Result<()> {
    /// let layout = ignis.descriptor_set_layout_builder()
    ///     .binding(0, vk::DescriptorType::UNIFORM_BUFFER, 1, vk::ShaderStageFlags::VERTEX)
    ///     .binding(1, vk::DescriptorType::COMBINED_IMAGE_SAMPLER, 1, vk::ShaderStageFlags::FRAGMENT)
    ///     .build()?;
    /// # Ok(())
    /// # }
    /// ```
    #[cfg(feature = "descriptors")]
    pub fn descriptor_set_layout_builder(&self) -> DescriptorSetLayoutBuilder {
        DescriptorSetLayoutBuilder::new(Arc::clone(&self.shared))
    }

    /// Begin building a descriptor pool.
    ///
    /// For most use cases, prefer [`create_descriptor_arena`](Self::create_descriptor_arena)
    /// which auto-grows on exhaustion, or [`create_descriptor_ring`](Self::create_descriptor_ring)
    /// for per-frame transient descriptors.
    #[cfg(feature = "descriptors")]
    pub fn descriptor_pool_builder(&self) -> DescriptorPoolBuilder {
        DescriptorPoolBuilder::new(Arc::clone(&self.shared))
    }

    /// Create an auto-growing descriptor pool (arena).
    ///
    /// When allocation fails due to pool exhaustion, automatically creates
    /// a new pool and retries. Eliminates the "forgot to make the pool
    /// big enough" class of bugs.
    ///
    /// Not thread-safe. Use one arena per recording thread, or wrap in a Mutex.
    ///
    /// # Arguments
    ///
    /// * `max_sets_per_pool` - Maximum descriptor sets per internal pool
    /// * `type_counts` - Descriptor counts per type per pool
    ///
    /// # Example
    ///
    /// ```rust,no_run
    /// # use ignis::*; use ash::vk;
    /// # fn example(ignis: &Ignis) -> Result<()> {
    /// let mut arena = ignis.create_descriptor_arena(256, &[
    ///     (vk::DescriptorType::UNIFORM_BUFFER, 256),
    ///     (vk::DescriptorType::COMBINED_IMAGE_SAMPLER, 512),
    /// ])?;
    ///
    /// let set = arena.allocate(layout)?; // auto-grows if full
    /// # Ok(())
    /// # }
    /// ```
    #[cfg(feature = "descriptors")]
    pub fn create_descriptor_arena(
        &self,
        max_sets_per_pool: u32,
        type_counts: &[(vk::DescriptorType, u32)],
    ) -> Result<DescriptorArena> {
        DescriptorArena::new(
            Arc::clone(&self.shared),
            max_sets_per_pool,
            type_counts,
        )
    }

    /// Create a per-frame descriptor set ring buffer.
    ///
    /// Maintains one [`DescriptorArena`] per frame in flight. At each frame
    /// boundary, the oldest arena is reset, recycling its descriptor sets.
    /// This is the standard pattern for transient per-frame descriptors
    /// (uniforms, material parameters, etc.).
    ///
    /// Sets that must survive across frames should NOT be allocated from the
    /// ring. Use a separate [`DescriptorArena`] for those.
    ///
    /// # Arguments
    ///
    /// * `frames_in_flight` - Number of arenas (typically 2 or 3)
    /// * `max_sets_per_pool` - Maximum sets per internal pool within each arena
    /// * `type_counts` - Descriptor counts per type per pool
    ///
    /// # Example
    ///
    /// ```rust,no_run
    /// # use ignis::*; use ash::vk;
    /// # fn example(ignis: &Ignis, layout: vk::DescriptorSetLayout) -> Result<()> {
    /// let mut ring = ignis.create_descriptor_ring(2, 256, &[
    ///     (vk::DescriptorType::UNIFORM_BUFFER, 256),
    /// ])?;
    ///
    /// // Each frame:
    /// ring.advance()?; // resets the arena from 2 frames ago
    /// let set = ring.allocate(layout)?;
    /// // ... write and use the set, recycled automatically later ...
    /// # Ok(())
    /// # }
    /// ```
    #[cfg(feature = "descriptors")]
    pub fn create_descriptor_ring(
        &self,
        frames_in_flight: u32,
        max_sets_per_pool: u32,
        type_counts: &[(vk::DescriptorType, u32)],
    ) -> Result<DescriptorRing> {
        DescriptorRing::new(
            Arc::clone(&self.shared),
            frames_in_flight,
            max_sets_per_pool,
            type_counts,
        )
    }

    // Interop primitives.

    /// Create a queue broker for safe inter-engine queue sharing.
    ///
    /// When ignis and another engine (wgpu, vulkano) must share the same
    /// `VkQueue`, the broker provides a mutex-guarded access pattern.
    /// Both sides acquire through the broker before submitting.
    ///
    /// If the device supports multiple queues in the same family, prefer
    /// giving each engine its own queue. Use the broker only when a single
    /// queue must be shared.
    ///
    /// # Errors
    ///
    /// Returns [`Error::NoSuitableQueueFamily`] if no queue matches.
    #[cfg(feature = "interop")]
    pub fn create_queue_broker(
        &self,
        queue_type: QueueType,
    ) -> Result<QueueBroker> {
        let q = self.queue(queue_type)?;
        let handle = unsafe {
            self.device()
                .get_device_queue(q.family_index(), q.queue_index())
        };
        Ok(QueueBroker::new(handle, q.family_index(), q.queue_index()))
    }

    /// Create an interop sync pair for cross-engine synchronization.
    ///
    /// Returns two semaphores: `a_done` (signaled by engine A, waited by B)
    /// and `b_done` (signaled by B, waited by A). This is the standard
    /// handoff pattern for sharing resources between engines.
    ///
    /// # Errors
    ///
    /// Returns a Vulkan error if semaphore creation fails.
    #[cfg(feature = "interop")]
    pub fn create_interop_sync(&self) -> Result<InteropSync> {
        InteropSync::new(Arc::clone(&self.shared))
    }

    // Debug and diagnostic tooling.

    /// Create a GPU hang detector with breadcrumb support.
    ///
    /// Spawns a watchdog thread that monitors submitted fences. If any
    /// fence fails to signal within the configured timeout, a rich
    /// diagnostic is produced showing the breadcrumb trail of completed
    /// operations (if a [`BreadcrumbBuffer`] was attached).
    ///
    /// # Example
    ///
    /// ```rust,no_run
    /// # use ignis::*; use std::time::Duration;
    /// # fn example(ignis: &Ignis) {
    /// let detector = ignis.create_hang_detector(
    ///     HangConfig {
    ///         timeout: Duration::from_secs(5),
    ///         check_interval: Duration::from_millis(100),
    ///     },
    ///     HangAction::Log,
    /// );
    /// # }
    /// ```
    #[cfg(feature = "debug-tools")]
    pub fn create_hang_detector(&self, config: HangConfig, action: HangAction) -> HangDetector {
        HangDetector::new(Arc::clone(&self.shared), config, action)
    }

    /// Create a breadcrumb buffer for GPU operation tracking.
    ///
    /// Allocates a small CPU-visible GPU buffer. Insert breadcrumbs
    /// into command buffers via [`BreadcrumbBuffer::insert`]. After a
    /// hang, [`BreadcrumbBuffer::readback`] reveals the last completed
    /// operation.
    ///
    /// # Errors
    ///
    /// Returns [`Error::NoSuitableMemoryType`] if no host-visible memory
    /// type exists, or a Vulkan error if buffer creation fails.
    #[cfg(feature = "debug-tools")]
    pub fn create_breadcrumb_buffer(&self) -> Result<BreadcrumbBuffer> {
        BreadcrumbBuffer::new(Arc::clone(&self.shared))
    }

    /// Create a memory budget monitor.
    ///
    /// Polls `VK_EXT_memory_budget` (when available) to track per-heap
    /// consumption against driver budgets. Emits warnings at configurable
    /// thresholds (default: 80%, 90%, 95%).
    ///
    /// If the extension is not available, falls back to using heap sizes
    /// from `VkPhysicalDeviceMemoryProperties` as the budget.
    #[cfg(feature = "debug-tools")]
    pub fn create_budget_monitor(&self, thresholds: BudgetThresholds) -> BudgetMonitor {
        BudgetMonitor::new(Arc::clone(&self.shared), thresholds)
    }

    /// Create a submission flight recorder (journal).
    ///
    /// Maintains a lock-free ring buffer of queue submissions (timestamps,
    /// queue identity, command buffers, semaphores, fences). On device lost
    /// or any failure, provides a chronological record of what was in flight.
    ///
    /// # Arguments
    ///
    /// * `capacity` - Maximum journal entries. Old entries are evicted when full.
    ///   256 is a reasonable default for most applications.
    #[cfg(feature = "debug-tools")]
    pub fn create_journal(&self, capacity: usize) -> SubmissionJournal {
        SubmissionJournal::new(capacity)
    }

    /// Create an object lifetime tracker.
    ///
    /// Registers Vulkan objects with `#[track_caller]` creation sites.
    /// At shutdown (or on demand), produces a leak report showing every
    /// live object with its creation location, age, and usage count.
    ///
    /// # Example
    ///
    /// ```rust,no_run
    /// # use ignis::*; use ash::vk;
    /// # fn example(ignis: &Ignis) {
    /// let tracker = ignis.create_lifetime_tracker();
    /// tracker.register(vk::ObjectType::PIPELINE, 0x42, Some("my_pipeline"));
    /// // ... later, check for leaks:
    /// if let Some(report) = tracker.report_leaks() {
    ///     eprintln!("{report}");
    /// }
    /// # }
    /// ```
    #[cfg(feature = "debug-tools")]
    pub fn create_lifetime_tracker(&self) -> LifetimeTracker {
        LifetimeTracker::new()
    }

    /// Create a production-grade hardened slab allocator.
    ///
    /// Structural hardening with near-zero overhead: size-class slabs,
    /// bitmap-based double-free detection, randomized slot placement,
    /// right-alignment for overflow detection, quarantine for UAF mitigation.
    ///
    /// Suitable for shipping builds. For debug builds with full guard bands
    /// and canary verification, use [`create_hardened_allocator`](Self::create_hardened_allocator).
    ///
    /// # Example
    ///
    /// ```rust,no_run
    /// # use ignis::*; use ash::vk;
    /// # fn example(ignis: &Ignis) -> Result<()> {
    /// // Production: structural hardening, near-zero overhead.
    /// let alloc = ignis.create_slab_allocator();
    ///
    /// // Debug: full diagnostics with slot history.
    /// let alloc_dbg = ignis.create_slab_allocator_with(SlabConfig::debug());
    /// # Ok(())
    /// # }
    /// ```
    #[cfg(feature = "slab-allocator")]
    pub fn create_slab_allocator(&self) -> Arc<dyn Allocator> {
        Arc::new(SlabAllocator::new(Arc::clone(&self.shared)))
    }

    /// Create a slab allocator with custom configuration.
    #[cfg(feature = "slab-allocator")]
    pub fn create_slab_allocator_with(&self, config: SlabConfig) -> Arc<dyn Allocator> {
        Arc::new(SlabAllocator::with_config(
            Arc::clone(&self.shared),
            config,
        ))
    }

    /// Begin building a pipeline layout.
    pub fn pipeline_layout_builder(&self) -> PipelineLayoutBuilder {
        PipelineLayoutBuilder::new(Arc::clone(&self.shared))
    }

    /// Create an empty pipeline cache.
    pub fn create_pipeline_cache(&self) -> Result<PipelineCache> {
        PipelineCache::new(Arc::clone(&self.shared))
    }

    /// Create a pipeline cache from a file (falls back to empty if invalid).
    pub fn create_pipeline_cache_from_file(
        &self,
        path: impl AsRef<std::path::Path>,
    ) -> Result<PipelineCache> {
        PipelineCache::from_file(Arc::clone(&self.shared), path)
    }

    /// Create a staging ring buffer for CPU→GPU uploads.
    ///
    /// # Arguments
    ///
    /// * `frame_capacity` - Bytes per frame's staging buffer
    /// * `frames_in_flight` - Number of frame slots
    pub fn create_staging_ring(
        &self,
        frame_capacity: vk::DeviceSize,
        frames_in_flight: u32,
    ) -> Result<StagingRing> {
        let allocator = self.create_block_allocator();
        StagingRing::new(
            Arc::clone(&self.shared),
            allocator,
            frame_capacity,
            frames_in_flight,
        )
    }

    /// Create a per-frame bump allocator for transient GPU data.
    ///
    /// # Arguments
    ///
    /// * `capacity` - Bytes per frame
    /// * `frames_in_flight` - Number of frame slots
    /// * `usage` - Buffer usage flags for the backing buffers
    pub fn create_frame_allocator(
        &self,
        capacity: vk::DeviceSize,
        frames_in_flight: u32,
        usage: vk::BufferUsageFlags,
    ) -> Result<FrameAllocator> {
        let allocator = self.default_allocator();
        FrameAllocator::new(
            Arc::clone(&self.shared),
            Arc::clone(allocator),
            capacity,
            frames_in_flight,
            usage,
        )
    }

    /// Create a typed buffer with the default allocator.
    pub fn create_typed_buffer<T: Copy + Send>(
        &self,
        element_count: usize,
        usage: vk::BufferUsageFlags,
        location: MemoryLocation,
    ) -> Result<TypedBuffer<T>> {
        let allocator = self.default_allocator();
        TypedBuffer::new(
            Arc::clone(&self.shared),
            Arc::clone(allocator),
            element_count,
            usage,
            location,
        )
    }

    /// Create a reusable fence pool.
    pub fn create_fence_pool(&self) -> FencePool {
        FencePool::new(Arc::clone(&self.shared))
    }

    /// Create a debug utils wrapper for object naming and command labels.
    ///
    /// Requires `VK_EXT_debug_utils` to have been enabled on the instance.
    /// In managed mode with `debug-tools`, this is done automatically.
    #[cfg(feature = "debug-tools")]
    pub fn create_debug_utils(&self) -> DebugUtils {
        DebugUtils::new(&self.shared.instance, &self.shared.device)
    }

    /// Create a GPU timestamp profiler.
    ///
    /// # Arguments
    ///
    /// * `max_queries` - Maximum timestamp queries (each scope uses 2).
    #[cfg(feature = "debug-tools")]
    pub fn create_gpu_profiler(&self, max_queries: u32) -> Result<GpuProfiler> {
        GpuProfiler::new(&self.shared, max_queries)
    }
}

impl Drop for Ignis {
    fn drop(&mut self) {
        // Emit diagnostic session summary if any diagnostics were produced.
        let summary = diagnostic::session_summary();
        if !summary.is_empty() {
            eprint!("{summary}");
        }
    }
}

impl DeviceHandle for Ignis {
    #[inline]
    fn ash_instance(&self) -> &ash::Instance {
        &self.shared.instance
    }
    #[inline]
    fn ash_device(&self) -> &ash::Device {
        &self.shared.device
    }
    #[inline]
    fn physical_device(&self) -> vk::PhysicalDevice {
        self.shared.physical_device
    }
    #[inline]
    fn queue_family_properties(&self) -> &[vk::QueueFamilyProperties] {
        &self.shared.queue_family_props
    }
}

/// Convenience re-exports for the most common types.
///
/// ```rust
/// use ignis::prelude::*;
/// ```
pub mod prelude {
    pub use crate::error::{Error, Result, WithContext};
    pub use crate::{DeviceHandle, Ignis, ManagedConfig, QueueType};
    pub use crate::{AsyncQueue, GpuFuture, SubmitBuilder};
    pub use crate::{FrameContext, FrameSync, FencePool};
    pub use crate::{CommandPool, CommandRecorder, ParallelRecorder, ShaderModule};
    pub use crate::{Allocator, BlockAllocator, Buffer, BufferInfo, Image, ImageInfo, MemoryLocation};
    pub use crate::{ComputePipelineBuilder, GraphicsPipelineBuilder, RenderPassBuilder};
    pub use crate::{PipelineCache, PipelineLayoutBuilder, PipelineLayoutHandle};
    pub use crate::{TypedBuffer, StagingRing, FrameAllocator, ReadbackRequest};
    pub use crate::format::{dispatch_size, format_byte_size, format_aspect_mask};
}