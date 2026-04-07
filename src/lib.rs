#![warn(missing_docs)]
#![doc = include_str!("../README.md")]

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

// Core modules: device, queues, memory, rendering pipeline.
pub mod allocator;
pub mod command;
pub mod device;
pub mod error;
pub mod memory;
pub mod pipeline;
pub mod queue;
pub mod renderpass;
pub mod shader;
pub mod swapchain;
pub mod sync;
pub mod watcher;

// Resource tracking and synchronization helpers.
pub mod tracker;

// Debug and validation modules.
pub mod aliasing;
pub mod barrier_opt;
pub mod budget;
pub mod cmd_state;
pub mod descriptor_audit;
pub mod diagnostic;
pub mod hang_detector;
pub mod hardened;
pub mod journal;
pub mod lifetime;
pub mod pipeline_audit;
pub mod thread_audit;

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
pub use watcher::FenceWatcher;

// Core re-exports: command recording.
pub use command::{
    ColorAttachmentInfo, CommandBufferInheritance, CommandPool, CommandRecorder,
    DepthStencilAttachmentInfo, DynamicRenderPassBuilder, ParallelRecorder,
};

// Core re-exports: memory and allocation.
pub use allocator::{Allocation, Allocator, BlockAllocator, DedicatedAllocator};
pub use memory::{Buffer, BufferInfo, Image, ImageInfo, MemoryLocation};
pub use swapchain::{Swapchain, SwapchainConfig, SwapchainSupport};

// Core re-exports: shaders, pipelines, render passes.
pub use pipeline::{
    ComputePipelineBuilder, GraphicsPipelineBuilder, RayTracingPipeline, RayTracingPipelineBuilder,
    ShaderBindingTableLayout, ShaderGroup,
};
pub use renderpass::{
    AttachmentConfig, AttachmentRef, RenderPassBuilder, RenderPassHandle, SubpassConfig,
    SubpassDependency,
};
pub use shader::ShaderModule;

// Re-exports: resource tracking.
pub use tracker::{ImageState, ImageTransition, ResourceTracker};

// Re-exports: hardened allocator.
pub use hardened::{
    CorruptionAction, CorruptionEvent, FreePattern, GuardRegion, HardenedAllocator, HardenedConfig,
    HardenedStats,
};

// Re-exports: debug and validation tools.
pub use aliasing::{AliasingDetector, AliasingIssue};
pub use barrier_opt::{BarrierAnalyzer, BarrierSuggestion, SuggestionKind};
pub use budget::{BudgetMonitor, BudgetSnapshot, BudgetThresholds, HeapStatus};
pub use cmd_state::{RecordingState, StateErrorAction, ValidatedRecorder};
pub use descriptor_audit::{BoundResource, DescriptorAuditor, DescriptorIssue};
pub use hang_detector::{BreadcrumbBuffer, HangAction, HangConfig, HangDetector};
pub use journal::{EntryStatus, SubmissionJournal};
pub use lifetime::{LeakAction, LifetimeTracker};
pub use pipeline_audit::{PipelineAuditor, PipelineIssue};
pub use thread_audit::{AuditedPool, ThreadViolationAction};

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
/// synchronization primitives, command pools, pipeline builders, and more.
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

// SAFETY: Ignis contains only Arc-wrapped state and Vec of Arc.
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
        let queues = allocations
            .into_iter()
            .map(|alloc| {
                Arc::new(AsyncQueue::new(
                    Arc::clone(&shared),
                    alloc.handle,
                    alloc.family_index,
                    alloc.queue_index,
                    alloc.capabilities,
                ))
            })
            .collect();
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
        let queues = allocations
            .into_iter()
            .map(|alloc| {
                Arc::new(AsyncQueue::new(
                    Arc::clone(&shared),
                    alloc.handle,
                    alloc.family_index,
                    alloc.queue_index,
                    alloc.capabilities,
                ))
            })
            .collect();
        Ok(Self {
            shared,
            queues,
            default_allocator: OnceLock::new(),
        })
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
                && (queue_type == QueueType::Graphics || !caps.contains(vk::QueueFlags::GRAPHICS))
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

    /// Create a background fence watcher for efficient async polling.
    ///
    /// The watcher spawns a dedicated thread that periodically checks
    /// fence status and wakes async tasks. The `poll_interval` controls
    /// the check frequency.
    ///
    /// A reasonable default is `Duration::from_micros(200)`.
    pub fn create_fence_watcher(&self, poll_interval: Duration) -> Arc<FenceWatcher> {
        Arc::new(FenceWatcher::new(Arc::clone(&self.shared), poll_interval))
    }

    /// Returns (or lazily creates) the default block allocator.
    ///
    /// Shared across all convenience `create_buffer` / `create_image`
    /// calls. The allocator is created on first use and reused afterwards.
    fn default_allocator(&self) -> &Arc<dyn Allocator> {
        self.default_allocator
            .get_or_init(|| Arc::new(BlockAllocator::new(Arc::clone(&self.shared))))
    }

    /// Create a buffer with the default block allocator.
    ///
    /// Convenience method that creates a [`BlockAllocator`] internally.
    /// If you create many resources, prefer [`create_buffer_with`](Self::create_buffer_with)
    /// and a shared allocator to avoid creating a new allocator per call.
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
        Buffer::new(Arc::clone(&self.shared), Arc::clone(allocator), info)
    }

    /// Create an image with the default block allocator.
    ///
    /// Convenience method. For bulk creation prefer
    /// [`create_image_with`](Self::create_image_with) and a shared allocator.
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
        Image::new(Arc::clone(&self.shared), Arc::clone(allocator), info)
    }

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
    pub fn query_swapchain_support(&self, surface: vk::SurfaceKHR) -> Result<SwapchainSupport> {
        Swapchain::query_support(&self.shared, surface)
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
    /// Returns [`Error::InvalidSpirv`] if the data is empty or misaligned,
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

    /// Returns ray tracing pipeline properties if the extension is enabled.
    ///
    /// Contains shader group handle sizes, alignment requirements, and
    /// maximum recursion depth. Returns `None` if the device was not
    /// created with ray tracing support.
    ///
    /// # Example
    ///
    /// ```rust,no_run
    /// # use ignis::*;
    /// # fn example(ignis: &Ignis) {
    /// if let Some(props) = ignis.ray_tracing_properties() {
    ///     println!("handle size: {}", props.shader_group_handle_size);
    ///     println!("max recursion: {}", props.max_ray_recursion_depth);
    /// }
    /// # }
    /// ```
    #[inline]
    pub fn ray_tracing_properties(&self) -> Option<&crate::device::RayTracingProperties> {
        self.shared.rt_properties.as_ref()
    }

    /// Returns `true` if the device was created with ray tracing support.
    #[inline]
    pub fn supports_ray_tracing(&self) -> bool {
        self.shared.rt_pipeline_fn.is_some()
    }

    /// Access the acceleration structure extension function loader.
    ///
    /// Returns `None` if ray tracing is not enabled. Useful for building
    /// and managing acceleration structures directly through ash.
    #[inline]
    pub fn acceleration_structure_fn(&self) -> Option<&ash::khr::acceleration_structure::Device> {
        self.shared.accel_struct_fn.as_ref()
    }

    /// Access the ray tracing pipeline extension function loader.
    ///
    /// Returns `None` if ray tracing is not enabled. Useful for
    /// `cmd_trace_rays` and other RT pipeline operations through ash.
    #[inline]
    pub fn ray_tracing_pipeline_fn(&self) -> Option<&ash::khr::ray_tracing_pipeline::Device> {
        self.shared.rt_pipeline_fn.as_ref()
    }
    /// Create the default block allocator (256 MiB blocks).
    ///
    /// This is the recommended allocator for most workloads. Returns
    /// an `Arc` suitable for passing to [`create_buffer`](Self::create_buffer_with)
    /// and [`create_image`](Self::create_image_with).
    pub fn create_block_allocator(&self) -> Arc<dyn Allocator> {
        Arc::new(BlockAllocator::new(Arc::clone(&self.shared)))
    }

    /// Create a block allocator with a custom block size.
    pub fn create_block_allocator_with_size(
        &self,
        block_size: vk::DeviceSize,
    ) -> Arc<dyn Allocator> {
        Arc::new(BlockAllocator::with_block_size(
            Arc::clone(&self.shared),
            block_size,
        ))
    }

    /// Create a dedicated allocator (one VkDeviceMemory per resource).
    ///
    /// Only suitable for a small number of large resources.
    pub fn create_dedicated_allocator(&self) -> Arc<dyn Allocator> {
        Arc::new(DedicatedAllocator::new(Arc::clone(&self.shared)))
    }

    /// Create a new empty resource tracker.
    pub fn create_resource_tracker(&self) -> ResourceTracker {
        ResourceTracker::new()
    }

    /// Create a hardened allocator wrapping the default block allocator.
    ///
    /// For development/testing builds. Adds guard bands, canary
    /// verification, and quarantine to every allocation.
    ///
    /// # Example
    ///
    /// ```rust,no_run
    /// # use ignis::*; use ash::vk;
    /// # fn example(ignis: &Ignis) -> Result<()> {
    /// let alloc = ignis.create_hardened_allocator(HardenedConfig::default());
    ///
    /// // Every buffer/image now has guard bands and quarantine.
    /// let buf = ignis.create_buffer_with(&alloc, &BufferInfo::staging(1024))?;
    /// # Ok(())
    /// # }
    /// ```
    pub fn create_hardened_allocator(&self, config: HardenedConfig) -> Arc<dyn Allocator> {
        let inner = self.create_block_allocator();
        Arc::new(HardenedAllocator::new(
            Arc::clone(&self.shared),
            inner,
            config,
        ))
    }

    /// Create a hardened allocator wrapping a specific inner allocator.
    ///
    /// Composable: wrap any allocator (BlockAllocator, DedicatedAllocator,
    /// or a foreign gpu-allocator/vk-mem bridge).
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

    /// Create a hang detector with default configuration.
    pub fn create_hang_detector(&self, config: HangConfig, action: HangAction) -> HangDetector {
        HangDetector::new(Arc::clone(&self.shared), config, action)
    }

    /// Create a breadcrumb buffer for GPU operation tracking.
    pub fn create_breadcrumb_buffer(&self) -> Result<BreadcrumbBuffer> {
        BreadcrumbBuffer::new(Arc::clone(&self.shared))
    }

    /// Create a memory budget monitor.
    pub fn create_budget_monitor(&self, thresholds: BudgetThresholds) -> BudgetMonitor {
        BudgetMonitor::new(Arc::clone(&self.shared), thresholds)
    }

    /// Create a submission journal with the given capacity.
    pub fn create_journal(&self, capacity: usize) -> SubmissionJournal {
        SubmissionJournal::new(capacity)
    }

    /// Create a lifetime tracker.
    pub fn create_lifetime_tracker(&self) -> LifetimeTracker {
        LifetimeTracker::new()
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
