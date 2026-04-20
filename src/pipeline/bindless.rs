//! Bindless descriptor heap using `VK_EXT_descriptor_indexing`.
//!
//! Allocates one large descriptor set with update-after-bind and
//! partially-bound flags, and hands out integer slot handles. Shaders
//! access resources via `nonuniformEXT(handle)` array indexing into the
//! heap's descriptor arrays.
//!
//! # Features
//!
//! - Generation-counted [`BindlessHandle`]s that catch use-after-free
//!   at register/free/update time.
//! - [`update_*`](BindlessHeap::update_sampled_image) methods for
//!   rebinding a slot without losing its handle. Useful when swapchain
//!   images are recreated on resize: the shader keeps the same bindless
//!   index and only the backing view changes.
//! - Structured [`BindlessError`] enum distinguishing exhaustion from
//!   stale handle use.
//!
//! # Requirements
//!
//! Device must enable descriptor indexing:
//!
//! - `descriptor_indexing` (Vulkan 1.2 core) or
//!   `VK_EXT_descriptor_indexing` extension
//! - `descriptor_binding_partially_bound`
//! - `descriptor_binding_update_unused_while_pending`
//! - `runtime_descriptor_array`
//! - `shader_sampled_image_array_non_uniform_indexing`
//! - `shader_storage_image_array_non_uniform_indexing`
//! - `shader_storage_buffer_array_non_uniform_indexing`
//! - `descriptor_binding_sampled_image_update_after_bind`
//! - `descriptor_binding_storage_image_update_after_bind`
//! - `descriptor_binding_storage_buffer_update_after_bind`
//!
//! Enable all of these via [`ManagedConfig::enable_descriptor_indexing`].
//!
//! # Shader-side usage
//!
//! ```glsl
//! #extension GL_EXT_nonuniform_qualifier : require
//!
//! layout(set = 0, binding = 0) uniform texture2D u_textures[];
//! layout(set = 0, binding = 2) uniform sampler u_samplers[];
//!
//! // push constant contains BindlessHandle::raw() values
//! layout(push_constant) uniform PC {
//!     uint tex_handle;
//!     uint sampler_handle;
//! } pc;
//!
//! void main() {
//!     vec4 c = texture(
//!         sampler2D(
//!             u_textures[nonuniformEXT(pc.tex_handle)],
//!             u_samplers[nonuniformEXT(pc.sampler_handle)]
//!         ),
//!         v_uv
//!     );
//! }
//! ```

use std::sync::{Arc, Mutex};

use ash::vk;

use crate::device::SharedState;
use crate::error::{Error, Result};

//
// Handle + error types
//

/// Opaque handle into a bindless heap.
///
/// Combines a slot index with a generation counter. The generation
/// increments every time the slot is freed, so handles retained past a
/// free compare unequal to the current occupant and the heap detects
/// the mismatch in [`update_*`](BindlessHeap::update_sampled_image) and
/// [`free_*`](BindlessHeap::free_sampled_image).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct BindlessHandle {
    slot: u32,
    generation: u32,
}

impl BindlessHandle {
    /// Slot index. Pass to shaders via push constant or uniform.
    ///
    /// The generation is not exposed to shaders; they index the
    /// descriptor array directly by slot. Stale handle detection is a
    /// CPU-side safety net, not a GPU-side access barrier.
    pub fn raw(self) -> u32 {
        self.slot
    }

    /// Generation counter.
    pub fn generation(self) -> u32 {
        self.generation
    }

    /// Construct from raw parts. Useful when rehydrating handles from
    /// serialization or tests. Callers are responsible for ensuring
    /// the (slot, generation) pair is valid for the target heap.
    pub fn from_parts(slot: u32, generation: u32) -> Self {
        Self { slot, generation }
    }
}

/// Errors from bindless heap operations.
#[derive(Debug, Clone, Copy)]
pub enum BindlessError {
    /// Handle's generation does not match the slot's current generation.
    /// The slot was freed and possibly reallocated.
    StaleHandle {
        /// Slot index.
        slot: u32,
        /// Generation the handle was constructed with.
        expected_generation: u32,
        /// Generation the slot currently has.
        current_generation: u32,
    },
    /// No free slots remaining in the requested binding.
    Exhausted {
        /// Human-readable binding name.
        binding: &'static str,
        /// Total slot capacity of that binding.
        capacity: u32,
    },
    /// A lower-level Vulkan operation failed while setting up the heap.
    Vulkan(vk::Result),
}

impl std::fmt::Display for BindlessError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::StaleHandle {
                slot,
                expected_generation,
                current_generation,
            } => write!(
                f,
                "bindless handle is stale: slot={slot}, expected gen={expected_generation}, current gen={current_generation}"
            ),
            Self::Exhausted { binding, capacity } => write!(
                f,
                "bindless heap exhausted: binding {binding} has no free slots (capacity {capacity})"
            ),
            Self::Vulkan(e) => write!(f, "bindless heap Vulkan error: {e:?}"),
        }
    }
}

impl std::error::Error for BindlessError {}

impl From<vk::Result> for BindlessError {
    fn from(value: vk::Result) -> Self {
        BindlessError::Vulkan(value)
    }
}

impl From<BindlessError> for Error {
    fn from(value: BindlessError) -> Self {
        match value {
            BindlessError::Vulkan(r) => Error::Vulkan(r),
            BindlessError::StaleHandle { .. } => {
                Error::InvalidConfig("bindless stale handle detected")
            }
            BindlessError::Exhausted { .. } => Error::InvalidConfig("bindless heap exhausted"),
        }
    }
}

//
// Config
//

/// Configuration for a bindless heap.
#[derive(Debug, Clone)]
pub struct BindlessConfig {
    /// Number of sampled image slots.
    pub sampled_images: u32,
    /// Number of storage image slots.
    pub storage_images: u32,
    /// Number of sampler slots.
    pub samplers: u32,
    /// Number of storage buffer slots.
    pub storage_buffers: u32,
}

impl Default for BindlessConfig {
    fn default() -> Self {
        Self {
            sampled_images: 16384,
            storage_images: 1024,
            samplers: 1024,
            storage_buffers: 4096,
        }
    }
}

/// Binding indices within the bindless set. Shader code must match.
pub const BINDING_SAMPLED_IMAGE: u32 = 0;
pub const BINDING_STORAGE_IMAGE: u32 = 1;
pub const BINDING_SAMPLER: u32 = 2;
pub const BINDING_STORAGE_BUFFER: u32 = 3;

//
// Generation-tracked free list
//

/// Per-binding slot manager with per-slot generation counters.
struct FreeList {
    /// Index of the next never-used slot.
    next_fresh: u32,
    /// Total slot count.
    capacity: u32,
    /// Stack of slots available for reuse.
    free: Vec<u32>,
    /// Generation counter per slot. Incremented on free.
    generation: Vec<u32>,
    /// Name for diagnostics.
    name: &'static str,
}

impl FreeList {
    fn new(capacity: u32, name: &'static str) -> Self {
        Self {
            next_fresh: 0,
            capacity,
            free: Vec::new(),
            generation: vec![0u32; capacity as usize],
            name,
        }
    }

    fn alloc(&mut self) -> std::result::Result<(u32, u32), BindlessError> {
        let slot = if let Some(v) = self.free.pop() {
            v
        } else if self.next_fresh < self.capacity {
            let v = self.next_fresh;
            self.next_fresh += 1;
            v
        } else {
            return Err(BindlessError::Exhausted {
                binding: self.name,
                capacity: self.capacity,
            });
        };
        let gen = self.generation[slot as usize];
        Ok((slot, gen))
    }

    fn free(&mut self, slot: u32, expected_generation: u32) -> std::result::Result<(), BindlessError> {
        let current = self.generation[slot as usize];
        if current != expected_generation {
            return Err(BindlessError::StaleHandle {
                slot,
                expected_generation,
                current_generation: current,
            });
        }
        // Wrapping_add: after 2^32 frees the generation wraps, but that
        // would require billions of free/realloc cycles on the same slot.
        // No realistic workload hits this.
        self.generation[slot as usize] = current.wrapping_add(1);
        self.free.push(slot);
        Ok(())
    }

    fn validate(
        &self,
        slot: u32,
        expected_generation: u32,
    ) -> std::result::Result<(), BindlessError> {
        let current = self.generation[slot as usize];
        if current != expected_generation {
            return Err(BindlessError::StaleHandle {
                slot,
                expected_generation,
                current_generation: current,
            });
        }
        Ok(())
    }

    fn live_count(&self) -> u32 {
        self.next_fresh - self.free.len() as u32
    }
}

//
// Heap
//

/// The bindless descriptor heap.
pub struct BindlessHeap {
    shared: Arc<SharedState>,
    layout: vk::DescriptorSetLayout,
    pool: vk::DescriptorPool,
    set: vk::DescriptorSet,
    config: BindlessConfig,
    sampled_free: Mutex<FreeList>,
    storage_image_free: Mutex<FreeList>,
    sampler_free: Mutex<FreeList>,
    storage_buffer_free: Mutex<FreeList>,
}

impl BindlessHeap {
    /// Create a new bindless heap.
    pub fn new(shared: Arc<SharedState>, config: BindlessConfig) -> Result<Self> {
        let device = &shared.device;

        let bindings = [
            vk::DescriptorSetLayoutBinding::default()
                .binding(BINDING_SAMPLED_IMAGE)
                .descriptor_type(vk::DescriptorType::SAMPLED_IMAGE)
                .descriptor_count(config.sampled_images)
                .stage_flags(vk::ShaderStageFlags::ALL),
            vk::DescriptorSetLayoutBinding::default()
                .binding(BINDING_STORAGE_IMAGE)
                .descriptor_type(vk::DescriptorType::STORAGE_IMAGE)
                .descriptor_count(config.storage_images)
                .stage_flags(vk::ShaderStageFlags::ALL),
            vk::DescriptorSetLayoutBinding::default()
                .binding(BINDING_SAMPLER)
                .descriptor_type(vk::DescriptorType::SAMPLER)
                .descriptor_count(config.samplers)
                .stage_flags(vk::ShaderStageFlags::ALL),
            vk::DescriptorSetLayoutBinding::default()
                .binding(BINDING_STORAGE_BUFFER)
                .descriptor_type(vk::DescriptorType::STORAGE_BUFFER)
                .descriptor_count(config.storage_buffers)
                .stage_flags(vk::ShaderStageFlags::ALL),
        ];

        let binding_flags_value = vk::DescriptorBindingFlags::PARTIALLY_BOUND
            | vk::DescriptorBindingFlags::UPDATE_AFTER_BIND
            | vk::DescriptorBindingFlags::UPDATE_UNUSED_WHILE_PENDING;
        let binding_flags = [binding_flags_value; 4];
        let mut binding_flags_info = vk::DescriptorSetLayoutBindingFlagsCreateInfo::default()
            .binding_flags(&binding_flags);

        let layout_ci = vk::DescriptorSetLayoutCreateInfo::default()
            .bindings(&bindings)
            .flags(vk::DescriptorSetLayoutCreateFlags::UPDATE_AFTER_BIND_POOL)
            .push_next(&mut binding_flags_info);
        let layout = unsafe { device.create_descriptor_set_layout(&layout_ci, None)? };

        let pool_sizes = [
            vk::DescriptorPoolSize {
                ty: vk::DescriptorType::SAMPLED_IMAGE,
                descriptor_count: config.sampled_images,
            },
            vk::DescriptorPoolSize {
                ty: vk::DescriptorType::STORAGE_IMAGE,
                descriptor_count: config.storage_images,
            },
            vk::DescriptorPoolSize {
                ty: vk::DescriptorType::SAMPLER,
                descriptor_count: config.samplers,
            },
            vk::DescriptorPoolSize {
                ty: vk::DescriptorType::STORAGE_BUFFER,
                descriptor_count: config.storage_buffers,
            },
        ];
        let pool_ci = vk::DescriptorPoolCreateInfo::default()
            .max_sets(1)
            .pool_sizes(&pool_sizes)
            .flags(vk::DescriptorPoolCreateFlags::UPDATE_AFTER_BIND);
        let pool = unsafe { device.create_descriptor_pool(&pool_ci, None)? };

        let alloc_info = vk::DescriptorSetAllocateInfo::default()
            .descriptor_pool(pool)
            .set_layouts(std::slice::from_ref(&layout));
        let set = unsafe { device.allocate_descriptor_sets(&alloc_info)? }[0];

        Ok(Self {
            shared,
            layout,
            pool,
            set,
            sampled_free: Mutex::new(FreeList::new(config.sampled_images, "sampled_images")),
            storage_image_free: Mutex::new(FreeList::new(
                config.storage_images,
                "storage_images",
            )),
            sampler_free: Mutex::new(FreeList::new(config.samplers, "samplers")),
            storage_buffer_free: Mutex::new(FreeList::new(
                config.storage_buffers,
                "storage_buffers",
            )),
            config,
        })
    }

    /// Descriptor set layout. Use when creating pipeline layouts.
    pub fn layout(&self) -> vk::DescriptorSetLayout {
        self.layout
    }

    /// Descriptor set. Bind at a fixed slot (commonly 0) in every pipeline.
    pub fn set(&self) -> vk::DescriptorSet {
        self.set
    }

    /// Heap configuration snapshot.
    pub fn config(&self) -> &BindlessConfig {
        &self.config
    }

    /// Number of currently live (allocated and not freed) slots per binding.
    pub fn live_counts(&self) -> BindlessLiveCounts {
        BindlessLiveCounts {
            sampled_images: self.sampled_free.lock().unwrap().live_count(),
            storage_images: self.storage_image_free.lock().unwrap().live_count(),
            samplers: self.sampler_free.lock().unwrap().live_count(),
            storage_buffers: self.storage_buffer_free.lock().unwrap().live_count(),
        }
    }

    // Sampled images 

    /// Register a sampled image.
    pub fn register_sampled_image(
        &self,
        view: vk::ImageView,
        layout: vk::ImageLayout,
    ) -> std::result::Result<BindlessHandle, BindlessError> {
        let (slot, generation) = self.sampled_free.lock().unwrap().alloc()?;
        self.write_sampled_image(slot, view, layout);
        Ok(BindlessHandle { slot, generation })
    }

    /// Replace the image view at a slot without freeing it.
    ///
    /// Useful when a render target is recreated on window resize: keep
    /// the handle, swap the backing view. Validates the handle's
    /// generation to catch use-after-free.
    pub fn update_sampled_image(
        &self,
        handle: BindlessHandle,
        view: vk::ImageView,
        layout: vk::ImageLayout,
    ) -> std::result::Result<(), BindlessError> {
        self.sampled_free
            .lock()
            .unwrap()
            .validate(handle.slot, handle.generation)?;
        self.write_sampled_image(handle.slot, view, layout);
        Ok(())
    }

    /// Free a sampled image slot. Subsequent use of the handle will
    /// return [`BindlessError::StaleHandle`].
    pub fn free_sampled_image(
        &self,
        handle: BindlessHandle,
    ) -> std::result::Result<(), BindlessError> {
        self.sampled_free
            .lock()
            .unwrap()
            .free(handle.slot, handle.generation)
    }

    fn write_sampled_image(&self, slot: u32, view: vk::ImageView, layout: vk::ImageLayout) {
        let image_info = vk::DescriptorImageInfo {
            sampler: vk::Sampler::null(),
            image_view: view,
            image_layout: layout,
        };
        let write = vk::WriteDescriptorSet::default()
            .dst_set(self.set)
            .dst_binding(BINDING_SAMPLED_IMAGE)
            .dst_array_element(slot)
            .descriptor_type(vk::DescriptorType::SAMPLED_IMAGE)
            .image_info(std::slice::from_ref(&image_info));
        unsafe {
            self.shared
                .device
                .update_descriptor_sets(std::slice::from_ref(&write), &[]);
        }
    }

    // Storage images 

    /// Register a storage image (writable from shaders).
    pub fn register_storage_image(
        &self,
        view: vk::ImageView,
    ) -> std::result::Result<BindlessHandle, BindlessError> {
        let (slot, generation) = self.storage_image_free.lock().unwrap().alloc()?;
        self.write_storage_image(slot, view);
        Ok(BindlessHandle { slot, generation })
    }

    /// Replace the storage image view at a slot.
    pub fn update_storage_image(
        &self,
        handle: BindlessHandle,
        view: vk::ImageView,
    ) -> std::result::Result<(), BindlessError> {
        self.storage_image_free
            .lock()
            .unwrap()
            .validate(handle.slot, handle.generation)?;
        self.write_storage_image(handle.slot, view);
        Ok(())
    }

    /// Free a storage image slot.
    pub fn free_storage_image(
        &self,
        handle: BindlessHandle,
    ) -> std::result::Result<(), BindlessError> {
        self.storage_image_free
            .lock()
            .unwrap()
            .free(handle.slot, handle.generation)
    }

    fn write_storage_image(&self, slot: u32, view: vk::ImageView) {
        let image_info = vk::DescriptorImageInfo {
            sampler: vk::Sampler::null(),
            image_view: view,
            image_layout: vk::ImageLayout::GENERAL,
        };
        let write = vk::WriteDescriptorSet::default()
            .dst_set(self.set)
            .dst_binding(BINDING_STORAGE_IMAGE)
            .dst_array_element(slot)
            .descriptor_type(vk::DescriptorType::STORAGE_IMAGE)
            .image_info(std::slice::from_ref(&image_info));
        unsafe {
            self.shared
                .device
                .update_descriptor_sets(std::slice::from_ref(&write), &[]);
        }
    }

    // Samplers──

    /// Register a sampler.
    pub fn register_sampler(
        &self,
        sampler: vk::Sampler,
    ) -> std::result::Result<BindlessHandle, BindlessError> {
        let (slot, generation) = self.sampler_free.lock().unwrap().alloc()?;
        self.write_sampler(slot, sampler);
        Ok(BindlessHandle { slot, generation })
    }

    /// Replace the sampler at a slot.
    pub fn update_sampler(
        &self,
        handle: BindlessHandle,
        sampler: vk::Sampler,
    ) -> std::result::Result<(), BindlessError> {
        self.sampler_free
            .lock()
            .unwrap()
            .validate(handle.slot, handle.generation)?;
        self.write_sampler(handle.slot, sampler);
        Ok(())
    }

    /// Free a sampler slot.
    pub fn free_sampler(
        &self,
        handle: BindlessHandle,
    ) -> std::result::Result<(), BindlessError> {
        self.sampler_free
            .lock()
            .unwrap()
            .free(handle.slot, handle.generation)
    }

    fn write_sampler(&self, slot: u32, sampler: vk::Sampler) {
        let image_info = vk::DescriptorImageInfo {
            sampler,
            image_view: vk::ImageView::null(),
            image_layout: vk::ImageLayout::UNDEFINED,
        };
        let write = vk::WriteDescriptorSet::default()
            .dst_set(self.set)
            .dst_binding(BINDING_SAMPLER)
            .dst_array_element(slot)
            .descriptor_type(vk::DescriptorType::SAMPLER)
            .image_info(std::slice::from_ref(&image_info));
        unsafe {
            self.shared
                .device
                .update_descriptor_sets(std::slice::from_ref(&write), &[]);
        }
    }

    // Storage buffers 

    /// Register a storage buffer range.
    pub fn register_storage_buffer(
        &self,
        buffer: vk::Buffer,
        offset: vk::DeviceSize,
        range: vk::DeviceSize,
    ) -> std::result::Result<BindlessHandle, BindlessError> {
        let (slot, generation) = self.storage_buffer_free.lock().unwrap().alloc()?;
        self.write_storage_buffer(slot, buffer, offset, range);
        Ok(BindlessHandle { slot, generation })
    }

    /// Replace the storage buffer at a slot.
    pub fn update_storage_buffer(
        &self,
        handle: BindlessHandle,
        buffer: vk::Buffer,
        offset: vk::DeviceSize,
        range: vk::DeviceSize,
    ) -> std::result::Result<(), BindlessError> {
        self.storage_buffer_free
            .lock()
            .unwrap()
            .validate(handle.slot, handle.generation)?;
        self.write_storage_buffer(handle.slot, buffer, offset, range);
        Ok(())
    }

    /// Free a storage buffer slot.
    pub fn free_storage_buffer(
        &self,
        handle: BindlessHandle,
    ) -> std::result::Result<(), BindlessError> {
        self.storage_buffer_free
            .lock()
            .unwrap()
            .free(handle.slot, handle.generation)
    }

    fn write_storage_buffer(
        &self,
        slot: u32,
        buffer: vk::Buffer,
        offset: vk::DeviceSize,
        range: vk::DeviceSize,
    ) {
        let buf_info = vk::DescriptorBufferInfo {
            buffer,
            offset,
            range,
        };
        let write = vk::WriteDescriptorSet::default()
            .dst_set(self.set)
            .dst_binding(BINDING_STORAGE_BUFFER)
            .dst_array_element(slot)
            .descriptor_type(vk::DescriptorType::STORAGE_BUFFER)
            .buffer_info(std::slice::from_ref(&buf_info));
        unsafe {
            self.shared
                .device
                .update_descriptor_sets(std::slice::from_ref(&write), &[]);
        }
    }
}

impl Drop for BindlessHeap {
    fn drop(&mut self) {
        unsafe {
            self.shared.device.destroy_descriptor_pool(self.pool, None);
            self.shared
                .device
                .destroy_descriptor_set_layout(self.layout, None);
        }
    }
}

/// Snapshot of live slot counts per binding.
///
/// Returned by [`BindlessHeap::live_counts`] for diagnostics or
/// integration with the memory budget monitor.
#[derive(Debug, Clone, Copy)]
pub struct BindlessLiveCounts {
    /// Currently allocated sampled image slots.
    pub sampled_images: u32,
    /// Currently allocated storage image slots.
    pub storage_images: u32,
    /// Currently allocated sampler slots.
    pub samplers: u32,
    /// Currently allocated storage buffer slots.
    pub storage_buffers: u32,
}