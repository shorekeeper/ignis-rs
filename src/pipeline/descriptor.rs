//! Descriptor set and pool builder.
//!
//! Eliminates the pain of constructing `VkWriteDescriptorSet` arrays
//! and managing their lifetimes.

use std::sync::Arc;

use ash::vk;

use crate::device::SharedState;
use crate::error::{Error, Result};

/// Builder for a descriptor set layout.
pub struct DescriptorSetLayoutBuilder {
    shared: Arc<SharedState>,
    bindings: Vec<vk::DescriptorSetLayoutBinding<'static>>,
}

impl DescriptorSetLayoutBuilder {
    pub(crate) fn new(shared: Arc<SharedState>) -> Self {
        Self {
            shared,
            bindings: Vec::new(),
        }
    }

    /// Add a binding.
    pub fn binding(
        mut self,
        binding: u32,
        descriptor_type: vk::DescriptorType,
        count: u32,
        stage_flags: vk::ShaderStageFlags,
    ) -> Self {
        self.bindings.push(
            vk::DescriptorSetLayoutBinding::default()
                .binding(binding)
                .descriptor_type(descriptor_type)
                .descriptor_count(count)
                .stage_flags(stage_flags),
        );
        self
    }

    /// Build the layout.
    pub fn build(self) -> Result<vk::DescriptorSetLayout> {
        let ci = vk::DescriptorSetLayoutCreateInfo::default().bindings(&self.bindings);
        let layout = unsafe { self.shared.device.create_descriptor_set_layout(&ci, None)? };
        Ok(layout)
    }
}

/// Builder for a descriptor pool.
pub struct DescriptorPoolBuilder {
    shared: Arc<SharedState>,
    sizes: Vec<vk::DescriptorPoolSize>,
    max_sets: u32,
    flags: vk::DescriptorPoolCreateFlags,
}

impl DescriptorPoolBuilder {
    pub(crate) fn new(shared: Arc<SharedState>) -> Self {
        Self {
            shared,
            sizes: Vec::new(),
            max_sets: 1,
            flags: vk::DescriptorPoolCreateFlags::empty(),
        }
    }

    /// Set the maximum number of sets that can be allocated.
    pub fn max_sets(mut self, n: u32) -> Self {
        self.max_sets = n;
        self
    }

    /// Add a pool size entry.
    pub fn pool_size(mut self, ty: vk::DescriptorType, count: u32) -> Self {
        self.sizes.push(vk::DescriptorPoolSize {
            ty,
            descriptor_count: count,
        });
        self
    }

    /// Allow individual descriptor set freeing.
    pub fn free_descriptor_set(mut self) -> Self {
        self.flags |= vk::DescriptorPoolCreateFlags::FREE_DESCRIPTOR_SET;
        self
    }

    /// Build the pool.
    pub fn build(self) -> Result<vk::DescriptorPool> {
        let ci = vk::DescriptorPoolCreateInfo::default()
            .max_sets(self.max_sets)
            .pool_sizes(&self.sizes)
            .flags(self.flags);
        let pool = unsafe { self.shared.device.create_descriptor_pool(&ci, None)? };
        Ok(pool)
    }
}

/// Allocate descriptor sets from a pool.
pub fn allocate_descriptor_sets(
    device: &ash::Device,
    pool: vk::DescriptorPool,
    layouts: &[vk::DescriptorSetLayout],
) -> Result<Vec<vk::DescriptorSet>> {
    let ai = vk::DescriptorSetAllocateInfo::default()
        .descriptor_pool(pool)
        .set_layouts(layouts);
    let sets = unsafe { device.allocate_descriptor_sets(&ai)? };
    Ok(sets)
}

/// A single descriptor write prepared by [`DescriptorWriter`].
enum PreparedWrite {
    Buffer {
        binding: u32,
        ty: vk::DescriptorType,
        info: vk::DescriptorBufferInfo,
    },
    Image {
        binding: u32,
        ty: vk::DescriptorType,
        info: vk::DescriptorImageInfo,
    },
}

/// Builder for writing descriptors to a set.
///
/// Manages the lifetime of intermediate `DescriptorBufferInfo` and
/// `DescriptorImageInfo` structs that must live until
/// `vkUpdateDescriptorSets` is called.
///
/// # Example
///
/// ```rust,no_run
/// # use ignis::descriptor::*; use ash::vk;
/// # fn example(device: &ash::Device, set: vk::DescriptorSet,
/// #            buffer: vk::Buffer, view: vk::ImageView,
/// #            sampler: vk::Sampler) {
/// DescriptorWriter::new(set)
///     .buffer(0, vk::DescriptorType::UNIFORM_BUFFER, buffer, 0, 256)
///     .image(1, vk::DescriptorType::COMBINED_IMAGE_SAMPLER, view, sampler,
///            vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL)
///     .write(device);
/// # }
/// ```
pub struct DescriptorWriter {
    set: vk::DescriptorSet,
    writes: Vec<PreparedWrite>,
}

impl DescriptorWriter {
    /// Create a writer targeting the given descriptor set.
    pub fn new(set: vk::DescriptorSet) -> Self {
        Self {
            set,
            writes: Vec::new(),
        }
    }

    /// Write a buffer descriptor.
    pub fn buffer(
        mut self,
        binding: u32,
        ty: vk::DescriptorType,
        buffer: vk::Buffer,
        offset: vk::DeviceSize,
        range: vk::DeviceSize,
    ) -> Self {
        self.writes.push(PreparedWrite::Buffer {
            binding,
            ty,
            info: vk::DescriptorBufferInfo {
                buffer,
                offset,
                range,
            },
        });
        self
    }

    /// Write an image/sampler descriptor.
    pub fn image(
        mut self,
        binding: u32,
        ty: vk::DescriptorType,
        image_view: vk::ImageView,
        sampler: vk::Sampler,
        layout: vk::ImageLayout,
    ) -> Self {
        self.writes.push(PreparedWrite::Image {
            binding,
            ty,
            info: vk::DescriptorImageInfo {
                sampler,
                image_view,
                image_layout: layout,
            },
        });
        self
    }

    /// Execute the writes.
    ///
    /// All intermediate structs live on the stack until this call completes.
    pub fn write(self, device: &ash::Device) {
        // Collect buffer and image infos into stable Vecs first,
        // then build WriteDescriptorSet referencing them.
        let mut buffer_infos: Vec<vk::DescriptorBufferInfo> = Vec::new();
        let mut image_infos: Vec<vk::DescriptorImageInfo> = Vec::new();

        // Index map: (kind, index_in_vec) per write.
        struct WriteRef {
            binding: u32,
            ty: vk::DescriptorType,
            kind: u8, // 0=buffer, 1=image
            index: usize,
        }
        let mut refs: Vec<WriteRef> = Vec::new();

        for w in &self.writes {
            match w {
                PreparedWrite::Buffer { binding, ty, info } => {
                    let idx = buffer_infos.len();
                    buffer_infos.push(*info);
                    refs.push(WriteRef {
                        binding: *binding,
                        ty: *ty,
                        kind: 0,
                        index: idx,
                    });
                }
                PreparedWrite::Image { binding, ty, info } => {
                    let idx = image_infos.len();
                    image_infos.push(*info);
                    refs.push(WriteRef {
                        binding: *binding,
                        ty: *ty,
                        kind: 1,
                        index: idx,
                    });
                }
            }
        }

        let vk_writes: Vec<vk::WriteDescriptorSet<'_>> = refs
            .iter()
            .map(|r| {
                let mut w = vk::WriteDescriptorSet::default()
                    .dst_set(self.set)
                    .dst_binding(r.binding)
                    .descriptor_type(r.ty)
                    .dst_array_element(0);
                match r.kind {
                    0 => {
                        w = w.buffer_info(std::slice::from_ref(&buffer_infos[r.index]));
                    }
                    1 => {
                        w = w.image_info(std::slice::from_ref(&image_infos[r.index]));
                    }
                    _ => unreachable!(),
                }
                w
            })
            .collect();

        unsafe {
            device.update_descriptor_sets(&vk_writes, &[]);
        }
    }
}

/// Auto-growing descriptor pool.
///
/// When allocation fails due to pool exhaustion, automatically creates
/// a new pool and retries. Eliminates the "forgot to make the pool
/// big enough" class of bugs.
pub struct DescriptorArena {
    shared: Arc<SharedState>,
    pools: Vec<vk::DescriptorPool>,
    current_pool: usize,
    max_sets_per_pool: u32,
    type_counts: Vec<(vk::DescriptorType, u32)>,
}

impl DescriptorArena {
    /// Create an arena with the given per-pool capacity.
    ///
    /// `type_counts` specifies how many descriptors of each type are
    /// available per pool. When a pool runs out, a new one with the
    /// same capacity is created.
    ///
    /// # Example
    ///
    /// ```rust,no_run
    /// # use ignis::descriptor::*; use ash::vk;
    /// # use std::sync::Arc;
    /// # fn example(shared: Arc<ignis::device::SharedState>) {
    /// let arena = DescriptorArena::new(
    ///     shared,
    ///     256, // max sets per pool
    ///     &[
    ///         (vk::DescriptorType::UNIFORM_BUFFER, 256),
    ///         (vk::DescriptorType::COMBINED_IMAGE_SAMPLER, 512),
    ///     ],
    /// ).unwrap();
    /// # }
    /// ```
    pub fn new(
        shared: Arc<SharedState>,
        max_sets_per_pool: u32,
        type_counts: &[(vk::DescriptorType, u32)],
    ) -> Result<Self> {
        let pool_sizes: Vec<vk::DescriptorPoolSize> = type_counts
            .iter()
            .map(|(ty, count)| vk::DescriptorPoolSize {
                ty: *ty,
                descriptor_count: *count,
            })
            .collect();

        let ci = vk::DescriptorPoolCreateInfo::default()
            .max_sets(max_sets_per_pool)
            .pool_sizes(&pool_sizes);

        let first_pool = unsafe { shared.device.create_descriptor_pool(&ci, None)? };

        Ok(Self {
            shared,
            pools: vec![first_pool],
            current_pool: 0,
            max_sets_per_pool,
            type_counts: type_counts.to_vec(),
        })
    }

    /// Allocate a descriptor set. If the current pool is exhausted,
    /// creates a new pool and retries.
    pub fn allocate(&mut self, layout: vk::DescriptorSetLayout) -> Result<vk::DescriptorSet> {
        // Try current pool.
        let ai = vk::DescriptorSetAllocateInfo::default()
            .descriptor_pool(self.pools[self.current_pool])
            .set_layouts(std::slice::from_ref(&layout));

        match unsafe { self.shared.device.allocate_descriptor_sets(&ai) } {
            Ok(sets) => return Ok(sets[0]),
            Err(vk::Result::ERROR_OUT_OF_POOL_MEMORY | vk::Result::ERROR_FRAGMENTED_POOL) => {
                // Pool full. Create a new one.
            }
            Err(e) => return Err(Error::from(e)),
        }

        // Create new pool.
        let pool_sizes: Vec<vk::DescriptorPoolSize> = self
            .type_counts
            .iter()
            .map(|(ty, count)| vk::DescriptorPoolSize {
                ty: *ty,
                descriptor_count: *count,
            })
            .collect();

        let ci = vk::DescriptorPoolCreateInfo::default()
            .max_sets(self.max_sets_per_pool)
            .pool_sizes(&pool_sizes);

        let new_pool = unsafe { self.shared.device.create_descriptor_pool(&ci, None)? };
        self.pools.push(new_pool);
        self.current_pool = self.pools.len() - 1;

        // Retry with new pool.
        let ai = vk::DescriptorSetAllocateInfo::default()
            .descriptor_pool(self.pools[self.current_pool])
            .set_layouts(std::slice::from_ref(&layout));

        let sets = unsafe { self.shared.device.allocate_descriptor_sets(&ai)? };
        Ok(sets[0])
    }

    /// Allocate multiple sets.
    pub fn allocate_many(
        &mut self,
        layouts: &[vk::DescriptorSetLayout],
    ) -> Result<Vec<vk::DescriptorSet>> {
        let mut result = Vec::with_capacity(layouts.len());
        for &layout in layouts {
            result.push(self.allocate(layout)?);
        }
        Ok(result)
    }

    /// Reset all pools, recycling all allocated sets.
    /// Call when the GPU is done with all sets from this arena.
    pub fn reset(&mut self) -> Result<()> {
        for &pool in &self.pools {
            unsafe {
                self.shared
                    .device
                    .reset_descriptor_pool(pool, vk::DescriptorPoolResetFlags::empty())?;
            }
        }
        self.current_pool = 0;
        Ok(())
    }

    /// Number of pools allocated (1 = no growth, 2+ = at least one overflow).
    pub fn pool_count(&self) -> usize {
        self.pools.len()
    }
}

impl Drop for DescriptorArena {
    fn drop(&mut self) {
        for &pool in &self.pools {
            unsafe {
                self.shared.device.destroy_descriptor_pool(pool, None);
            }
        }
    }
}

/// Per-frame descriptor set ring buffer.
///
/// Maintains one [`DescriptorArena`] per frame in flight. At each frame
/// boundary, the oldest arena is reset, recycling its descriptor sets
/// for the new frame. This is the standard pattern for transient per-frame
/// descriptors (uniforms, material parameters, etc.).
///
/// # Persistent Descriptors
///
/// Sets that must survive across frames should NOT be allocated from the
/// ring. Use a separate [`DescriptorArena`] or raw pool for those.
///
/// # Example
///
/// ```rust,no_run
/// # use ignis::descriptor::*; use ash::vk;
/// # use std::sync::Arc;
/// # fn example(shared: Arc<ignis::device::SharedState>,
/// #            layout: vk::DescriptorSetLayout) {
/// let mut ring = DescriptorRing::new(
///     shared, 2, 256,
///     &[(vk::DescriptorType::UNIFORM_BUFFER, 256)],
/// ).unwrap();
///
/// // Each frame:
/// ring.advance().unwrap(); // resets the arena from 2 frames ago
/// let set = ring.allocate(layout).unwrap();
/// // ... write and use the set ...
/// // set is automatically recycled 2 frames later
/// # }
/// ```
pub struct DescriptorRing {
    arenas: Vec<DescriptorArena>,
    current: usize,
}

impl DescriptorRing {
    /// Create a ring with `frames_in_flight` arenas.
    pub fn new(
        shared: Arc<SharedState>,
        frames_in_flight: u32,
        max_sets_per_pool: u32,
        type_counts: &[(vk::DescriptorType, u32)],
    ) -> Result<Self> {
        let mut arenas = Vec::with_capacity(frames_in_flight as usize);
        for _ in 0..frames_in_flight {
            arenas.push(DescriptorArena::new(
                Arc::clone(&shared),
                max_sets_per_pool,
                type_counts,
            )?);
        }
        Ok(Self { arenas, current: 0 })
    }

    /// Advance to the next frame.
    ///
    /// Resets the arena that is about to be reused (the oldest one).
    /// Call after the corresponding frame's fence has signaled.
    pub fn advance(&mut self) -> Result<()> {
        self.current = (self.current + 1) % self.arenas.len();
        self.arenas[self.current].reset()
    }

    /// Allocate a descriptor set for the current frame.
    pub fn allocate(&mut self, layout: vk::DescriptorSetLayout) -> Result<vk::DescriptorSet> {
        self.arenas[self.current].allocate(layout)
    }

    /// Current frame index (0-based).
    pub fn current_frame(&self) -> usize {
        self.current
    }
}
