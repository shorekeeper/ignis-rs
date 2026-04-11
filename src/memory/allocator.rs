//! GPU memory allocation with per-memory-type sharding.
//!
//! [`BlockAllocator`] uses one lock per Vulkan memory type (max 32),
//! eliminating cross-type contention. Within a type, a next-fit hint
//! provides amortized O(1) allocation for the common case.

use std::sync::Arc;

use ash::vk;

use crate::device::SharedState;
use crate::error::{Error, Result};
use super::resources::MemoryLocation;

/// Maximum memory types per the Vulkan spec.
const MAX_MEMORY_TYPES: usize = 32;

/// Align `value` up to `alignment`. Alignment must be a power of two.
#[inline]
pub(crate) fn align_up(value: vk::DeviceSize, alignment: vk::DeviceSize) -> vk::DeviceSize {
    debug_assert!(alignment.is_power_of_two());
    (value + alignment - 1) & !(alignment - 1)
}

/// Result of a memory allocation. Plain data, does NOT free on drop.
/// Memory is freed by calling [`Allocator::free`].
#[derive(Debug, Clone)]
pub struct Allocation {
    /// `VkDeviceMemory` containing this allocation.
    pub memory: vk::DeviceMemory,
    /// Byte offset within `memory`.
    pub offset: vk::DeviceSize,
    /// Size in bytes.
    pub size: vk::DeviceSize,
    /// Mapped pointer (base of this allocation's region), if host-visible.
    pub mapped_ptr: Option<*mut u8>,
    /// Memory type index. Used by [`BlockAllocator::free`] for O(1) pool lookup.
    pub memory_type_index: u32,
}

unsafe impl Send for Allocation {}
unsafe impl Sync for Allocation {}

/// Trait for GPU memory allocators.
pub trait Allocator: Send + Sync {
    /// Allocate memory satisfying the requirements and location.
    fn allocate(
        &self,
        requirements: &vk::MemoryRequirements,
        location: MemoryLocation,
    ) -> Result<Allocation>;

    /// Free a previously returned allocation.
    fn free(&self, allocation: &Allocation);

    /// Human-readable name.
    fn name(&self) -> &str;
}

/// One `VkDeviceMemory` per allocation.
pub struct DedicatedAllocator {
    shared: Arc<SharedState>,
}

impl DedicatedAllocator {
    /// Create a new dedicated allocator.
    pub fn new(shared: Arc<SharedState>) -> Self {
        Self { shared }
    }
}

impl Allocator for DedicatedAllocator {
    fn allocate(
        &self,
        requirements: &vk::MemoryRequirements,
        location: MemoryLocation,
    ) -> Result<Allocation> {
        let idx = find_memory_type_index(
            &self.shared.memory_properties,
            requirements,
            location,
        )?;

        let alloc_info = vk::MemoryAllocateInfo::default()
            .allocation_size(requirements.size)
            .memory_type_index(idx);

        let memory = unsafe { self.shared.device.allocate_memory(&alloc_info, None)? };

        let flags = self.shared.memory_properties.memory_types[idx as usize].property_flags;
        let mapped_ptr = if flags.contains(vk::MemoryPropertyFlags::HOST_VISIBLE) {
            let ptr = unsafe {
                self.shared.device.map_memory(
                    memory,
                    0,
                    vk::WHOLE_SIZE,
                    vk::MemoryMapFlags::empty(),
                )?
            };
            Some(ptr.cast::<u8>())
        } else {
            None
        };

        Ok(Allocation {
            memory,
            offset: 0,
            size: requirements.size,
            mapped_ptr,
            memory_type_index: idx,
        })
    }

    fn free(&self, allocation: &Allocation) {
        unsafe {
            if allocation.mapped_ptr.is_some() {
                self.shared.device.unmap_memory(allocation.memory);
            }
            self.shared.device.free_memory(allocation.memory, None);
        }
    }

    fn name(&self) -> &'static str {
        "DedicatedAllocator"
    }
}

/// Default block size: 256 MiB.
pub const DEFAULT_BLOCK_SIZE: vk::DeviceSize = 256 * 1024 * 1024;

/// A contiguous free region within a block.
#[derive(Debug, Clone, Copy)]
struct FreeRegion {
    offset: vk::DeviceSize,
    size: vk::DeviceSize,
}

/// A single large `VkDeviceMemory`.
struct Block {
    memory: vk::DeviceMemory,
    total_size: vk::DeviceSize,
    mapped_base: Option<*mut u8>,
    free_list: Vec<FreeRegion>,
    /// Next-fit hint: index into `free_list` to start searching from.
    /// Provides amortized O(1) when allocations are sequential.
    next_fit: usize,
}

unsafe impl Send for Block {}

/// Per-memory-type pool of blocks. Protected by its own mutex.
struct MemoryPool {
    memory_type_index: u32,
    is_host_visible: bool,
    blocks: Vec<Block>,
}

/// Suballocating block allocator with per-memory-type sharding.
///
/// Each Vulkan memory type gets its own lock. Threads working with
/// different memory types (e.g., staging buffers vs GPU-only meshes)
/// never contend. Within one type, the next-fit hint provides
/// amortized O(1) allocation.
///
/// # Scalability
///
/// With 8 threads:
/// - Threads using different memory types: zero contention
/// - Threads using the same type: serialize on that type's lock
///   (unavoidable - Vulkan memory types are the narrowest shard)
/// - Lock hold time is O(1) amortized (next-fit hint)
pub struct BlockAllocator {
    shared: Arc<SharedState>,
    block_size: vk::DeviceSize,
    /// One mutex per memory type. Only indices `0..memory_type_count` are used.
    pools: Vec<std::sync::Mutex<Option<MemoryPool>>>,
}

impl BlockAllocator {
    /// Create with default block size (256 MiB).
    pub fn new(shared: Arc<SharedState>) -> Self {
        Self::with_block_size(shared, DEFAULT_BLOCK_SIZE)
    }

    /// Create with custom block size.
    pub fn with_block_size(shared: Arc<SharedState>, block_size: vk::DeviceSize) -> Self {
        let type_count = shared.memory_properties.memory_type_count as usize;
        let mut pools = Vec::with_capacity(type_count);
        for _ in 0..type_count {
            pools.push(std::sync::Mutex::new(None));
        }
        // Pad to MAX_MEMORY_TYPES for safety.
        while pools.len() < MAX_MEMORY_TYPES {
            pools.push(std::sync::Mutex::new(None));
        }
        Self {
            shared,
            block_size,
            pools,
        }
    }

    fn allocate_block(
        &self,
        mem_type: u32,
        size: vk::DeviceSize,
        host_visible: bool,
    ) -> Result<Block> {
        let alloc_info = vk::MemoryAllocateInfo::default()
            .allocation_size(size)
            .memory_type_index(mem_type);

        let memory = unsafe { self.shared.device.allocate_memory(&alloc_info, None)? };

        let mapped_base = if host_visible {
            match unsafe {
                self.shared.device.map_memory(
                    memory,
                    0,
                    vk::WHOLE_SIZE,
                    vk::MemoryMapFlags::empty(),
                )
            } {
                Ok(p) => Some(p.cast::<u8>()),
                Err(e) => {
                    unsafe { self.shared.device.free_memory(memory, None) };
                    return Err(Error::Vulkan(e));
                }
            }
        } else {
            None
        };

        Ok(Block {
            memory,
            total_size: size,
            mapped_base,
            free_list: vec![FreeRegion { offset: 0, size }],
            next_fit: 0,
        })
    }
}

impl Allocator for BlockAllocator {
    fn allocate(
        &self,
        requirements: &vk::MemoryRequirements,
        location: MemoryLocation,
    ) -> Result<Allocation> {
        let mem_type = find_memory_type_index(
            &self.shared.memory_properties,
            requirements,
            location,
        )?;

        let flags =
            self.shared.memory_properties.memory_types[mem_type as usize].property_flags;
        let host_visible = flags.contains(vk::MemoryPropertyFlags::HOST_VISIBLE);

        let alloc_size = requirements.size;
        let alignment = requirements.alignment.max(1);

        // Lock only this memory type's pool. Other types are unaffected.
        let mut pool_guard = self.pools[mem_type as usize].lock().unwrap();
        let pool = pool_guard.get_or_insert_with(|| MemoryPool {
            memory_type_index: mem_type,
            is_host_visible: host_visible,
            blocks: Vec::new(),
        });

        // Oversized: dedicated block.
        if alloc_size > self.block_size {
            let block = self.allocate_block(mem_type, alloc_size, host_visible)?;
            let mapped_ptr = block.mapped_base;
            let memory = block.memory;
            let mut b = block;
            b.free_list.clear();
            pool.blocks.push(b);

            return Ok(Allocation {
                memory,
                offset: 0,
                size: alloc_size,
                mapped_ptr,
                memory_type_index: mem_type,
            });
        }

        // Try existing blocks.
        for block in &mut pool.blocks {
            if let Some(alloc) = try_allocate_from_block(block, alloc_size, alignment, mem_type) {
                return Ok(alloc);
            }
        }

        // New block.
        let new_block = self.allocate_block(mem_type, self.block_size, host_visible)?;
        pool.blocks.push(new_block);
        let block = pool.blocks.last_mut().unwrap();

        try_allocate_from_block(block, alloc_size, alignment, mem_type)
            .ok_or(Error::NoSuitableMemoryType)
    }

    fn free(&self, allocation: &Allocation) {
        // O(1) pool lookup via memory_type_index.
        let mut pool_guard = self.pools[allocation.memory_type_index as usize]
            .lock()
            .unwrap();

        if let Some(pool) = pool_guard.as_mut() {
            for block in &mut pool.blocks {
                if block.memory == allocation.memory {
                    free_in_block(block, allocation.offset, allocation.size);
                    return;
                }
            }
        }
    }

    fn name(&self) -> &'static str {
        "BlockAllocator"
    }
}

impl Drop for BlockAllocator {
    fn drop(&mut self) {
        for slot in &self.pools {
            if let Some(pool) = slot.lock().unwrap().as_ref() {
                for block in &pool.blocks {
                    unsafe {
                        if block.mapped_base.is_some() {
                            self.shared.device.unmap_memory(block.memory);
                        }
                        self.shared.device.free_memory(block.memory, None);
                    }
                }
            }
        }
    }
}

/// Next-fit allocation: start from hint, wrap around.
fn try_allocate_from_block(
    block: &mut Block,
    size: vk::DeviceSize,
    alignment: vk::DeviceSize,
    mem_type: u32,
) -> Option<Allocation> {
    let len = block.free_list.len();
    if len == 0 {
        return None;
    }

    let start = block.next_fit.min(len - 1);

    // First pass: from hint to end.
    for i in start..len {
        if let Some(alloc) = try_region(block, i, size, alignment, mem_type) {
            return Some(alloc);
        }
    }
    // Second pass: from 0 to hint (wrap around).
    for i in 0..start {
        if let Some(alloc) = try_region(block, i, size, alignment, mem_type) {
            return Some(alloc);
        }
    }
    None
}

fn try_region(
    block: &mut Block,
    i: usize,
    size: vk::DeviceSize,
    alignment: vk::DeviceSize,
    mem_type: u32,
) -> Option<Allocation> {
    let region = block.free_list[i];
    let aligned_offset = align_up(region.offset, alignment);
    let padding = aligned_offset - region.offset;
    let region_end = region.offset + region.size;

    if aligned_offset + size > region_end {
        return None;
    }

    // Remove the region.
    block.free_list.remove(i);

    // Leftover before (wasted padding).
    let mut insert_at = i;
    if padding > 0 {
        block.free_list.insert(
            insert_at,
            FreeRegion {
                offset: region.offset,
                size: padding,
            },
        );
        insert_at += 1;
    }

    // Leftover after.
    let after_offset = aligned_offset + size;
    let after_size = region_end - after_offset;
    if after_size > 0 {
        let pos = block
            .free_list
            .iter()
            .position(|r| r.offset > after_offset)
            .unwrap_or(block.free_list.len());
        block.free_list.insert(
            pos,
            FreeRegion {
                offset: after_offset,
                size: after_size,
            },
        );
        // Set next_fit hint to the region AFTER the allocated space.
        block.next_fit = pos;
    } else {
        block.next_fit = insert_at;
    }

    let mapped_ptr = block
        .mapped_base
        .map(|base| unsafe { base.add(aligned_offset as usize) });

    Some(Allocation {
        memory: block.memory,
        offset: aligned_offset,
        size,
        mapped_ptr,
        memory_type_index: mem_type,
    })
}

/// Return freed region to free list with coalescing.
fn free_in_block(block: &mut Block, offset: vk::DeviceSize, size: vk::DeviceSize) {
    let new_end = offset + size;
    let pos = block
        .free_list
        .iter()
        .position(|r| r.offset >= offset)
        .unwrap_or(block.free_list.len());

    let merge_prev = pos > 0 && {
        let prev = &block.free_list[pos - 1];
        prev.offset + prev.size == offset
    };

    let merge_next = pos < block.free_list.len() && block.free_list[pos].offset == new_end;

    match (merge_prev, merge_next) {
        (true, true) => {
            let next_end = block.free_list[pos].offset + block.free_list[pos].size;
            block.free_list.remove(pos);
            block.free_list[pos - 1].size = next_end - block.free_list[pos - 1].offset;
        }
        (true, false) => {
            block.free_list[pos - 1].size += size;
        }
        (false, true) => {
            block.free_list[pos].offset = offset;
            block.free_list[pos].size += size;
        }
        (false, false) => {
            block.free_list.insert(pos, FreeRegion { offset, size });
        }
    }
}

/// Two-pass memory type search: required+preferred, then required-only.
pub(crate) fn find_memory_type_index(
    mem_props: &vk::PhysicalDeviceMemoryProperties,
    requirements: &vk::MemoryRequirements,
    location: MemoryLocation,
) -> Result<u32> {
    let required = location.required_flags();
    let preferred = location.preferred_flags();

    for i in 0..mem_props.memory_type_count {
        if requirements.memory_type_bits & (1 << i) == 0 {
            continue;
        }
        let flags = mem_props.memory_types[i as usize].property_flags;
        if flags.contains(required | preferred) {
            return Ok(i);
        }
    }

    for i in 0..mem_props.memory_type_count {
        if requirements.memory_type_bits & (1 << i) == 0 {
            continue;
        }
        let flags = mem_props.memory_types[i as usize].property_flags;
        if flags.contains(required) {
            return Ok(i);
        }
    }

    Err(Error::NoSuitableMemoryType)
}