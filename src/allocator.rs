//! GPU memory allocation strategies.
//!
//! Provides an [`Allocator`] trait with two built-in implementations:
//!
//! - [`DedicatedAllocator`]: one `VkDeviceMemory` per resource. Simple,
//!   but subject to the driver's allocation count limit (~4096).
//!   Suitable for a handful of large resources or rapid prototyping.
//!
//! - [`BlockAllocator`]: suballocates from large memory blocks using a
//!   sorted free-list with coalescing. Stays well under driver limits
//!   for thousands of resources. Suitable for production workloads.
//!
//! For AAA production, consider implementing the [`Allocator`] trait as
//! a thin wrapper around `gpu-allocator` or `vk-mem-rs` (VMA), receiving
//! all the benefits of ignis's queue orchestration with a battle-tested
//! memory backend.
//!
//! # Interoperability
//!
//! If you use `wgpu` or `vulkano`, their internal allocators manage memory
//! for resources created through their APIs. Use ignis's allocator only
//! for resources created through ignis (buffers, images in `memory.rs`).
//! The two allocators operate on separate `VkDeviceMemory` objects and
//! do not interfere with each other.

use std::collections::HashMap;
use std::sync::{Arc, Mutex};

use ash::vk;

use crate::device::SharedState;
use crate::error::{Error, Result};
use crate::memory::MemoryLocation;

/// Align `value` up to `alignment`. Alignment must be a power of two.
#[inline]
pub(crate) fn align_up(value: vk::DeviceSize, alignment: vk::DeviceSize) -> vk::DeviceSize {
    debug_assert!(
        alignment.is_power_of_two(),
        "alignment must be power of two"
    );
    (value + alignment - 1) & !(alignment - 1)
}

/// Result of a memory allocation.
///
/// Represents a region of a `VkDeviceMemory` object. For dedicated
/// allocations the region spans the entire memory object. For block
/// allocations it is a subregion at `offset` within a larger object.
///
/// This struct is plain data and does NOT free memory on drop. Memory
/// is freed by calling [`Allocator::free`] or through an owning wrapper
/// like [`Buffer`](crate::Buffer) / [`Image`](crate::Image).
#[derive(Debug, Clone)]
pub struct Allocation {
    /// `VkDeviceMemory` handle containing this allocation.
    pub memory: vk::DeviceMemory,
    /// Byte offset within `memory` where this allocation starts.
    pub offset: vk::DeviceSize,
    /// Size of this allocation in bytes.
    pub size: vk::DeviceSize,
    /// If the memory is host-visible, a pointer to the beginning of this
    /// allocation's region (i.e. the block's mapped base + `offset`).
    /// `None` for device-local memory.
    pub mapped_ptr: Option<*mut u8>,
}

// SAFETY: the raw pointer in `mapped_ptr` points to persistently mapped
// Vulkan memory. Access is externally synchronized by the caller, same
// as all Vulkan mapped memory.
unsafe impl Send for Allocation {}
unsafe impl Sync for Allocation {}

/// Trait for GPU memory allocators.
///
/// Implement this to integrate a custom allocator (e.g. `gpu-allocator`,
/// `vk-mem`) with ignis's [`Buffer`](crate::Buffer) and
/// [`Image`](crate::Image) wrappers.
///
/// # Thread Safety
///
/// Implementations must be `Send + Sync`. The [`BlockAllocator`] uses
/// internal locking. If wrapping a foreign allocator that is not
/// thread-safe, add a `Mutex` in your implementation.
pub trait Allocator: Send + Sync {
    /// Allocate memory satisfying the given requirements and location.
    fn allocate(
        &self,
        requirements: &vk::MemoryRequirements,
        location: MemoryLocation,
    ) -> Result<Allocation>;

    /// Free a previously returned allocation.
    ///
    /// After this call, the `allocation`'s memory handle and mapped pointer
    /// must not be used by the caller.
    fn free(&self, allocation: &Allocation);

    /// Human-readable name for debug output.
    fn name(&self) -> &str;
}

/// One `VkDeviceMemory` per allocation. Simple, but limited to ~4096
/// total allocations by most drivers.
///
/// Prefer [`BlockAllocator`] for any non-trivial workload.
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
        let mem_type_index =
            find_memory_type_index(&self.shared.memory_properties, requirements, location)?;

        let alloc_info = vk::MemoryAllocateInfo::default()
            .allocation_size(requirements.size)
            .memory_type_index(mem_type_index);

        let memory = unsafe { self.shared.device.allocate_memory(&alloc_info, None)? };

        let flags =
            self.shared.memory_properties.memory_types[mem_type_index as usize].property_flags;
        let mapped_ptr = if flags.contains(vk::MemoryPropertyFlags::HOST_VISIBLE) {
            let ptr = unsafe {
                self.shared.device.map_memory(
                    memory,
                    0,
                    vk::WHOLE_SIZE,
                    vk::MemoryMapFlags::empty(),
                )?
            };
            Some(ptr as *mut u8)
        } else {
            None
        };

        Ok(Allocation {
            memory,
            offset: 0,
            size: requirements.size,
            mapped_ptr,
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

    fn name(&self) -> &str {
        "DedicatedAllocator"
    }
}

/// Default block size: 256 MiB.
pub const DEFAULT_BLOCK_SIZE: vk::DeviceSize = 256 * 1024 * 1024;

/// Suballocating block allocator.
///
/// Allocates large `VkDeviceMemory` blocks (default 256 MiB) and
/// suballocates from them using a sorted free-list with coalescing.
/// This keeps the total `VkDeviceMemory` count low regardless of how
/// many individual buffers or images are created.
///
/// Allocations larger than `block_size` automatically fall back to
/// a dedicated `VkDeviceMemory`.
///
/// Host-visible blocks are persistently mapped at creation time.
/// Individual allocations receive a pointer into the mapped region
/// at their offset.
///
/// # Drop Behavior
///
/// When the allocator is dropped, all blocks are unmapped and freed.
/// Any outstanding allocations become invalid. In practice this is not
/// an issue because [`Buffer`](crate::Buffer) and [`Image`](crate::Image)
/// hold an `Arc<dyn Allocator>`, keeping the allocator alive until all
/// resources are dropped.
pub struct BlockAllocator {
    shared: Arc<SharedState>,
    block_size: vk::DeviceSize,
    pools: Mutex<HashMap<u32, MemoryPool>>,
}

/// Per-memory-type pool of blocks.
struct MemoryPool {
    /// Stored for potential future defragmentation and diagnostics.
    #[allow(dead_code)]
    memory_type_index: u32,
    /// Stored for determining whether new blocks need mapping.
    #[allow(dead_code)]
    is_host_visible: bool,
    blocks: Vec<Block>,
}

/// A single large memory allocation from which sub-regions are carved.
struct Block {
    memory: vk::DeviceMemory,
    /// Total capacity of this block. Retained for defragmentation
    /// analysis and diagnostic reporting.
    #[allow(dead_code)]
    total_size: vk::DeviceSize,
    /// Persistently mapped base pointer, or `None` for device-local.
    mapped_base: Option<*mut u8>,
    /// Free regions sorted by ascending offset.
    free_list: Vec<FreeRegion>,
}

/// A contiguous free region within a block.
#[derive(Debug, Clone, Copy)]
struct FreeRegion {
    offset: vk::DeviceSize,
    size: vk::DeviceSize,
}

// SAFETY: Block contains only Vulkan handles, a raw pointer to
// persistently mapped memory, and plain data. Access is synchronized
// by the Mutex in BlockAllocator.
unsafe impl Send for Block {}

impl BlockAllocator {
    /// Create a block allocator with the default block size (256 MiB).
    pub fn new(shared: Arc<SharedState>) -> Self {
        Self::with_block_size(shared, DEFAULT_BLOCK_SIZE)
    }

    /// Create a block allocator with a custom block size.
    ///
    /// Larger blocks reduce the number of `vkAllocateMemory` calls but
    /// increase peak memory usage. 64-256 MiB is typical.
    pub fn with_block_size(shared: Arc<SharedState>, block_size: vk::DeviceSize) -> Self {
        Self {
            shared,
            block_size,
            pools: Mutex::new(HashMap::new()),
        }
    }

    /// Allocate a new block of the given size for the given memory type.
    fn allocate_block(
        &self,
        mem_type_index: u32,
        size: vk::DeviceSize,
        host_visible: bool,
    ) -> Result<Block> {
        let alloc_info = vk::MemoryAllocateInfo::default()
            .allocation_size(size)
            .memory_type_index(mem_type_index);

        let memory = unsafe { self.shared.device.allocate_memory(&alloc_info, None)? };

        let mapped_base = if host_visible {
            let ptr = unsafe {
                self.shared.device.map_memory(
                    memory,
                    0,
                    vk::WHOLE_SIZE,
                    vk::MemoryMapFlags::empty(),
                )
            };
            match ptr {
                Ok(p) => Some(p as *mut u8),
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
        })
    }
}

impl Allocator for BlockAllocator {
    fn allocate(
        &self,
        requirements: &vk::MemoryRequirements,
        location: MemoryLocation,
    ) -> Result<Allocation> {
        let mem_type_index =
            find_memory_type_index(&self.shared.memory_properties, requirements, location)?;

        let flags =
            self.shared.memory_properties.memory_types[mem_type_index as usize].property_flags;
        let host_visible = flags.contains(vk::MemoryPropertyFlags::HOST_VISIBLE);

        let alloc_size = requirements.size;
        let alignment = requirements.alignment.max(1);

        // Oversized allocations get their own dedicated block.
        if alloc_size > self.block_size {
            let block = self.allocate_block(mem_type_index, alloc_size, host_visible)?;
            let mapped_ptr = block.mapped_base;
            let memory = block.memory;

            let mut pools = self.pools.lock().unwrap();
            let pool = pools.entry(mem_type_index).or_insert_with(|| MemoryPool {
                memory_type_index: mem_type_index,
                is_host_visible: host_visible,
                blocks: Vec::new(),
            });

            // Mark the block as fully used (empty free list).
            let mut dedicated_block = block;
            dedicated_block.free_list.clear();
            pool.blocks.push(dedicated_block);

            return Ok(Allocation {
                memory,
                offset: 0,
                size: alloc_size,
                mapped_ptr,
            });
        }

        let mut pools = self.pools.lock().unwrap();
        let pool = pools.entry(mem_type_index).or_insert_with(|| MemoryPool {
            memory_type_index: mem_type_index,
            is_host_visible: host_visible,
            blocks: Vec::new(),
        });

        // Try to suballocate from an existing block.
        for block in &mut pool.blocks {
            if let Some(alloc) = try_allocate_from_block(block, alloc_size, alignment) {
                return Ok(alloc);
            }
        }

        // No existing block has space. Allocate a new one.
        let new_block = self.allocate_block(mem_type_index, self.block_size, host_visible)?;
        pool.blocks.push(new_block);

        let block = pool.blocks.last_mut().unwrap();
        try_allocate_from_block(block, alloc_size, alignment).ok_or(Error::NoSuitableMemoryType)
    }

    fn free(&self, allocation: &Allocation) {
        let mut pools = self.pools.lock().unwrap();

        // Find the pool and block that own this allocation.
        for pool in pools.values_mut() {
            for block in &mut pool.blocks {
                if block.memory == allocation.memory {
                    free_in_block(block, allocation.offset, allocation.size);
                    return;
                }
            }
        }

        // Should not happen if the allocation came from this allocator.
        #[cfg(debug_assertions)]
        panic!(
            "BlockAllocator::free: allocation at memory {:?} offset {} not found",
            allocation.memory, allocation.offset
        );
    }

    fn name(&self) -> &str {
        "BlockAllocator"
    }
}

impl Drop for BlockAllocator {
    fn drop(&mut self) {
        let pools = self.pools.get_mut().unwrap();
        for pool in pools.values() {
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

/// Try to allocate `size` bytes with `alignment` from a block's free list.
///
/// Uses first-fit: scans regions in offset order, picks the first one that
/// fits after alignment. Splits the region, keeping leftover space as free.
fn try_allocate_from_block(
    block: &mut Block,
    size: vk::DeviceSize,
    alignment: vk::DeviceSize,
) -> Option<Allocation> {
    for i in 0..block.free_list.len() {
        let region = block.free_list[i];

        let aligned_offset = align_up(region.offset, alignment);
        let padding = aligned_offset - region.offset;
        let region_end = region.offset + region.size;

        if aligned_offset + size > region_end {
            continue;
        }

        // This region fits. Remove it and insert leftover pieces.
        block.free_list.remove(i);

        // Leftover before the aligned offset (wasted padding).
        if padding > 0 {
            block.free_list.insert(
                i,
                FreeRegion {
                    offset: region.offset,
                    size: padding,
                },
            );
        }

        // Leftover after the allocation.
        let after_offset = aligned_offset + size;
        let after_size = region_end - after_offset;
        if after_size > 0 {
            // Insert at the correct sorted position.
            let insert_pos = block
                .free_list
                .iter()
                .position(|r| r.offset > after_offset)
                .unwrap_or(block.free_list.len());
            block.free_list.insert(
                insert_pos,
                FreeRegion {
                    offset: after_offset,
                    size: after_size,
                },
            );
        }

        let mapped_ptr = block
            .mapped_base
            .map(|base| unsafe { base.add(aligned_offset as usize) });

        return Some(Allocation {
            memory: block.memory,
            offset: aligned_offset,
            size,
            mapped_ptr,
        });
    }

    None
}

/// Return a freed region to a block's free list, coalescing with neighbors.
fn free_in_block(block: &mut Block, offset: vk::DeviceSize, size: vk::DeviceSize) {
    let new_end = offset + size;

    // Find sorted insertion position.
    let pos = block
        .free_list
        .iter()
        .position(|r| r.offset >= offset)
        .unwrap_or(block.free_list.len());

    // Check if we can merge with the previous region.
    let merge_prev = if pos > 0 {
        let prev = &block.free_list[pos - 1];
        prev.offset + prev.size == offset
    } else {
        false
    };

    // Check if we can merge with the next region.
    let merge_next = if pos < block.free_list.len() {
        block.free_list[pos].offset == new_end
    } else {
        false
    };

    match (merge_prev, merge_next) {
        (true, true) => {
            // Merge previous, new, and next into one.
            let next_end = block.free_list[pos].offset + block.free_list[pos].size;
            block.free_list.remove(pos); // remove next
            block.free_list[pos - 1].size = next_end - block.free_list[pos - 1].offset;
        }
        (true, false) => {
            // Extend previous to include new.
            block.free_list[pos - 1].size += size;
        }
        (false, true) => {
            // Extend next backward to include new.
            block.free_list[pos].offset = offset;
            block.free_list[pos].size += size;
        }
        (false, false) => {
            // Insert as a new free region.
            block.free_list.insert(pos, FreeRegion { offset, size });
        }
    }
}

/// Find a memory type index satisfying the requirements and location.
///
/// Two-pass: first tries required + preferred flags, then required only.
pub(crate) fn find_memory_type_index(
    mem_props: &vk::PhysicalDeviceMemoryProperties,
    requirements: &vk::MemoryRequirements,
    location: MemoryLocation,
) -> Result<u32> {
    let required = location.required_flags();
    let preferred = location.preferred_flags();

    // Pass 1: required + preferred.
    for i in 0..mem_props.memory_type_count {
        if requirements.memory_type_bits & (1 << i) == 0 {
            continue;
        }
        let flags = mem_props.memory_types[i as usize].property_flags;
        if flags.contains(required | preferred) {
            return Ok(i);
        }
    }

    // Pass 2: required only.
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

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn free_list_coalescing() {
        let mut block = Block {
            memory: vk::DeviceMemory::null(),
            total_size: 1024,
            mapped_base: None,
            free_list: vec![FreeRegion {
                offset: 0,
                size: 1024,
            }],
        };

        // Allocate three 256-byte regions.
        let a = try_allocate_from_block(&mut block, 256, 1).unwrap();
        assert_eq!(a.offset, 0);
        let b = try_allocate_from_block(&mut block, 256, 1).unwrap();
        assert_eq!(b.offset, 256);
        let c = try_allocate_from_block(&mut block, 256, 1).unwrap();
        assert_eq!(c.offset, 512);

        // Free list should have one region: [768..1024).
        assert_eq!(block.free_list.len(), 1);
        assert_eq!(block.free_list[0].offset, 768);

        // Free b (middle), creating a gap.
        free_in_block(&mut block, 256, 256);
        // Now: [256..512), [768..1024) - two regions.
        assert_eq!(block.free_list.len(), 2);

        // Free a (first), should coalesce with the gap after it.
        free_in_block(&mut block, 0, 256);
        // Now: [0..512), [768..1024) - two regions, first merged.
        assert_eq!(block.free_list.len(), 2);
        assert_eq!(block.free_list[0].offset, 0);
        assert_eq!(block.free_list[0].size, 512);

        // Free c, should coalesce into one contiguous region.
        free_in_block(&mut block, 512, 256);
        // Now: [0..1024) - fully free.
        assert_eq!(block.free_list.len(), 1);
        assert_eq!(block.free_list[0].offset, 0);
        assert_eq!(block.free_list[0].size, 1024);
    }

    #[test]
    fn alignment_handling() {
        let mut block = Block {
            memory: vk::DeviceMemory::null(),
            total_size: 4096,
            mapped_base: None,
            free_list: vec![FreeRegion {
                offset: 0,
                size: 4096,
            }],
        };

        // Allocate 100 bytes, eating offset 0.
        let _a = try_allocate_from_block(&mut block, 100, 1).unwrap();

        // Allocate 64 bytes with 256-byte alignment.
        // Must skip to offset 256.
        let b = try_allocate_from_block(&mut block, 64, 256).unwrap();
        assert_eq!(b.offset, 256);

        // Free list should contain: [100..256) and [320..4096).
        assert_eq!(block.free_list.len(), 2);
        assert_eq!(block.free_list[0].offset, 100);
        assert_eq!(block.free_list[0].size, 156);
        assert_eq!(block.free_list[1].offset, 320);
    }
}
