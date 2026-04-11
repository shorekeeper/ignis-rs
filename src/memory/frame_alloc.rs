//! Per-frame bump allocator for transient GPU data.
//!
//! [`FrameAllocator`] maintains one large buffer per frame-in-flight.
//! Each frame, the cursor is reset and data is written sequentially.
//! This is the fastest pattern for per-frame uniform buffers,
//! dynamic vertex data, and push-constant staging.
//!
//! # Example
//!
//! ```rust,no_run
//! # use ignis::*; use ash::vk;
//! # fn example(ignis: &Ignis) -> Result<()> {
//! let mut frame_alloc = ignis.create_frame_allocator(
//!     1024 * 1024,
//!     2,
//!     vk::BufferUsageFlags::UNIFORM_BUFFER | vk::BufferUsageFlags::VERTEX_BUFFER,
//! )?;
//!
//! // Each frame:
//! frame_alloc.advance();
//!
//! let (offset, ptr) = frame_alloc.push_bytes(256, 256)?;
//! // Write data through ptr, bind buffer at offset.
//! # Ok(())
//! # }
//! ```

use std::sync::Arc;

use ash::vk;

use crate::device::SharedState;
use crate::error::{Error, Result};
use super::allocator::{Allocation, Allocator};
use super::resources::MemoryLocation;

/// A single frame's bump-allocated buffer.
struct FrameBuffer {
    buffer: vk::Buffer,
    memory: Allocation,
    mapped: *mut u8,
    capacity: vk::DeviceSize,
    cursor: vk::DeviceSize,
}

unsafe impl Send for FrameBuffer {}

/// Per-frame bump allocator for transient GPU data.
///
/// Created via [`Ignis::create_frame_allocator`](crate::Ignis::create_frame_allocator).
pub struct FrameAllocator {
    shared: Arc<SharedState>,
    allocator: Arc<dyn Allocator>,
    frames: Vec<FrameBuffer>,
    current: usize,
}

impl FrameAllocator {
    /// Create a new frame allocator.
    pub fn new(
        shared: Arc<SharedState>,
        allocator: Arc<dyn Allocator>,
        capacity: vk::DeviceSize,
        frames_in_flight: u32,
        usage: vk::BufferUsageFlags,
    ) -> Result<Self> {
        let mut frames = Vec::with_capacity(frames_in_flight as usize);
        for _ in 0..frames_in_flight {
            let ci = vk::BufferCreateInfo::default()
                .size(capacity)
                .usage(usage)
                .sharing_mode(vk::SharingMode::EXCLUSIVE);
            let buffer = unsafe { shared.device.create_buffer(&ci, None)? };
            let req = unsafe { shared.device.get_buffer_memory_requirements(buffer) };
            let memory = allocator.allocate(&req, MemoryLocation::CpuToGpu)?;
            unsafe {
                shared
                    .device
                    .bind_buffer_memory(buffer, memory.memory, memory.offset)?;
            }
            let mapped = memory.mapped_ptr.ok_or(Error::NoSuitableMemoryType)?;
            frames.push(FrameBuffer {
                buffer,
                memory,
                mapped,
                capacity,
                cursor: 0,
            });
        }
        Ok(Self {
            shared,
            allocator,
            frames,
            current: 0,
        })
    }

    /// Advance to the next frame slot and reset its cursor.
    pub fn advance(&mut self) {
        self.current = (self.current + 1) % self.frames.len();
        self.frames[self.current].cursor = 0;
    }

    /// Push raw bytes with the given alignment.
    ///
    /// Returns `(offset, mapped_pointer)` within the current frame's buffer.
    pub fn push_bytes(
        &mut self,
        size: vk::DeviceSize,
        alignment: vk::DeviceSize,
    ) -> Result<(vk::DeviceSize, *mut u8)> {
        let frame = &mut self.frames[self.current];
        let alignment = alignment.max(1);
        let aligned = (frame.cursor + alignment - 1) & !(alignment - 1);
        if aligned + size > frame.capacity {
            return Err(Error::InvalidConfig("frame allocator capacity exceeded"));
        }
        let ptr = unsafe { frame.mapped.add(aligned as usize) };
        frame.cursor = aligned + size;
        Ok((aligned, ptr))
    }

    /// Push a typed value. Returns the byte offset within the buffer.
    ///
    /// # Safety
    ///
    /// `T` must be a plain-old-data type safe to transmit as raw bytes.
    pub unsafe fn push<T: Copy>(&mut self, value: &T) -> Result<vk::DeviceSize> {
        let size = std::mem::size_of::<T>() as vk::DeviceSize;
        let alignment = std::mem::align_of::<T>() as vk::DeviceSize;
        let (offset, ptr) = self.push_bytes(size, alignment)?;
        std::ptr::copy_nonoverlapping(value as *const T as *const u8, ptr, size as usize);
        Ok(offset)
    }

    /// The buffer handle for the current frame.
    #[inline]
    pub fn buffer(&self) -> vk::Buffer {
        self.frames[self.current].buffer
    }

    /// Remaining capacity in the current frame.
    #[inline]
    pub fn remaining(&self) -> vk::DeviceSize {
        let f = &self.frames[self.current];
        f.capacity.saturating_sub(f.cursor)
    }
}

impl Drop for FrameAllocator {
    fn drop(&mut self) {
        for frame in &self.frames {
            unsafe {
                self.shared.device.destroy_buffer(frame.buffer, None);
            }
            self.allocator.free(&frame.memory);
        }
    }
}