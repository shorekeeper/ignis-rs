//! Staging ring buffer for efficient CPU→GPU uploads.
//!
//! [`StagingRing`] manages a ring of staging buffers with per-frame
//! fence tracking. Data is written to the ring, copy commands are
//! recorded, and the ring advances each frame to reclaim old regions.
//!
//! # Example
//!
//! ```rust,no_run
//! # use ignis::*; use ash::vk;
//! # fn example(ignis: &Ignis, queue: &AsyncQueue,
//! #            pool: &CommandPool, dst: &Buffer) -> Result<()> {
//! let mut ring = ignis.create_staging_ring(4 * 1024 * 1024, 2)?;
//!
//! // Each frame:
//! ring.begin_frame()?;
//!
//! let data = [1.0f32, 2.0, 3.0, 4.0];
//! let bytes = bytemuck_or_manual_cast(&data);
//! let region = ring.push(bytes)?;
//!
//! let cmd = pool.allocate_primary()?;
//! let rec = pool.begin_primary(cmd)?;
//! rec.copy_buffer(
//!     region.buffer,
//!     dst.handle(),
//!     &[vk::BufferCopy {
//!         src_offset: region.offset,
//!         dst_offset: 0,
//!         size: region.size,
//!     }],
//! );
//! let cmd = rec.end()?;
//! queue.submit_simple(cmd)?.wait()?;
//! # Ok(())
//! # }
//! ```

use std::sync::Arc;

use ash::vk;

use crate::device::SharedState;
use crate::error::{Error, Result};
use super::allocator::{Allocation, Allocator};
use super::resources::MemoryLocation;

/// A region within the staging ring containing uploaded data.
#[derive(Debug, Clone, Copy)]
pub struct StagingRegion {
    /// The staging buffer handle (for use in copy commands).
    pub buffer: vk::Buffer,
    /// Byte offset within the buffer.
    pub offset: vk::DeviceSize,
    /// Size in bytes.
    pub size: vk::DeviceSize,
}

/// A single frame's staging buffer with bump allocation.
struct StagingFrame {
    buffer: vk::Buffer,
    memory: Allocation,
    mapped: *mut u8,
    capacity: vk::DeviceSize,
    cursor: vk::DeviceSize,
}

unsafe impl Send for StagingFrame {}

/// Per-frame ring of staging buffers.
///
/// Created via [`Ignis::create_staging_ring`](crate::Ignis::create_staging_ring).
pub struct StagingRing {
    shared: Arc<SharedState>,
    allocator: Arc<dyn Allocator>,
    frames: Vec<StagingFrame>,
    current_frame: usize,
    frame_capacity: vk::DeviceSize,
}

impl StagingRing {
    /// Create a new staging ring.
    ///
    /// # Arguments
    ///
    /// * `shared` - Device state
    /// * `allocator` - Backing allocator for staging buffers
    /// * `frame_capacity` - Bytes per frame's staging buffer
    /// * `frames_in_flight` - Number of frame slots
    pub fn new(
        shared: Arc<SharedState>,
        allocator: Arc<dyn Allocator>,
        frame_capacity: vk::DeviceSize,
        frames_in_flight: u32,
    ) -> Result<Self> {
        let mut frames = Vec::with_capacity(frames_in_flight as usize);
        for _ in 0..frames_in_flight {
            frames.push(Self::create_frame(&shared, &allocator, frame_capacity)?);
        }
        Ok(Self {
            shared,
            allocator,
            frames,
            current_frame: 0,
            frame_capacity,
        })
    }

    fn create_frame(
        shared: &SharedState,
        allocator: &Arc<dyn Allocator>,
        capacity: vk::DeviceSize,
    ) -> Result<StagingFrame> {
        let ci = vk::BufferCreateInfo::default()
            .size(capacity)
            .usage(vk::BufferUsageFlags::TRANSFER_SRC)
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
        Ok(StagingFrame {
            buffer,
            memory,
            mapped,
            capacity,
            cursor: 0,
        })
    }

    /// Reset the current frame's cursor. Call at the start of each frame
    /// after the corresponding GPU work from `frames_in_flight` ago has
    /// completed.
    pub fn begin_frame(&mut self) -> Result<()> {
        self.current_frame = (self.current_frame + 1) % self.frames.len();
        self.frames[self.current_frame].cursor = 0;
        Ok(())
    }

    /// Push data into the staging ring, returning a region descriptor
    /// for use in copy commands.
    ///
    /// # Errors
    ///
    /// Returns [`Error::InvalidConfig`] if the data exceeds the frame capacity.
    pub fn push(&mut self, data: &[u8]) -> Result<StagingRegion> {
        let size = data.len() as vk::DeviceSize;
        let frame = &mut self.frames[self.current_frame];

        // Align to 16 bytes for transfer efficiency.
        let aligned_cursor = (frame.cursor + 15) & !15;

        if aligned_cursor + size > frame.capacity {
            return Err(Error::InvalidConfig(
                "staging ring frame capacity exceeded",
            ));
        }

        unsafe {
            std::ptr::copy_nonoverlapping(
                data.as_ptr(),
                frame.mapped.add(aligned_cursor as usize),
                data.len(),
            );
        }

        let region = StagingRegion {
            buffer: frame.buffer,
            offset: aligned_cursor,
            size,
        };

        frame.cursor = aligned_cursor + size;
        Ok(region)
    }

    /// Remaining bytes in the current frame.
    pub fn remaining(&self) -> vk::DeviceSize {
        let frame = &self.frames[self.current_frame];
        frame.capacity.saturating_sub(frame.cursor)
    }

    /// Total capacity per frame.
    pub fn frame_capacity(&self) -> vk::DeviceSize {
        self.frame_capacity
    }
}

impl Drop for StagingRing {
    fn drop(&mut self) {
        for frame in &self.frames {
            unsafe {
                self.shared.device.destroy_buffer(frame.buffer, None);
            }
            self.allocator.free(&frame.memory);
        }
    }
}