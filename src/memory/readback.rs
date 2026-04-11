//! Async GPU→CPU readback utility.
//!
//! [`ReadbackRequest`] bundles the staging buffer allocation, copy
//! command recording, and submission into a single ergonomic API.
//!
//! # Example
//!
//! ```rust,no_run
//! # use ignis::*; use ash::vk;
//! # fn example(ignis: &Ignis, pool: &CommandPool,
//! #            queue: &AsyncQueue, src: &Buffer) -> Result<()> {
//! let mut readback = ReadbackRequest::new(ignis.shared_state(), src.handle(), 0, 256)?;
//! let cmd = pool.allocate_primary()?;
//! let rec = pool.begin_primary(cmd)?;
//! readback.record(&rec);
//! let cmd = rec.end()?;
//! let future = queue.submit_simple(cmd)?;
//! future.wait()?;
//! let data: &[u8] = readback.data();
//! # Ok(())
//! # }
//! ```

use std::sync::Arc;

use ash::vk;

use super::allocator::{Allocation, Allocator, BlockAllocator};
use super::resources::MemoryLocation;
use crate::command::CommandRecorder;
use crate::device::SharedState;
use crate::error::Result;

/// A prepared readback from a GPU buffer to CPU memory.
pub struct ReadbackRequest {
    shared: Arc<SharedState>,
    allocator: Arc<dyn Allocator>,
    staging_buffer: vk::Buffer,
    staging_alloc: Allocation,
    src_buffer: vk::Buffer,
    src_offset: vk::DeviceSize,
    size: vk::DeviceSize,
}

impl ReadbackRequest {
    /// Prepare a readback of `size` bytes from `src_buffer` at `src_offset`.
    pub fn new(
        shared: &Arc<SharedState>,
        src_buffer: vk::Buffer,
        src_offset: vk::DeviceSize,
        size: vk::DeviceSize,
    ) -> Result<Self> {
        let allocator: Arc<dyn Allocator> = Arc::new(BlockAllocator::new(Arc::clone(shared)));
        let ci = vk::BufferCreateInfo::default()
            .size(size)
            .usage(vk::BufferUsageFlags::TRANSFER_DST)
            .sharing_mode(vk::SharingMode::EXCLUSIVE);
        let staging_buffer = unsafe { shared.device.create_buffer(&ci, None)? };
        let req = unsafe { shared.device.get_buffer_memory_requirements(staging_buffer) };
        let staging_alloc = allocator.allocate(&req, MemoryLocation::GpuToCpu)?;
        unsafe {
            shared.device.bind_buffer_memory(
                staging_buffer,
                staging_alloc.memory,
                staging_alloc.offset,
            )?;
        }
        Ok(Self {
            shared: Arc::clone(shared),
            allocator,
            staging_buffer,
            staging_alloc,
            src_buffer,
            src_offset,
            size,
        })
    }

    /// Record the copy command into the given recorder.
    pub fn record(&self, rec: &CommandRecorder<'_>) {
        rec.copy_buffer(
            self.src_buffer,
            self.staging_buffer,
            &[vk::BufferCopy {
                src_offset: self.src_offset,
                dst_offset: 0,
                size: self.size,
            }],
        );
    }

    /// Access the readback data after the GPU work has completed.
    ///
    /// # Panics
    ///
    /// Panics if the staging memory is not host-visible (should not happen).
    pub fn data(&self) -> &[u8] {
        let ptr = self
            .staging_alloc
            .mapped_ptr
            .expect("staging buffer must be host-visible");
        unsafe { std::slice::from_raw_parts(ptr, self.size as usize) }
    }
}

impl Drop for ReadbackRequest {
    fn drop(&mut self) {
        unsafe {
            self.shared.device.destroy_buffer(self.staging_buffer, None);
        }
        self.allocator.free(&self.staging_alloc);
    }
}
