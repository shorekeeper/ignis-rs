//! GPU resource wrappers using the allocator system.
//!
//! [`Buffer`] and [`Image`] combine a Vulkan resource handle with an
//! allocation from an [`Allocator`]. They automatically destroy the
//! resource and free the allocation on drop.
//!
//! # Allocator Ownership
//!
//! Each resource holds an `Arc<dyn Allocator>`, keeping the allocator
//! alive as long as any resource exists. This means you can drop the
//! [`Ignis`](crate::Ignis) context or the allocator itself while
//! resources are still alive - the allocator's memory blocks will be
//! freed when the last resource is dropped.

use std::sync::Arc;

use ash::vk;

#[cfg(feature = "tracking")]
use crate::tracking::deletion::{DeletionQueue, DeletionGuard};
use super::allocator::{Allocation, Allocator};
use crate::device::SharedState;
use crate::error::{Error, Result};

/// Desired memory placement for an allocation.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum MemoryLocation {
    /// Device-local memory, not host-accessible.
    /// Best for static meshes, textures, and render targets.
    GpuOnly,
    /// Host-visible, coherent. Prefers device-local (ReBAR/SAM).
    /// Ideal for uniform buffers, staging data, or frequently updated geometry.
    CpuToGpu,
    /// Host-visible, coherent. For GPU -> CPU readback.
    GpuToCpu,
}

impl MemoryLocation {
    /// Required memory property flags.
    #[inline]
    pub fn required_flags(self) -> vk::MemoryPropertyFlags {
        match self {
            Self::GpuOnly => vk::MemoryPropertyFlags::DEVICE_LOCAL,
            Self::CpuToGpu | Self::GpuToCpu => {
                vk::MemoryPropertyFlags::HOST_VISIBLE | vk::MemoryPropertyFlags::HOST_COHERENT
            }
        }
    }

    /// Additional preferred flags for better performance.
    #[inline]
    pub fn preferred_flags(self) -> vk::MemoryPropertyFlags {
        match self {
            Self::CpuToGpu => vk::MemoryPropertyFlags::DEVICE_LOCAL,
            _ => vk::MemoryPropertyFlags::empty(),
        }
    }
}

/// Configuration for creating a [`Buffer`].
#[derive(Debug, Clone)]
pub struct BufferInfo {
    /// Size in bytes.
    pub size: vk::DeviceSize,
    /// Vulkan buffer usage flags.
    pub usage: vk::BufferUsageFlags,
    /// Desired memory placement.
    pub location: MemoryLocation,
    /// Sharing mode.
    pub sharing_mode: vk::SharingMode,
}

impl BufferInfo {
    /// Vertex buffer.
    pub fn vertex(size: vk::DeviceSize, location: MemoryLocation) -> Self {
        Self {
            size,
            usage: vk::BufferUsageFlags::VERTEX_BUFFER,
            location,
            sharing_mode: vk::SharingMode::EXCLUSIVE,
        }
    }

    /// Index buffer.
    pub fn index(size: vk::DeviceSize, location: MemoryLocation) -> Self {
        Self {
            size,
            usage: vk::BufferUsageFlags::INDEX_BUFFER,
            location,
            sharing_mode: vk::SharingMode::EXCLUSIVE,
        }
    }

    /// Uniform buffer (host-visible for frequent updates).
    pub fn uniform(size: vk::DeviceSize) -> Self {
        Self {
            size,
            usage: vk::BufferUsageFlags::UNIFORM_BUFFER,
            location: MemoryLocation::CpuToGpu,
            sharing_mode: vk::SharingMode::EXCLUSIVE,
        }
    }

    /// Staging (transfer source) buffer.
    pub fn staging(size: vk::DeviceSize) -> Self {
        Self {
            size,
            usage: vk::BufferUsageFlags::TRANSFER_SRC,
            location: MemoryLocation::CpuToGpu,
            sharing_mode: vk::SharingMode::EXCLUSIVE,
        }
    }

    /// Storage buffer.
    pub fn storage(size: vk::DeviceSize, location: MemoryLocation) -> Self {
        Self {
            size,
            usage: vk::BufferUsageFlags::STORAGE_BUFFER,
            location,
            sharing_mode: vk::SharingMode::EXCLUSIVE,
        }
    }
}

/// An owned Vulkan buffer backed by an allocator.
///
/// Manages the `VkBuffer` handle and memory allocation as a unit.
/// On drop, destroys the buffer and returns the allocation to the
/// allocator.
///
/// If the memory is host-visible, the allocation provides a persistently
/// mapped pointer accessible through [`mapped_ptr`](Buffer::mapped_ptr),
/// [`write`](Buffer::write), and [`write_struct`](Buffer::write_struct).
pub struct Buffer {
    shared: Arc<SharedState>,
    allocator: Arc<dyn Allocator>,
    handle: vk::Buffer,
    allocation: Allocation,
    size: vk::DeviceSize,
}

// SAFETY: Buffer contains Arc (Send+Sync), Vulkan handles (opaque u64),
// and an Allocation whose raw pointer is externally synchronized by the
// user (same as any Vulkan mapped memory).
unsafe impl Send for Buffer {}
unsafe impl Sync for Buffer {}

impl Buffer {
    /// Create a buffer using the given allocator.
    pub fn new(
        shared: Arc<SharedState>,
        allocator: Arc<dyn Allocator>,
        info: &BufferInfo,
    ) -> Result<Self> {
        let buffer_ci = vk::BufferCreateInfo::default()
            .size(info.size)
            .usage(info.usage)
            .sharing_mode(info.sharing_mode);

        let handle = unsafe { shared.device.create_buffer(&buffer_ci, None)? };
        let mem_req = unsafe { shared.device.get_buffer_memory_requirements(handle) };

        let allocation = match allocator.allocate(&mem_req, info.location) {
            Ok(a) => a,
            Err(e) => {
                unsafe { shared.device.destroy_buffer(handle, None) };
                return Err(e);
            }
        };

        if let Err(e) = unsafe {
            shared
                .device
                .bind_buffer_memory(handle, allocation.memory, allocation.offset)
        } {
            allocator.free(&allocation);
            unsafe { shared.device.destroy_buffer(handle, None) };
            return Err(Error::Vulkan(e));
        }

        Ok(Self {
            shared,
            allocator,
            handle,
            allocation,
            size: info.size,
        })
    }

    /// Raw buffer handle.
    #[inline]
    pub fn handle(&self) -> vk::Buffer {
        self.handle
    }

    /// Buffer size in bytes.
    #[inline]
    pub fn size(&self) -> vk::DeviceSize {
        self.size
    }

    /// Raw allocation memory handle.
    #[inline]
    pub fn memory(&self) -> vk::DeviceMemory {
        self.allocation.memory
    }

    /// Byte offset of this buffer within its memory object.
    #[inline]
    pub fn memory_offset(&self) -> vk::DeviceSize {
        self.allocation.offset
    }

    /// Returns the mapped pointer if the memory is host-visible.
    ///
    /// # Safety
    ///
    /// The caller must not create data races between CPU writes and GPU
    /// reads. Use fences or frame synchronization.
    #[inline]
    pub fn mapped_ptr(&self) -> Option<*mut u8> {
        self.allocation.mapped_ptr
    }

    /// Returns a read-only byte slice of the mapped memory region.
    ///
    /// Returns `None` if the buffer is not host-visible.
    ///
    /// # Aliasing Note
    ///
    /// This returns `&[u8]` through `&self`. Do not hold the returned
    /// slice across a [`write`](Buffer::write) call on the same buffer,
    /// as that constitutes aliased access to the same memory region.
    /// Use the slice, drop it, then write.
    pub fn mapped_slice(&self) -> Option<&[u8]> {
        self.allocation
            .mapped_ptr
            .map(|ptr| unsafe { std::slice::from_raw_parts(ptr, self.size as usize) })
    }

    /// Write bytes at the given offset.
    ///
    /// # Panics
    ///
    /// Panics if the buffer is not host-visible or if the write
    /// would exceed the buffer bounds.
    pub fn write(&self, offset: usize, data: &[u8]) {
        let ptr = self
            .allocation
            .mapped_ptr
            .expect("buffer is not host-visible");
        assert!(
            offset + data.len() <= self.size as usize,
            "write exceeds buffer bounds"
        );
        unsafe {
            std::ptr::copy_nonoverlapping(data.as_ptr(), ptr.add(offset), data.len());
        }
    }

    /// Write a typed value at byte offset zero.
    ///
    /// # Safety
    ///
    /// `T` must be a plain-old-data type safe to transmit as raw bytes.
    /// The buffer must be host-visible and large enough.
    pub unsafe fn write_struct<T: Copy>(&self, value: &T) {
        let bytes =
            std::slice::from_raw_parts(value as *const T as *const u8, std::mem::size_of::<T>());
        self.write(0, bytes);
    }

    /// Get the device address of this buffer.
    ///
    /// Requires `SHADER_DEVICE_ADDRESS` in usage flags and Vulkan 1.2+.
    pub fn device_address(&self) -> vk::DeviceAddress {
        let info = vk::BufferDeviceAddressInfo::default().buffer(self.handle);
        unsafe { self.shared.device.get_buffer_device_address(&info) }
    }

    /// Retire this buffer into a deletion queue for deferred destruction.
    ///
    /// The `guard` determines when it is safe to actually destroy the
    /// buffer (e.g., after a timeline semaphore reaches a specific value).
    ///
    /// This consumes the buffer without calling Drop.
    #[cfg(feature = "tracking")]
    pub fn retire(self, dq: &DeletionQueue, guard: DeletionGuard) {
        let handle = self.handle;
        let allocator = Arc::clone(&self.allocator);
        let allocation = self.allocation.clone();
        std::mem::forget(self);
        dq.retire_buffer_after(handle, Some((allocator, allocation)), guard);
    }

    /// Convert into raw parts without destroying. The caller assumes
    /// responsibility for eventually destroying the buffer and freeing
    /// the allocation.
    #[cfg(feature = "tracking")]
    pub fn into_raw(self) -> (vk::Buffer, Arc<dyn Allocator>, Allocation) {
        let handle = self.handle;
        let allocator = Arc::clone(&self.allocator);
        let allocation = self.allocation.clone();
        std::mem::forget(self);
        (handle, allocator, allocation)
    }
}

impl Drop for Buffer {
    fn drop(&mut self) {
        unsafe {
            self.shared.device.destroy_buffer(self.handle, None);
        }
        self.allocator.free(&self.allocation);
    }
}

/// Configuration for creating an [`Image`].
#[derive(Debug, Clone)]
pub struct ImageInfo {
    /// Image dimensions.
    pub extent: vk::Extent3D,
    /// Pixel format.
    pub format: vk::Format,
    /// Usage flags.
    pub usage: vk::ImageUsageFlags,
    /// Memory placement.
    pub location: MemoryLocation,
    /// Image type.
    pub image_type: vk::ImageType,
    /// Mip levels.
    pub mip_levels: u32,
    /// Array layers.
    pub array_layers: u32,
    /// Sample count.
    pub samples: vk::SampleCountFlags,
    /// Tiling mode.
    pub tiling: vk::ImageTiling,
    /// Initial layout.
    pub initial_layout: vk::ImageLayout,
}

impl Default for ImageInfo {
    fn default() -> Self {
        Self {
            extent: vk::Extent3D {
                width: 1,
                height: 1,
                depth: 1,
            },
            format: vk::Format::R8G8B8A8_UNORM,
            usage: vk::ImageUsageFlags::SAMPLED,
            location: MemoryLocation::GpuOnly,
            image_type: vk::ImageType::TYPE_2D,
            mip_levels: 1,
            array_layers: 1,
            samples: vk::SampleCountFlags::TYPE_1,
            tiling: vk::ImageTiling::OPTIMAL,
            initial_layout: vk::ImageLayout::UNDEFINED,
        }
    }
}

impl ImageInfo {
    /// 2D texture.
    pub fn texture_2d(
        width: u32,
        height: u32,
        format: vk::Format,
        usage: vk::ImageUsageFlags,
    ) -> Self {
        Self {
            extent: vk::Extent3D {
                width,
                height,
                depth: 1,
            },
            format,
            usage,
            ..Default::default()
        }
    }

    /// Depth attachment.
    pub fn depth(width: u32, height: u32, format: vk::Format) -> Self {
        Self {
            extent: vk::Extent3D {
                width,
                height,
                depth: 1,
            },
            format,
            usage: vk::ImageUsageFlags::DEPTH_STENCIL_ATTACHMENT,
            ..Default::default()
        }
    }
}

/// An owned Vulkan image backed by an allocator.
///
/// Does NOT create image views automatically. Use
/// [`create_view`](Image::create_view) or create views through ash.
pub struct Image {
    shared: Arc<SharedState>,
    allocator: Arc<dyn Allocator>,
    handle: vk::Image,
    allocation: Allocation,
    extent: vk::Extent3D,
    format: vk::Format,
    mip_levels: u32,
    array_layers: u32,
}

unsafe impl Send for Image {}
unsafe impl Sync for Image {}

impl Image {
    /// Create an image using the given allocator.
    pub fn new(
        shared: Arc<SharedState>,
        allocator: Arc<dyn Allocator>,
        info: &ImageInfo,
    ) -> Result<Self> {
        let image_ci = vk::ImageCreateInfo::default()
            .image_type(info.image_type)
            .format(info.format)
            .extent(info.extent)
            .mip_levels(info.mip_levels)
            .array_layers(info.array_layers)
            .samples(info.samples)
            .tiling(info.tiling)
            .usage(info.usage)
            .sharing_mode(vk::SharingMode::EXCLUSIVE)
            .initial_layout(info.initial_layout);

        let handle = unsafe { shared.device.create_image(&image_ci, None)? };
        let mem_req = unsafe { shared.device.get_image_memory_requirements(handle) };

        let allocation = match allocator.allocate(&mem_req, info.location) {
            Ok(a) => a,
            Err(e) => {
                unsafe { shared.device.destroy_image(handle, None) };
                return Err(e);
            }
        };

        if let Err(e) = unsafe {
            shared
                .device
                .bind_image_memory(handle, allocation.memory, allocation.offset)
        } {
            allocator.free(&allocation);
            unsafe { shared.device.destroy_image(handle, None) };
            return Err(Error::Vulkan(e));
        }

        Ok(Self {
            shared,
            allocator,
            handle,
            allocation,
            extent: info.extent,
            format: info.format,
            mip_levels: info.mip_levels,
            array_layers: info.array_layers,
        })
    }

    /// Raw image handle.
    #[inline]
    pub fn handle(&self) -> vk::Image {
        self.handle
    }

    /// Image format.
    #[inline]
    pub fn format(&self) -> vk::Format {
        self.format
    }

    /// Image extent.
    #[inline]
    pub fn extent(&self) -> vk::Extent3D {
        self.extent
    }

    /// Number of mip levels.
    #[inline]
    pub fn mip_levels(&self) -> u32 {
        self.mip_levels
    }

    /// Number of array layers.
    #[inline]
    pub fn array_layers(&self) -> u32 {
        self.array_layers
    }

    /// Create a simple full-range `VkImageView`.
    pub fn create_view(&self, aspect: vk::ImageAspectFlags) -> Result<vk::ImageView> {
        let view_type = if self.array_layers > 1 {
            vk::ImageViewType::TYPE_2D_ARRAY
        } else {
            vk::ImageViewType::TYPE_2D
        };

        let ci = vk::ImageViewCreateInfo::default()
            .image(self.handle)
            .view_type(view_type)
            .format(self.format)
            .subresource_range(vk::ImageSubresourceRange {
                aspect_mask: aspect,
                base_mip_level: 0,
                level_count: self.mip_levels,
                base_array_layer: 0,
                layer_count: self.array_layers,
            });

        let view = unsafe { self.shared.device.create_image_view(&ci, None)? };
        Ok(view)
    }

    /// Retire this image into a deletion queue.
    #[cfg(feature = "tracking")]
    pub fn retire(self, dq: &DeletionQueue, guard: DeletionGuard) {
        let handle = self.handle;
        let allocator = Arc::clone(&self.allocator);
        let allocation = self.allocation.clone();
        std::mem::forget(self);
        dq.retire_image_after(handle, Some((allocator, allocation)), guard);
    }

    /// Convert into raw parts.
    #[cfg(feature = "tracking")]
    pub fn into_raw(self) -> (vk::Image, Arc<dyn Allocator>, Allocation) {
        let handle = self.handle;
        let allocator = Arc::clone(&self.allocator);
        let allocation = self.allocation.clone();
        std::mem::forget(self);
        (handle, allocator, allocation)
    }
}

impl Drop for Image {
    fn drop(&mut self) {
        unsafe {
            self.shared.device.destroy_image(self.handle, None);
        }
        self.allocator.free(&self.allocation);
    }
}
