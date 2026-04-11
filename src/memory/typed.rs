//! Type-safe buffer wrapper.
//!
//! [`TypedBuffer<T>`] wraps a [`Buffer`] with element-level access,
//! bounds checking, and ergonomic read/write methods.
//!
//! # Example
//!
//! ```rust,no_run
//! # use ignis::*; use ash::vk;
//! # fn example(ignis: &Ignis) -> Result<()> {
//! let buf: TypedBuffer<[f32; 4]> = ignis.create_typed_buffer(
//!     64,
//!     vk::BufferUsageFlags::UNIFORM_BUFFER,
//!     MemoryLocation::CpuToGpu,
//! )?;
//! buf.write(0, &[1.0, 2.0, 3.0, 4.0]);
//! # Ok(())
//! # }
//! ```

use std::marker::PhantomData;
use std::sync::Arc;

use ash::vk;

use super::resources::{Buffer, BufferInfo, MemoryLocation};
use crate::device::SharedState;
use crate::error::Result;
use crate::memory::allocator::Allocator;

/// A buffer with typed element access.
pub struct TypedBuffer<T: Copy + Send> {
    inner: Buffer,
    element_count: usize,
    _marker: PhantomData<T>,
}

impl<T: Copy + Send> TypedBuffer<T> {
    /// Create a typed buffer using the given allocator.
    pub fn new(
        shared: Arc<SharedState>,
        allocator: Arc<dyn Allocator>,
        element_count: usize,
        usage: vk::BufferUsageFlags,
        location: MemoryLocation,
    ) -> Result<Self> {
        let byte_size = (element_count * std::mem::size_of::<T>()) as vk::DeviceSize;
        let info = BufferInfo {
            size: byte_size,
            usage,
            location,
            sharing_mode: vk::SharingMode::EXCLUSIVE,
        };
        let inner = Buffer::new(shared, allocator, &info)?;
        Ok(Self {
            inner,
            element_count,
            _marker: PhantomData,
        })
    }

    /// Number of `T` elements this buffer can hold.
    #[inline]
    pub fn element_count(&self) -> usize {
        self.element_count
    }

    /// Size in bytes.
    #[inline]
    pub fn byte_size(&self) -> vk::DeviceSize {
        self.inner.size()
    }

    /// Write a single element at the given index.
    ///
    /// # Panics
    ///
    /// Panics if the buffer is not host-visible or if `index >= element_count`.
    pub fn write(&self, index: usize, value: &T) {
        assert!(index < self.element_count, "index out of bounds");
        let offset = index * std::mem::size_of::<T>();
        let bytes = unsafe {
            std::slice::from_raw_parts(value as *const T as *const u8, std::mem::size_of::<T>())
        };
        self.inner.write(offset, bytes);
    }

    /// Write a slice of elements starting at the given index.
    ///
    /// # Panics
    ///
    /// Panics if the write would exceed bounds or the buffer is not mapped.
    pub fn write_slice(&self, start_index: usize, values: &[T]) {
        assert!(
            start_index + values.len() <= self.element_count,
            "write_slice exceeds buffer bounds"
        );
        let offset = start_index * std::mem::size_of::<T>();
        let bytes = unsafe {
            std::slice::from_raw_parts(
                values.as_ptr() as *const u8,
                values.len() * std::mem::size_of::<T>(),
            )
        };
        self.inner.write(offset, bytes);
    }

    /// Read a single element at the given index.
    ///
    /// # Panics
    ///
    /// Panics if the buffer is not host-visible or if `index >= element_count`.
    pub fn read(&self, index: usize) -> T {
        assert!(index < self.element_count, "index out of bounds");
        let slice = self.inner.mapped_slice().expect("buffer is not host-visible");
        let offset = index * std::mem::size_of::<T>();
        unsafe { (slice.as_ptr().add(offset) as *const T).read() }
    }

    /// Access the underlying buffer.
    #[inline]
    pub fn buffer(&self) -> &Buffer {
        &self.inner
    }

    /// Raw buffer handle.
    #[inline]
    pub fn handle(&self) -> vk::Buffer {
        self.inner.handle()
    }
}