//! GPU memory allocation and resource management.
//!
//! Always available:
//! - [`allocator`]: [`Allocator`](allocator::Allocator) trait,
//!   [`BlockAllocator`](allocator::BlockAllocator),
//!   [`DedicatedAllocator`](allocator::DedicatedAllocator)
//! - [`resources`]: RAII wrappers [`Buffer`](resources::Buffer),
//!   [`Image`](resources::Image), [`MemoryLocation`](resources::MemoryLocation)
//!
//! Feature-gated:
//! - [`slab`]: Production hardened [`SlabAllocator`](slab::SlabAllocator)
//!   (feature `slab-allocator`)

/// Core allocator trait and implementations.
pub mod allocator;

/// RAII wrappers for VkBuffer and VkImage.
pub mod resources;

/// Staging ring buffer for efficient CPU→GPU uploads.
pub mod staging;

/// Per-frame bump allocator for transient GPU data.
pub mod frame_alloc;

/// Type-safe buffer wrapper.
pub mod typed;

/// Async GPU→CPU readback utility.
pub mod readback;

/// Production hardened slab allocator.
#[cfg(feature = "slab-allocator")]
pub mod slab;
