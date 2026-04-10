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

/// Production hardened slab allocator.
#[cfg(feature = "slab-allocator")]
pub mod slab;