//! Resource state tracking, deferred deletion, and completion monitoring.
//!
//! Always available (tiny, deeply integrated with queue submission):
//! - [`timeline`]: Timeline semaphore wrappers for O(1) async completion
//! - [`watcher`]: Legacy fence polling (Vulkan 1.1 fallback)
//!
//! Feature-gated:
//! - [`tracker`]: Per-subresource image layout tracking and buffer barriers
//!   (feature `tracking`)
//! - [`deletion`]: Timeline-based deferred resource destruction
//!   (feature `tracking`)

/// Timeline semaphore wrappers (Vulkan 1.2+).
pub mod timeline;

/// Legacy fence polling background thread (Vulkan 1.1 fallback).
pub mod watcher;

/// Per-subresource image and buffer state tracking with barrier computation.
#[cfg(feature = "tracking")]
pub mod tracker;

/// Timeline-based deferred GPU resource destruction.
#[cfg(feature = "tracking")]
pub mod deletion;
/// Mipmap generation utility via blit chain.
#[cfg(feature = "tracking")]
pub mod mipmap;
