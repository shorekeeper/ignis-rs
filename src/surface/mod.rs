//! Window surface and swapchain management.
//!
//! Available when the `swapchain` feature is enabled.
//!
//! The user creates the `VkSurfaceKHR` externally (via winit, SDL, raw
//! platform extensions) and passes it to [`Swapchain::new`](swapchain::Swapchain::new).

/// Swapchain lifecycle management.
pub mod swapchain;
