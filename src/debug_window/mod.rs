//! Real-time CPU-rasterized debug window.
//!
//! Provides a self-contained, dependency-free (apart from `ash`) debug
//! viewer that opens its own native window, builds a Vulkan swapchain on
//! it, and renders memory layout and resource timeline panels live as
//! the host application runs.
//!
//! The window runs on a dedicated worker thread; its existence is
//! invisible to the rest of the application beyond the `DebugWindow`
//! handle. Drop the handle and the window closes itself; close the
//! window with the system close button and the handle reports
//! [`DebugWindow::is_closed`] as true.
//!
//! # Platform Support
//!
//! - **Windows**: native via `user32` / `kernel32` raw FFI. No winit, no
//!   raw-window-handle, no third-party crates.
//! - **Linux / macOS**: not implemented. `DebugWindowBuilder::open`
//!   returns [`Error::InvalidConfig`] on these platforms with a clear
//!   message. PRs welcome; the rendering pipeline (raster + font +
//!   panels) is platform-agnostic and only the window glue needs work.
//!
//! # Rendering
//!
//! No graphics pipeline. Each frame the worker thread:
//!
//! 1. Calls each registered data source's snapshot method.
//! 2. Clears a CPU-side BGRA bitmap (`Vec<u8>`).
//! 3. Renders panels into the bitmap (text, rectangles, lines).
//! 4. Copies the bitmap into a host-visible staging buffer.
//! 5. Issues `vkCmdCopyBufferToImage` from staging into the current
//!    swapchain image, with appropriate layout transitions.
//! 6. Presents.
//!
//! Pipeline-free rendering keeps total code small, eliminates an entire
//! class of validation issues, and makes the debug window run on every
//! Vulkan-capable device including software renderers.
//!
//! # Feature Gating
//!
//! Module is compiled only when the `debug-window` Cargo feature is
//! active. That feature also implies `swapchain` and `debug-tools`.
//!
//! # Example
//!
//! ```rust,no_run
//! # use ignis::*;
//! # use ignis::debug_window::*;
//! # use std::sync::Arc;
//! # fn example(ignis: &Ignis, profiler: Arc<AllocationProfiler>,
//! #            trace: Arc<ResourceTrace>) -> Result<()> {
//! let win = DebugWindow::builder()
//!     .title("Renderer Diagnostics")
//!     .size(1400, 800)
//!     .memory_source(profiler)
//!     .trace_source(trace)
//!     .open(ignis)?;
//!
//! // ... run application ...
//!
//! drop(win); // closes the window when the handle drops
//! # Ok(())
//! # }
//! ```

mod font;
mod panels;
mod raster;
mod window;

pub use self::window::{DebugWindow, DebugWindowBuilder};