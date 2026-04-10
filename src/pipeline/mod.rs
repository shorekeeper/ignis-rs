//! Pipeline builders, render passes, and descriptor management.
//!
//! Always available:
//! - [`builders`]: Graphics, compute, and ray tracing pipeline builders
//! - [`renderpass`]: Render pass builder with attachments/subpasses/dependencies
//!
//! Feature-gated:
//! - [`descriptor`]: Descriptor set layout/pool builders, arena, ring buffer
//!   (feature `descriptors`)

/// Graphics, compute, and ray tracing pipeline builders.
pub mod builders;

/// Render pass construction.
pub mod renderpass;

/// Descriptor set layout, pool, arena, and ring buffer builders.
#[cfg(feature = "descriptors")]
pub mod descriptor;