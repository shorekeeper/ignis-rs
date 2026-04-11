//! Pipeline cache persistence.
//!
//! Wraps `VkPipelineCache` with disk save/load and RAII cleanup.
//!
//! # Example
//!
//! ```rust,no_run
//! # use ignis::*;
//! # fn example(ignis: &Ignis) -> Result<()> {
//! let cache = ignis.create_pipeline_cache_from_file("shader_cache.bin")?;
//! let pipeline = ignis.compute_pipeline_builder()
//!     .shader(module.handle(), "main")
//!     .layout(layout)
//!     .cache(cache.handle())
//!     .build()?;
//! cache.save("shader_cache.bin")?;
//! # Ok(())
//! # }
//! ```

use std::path::Path;
use std::sync::Arc;

use ash::vk;

use crate::device::SharedState;
use crate::error::{Error, Result};

/// An owned `VkPipelineCache` with disk persistence.
pub struct PipelineCache {
    shared: Arc<SharedState>,
    handle: vk::PipelineCache,
}

impl PipelineCache {
    /// Create an empty pipeline cache.
    pub fn new(shared: Arc<SharedState>) -> Result<Self> {
        let ci = vk::PipelineCacheCreateInfo::default();
        let handle = unsafe { shared.device.create_pipeline_cache(&ci, None)? };
        Ok(Self { shared, handle })
    }

    /// Create a pipeline cache from previously saved data.
    ///
    /// If the file does not exist or is invalid, falls back to an empty cache.
    pub fn from_file(shared: Arc<SharedState>, path: impl AsRef<Path>) -> Result<Self> {
        let data = std::fs::read(path.as_ref()).unwrap_or_default();
        let ci = vk::PipelineCacheCreateInfo::default().initial_data(&data);
        let handle = unsafe { shared.device.create_pipeline_cache(&ci, None)? };
        Ok(Self { shared, handle })
    }

    /// Save the cache data to a file.
    ///
    /// The file can be loaded later via [`from_file`](Self::from_file)
    /// to skip shader compilation on subsequent runs.
    pub fn save(&self, path: impl AsRef<Path>) -> Result<()> {
        let data = unsafe { self.shared.device.get_pipeline_cache_data(self.handle)? };
        std::fs::write(path.as_ref(), &data)
            .map_err(|_| Error::InvalidConfig("failed to write pipeline cache file"))?;
        Ok(())
    }

    /// Merge another cache into this one.
    pub fn merge(&self, other: &PipelineCache) -> Result<()> {
        unsafe {
            self.shared
                .device
                .merge_pipeline_caches(self.handle, &[other.handle])?;
        }
        Ok(())
    }

    /// Get the raw handle for use with pipeline builders.
    #[inline]
    pub fn handle(&self) -> vk::PipelineCache {
        self.handle
    }
}

impl Drop for PipelineCache {
    fn drop(&mut self) {
        unsafe {
            self.shared
                .device
                .destroy_pipeline_cache(self.handle, None);
        }
    }
}