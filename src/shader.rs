//! Shader module management.
//!
//! Wraps `VkShaderModule` creation and destruction.

use std::sync::Arc;

use ash::vk;

use crate::device::SharedState;
use crate::error::{Error, Result};

/// A compiled SPIR-V shader module with automatic cleanup.
///
/// Created via [`Ignis::create_shader_module`](crate::Ignis::create_shader_module).
///
/// The shader module should be kept alive until all pipelines referencing
/// it have been created. After pipeline creation, the module can be safely
/// dropped.
pub struct ShaderModule {
    shared: Arc<SharedState>,
    handle: vk::ShaderModule,
}

impl ShaderModule {
    /// Create a shader module from SPIR-V data.
    ///
    /// # Arguments
    ///
    /// * `shared` - Device state
    /// * `spirv` - Valid SPIR-V bytecode as `u32` words
    pub(crate) fn new(shared: Arc<SharedState>, spirv: &[u32]) -> Result<Self> {
        if spirv.is_empty() {
            return Err(Error::InvalidSpirv);
        }

        // Quick magic number check.
        if spirv[0] != 0x07230203 {
            return Err(Error::InvalidSpirv);
        }

        let create_info = vk::ShaderModuleCreateInfo::default().code(spirv);

        // SAFETY: spirv data is validated above, device is valid.
        let handle = unsafe { shared.device.create_shader_module(&create_info, None)? };

        Ok(Self { shared, handle })
    }

    /// Get the raw shader module handle.
    #[inline]
    pub fn handle(&self) -> vk::ShaderModule {
        self.handle
    }
}

impl Drop for ShaderModule {
    fn drop(&mut self) {
        unsafe {
            self.shared.device.destroy_shader_module(self.handle, None);
        }
    }
}
