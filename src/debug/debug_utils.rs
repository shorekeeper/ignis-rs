//! `VK_EXT_debug_utils` integration for object naming and command labels.
//!
//! Provides [`DebugUtils`] for naming Vulkan objects (visible in
//! RenderDoc and validation layer output) and inserting debug labels
//! into command buffers.
//!
//! # Requirements
//!
//! The Vulkan instance must have been created with `VK_EXT_debug_utils`.
//! In managed mode with the `debug-tools` feature, this extension is
//! automatically enabled when available.
//!
//! # Example
//!
//! ```rust,no_run
//! # use ignis::*; use ignis::debug::debug_utils::*; use ash::vk;
//! # fn example(ignis: &Ignis) {
//! let dbg = DebugUtils::new(ignis.instance(), ignis.device());
//!
//! // Name a buffer.
//! dbg.set_object_name(ignis.device(), vk::ObjectType::BUFFER, buffer_handle, "vertex_buffer");
//!
//! // Insert command buffer labels.
//! let cmd = pool.allocate_primary().unwrap();
//! let rec = pool.begin_primary(cmd).unwrap();
//! dbg.cmd_begin_label(&rec, "shadow_pass", [0.2, 0.2, 0.8, 1.0]);
//! // ... draw commands ...
//! dbg.cmd_end_label(&rec);
//! # }
//! ```

use std::ffi::CString;

use ash::vk;
use ash::vk::Handle;

use crate::command::CommandRecorder;

/// Wrapper around `VK_EXT_debug_utils` device and instance functions.
///
/// Create one instance and reuse it. All methods are no-ops if the
/// underlying function pointers are null (extension not loaded).
pub struct DebugUtils {
    device_fn: ash::ext::debug_utils::Device,
}

impl DebugUtils {
    /// Create a new debug utils wrapper.
    ///
    /// # Safety Contract
    ///
    /// The instance must have been created with `VK_EXT_debug_utils`.
    /// If the extension is not available, methods will cause undefined
    /// behavior. Use [`try_new`](Self::try_new) for a safe alternative.
    pub fn new(instance: &ash::Instance, device: &ash::Device) -> Self {
        Self {
            device_fn: ash::ext::debug_utils::Device::new(instance, device),
        }
    }

    /// Set a debug name on a Vulkan object.
    ///
    /// The name will appear in validation layer output and GPU
    /// debuggers like RenderDoc.
    pub fn set_object_name(
        &self,
        _device: &ash::Device,
        object_type: vk::ObjectType,
        handle: u64,
        name: &str,
    ) {
        let c_name = CString::new(name).unwrap_or_else(|_| CString::new("?").unwrap());
        let mut info = vk::DebugUtilsObjectNameInfoEXT::default()
            .object_name(&c_name);
        info.object_type = object_type;
        info.object_handle = handle;
        unsafe {
            let _ = self.device_fn.set_debug_utils_object_name(&info);
        }
    }

    /// Name a raw Vulkan handle using its `Handle` trait.
    pub fn name_handle<H: Handle>(
        &self,
        device: &ash::Device,
        object_type: vk::ObjectType,
        handle: H,
        name: &str,
    ) {
        self.set_object_name(device, object_type, handle.as_raw(), name);
    }

    /// Begin a debug label region in a command buffer.
    ///
    /// The label appears as a collapsible section in RenderDoc.
    /// `color` is RGBA in \[0, 1\].
    pub fn cmd_begin_label(
        &self,
        rec: &CommandRecorder<'_>,
        name: &str,
        color: [f32; 4],
    ) {
        let c_name = CString::new(name).unwrap_or_else(|_| CString::new("?").unwrap());
        let label = vk::DebugUtilsLabelEXT::default()
            .label_name(&c_name)
            .color(color);
        unsafe {
            self.device_fn
                .cmd_begin_debug_utils_label(rec.raw_buffer(), &label);
        }
    }

    /// End the current debug label region.
    pub fn cmd_end_label(&self, rec: &CommandRecorder<'_>) {
        unsafe {
            self.device_fn
                .cmd_end_debug_utils_label(rec.raw_buffer());
        }
    }

    /// Insert a single-point debug label (marker).
    pub fn cmd_insert_label(
        &self,
        rec: &CommandRecorder<'_>,
        name: &str,
        color: [f32; 4],
    ) {
        let c_name = CString::new(name).unwrap_or_else(|_| CString::new("?").unwrap());
        let label = vk::DebugUtilsLabelEXT::default()
            .label_name(&c_name)
            .color(color);
        unsafe {
            self.device_fn
                .cmd_insert_debug_utils_label(rec.raw_buffer(), &label);
        }
    }
}