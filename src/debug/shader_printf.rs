//! VK_EXT_debug_printf integration.
//!
//! Enables `debugPrintfEXT(...)` calls in shaders and routes their output
//! through the same diagnostic styling the rest of the crate uses.
//!
//! # Requirements
//!
//! - Instance created with `VK_EXT_debug_utils`
//! - Device created with `VK_KHR_shader_non_semantic_info`
//! - Validation layer loaded with `VK_LAYER_KHRONOS_validation`
//! - Layer setting `khronos_validation.enables = VK_VALIDATION_FEATURE_ENABLE_DEBUG_PRINTF_EXT`
//!
//! All of the above are wired automatically when `ManagedConfig::enable_shader_printf(true)`
//! is set. External mode users must set them up themselves.
//!
//! # GLSL usage
//!
//! ```glsl
//! #extension GL_EXT_debug_printf : enable
//! debugPrintfEXT("density=%f at voxel=%v3i", d, vox);
//! ```

use std::sync::Mutex;

use ash::vk;

/// A parsed printf message from a shader.
#[derive(Debug, Clone)]
pub struct ShaderPrintfMessage {
    /// Shader stage that emitted the print, or "unknown" if not deducible.
    pub shader_stage: &'static str,
    /// Source location if the validation layer included it.
    pub location: Option<String>,
    /// The formatted message body as the shader produced it.
    pub formatted: String,
    /// Raw message id number from the validation layer (for correlation).
    pub message_id: i32,
}

/// Callback type for shader printf messages.
pub type ShaderPrintfHandler = Box<dyn Fn(&ShaderPrintfMessage) + Send + Sync>;

/// Global registration point for the printf handler.
///
/// Stored globally because the Vulkan debug utils callback is a C function
/// pointer with a user_data pointer; we keep a single process-wide Mutex
/// rather than threading user_data through the instance.
pub(crate) struct PrintfRegistry {
    handler: Mutex<Option<ShaderPrintfHandler>>,
}

impl PrintfRegistry {
    const fn new() -> Self {
        Self {
            handler: Mutex::new(None),
        }
    }

    pub(crate) fn set(&self, handler: ShaderPrintfHandler) {
        *self.handler.lock().unwrap() = Some(handler);
    }

    pub(crate) fn clear(&self) {
        *self.handler.lock().unwrap() = None;
    }

    pub(crate) fn dispatch(&self, msg: &ShaderPrintfMessage) {
        if let Some(h) = self.handler.lock().unwrap().as_ref() {
            h(msg);
        }
    }
}

pub static PRINTF_REGISTRY: PrintfRegistry = PrintfRegistry::new();

/// Attempt to parse a validation layer message that originated from
/// `debugPrintfEXT`. Returns `None` if the message is not a printf.
///
/// Different Vulkan SDK versions use different message id names for
/// printf: older SDKs emit `UNASSIGNED-DEBUG-PRINTF`, newer ones emit
/// `WARNING-DEBUG-PRINTF`. Both the id name and the raw message body are
/// checked so the parser stays stable across SDK upgrades.
pub(crate) fn try_parse_printf(
    raw: &str,
    message_id: i32,
    message_id_name: &str,
) -> Option<ShaderPrintfMessage> {
    let is_printf = message_id_name.contains("DEBUG-PRINTF")
        || message_id_name.contains("DebugPrintf")
        || raw.contains("DEBUG-PRINTF");
    if !is_printf {
        return None;
    }

    // The layer emits lines like:
    //   "Object 0: ... | MessageID = 0x... | <actual printf body>"
    // The body we want is everything after the last " | " separator.
    let payload = raw.rsplit_once(" | ").map(|(_, t)| t).unwrap_or(raw);
    let payload = payload.trim();

    // Deduce stage from object labels the layer usually dumps.
    let stage = if raw.contains("VERTEX") {
        "VERTEX"
    } else if raw.contains("FRAGMENT") {
        "FRAGMENT"
    } else if raw.contains("COMPUTE") {
        "COMPUTE"
    } else if raw.contains("RAYGEN") {
        "RAYGEN"
    } else if raw.contains("MISS") {
        "MISS"
    } else if raw.contains("CLOSEST_HIT") {
        "CLOSEST_HIT"
    } else {
        "unknown"
    };

    let location = raw
        .find("Shader handle")
        .map(|i| raw[i..].lines().next().unwrap_or("").trim().to_string());

    Some(ShaderPrintfMessage {
        shader_stage: stage,
        location,
        formatted: payload.to_string(),
        message_id,
    })
}

/// Format a shader printf message using the standard diagnostic style.
pub fn format_message(msg: &ShaderPrintfMessage) -> String {
    let s = crate::diagnostic::Style::detect();
    let mut o = String::with_capacity(256);
    let header = format!(
        " {} {} {}",
        s.bold_cyan("[shader]"),
        s.bold(&format!("stage={}", msg.shader_stage)),
        s.bright_white(&msg.formatted),
    );
    o.push_str(&header);
    o.push('\n');
    if let Some(loc) = &msg.location {
        o.push_str(&format!("   {}\n", s.dim(loc)));
    }
    o
}

/// The device extension string that must be enabled for debugPrintfEXT.
pub const SHADER_NON_SEMANTIC_INFO_EXT: &std::ffi::CStr = ash::khr::shader_non_semantic_info::NAME;

/// The validation layer feature enable flag value for debug printf.
/// Equals `VK_VALIDATION_FEATURE_ENABLE_DEBUG_PRINTF_EXT`.
pub const VALIDATION_FEATURE_ENABLE_DEBUG_PRINTF: vk::ValidationFeatureEnableEXT =
    vk::ValidationFeatureEnableEXT::DEBUG_PRINTF;
